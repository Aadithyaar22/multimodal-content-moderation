"""Moderation record storage.

MongoDB when MONGODB_URI is set, an in-memory dict otherwise. The fallback is
deliberate: the demo, the local dev loop and CI all need a working API without
a database, and failing to start because a log sink is absent would be the wrong
trade for a decision-support tool whose primary output is the verdict itself.

The in-memory store is bounded. An unbounded dict in a long-running container is
a slow leak, and the queue only ever shows the most recent items anyway.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from mcm.utils.logging import get_logger

log = get_logger(__name__)

MAX_MEMORY_ITEMS = 500

# An entry is a 512-float embedding plus a few small metadata fields — a few
# KB each, so this cap exists for hygiene (an unbounded list in a long-running
# process) rather than because the memory cost is otherwise a concern.
MAX_IMAGE_INDEX = 2000

# CLIP ViT-B/32 cosine similarity, measured empirically against real Hateful
# Memes images: the same image reloaded scores 1.00, JPEG-recompressed 0.988,
# downsized-then-upsized 0.973 — the kind of degradation an actual re-upload
# produces. Different images sharing the same meme-template visual style (the
# hardest case, since the domain is stylistically homogeneous) top out around
# 0.74. 0.90 sits with a wide margin on both sides rather than splitting a
# close call.
IMAGE_REUSE_THRESHOLD = 0.90


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Store:
    """Records analyses and decisions."""

    def __init__(self, uri: str | None = None):
        self.uri = uri or os.getenv("MONGODB_URI") or ""
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Image-reuse index: one entry per analyzed image, bounded the same way
        # as _memory. Kept separate from the moderation records themselves
        # because it is queried completely differently — a linear similarity
        # scan over every embedding, not a lookup by id or a status/priority
        # filter — and mixing the two would mean every /queue call scanning
        # past 512-float vectors it never uses.
        self._image_index: list[dict[str, Any]] = []
        # FastAPI runs sync path operations in a thread pool, and Cloud Run is
        # configured for concurrency=4 per instance, so the memory fallback is
        # genuinely accessed from multiple threads at once. A dict's individual
        # operations are each atomic under the GIL, but put()'s eviction is a
        # read-check-act sequence across two calls (len() then popitem()); two
        # threads interleaved there can both decide eviction is needed and pop
        # two items for one insert, silently dropping a moderator's item from
        # the queue. The lock makes each public method atomic as a whole.
        self._lock = threading.Lock()
        self._collection = None
        self._image_collection = None

        if self.uri:
            try:
                from pymongo import MongoClient

                client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
                client.admin.command("ping")
                self._collection = client.get_database("mcm").get_collection("items")
                self._collection.create_index("item_id", unique=True)
                self._collection.create_index([("status", 1), ("priority_score", -1)])
                self._image_collection = client.get_database("mcm").get_collection(
                    "image_embeddings"
                )
                self._image_collection.create_index("item_id", unique=True)
                log.info("moderation store: mongodb")
            except Exception as e:  # noqa: BLE001
                # A database that is configured but unreachable should not take
                # the API down; it degrades to memory and says so loudly.
                log.warning("mongodb unavailable (%s); falling back to memory", e)
                self._collection = None
                self._image_collection = None

        if self._collection is None:
            log.info("moderation store: in-memory (set MONGODB_URI to persist)")

    @property
    def backend(self) -> str:
        return "mongodb" if self._collection is not None else "memory"

    def put(self, item: dict[str, Any]) -> None:
        if self._collection is not None:
            self._collection.replace_one({"item_id": item["item_id"]}, item, upsert=True)
            return
        with self._lock:
            self._memory[item["item_id"]] = item
            self._memory.move_to_end(item["item_id"])
            while len(self._memory) > MAX_MEMORY_ITEMS:
                self._memory.popitem(last=False)

    def get(self, item_id: str) -> dict[str, Any] | None:
        if self._collection is not None:
            return self._collection.find_one({"item_id": item_id}, {"_id": 0})
        with self._lock:
            return self._memory.get(item_id)

    def update(self, item_id: str, patch: dict[str, Any]) -> None:
        if self._collection is not None:
            self._collection.update_one({"item_id": item_id}, {"$set": patch})
            return
        with self._lock:
            if item_id in self._memory:
                self._memory[item_id].update(patch)

    def query(
        self,
        status: str = "pending",
        head: str | None = None,
        min_priority: float = 0.0,
        emergent_only: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Ranked queue. Ordering is by priority, never by arrival time.

        The returned count reflects exactly the filters passed in — it answers
        "how many rows matched this query" for pagination, not "how many items
        are pending" (that is ``count_pending``). Conflating the two used to
        surface as ``total_pending: 3`` when a caller asked for
        ``status=resolved`` and got 3 resolved rows back.

        ``head`` matches against ``active_heads`` — every head that
        independently clears threshold on this item — not ``top_head`` alone.
        A real example is why: a meme scoring toxicity=0.76 (correctly
        harmful) had top_head="misinformation" because that head happened to
        read 0.94, a 3-way score that is not on the same scale as a 2-way
        one. Filtering on top_head would have made that item invisible to a
        moderator asking specifically for harassment cases.
        """
        if self._collection is not None:
            q: dict[str, Any] = {}
            if status != "all":
                q["status"] = status
            if head:
                # Mongo's equality match against an array field is an implicit
                # "any element equals this" — exactly array-contains, no
                # $elemMatch needed for a single scalar comparison.
                q["active_heads"] = head
            if emergent_only:
                q["is_emergent"] = True
            if min_priority > 0:
                q["priority_score"] = {"$gte": min_priority}
            total = self._collection.count_documents(q)
            cursor = (
                self._collection.find(q, {"_id": 0})
                .sort("priority_score", -1)
                .skip(offset)
                .limit(limit)
            )
            return list(cursor), total

        with self._lock:
            items = list(self._memory.values())
        if status != "all":
            items = [i for i in items if i.get("status") == status]
        if head:
            # Fall back to top_head only for a record predating active_heads —
            # in-memory records are never that old in practice (a redeploy
            # wipes the store), but a stale MongoDB document could be, and
            # falling back keeps it findable under its one known head rather
            # than silently dropping out of every head-filtered query.
            #
            # `in` binds tighter than `or`, so `head in a or b` parses as
            # `(head in a) or b` — with a non-empty fallback list that is
            # truthy regardless of head, making the filter match everything.
            # The parens below are load-bearing, not stylistic.
            items = [i for i in items if head in (i.get("active_heads") or [i.get("top_head")])]
        if emergent_only:
            items = [i for i in items if i.get("is_emergent")]
        if min_priority > 0:
            items = [i for i in items if i.get("priority_score", 0) >= min_priority]
        items.sort(key=lambda i: i.get("priority_score", 0), reverse=True)
        return items[offset : offset + limit], len(items)

    def find_similar_image(
        self, embedding: list[float], threshold: float = IMAGE_REUSE_THRESHOLD
    ) -> dict[str, Any] | None:
        """Best match for a normalized CLIP embedding, or None below threshold.

        A full scan, not an index lookup. At the scale this project actually
        runs at — hundreds to low thousands of images — a linear numpy pass is
        faster than the engineering cost of standing up a vector database, and
        MongoDB's own driver has no native similarity search on a free tier
        without Atlas Search. If this index ever needs to hold millions of
        entries, replace this method's body, not its callers.
        """
        import numpy as np

        if self._image_collection is not None:
            docs = list(self._image_collection.find({}, {"_id": 0}))
        else:
            with self._lock:
                docs = list(self._image_index)

        if not docs:
            return None

        vectors = np.array([d["embedding"] for d in docs], dtype=np.float32)
        query = np.array(embedding, dtype=np.float32)
        # Embeddings are stored pre-normalized (see add_image_embedding), so a
        # dot product is the cosine similarity directly.
        similarities = vectors @ query
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < threshold:
            return None
        match = docs[best_idx]
        return {
            "item_id": match["item_id"],
            "similarity": round(best_score, 4),
            "created_at": match["created_at"],
            "text": match.get("text", ""),
        }

    def add_image_embedding(
        self, item_id: str, embedding: list[float], created_at: str, text: str
    ) -> None:
        """Register an image so future analyses can be matched against it.

        Always called after find_similar_image, never before — an image must
        not be able to match against itself.
        """
        doc = {"item_id": item_id, "embedding": embedding, "created_at": created_at, "text": text}
        if self._image_collection is not None:
            self._image_collection.replace_one({"item_id": item_id}, doc, upsert=True)
            return
        with self._lock:
            self._image_index.append(doc)
            while len(self._image_index) > MAX_IMAGE_INDEX:
                self._image_index.pop(0)

    def count_pending(self) -> int:
        """Total pending items, independent of whatever filters a caller applied.

        This is the "how many are waiting" figure for a badge or header count,
        and must not shrink just because the caller is looking at a filtered
        view of the queue.
        """
        if self._collection is not None:
            return self._collection.count_documents({"status": "pending"})
        with self._lock:
            return sum(1 for i in self._memory.values() if i.get("status") == "pending")

    def aggregate_stats(self) -> dict[str, Any]:
        """Dashboard figures, computed over whatever records exist."""
        if self._collection is not None:
            items = list(self._collection.find({}, {"_id": 0}))
        else:
            with self._lock:
                items = list(self._memory.values())

        pending = [i for i in items if i.get("status") == "pending"]
        resolved = [i for i in items if i.get("status") == "resolved"]
        times = [
            i["time_to_decision_seconds"]
            for i in resolved
            if i.get("time_to_decision_seconds") is not None
        ]
        agreed = [
            i["agreed_with_model"]
            for i in resolved
            if i.get("agreed_with_model") is not None
        ]
        useful = [
            i["explanation_was_useful"]
            for i in resolved
            if i.get("explanation_was_useful") is not None
        ]

        return {
            "queue": {
                "pending": len(pending),
                "resolved_24h": len(resolved),
                "median_time_to_decision_s": int(_median(times)) if times else 0,
            },
            "model": {
                "emergent_case_rate": _rate([bool(i.get("is_emergent")) for i in items]),
                # Rates over the subset who answered, not over every decision;
                # both fields are optional by design.
                "agreement_rate": _rate(agreed),
                "explanation_useful_rate": _rate(useful),
            },
            "distribution": _distribution(items),
        }


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def _distribution(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    tox = {"benign": 0, "harmful": 0}
    mis = {"true": 0, "satire": 0, "misleading": 0}
    for i in items:
        for head, bucket in (("toxicity", tox), ("misinformation", mis)):
            label = (i.get("heads") or {}).get(head, {}).get("label")
            if label in bucket:
                bucket[label] += 1

    def norm(d: dict[str, int]) -> dict[str, float]:
        total = sum(d.values())
        return {k: round(v / total, 4) if total else 0.0 for k, v in d.items()}

    return {"toxicity": norm(tox), "misinformation": norm(mis)}


def parse_utc(ts: str) -> float:
    """ISO-8601 UTC timestamp -> epoch seconds.

    Uses calendar.timegm rather than time.mktime. mktime interprets a
    struct_time as *local* time, so parsing a UTC string with it shifts every
    result by the host's offset — which showed up as items in the review queue
    reporting an age of "5h ago" seconds after being created, on a UTC+5:30
    machine. It went unnoticed in the decision endpoint only because both
    timestamps there carry the same offset and it cancels.
    """
    import calendar
    import time as _time

    return calendar.timegm(_time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
