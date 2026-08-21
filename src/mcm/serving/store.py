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
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from mcm.utils.logging import get_logger

log = get_logger(__name__)

MAX_MEMORY_ITEMS = 500


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Store:
    """Records analyses and decisions."""

    def __init__(self, uri: str | None = None):
        self.uri = uri or os.getenv("MONGODB_URI") or ""
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._collection = None

        if self.uri:
            try:
                from pymongo import MongoClient

                client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
                client.admin.command("ping")
                self._collection = client.get_database("mcm").get_collection("items")
                self._collection.create_index("item_id", unique=True)
                self._collection.create_index([("status", 1), ("priority_score", -1)])
                log.info("moderation store: mongodb")
            except Exception as e:  # noqa: BLE001
                # A database that is configured but unreachable should not take
                # the API down; it degrades to memory and says so loudly.
                log.warning("mongodb unavailable (%s); falling back to memory", e)
                self._collection = None

        if self._collection is None:
            log.info("moderation store: in-memory (set MONGODB_URI to persist)")

    @property
    def backend(self) -> str:
        return "mongodb" if self._collection is not None else "memory"

    def put(self, item: dict[str, Any]) -> None:
        if self._collection is not None:
            self._collection.replace_one({"item_id": item["item_id"]}, item, upsert=True)
            return
        self._memory[item["item_id"]] = item
        self._memory.move_to_end(item["item_id"])
        while len(self._memory) > MAX_MEMORY_ITEMS:
            self._memory.popitem(last=False)

    def get(self, item_id: str) -> dict[str, Any] | None:
        if self._collection is not None:
            return self._collection.find_one({"item_id": item_id}, {"_id": 0})
        return self._memory.get(item_id)

    def update(self, item_id: str, patch: dict[str, Any]) -> None:
        if self._collection is not None:
            self._collection.update_one({"item_id": item_id}, {"$set": patch})
            return
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
        """Ranked queue. Ordering is by priority, never by arrival time."""
        if self._collection is not None:
            q: dict[str, Any] = {}
            if status != "all":
                q["status"] = status
            if head:
                q["top_head"] = head
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

        items = list(self._memory.values())
        if status != "all":
            items = [i for i in items if i.get("status") == status]
        if head:
            items = [i for i in items if i.get("top_head") == head]
        if emergent_only:
            items = [i for i in items if i.get("is_emergent")]
        if min_priority > 0:
            items = [i for i in items if i.get("priority_score", 0) >= min_priority]
        items.sort(key=lambda i: i.get("priority_score", 0), reverse=True)
        return items[offset : offset + limit], len(items)

    def aggregate_stats(self) -> dict[str, Any]:
        """Dashboard figures, computed over whatever records exist."""
        if self._collection is not None:
            items = list(self._collection.find({}, {"_id": 0}))
        else:
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
