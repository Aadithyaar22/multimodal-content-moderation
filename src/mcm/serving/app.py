"""FastAPI application implementing docs/api.md.

Two things shape the design.

Models load in a background thread at startup rather than blocking it, so the
container answers /health immediately and reports models_loaded=false while
weights arrive. On a free-tier host that wakes in 30-50s, a server that refuses
connections until loading finishes is indistinguishable from one that is down.

No endpoint removes content. /analyze scores and queues; the only state-changing
call records a human decision. auto_action is present in every verdict and
always null, so the contract itself states that nothing acts autonomously.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from mcm import __version__
from mcm.models.deepfake import combine_verdict
from mcm.serving import attributions as attributions_mod
from mcm.serving import explain as explain_mod
from mcm.serving.inference import (
    THRESHOLD,
    ModelBundle,
    emergent_signal,
    load_bundle,
    priority_score,
    run_arms,
    verdict_for,
)
from mcm.serving.ratelimit import enforce_rate_limit
from mcm.serving.schemas import (
    Attributions,
    DecisionRequest,
    DecisionResponse,
    Explanation,
    Health,
    ItemDetail,
    ModelCard,
    QueueResponse,
    Stats,
)
from mcm.serving.store import Store, parse_utc, utcnow
from mcm.utils.logging import get_logger

log = get_logger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_state: dict = {"bundle": None, "loading": True, "error": None, "started": time.time()}
_store = Store()
#: Raw image bytes, kept only so the UI can render the evidence it just
#: submitted. Bounded, and never the system of record.
_images: dict[str, bytes] = {}


def _load_models() -> None:
    try:
        _state["bundle"] = load_bundle()
    except Exception as e:  # noqa: BLE001
        log.exception("model load failed")
        _state["error"] = str(e)
    finally:
        _state["loading"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_models, daemon=True).start()
    yield


app = FastAPI(
    title="Vanguard — Multimodal Content Moderation",
    description="Decision support for human moderators. Nothing here removes content.",
    version=__version__,
    lifespan=lifespan,
)

# Origins come from the environment in production. The default covers local dev
# only, so a deployment that forgets to set this fails visibly in the browser
# rather than silently accepting requests from anywhere.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "MCM_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    # Vercel preview deployments get a generated subdomain per commit, so the
    # exact origin cannot be enumerated ahead of time.
    allow_origin_regex=os.getenv("MCM_ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run_deepfake(bundle: ModelBundle, pil):
    """Run the manipulation check, degrading to not-checked on any failure.

    The branch is auxiliary: a verdict without it is still complete, so an error
    here must never fail the request.
    """
    from mcm.models.deepfake import DeepfakeResult

    if bundle.deepfake is None:
        return DeepfakeResult(False, 0.0, "not_checked", reason="detector not loaded")
    if pil is None:
        return DeepfakeResult(False, 0.0, "not_checked", reason="no image supplied")
    try:
        with torch.no_grad():
            px = bundle.clip.preprocess_images([pil]).to(bundle.device)
            emb = bundle.clip.encode_pooled(pixel_values=px).image_emb
        return bundle.deepfake.check(pil, emb)
    except Exception:  # noqa: BLE001
        log.exception("deepfake check failed")
        return DeepfakeResult(False, 0.0, "not_checked", reason="detector error")


def _run_ocr(bundle: ModelBundle, pil) -> str | None:
    """Extract embedded meme text, degrading to None on any failure.

    Informational only: this is displayed alongside the caption for a
    moderator's own reading, never concatenated into the text the classifiers
    see. The heads were trained on Hateful Memes captions and Fakeddit titles,
    neither of which includes OCR'd meme text, so splicing it into the
    classifier input at inference time would feed the model a distribution it
    has never seen and was never evaluated against — an untested behaviour
    change dressed up as a feature. Extract-and-display is what the frontend
    and docs/api.md's own example already expect.
    """
    if bundle.ocr is None or pil is None:
        return None
    try:
        import numpy as np

        result = bundle.ocr.readtext(np.array(pil), detail=0)
        text = " ".join(result).strip()
        return text or None
    except Exception:  # noqa: BLE001
        log.exception("OCR failed")
        return None


def _bundle() -> ModelBundle:
    bundle = _state["bundle"]
    if bundle is None:
        # 503 with Retry-After, not 500: the container is healthy and the client
        # should wait rather than treat this as a failure.
        raise HTTPException(
            status_code=503,
            detail="models are still loading",
            headers={"Retry-After": "3"},
        )
    return bundle


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"http_{exc.status_code}", "message": exc.detail}},
        headers=exc.headers,
    )


@app.get("/api/v1/health", response_model=Health)
def health() -> Health:
    bundle: ModelBundle | None = _state["bundle"]
    return Health(
        status="error" if _state["error"] else "ok",
        models_loaded=bundle is not None and bundle.ready,
        warm=not _state["loading"],
        device=str(bundle.device) if bundle else "unknown",
        version=__version__,
        loaded_at=utcnow() if bundle else None,
        # Without this, "still loading" and "permanently failed" are the same
        # models_loaded=false response, and a client polling on that alone
        # cannot tell a 30s cold start from a crashed deployment — it just
        # keeps polling forever with nothing to show the reason.
        error=_state["error"],
    )


@app.post("/api/v1/analyze", response_model=ItemDetail, dependencies=[Depends(enforce_rate_limit)])
async def analyze(
    text: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    run_ocr: Annotated[bool, Form()] = True,
    source: Annotated[str | None, Form()] = None,
) -> ItemDetail:
    bundle = _bundle()
    if not text and image is None:
        raise HTTPException(422, "provide text, an image, or both")

    started = time.perf_counter()
    item_id = f"itm_{uuid.uuid4().hex[:12]}"

    pil: Image.Image | None = None
    if image is not None:
        raw = await image.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"image exceeds {MAX_UPLOAD_BYTES // 1_000_000}MB")
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(415, "file is not a decodable image") from None
        _images[item_id] = raw

    heads: dict = {}
    modality: dict[str, dict[str, float]] = {"cv_only": {}, "nlp_only": {}, "fusion": {}}
    timings: dict[str, int] = {}
    per_task: dict[str, tuple] = {}

    has_text = bool(text and text.strip())

    for task in bundle.tasks:
        arms, task_timings = run_arms(bundle, task, pil, text or "")
        per_task[task] = arms

        # A single-modality arm is only reported when its modality is actually
        # present. Run on a zeroed input it still emits a confident number —
        # measured at 0.69 for a text-only item — but that is the head's bias on
        # a null vector, not evidence about the content. Publishing it as
        # "vision-only analysis" of an item with no vision would fabricate a
        # signal, and the emergent comparison downstream would be reading it.
        if pil is not None:
            modality["cv_only"][task] = round(arms.cv_only, 4)
        if has_text:
            modality["nlp_only"][task] = round(arms.nlp_only, 4)
        modality["fusion"][task] = round(arms.fusion, 4)
        heads[task] = {
            "label": max(arms.fusion_probs, key=arms.fusion_probs.get),
            "score": round(arms.fusion, 4),
            "classes": {k: round(v, 4) for k, v in arms.fusion_probs.items()},
        }
        for k, v in task_timings.items():
            timings[k] = timings.get(k, 0) + v

    # Manipulation check. Outside the cross-attention block by design: it
    # reasons about pixel artefacts, which caption text says nothing about.
    deepfake = _run_deepfake(bundle, pil)

    ocr_started = time.perf_counter()
    ocr_text = _run_ocr(bundle, pil) if run_ocr else None
    timings["ocr"] = int((time.perf_counter() - ocr_started) * 1000)

    # The reported verdict follows whichever head scored highest; that is the
    # reason the item is in the queue at all.
    lead_task = max(per_task, key=lambda t: per_task[t].fusion)
    lead = per_task[lead_task]

    # Every head whose OWN fusion score clears threshold, independent of which
    # one happens to be highest. lead_task alone is not enough to file an item
    # under: comparing a 2-way toxicity score against a 3-way misinformation
    # score as if they sat on the same scale is not sound, and a real example
    # caught it directly — a meme scoring toxicity=0.76 (correctly harmful) was
    # filed under top_head="misinformation" because that head happened to read
    # 0.94. Filtering /queue on top_head alone would make that item invisible
    # to a moderator asking for harassment cases specifically. active_heads is
    # what /queue?head=... actually matches against; top_head remains the
    # single "most urgent" head for display and priority ordering.
    active_heads = sorted(t for t in per_task if per_task[t].fusion >= THRESHOLD)

    # Score-level combination, the only place the branch touches the verdict.
    combined, manipulated = combine_verdict(lead.fusion, deepfake)
    label, action = verdict_for(combined)

    # Emergence is only meaningful when both modalities were actually present.
    # With one missing there is no "neither alone" to establish, and the absent
    # arm's reading is a null-input bias rather than a score to compare against.
    both_modalities = pil is not None and has_text
    if both_modalities:
        is_emergent, delta = emergent_signal(lead)
    else:
        is_emergent, delta = False, 0.0

    timings["total"] = int((time.perf_counter() - started) * 1000)

    record = {
        "item_id": item_id,
        "created_at": utcnow(),
        "status": "pending",
        "source": source,
        "top_head": lead_task,
        "active_heads": active_heads,
        "is_emergent": is_emergent,
        "priority_score": round(priority_score(combined, is_emergent), 4),
        "input": {
            "text": text or "",
            "has_image": pil is not None,
            "image_url": f"/api/v1/items/{item_id}/image" if pil is not None else None,
            "ocr_text": ocr_text,
            "modalities": [m for m, on in (("image", pil is not None), ("text", bool(text))) if on],
        },
        "verdict": {
            "label": label,
            "confidence": round(combined, 4),
            "priority_score": round(priority_score(combined, is_emergent), 4),
            "recommended_action": action,
            "auto_action": None,
        },
        "heads": heads,
        "modality_scores": modality,
        "fusion_signal": {
            "is_emergent": is_emergent,
            "delta_over_best_unimodal": round(delta, 4),
            "note": (
                "Neither modality alone crosses the threshold; the signal "
                "appears only jointly."
                if is_emergent
                else (
                    None
                    if both_modalities
                    else "Only one modality was supplied, so no cross-modal "
                    "comparison was made."
                )
            ),
        },
        "deepfake": deepfake.to_dict(),
        "manipulation_flagged": manipulated,
        "explanation_status": "pending",
        "latency_ms": timings,
        "explanation": None,
        "attributions": None,
        "decisions": [],
    }

    _store.put(record)
    return ItemDetail(**record)


@app.get("/api/v1/items/{item_id}", response_model=ItemDetail)
def get_item(item_id: str) -> ItemDetail:
    record = _store.get(item_id)
    if not record:
        raise HTTPException(404, "item not found")
    return ItemDetail(**record)


@app.get("/api/v1/items/{item_id}/image")
def get_image(item_id: str) -> Response:
    raw = _images.get(item_id)
    if raw is None:
        raise HTTPException(404, "no image for this item")
    return Response(content=raw, media_type="image/jpeg")


@app.get("/api/v1/items/{item_id}/explanation", response_model=Explanation)
def get_explanation(item_id: str) -> Explanation:
    record = _store.get(item_id)
    if not record:
        raise HTTPException(404, "item not found")

    cached = record.get("explanation")
    if cached and cached.get("status") == "ready":
        return Explanation(**cached)

    # Every head that independently cleared threshold, not just top_head. An
    # item scoring toxicity=0.76 (harmful) and misinformation=0.94 (misleading)
    # is two separate reasons to look at it; a prompt built from top_head alone
    # would only ever ask the model to discuss the one that happened to score
    # higher, which is the same blind spot the /queue head filter had before
    # active_heads existed. Falls back to [top_head] for a record that
    # predates the field (a stale MongoDB document, in practice).
    active = record.get("active_heads") or [record["top_head"]]
    result = explain_mod.generate(
        {
            "heads": {
                task: {
                    "label": record["heads"][task]["label"],
                    "modality_scores": {
                        k: record["modality_scores"][k].get(task, 0.0)
                        for k in ("cv_only", "nlp_only", "fusion")
                    },
                }
                for task in active
            },
            "text": record["input"]["text"],
            "has_image": record["input"]["has_image"],
            "threshold": THRESHOLD,
            "is_emergent": record["fusion_signal"]["is_emergent"],
        }
    )
    payload = {"item_id": item_id, "generated_at": utcnow(), **result}
    _store.update(item_id, {"explanation": payload, "explanation_status": result["status"]})
    return Explanation(**payload)


@app.get("/api/v1/items/{item_id}/attributions", response_model=Attributions)
def get_attributions(item_id: str) -> Attributions:
    record = _store.get(item_id)
    if not record:
        raise HTTPException(404, "item not found")

    cached = record.get("attributions")
    if cached:
        return Attributions(**cached)

    bundle = _bundle()
    image = None
    raw = _images.get(item_id)
    if raw is not None:
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:  # noqa: BLE001
            image = None

    payload = attributions_mod.compute(bundle, record, image=image)
    # Cached because occlusion costs one forward pass per token; a moderator
    # reopening an item should not pay for it twice.
    _store.update(item_id, {"attributions": payload})
    return Attributions(**payload)


@app.post("/api/v1/items/{item_id}/decision", response_model=DecisionResponse)
def submit_decision(item_id: str, body: DecisionRequest) -> DecisionResponse:
    record = _store.get(item_id)
    if not record:
        raise HTTPException(404, "item not found")

    decided_at = utcnow()
    elapsed = max(0, int(parse_utc(decided_at) - parse_utc(record["created_at"])))

    decision = {**body.model_dump(), "decided_at": decided_at}
    _store.update(
        item_id,
        {
            "status": "resolved",
            "decisions": [*record.get("decisions", []), decision],
            "time_to_decision_seconds": elapsed,
            "agreed_with_model": body.agreed_with_model,
            "explanation_was_useful": body.explanation_was_useful,
        },
    )

    return DecisionResponse(
        item_id=item_id,
        status="resolved",
        action=body.action,
        decided_at=decided_at,
        time_to_decision_seconds=elapsed,
    )


@app.get("/api/v1/queue", response_model=QueueResponse)
def queue(
    status: str = "pending",
    head: str | None = None,
    min_priority: float = 0.0,
    emergent_only: bool = False,
    limit: int = Query(25, le=100),
    cursor: str | None = None,
) -> QueueResponse:
    offset = int(cursor) if cursor and cursor.isdigit() else 0
    records, total = _store.query(
        status=status,
        head=head,
        min_priority=min_priority,
        emergent_only=emergent_only,
        limit=limit,
        offset=offset,
    )

    now = time.time()
    items = []
    for r in records:
        created = parse_utc(r["created_at"])
        items.append(
            {
                "item_id": r["item_id"],
                "thumbnail_url": r["input"].get("image_url"),
                "text_preview": (r["input"]["text"] or "")[:160],
                "verdict": r["verdict"],
                "top_head": r["top_head"],
                "active_heads": r.get("active_heads", [r["top_head"]]),
                "is_emergent": r["is_emergent"],
                "status": r["status"],
                "created_at": r["created_at"],
                "age_seconds": max(0, int(now - created)),
            }
        )

    next_cursor = str(offset + limit) if offset + limit < total else None
    return QueueResponse(
        items=items,
        next_cursor=next_cursor,
        total_matching=total,
        # Always the system-wide pending count, not `total` — a moderator
        # filtering to "emergent only" must still see the true backlog size,
        # not the size of the slice they are currently looking at.
        total_pending=_store.count_pending(),
    )


@app.get("/api/v1/stats", response_model=Stats)
def stats() -> Stats:
    return Stats(**_store.aggregate_stats())


@app.get("/api/v1/model-card", response_model=ModelCard)
def model_card() -> ModelCard:
    bundle: ModelBundle | None = _state["bundle"]
    return ModelCard(
        architecture="Cross-attention fusion over frozen CLIP ViT-B/32",
        backbone="openai/clip-vit-base-patch32 (frozen)",
        trained_on=["Hateful Memes", "Fakeddit"],
        checkpoints={
            task: sorted(arms) for task, arms in (bundle.arms.items() if bundle else {})
        },
        # Stated rather than summarised: these are the measured results, and a
        # non-significant difference must be reported as non-significant.
        ablation={
            "hateful_memes": {
                "cv_only": 0.6217,
                "nlp_only": 0.6283,
                "late_fusion": 0.6910,
                "cross_attention": 0.7035,
                "cross_vs_late_p": 0.051,
            },
            "fakeddit": {
                "cv_only": 0.6863,
                "nlp_only": 0.7031,
                "late_fusion": 0.7732,
                "cross_attention": 0.7705,
                "cross_vs_late_p": 0.596,
            },
        },
        limitations=[
            "Cross-attention beats late fusion on Hateful Memes by 0.0125 macro-F1 "
            "at p=0.051, which is not significant at the conventional threshold.",
            "On Fakeddit the two are indistinguishable (p=0.596), consistent with "
            "that benchmark's text often carrying the label alone.",
            "Trained on 8,500 Hateful Memes examples; small for a transformer "
            "trained from scratch.",
            "The deepfake branch uses a ViT classifier, not the planned Xception "
            "on FaceForensics++, which is EULA-gated with no public checkpoint. "
            "It detects facial manipulation only, and is skipped with a stated "
            "reason when no face is present rather than scoring regardless.",
            "Token attribution is leave-one-out occlusion, not Shapley values.",
            "Decision support only. No endpoint removes content.",
        ],
    )
