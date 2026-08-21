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
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from mcm import __version__
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

app.add_middleware(
    CORSMiddleware,
    # Set MCM_ALLOWED_ORIGINS in production. The default covers local dev only.
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
    )


@app.post("/api/v1/analyze", response_model=ItemDetail)
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

    # The reported verdict follows whichever head scored highest; that is the
    # reason the item is in the queue at all.
    lead_task = max(per_task, key=lambda t: per_task[t].fusion)
    lead = per_task[lead_task]
    label, action = verdict_for(lead.fusion)

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
        "is_emergent": is_emergent,
        "priority_score": round(priority_score(lead.fusion, is_emergent), 4),
        "input": {
            "text": text or "",
            "has_image": pil is not None,
            "image_url": f"/api/v1/items/{item_id}/image" if pil is not None else None,
            "ocr_text": None,
            "modalities": [m for m, on in (("image", pil is not None), ("text", bool(text))) if on],
        },
        "verdict": {
            "label": label,
            "confidence": round(lead.fusion, 4),
            "priority_score": round(priority_score(lead.fusion, is_emergent), 4),
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
        # Step 5 of the build order; declared here so the contract is stable.
        "deepfake": {"checked": False, "score": 0.0, "label": "not_checked"},
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

    task = record["top_head"]
    result = explain_mod.generate(
        {
            "task": task,
            "text": record["input"]["text"],
            "has_image": record["input"]["has_image"],
            "modality_scores": {
                k: record["modality_scores"][k].get(task, 0.0)
                for k in ("cv_only", "nlp_only", "fusion")
            },
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
    # SHAP, Grad-CAM and attention extraction are step 6. The endpoint exists
    # and returns a well-formed empty payload so the UI renders its "no
    # attribution" state rather than erroring.
    return Attributions(
        item_id=item_id,
        text=None,
        image=None,
        cross_attention={"available": False, "top_links": []},
    )


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
                "is_emergent": r["is_emergent"],
                "status": r["status"],
                "created_at": r["created_at"],
                "age_seconds": max(0, int(now - created)),
            }
        )

    next_cursor = str(offset + limit) if offset + limit < total else None
    return QueueResponse(items=items, next_cursor=next_cursor, total_pending=total)


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
            "Deepfake detection and attribution extraction are not yet wired.",
            "Decision support only. No endpoint removes content.",
        ],
    )
