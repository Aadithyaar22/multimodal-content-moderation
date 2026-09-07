"""Response models for the moderation API.

Mirrors docs/api.md, which the frontend's lib/types.ts also mirrors. All three
must move together; the contract is the shared reference, not this file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VerdictLabel = Literal["benign", "review", "harmful"]
DecisionAction = Literal["approve", "remove", "escalate", "defer"]
ExplanationStatus = Literal["pending", "ready", "failed", "unavailable"]


class Health(BaseModel):
    status: str = "ok"
    models_loaded: bool
    warm: bool
    device: str
    version: str
    loaded_at: str | None = None
    #: Set only when status="error", i.e. loading finished and failed. A caller
    #: (or the frontend's health poll) must be able to tell that apart from
    #: "still loading" — both otherwise look identical as models_loaded=false.
    error: str | None = None


class HeadScore(BaseModel):
    label: str
    score: float
    classes: dict[str, float]


class FusionSignal(BaseModel):
    """The project's central claim, in wire form.

    ``is_emergent`` marks items where the fused score materially exceeds both
    unimodal scores — harm visible only in the combination.
    """

    is_emergent: bool
    delta_over_best_unimodal: float
    note: str | None = None


class ModalityScores(BaseModel):
    cv_only: dict[str, float] = Field(default_factory=dict)
    nlp_only: dict[str, float] = Field(default_factory=dict)
    fusion: dict[str, float] = Field(default_factory=dict)


class Verdict(BaseModel):
    label: VerdictLabel
    confidence: float
    priority_score: float
    recommended_action: str
    # Always null. The system is decision support; encoding that in the contract
    # keeps it from being quietly designed away.
    auto_action: None = None


class AnalysisInput(BaseModel):
    text: str
    has_image: bool
    image_url: str | None = None
    ocr_text: str | None = None
    modalities: list[str]


class DeepfakeResult(BaseModel):
    checked: bool
    score: float
    label: str
    #: Why the check was skipped, when it was. Stated rather than left implicit,
    #: so "not checked" is never mistaken for "checked and found authentic".
    reason: str | None = None
    face_confidence: float | None = None


class ImageReuse(BaseModel):
    """Has this image been analyzed before, possibly under a different claim.

    Deliberately informational, the same as DeepfakeResult and OcrText: this
    is a similarity signal, not a fused prediction. "Same image, different
    caption" and "same image, legitimately re-shared" are indistinguishable
    from a similarity score alone; a moderator reading first_seen_text next
    to the current caption can make that call, a fixed threshold cannot.
    """

    checked: bool
    is_reused: bool
    similarity: float = 0.0
    first_seen_item_id: str | None = None
    first_seen_at: str | None = None
    first_seen_text: str | None = None
    reason: str | None = None


class AnalysisResult(BaseModel):
    item_id: str
    created_at: str
    input: AnalysisInput
    verdict: Verdict
    heads: dict[str, HeadScore]
    modality_scores: ModalityScores
    fusion_signal: FusionSignal
    #: Every head whose own fusion score clears threshold, not just whichever
    #: one is highest. A queue filter or a moderator's own reading of "which
    #: kinds of harm apply here" should go by this, not by picking the single
    #: loudest head — see the note where this is computed in app.py.
    active_heads: list[str] = Field(default_factory=list)
    deepfake: DeepfakeResult
    image_reuse: ImageReuse
    explanation_status: ExplanationStatus
    latency_ms: dict[str, int]


class KeyFactor(BaseModel):
    modality: Literal["text", "image", "cross"]
    factor: str
    weight: float
    #: Which head this factor is about. An item can be flagged on more than
    #: one independent ground (toxicity and misinformation each clearing their
    #: own threshold); this is what lets a client group factors correctly
    #: instead of presenting a second head's numbers as if they belonged to
    #: the first.
    head: str


class Explanation(BaseModel):
    item_id: str
    status: ExplanationStatus
    narrative: str | None = None
    key_factors: list[KeyFactor] = Field(default_factory=list)
    model: str | None = None
    generated_at: str | None = None
    latency_ms: int | None = None


class TokenAttribution(BaseModel):
    token: str
    #: Signed: positive pushes toward harmful, negative toward benign.
    score: float


class ImageRegion(BaseModel):
    bbox: list[float]
    score: float
    label: str


class TextAttributions(BaseModel):
    method: str
    tokens: list[TokenAttribution]


class ImageAttributions(BaseModel):
    method: str
    heatmap_url: str
    regions: list[ImageRegion]
    #: Raw NxN Grad-CAM map, normalized to [0, 1]. Sent alongside the boxes so a
    #: client can render the continuous heatmap rather than only the regions.
    grid: list[list[float]] | None = None


class CrossAttentionLink(BaseModel):
    text_token: str
    image_region: list[float]
    weight: float


class CrossAttentionAttributions(BaseModel):
    available: bool
    top_links: list[CrossAttentionLink] = Field(default_factory=list)


class Attributions(BaseModel):
    item_id: str
    text: TextAttributions | None = None
    image: ImageAttributions | None = None
    cross_attention: CrossAttentionAttributions | None = None


class QueueItem(BaseModel):
    item_id: str
    thumbnail_url: str | None
    text_preview: str
    verdict: dict
    top_head: str
    #: Every head this item actually clears threshold on. `head=X` in
    #: GET /queue matches against this, not against top_head alone.
    active_heads: list[str] = Field(default_factory=list)
    is_emergent: bool
    status: str
    created_at: str
    age_seconds: int


class QueueResponse(BaseModel):
    items: list[QueueItem]
    next_cursor: str | None = None
    #: Rows matching this call's own filters — the count `next_cursor` paginates
    #: over. Not "how many are pending" when the caller filtered on something else.
    total_matching: int
    #: Total pending items system-wide, independent of this call's filters.
    #: A badge reading this must not shrink because the caller narrowed status
    #: or head — that reads as items disappearing rather than being filtered.
    total_pending: int


class DecisionRequest(BaseModel):
    action: DecisionAction
    moderator_id: str
    rationale: str | None = None
    agreed_with_model: bool | None = None
    explanation_was_useful: bool | None = None


class DecisionResponse(BaseModel):
    item_id: str
    status: str
    action: DecisionAction
    decided_at: str
    time_to_decision_seconds: int


class ItemDetail(AnalysisResult):
    explanation: Explanation | None = None
    attributions: Attributions | None = None
    decisions: list[dict] = Field(default_factory=list)
    status: str


class Stats(BaseModel):
    queue: dict
    model: dict
    distribution: dict


class ModelCard(BaseModel):
    architecture: str
    backbone: str
    trained_on: list[str]
    checkpoints: dict
    ablation: dict
    limitations: list[str]
