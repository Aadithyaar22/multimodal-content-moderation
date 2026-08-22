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


class AnalysisResult(BaseModel):
    item_id: str
    created_at: str
    input: AnalysisInput
    verdict: Verdict
    heads: dict[str, HeadScore]
    modality_scores: ModalityScores
    fusion_signal: FusionSignal
    deepfake: DeepfakeResult
    explanation_status: ExplanationStatus
    latency_ms: dict[str, int]


class KeyFactor(BaseModel):
    modality: Literal["text", "image", "cross"]
    factor: str
    weight: float


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
    is_emergent: bool
    status: str
    created_at: str
    age_seconds: int


class QueueResponse(BaseModel):
    items: list[QueueItem]
    next_cursor: str | None = None
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
