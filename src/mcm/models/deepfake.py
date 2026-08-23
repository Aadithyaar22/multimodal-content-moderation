"""Deepfake / manipulated-media detection.

Architecture note (PROJECT_CONTEXT.md Sec. 4, decision 2)
--------------------------------------------------------
This branch is deliberately *outside* the cross-attention block. It reasons about
pixel-level artefacts — compression inconsistencies, blending seams, generator
fingerprints — which have no relationship to what a caption says. Attending it to
language would model a correlation that does not exist, and would let caption
text move a judgement about whether pixels were synthesised. It joins the verdict
at score level instead, which is a decision about the right tool for the job
rather than an implementation shortcut.

Deviation from the plan
-----------------------
The plan specified Xception fine-tuned on FaceForensics++. That is not used here:
FaceForensics++ is EULA-gated, and no loadable Xception checkpoint trained on it
is publicly available. A ViT deepfake classifier is used instead. It is a
substitution of backbone and training set, not of task, and it is recorded in the
model card rather than left implicit.

The face gate
-------------
The detector is trained on faces. Run on a landscape, a screenshot or a diagram
it still emits a confident number, and that number is meaningless — the same
fabricated-signal failure that appeared twice already in this system, once for
single-modality arms scored on null inputs and once for attention read off a
register patch.

So the branch is gated: the frozen CLIP encoder already running on every request
scores the image against face and non-face prompts, and the detector runs only
when a face is plausibly present. Otherwise the result reports checked=False with
a reason, which is an honest "not applicable" rather than a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from mcm.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "dima806/deepfake_vs_real_image_detection"

#: Zero-shot prompts for the face gate. Encoded once at load, since they never
#: change, so the gate costs one image-to-text similarity at request time.
FACE_PROMPTS = [
    "a photograph of a person's face",
    "a close-up portrait of a human face",
    "a selfie of a person",
]
NON_FACE_PROMPTS = [
    "a landscape with no people in it",
    "a screenshot of text",
    "a picture of an object, with no people",
    "a chart or diagram",
]

#: How far face similarity must exceed non-face similarity before the detector is
#: considered applicable. Set above zero so an ambiguous image does not get a
#: confident manipulation verdict on the strength of a coin flip.
FACE_MARGIN = 0.01


@dataclass
class DeepfakeResult:
    checked: bool
    score: float
    label: str
    reason: str | None = None
    face_confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "score": round(self.score, 4),
            "label": self.label,
            "reason": self.reason,
            "face_confidence": (
                round(self.face_confidence, 4) if self.face_confidence is not None else None
            ),
        }


class DeepfakeDetector:
    """Face-manipulation classifier with a CLIP-based applicability gate."""

    def __init__(self, clip, model_name: str = DEFAULT_MODEL, device=None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device or clip.device_
        self.clip = clip
        log.info("loading deepfake detector %s", model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Read the fake class from the label map rather than assuming an index.
        # Published checkpoints disagree on ordering — one ships {0: Real,
        # 1: Fake} and another the exact reverse — so a hardcoded index inverts
        # the verdict silently on half of them.
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.fake_index = next(
            (i for i, name in id2label.items() if "fake" in name.lower()), 1
        )
        self.labels = id2label
        log.info("deepfake labels %s; fake class = %d", id2label, self.fake_index)

        self._face_bank = self._encode_prompts()

    @torch.no_grad()
    def _encode_prompts(self) -> tuple[torch.Tensor, torch.Tensor]:
        def encode(prompts: list[str]) -> torch.Tensor:
            inputs = {k: v.to(self.device) for k, v in self.clip.tokenize(prompts).items()}
            emb = self.clip.encode_pooled(text_inputs=inputs).text_emb
            return F.normalize(emb, dim=-1)

        return encode(FACE_PROMPTS), encode(NON_FACE_PROMPTS)

    @torch.no_grad()
    def face_likelihood(self, image_emb: torch.Tensor) -> float:
        """Margin by which the image looks more like a face than not."""
        emb = F.normalize(image_emb, dim=-1)
        face_bank, non_face_bank = self._face_bank
        face = (emb @ face_bank.T).max().item()
        non_face = (emb @ non_face_bank.T).max().item()
        return face - non_face

    @torch.no_grad()
    def check(self, image: Image.Image | None, image_emb: torch.Tensor | None) -> DeepfakeResult:
        if image is None:
            return DeepfakeResult(False, 0.0, "not_checked", reason="no image supplied")

        margin = self.face_likelihood(image_emb) if image_emb is not None else 1.0
        if margin < FACE_MARGIN:
            return DeepfakeResult(
                False,
                0.0,
                "not_applicable",
                reason=(
                    "no face detected; this detector is trained on facial "
                    "manipulation and its output would not be meaningful here"
                ),
                face_confidence=margin,
            )

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits.float()[0]
        probs = torch.softmax(logits, dim=-1)
        score = float(probs[self.fake_index])

        return DeepfakeResult(
            checked=True,
            score=score,
            label="manipulated" if score >= 0.5 else "authentic",
            face_confidence=margin,
        )


def combine_verdict(
    harm_score: float,
    deepfake: DeepfakeResult,
    weight: float = 0.25,
) -> tuple[float, bool]:
    """Score-level combination of the harm verdict with the manipulation check.

    Deliberately additive and bounded rather than a learned fusion. The branch
    was never trained jointly with the heads, so there is no principled way to
    learn a weighting here, and inventing one would imply a calibration this
    system does not have.

    Manipulation raises priority; it never lowers it. An authentic image is not
    evidence that content is harmless, so a low manipulation score must not
    discount a high harm score.
    """
    if not deepfake.checked:
        return harm_score, False
    lift = weight * max(0.0, deepfake.score - 0.5) * 2.0
    return min(1.0, harm_score + lift), deepfake.score >= 0.5
