"""Model bundle and the inference path behind /analyze.

One CLIP forward pass feeds every head. The backbone is frozen and shared, so
running the vision-only, language-only and fused arms costs one encode plus
three small matrix multiplies rather than three separate passes — which is what
makes it affordable to return the per-modality comparison on every request
instead of only in the ablation table.

That comparison is not diagnostic decoration. The emergent flag is derived from
it at request time: an item is emergent when both unimodal arms sit below
threshold and the fused arm clears it, which is exactly the case the project
exists to catch.

Each benchmark keeps its own trio of arms. A Hateful Memes model's
misinformation head never saw a misinformation label, so serving its output
would produce a confident number with nothing behind it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from mcm.config import CHECKPOINT_DIR
from mcm.data.schema import MISINFO_3_LABELS, TOXICITY_LABELS
from mcm.models.baselines import build_model
from mcm.models.clip_encoder import FrozenCLIP
from mcm.utils.device import get_device
from mcm.utils.logging import get_logger

log = get_logger(__name__)

TASK_DATASET = {"toxicity": "hateful_memes", "misinformation": "fakeddit"}
LABELS = {"toxicity": TOXICITY_LABELS, "misinformation": MISINFO_3_LABELS}

#: Scores at or above this are treated as crossing threshold, matching the
#: value the ablation used so the served flag agrees with the reported metric.
THRESHOLD = 0.5

#: How far the fused score must exceed the better unimodal one before the item
#: is called emergent. A bare inequality would fire on rounding noise; this is
#: a deliberate margin.
EMERGENT_MARGIN = 0.15


@dataclass
class ArmOutputs:
    cv_only: float
    nlp_only: float
    fusion: float
    fusion_probs: dict[str, float]


@dataclass
class ModelBundle:
    """Frozen CLIP plus the trained heads, loaded once per process."""

    clip: FrozenCLIP
    arms: dict[str, dict[str, torch.nn.Module]] = field(default_factory=dict)
    #: Per-arm temperature from scripts/calibrate.py, applied to logits before
    #: softmax. Defaults to 1.0 for any checkpoint predating calibration.
    temperatures: dict[str, dict[str, float]] = field(default_factory=dict)
    device: torch.device = field(default_factory=get_device)
    loaded_at: float = 0.0

    @property
    def ready(self) -> bool:
        return bool(self.arms)

    @property
    def tasks(self) -> list[str]:
        return sorted(self.arms)


def load_bundle(checkpoint_dir: Path | None = None) -> ModelBundle:
    """Load every checkpoint present. Missing tasks are skipped, not fatal.

    A partially-trained deployment should serve what it has rather than refuse
    to start; /health reports which tasks are live.
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    device = get_device()
    clip = FrozenCLIP(device=device)
    bundle = ModelBundle(clip=clip, device=device, loaded_at=time.time())

    for task, dataset in TASK_DATASET.items():
        arms: dict[str, torch.nn.Module] = {}
        temps: dict[str, float] = {}
        for arch in ("cv_only", "nlp_only", "cross_attention"):
            path = checkpoint_dir / f"{arch}__{dataset}.pt"
            if not path.exists():
                log.warning("missing checkpoint %s; %s/%s unavailable", path, task, arch)
                continue
            blob = torch.load(path, map_location=device, weights_only=False)
            cfg = blob["config"]
            model = _build(arch, cfg).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            arms[arch] = model
            temps[arch] = float(blob.get("temperature", 1.0))
            if temps[arch] == 1.0:
                log.warning(
                    "%s/%s has no fitted temperature; its probabilities will be "
                    "overconfident. Run scripts/calibrate.py",
                    task,
                    arch,
                )

        if len(arms) == 3:
            bundle.arms[task] = arms
            bundle.temperatures[task] = temps
            log.info(
                "loaded %s arms for task=%s (T=%s)",
                len(arms),
                task,
                {k: round(v, 2) for k, v in temps.items()},
            )
        elif arms:
            log.warning(
                "task=%s has only %s; needs all three arms for the modality "
                "comparison, so it is not served",
                task,
                sorted(arms),
            )

    return bundle


def _build(arch: str, cfg: dict) -> torch.nn.Module:
    if arch == "cross_attention":
        return build_model(
            arch,
            d_model=cfg.get("d_model", 512),
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 8),
            dropout=cfg.get("fusion_dropout", 0.1),
            head_hidden=cfg.get("hidden_dim", 256),
            head_dropout=cfg.get("dropout", 0.3),
        )
    return build_model(
        arch,
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
    )


@torch.no_grad()
def run_arms(
    bundle: ModelBundle,
    task: str,
    image: Image.Image | None,
    text: str,
) -> tuple[ArmOutputs, dict[str, int]]:
    """Score one item through all three arms of a task."""
    device = bundle.device
    arms = bundle.arms[task]
    temps = bundle.temperatures.get(task, {})
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    has_image = image is not None
    pixel_values = (
        bundle.clip.preprocess_images([image])
        if has_image
        # A zero image paired with image_mask=False is never attended to; its
        # value is irrelevant but zero makes an accidental unmasked read obvious.
        else torch.zeros(1, 3, 224, 224)
    ).to(device)
    text_inputs = {k: v.to(device) for k, v in bundle.clip.tokenize([text or ""]).items()}
    image_mask = torch.tensor([has_image], dtype=torch.bool, device=device)

    pooled = bundle.clip.encode_pooled(pixel_values=pixel_values, text_inputs=text_inputs)
    tokens = bundle.clip.encode_tokens(pixel_values=pixel_values, text_inputs=text_inputs)
    timings["encode"] = int((time.perf_counter() - t0) * 1000)

    image_emb = pooled.image_emb.clone()
    image_emb[~image_mask] = 0.0

    t1 = time.perf_counter()
    cv = _prob(
        arms["cv_only"](image_emb=image_emb, text_emb=pooled.text_emb, image_mask=image_mask),
        task,
        temps.get("cv_only", 1.0),
    )
    nlp = _prob(
        arms["nlp_only"](image_emb=image_emb, text_emb=pooled.text_emb, image_mask=image_mask),
        task,
        temps.get("nlp_only", 1.0),
    )
    timings["unimodal"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    fused_out = arms["cross_attention"](
        image_tokens=tokens.image_tokens,
        text_tokens=tokens.text_tokens,
        text_attention_mask=tokens.text_attention_mask,
        image_mask=image_mask,
    )
    timings["fusion"] = int((time.perf_counter() - t2) * 1000)

    probs = _softmax(fused_out, task, temps.get("cross_attention", 1.0))
    labels = LABELS[task]

    return (
        ArmOutputs(
            cv_only=cv,
            nlp_only=nlp,
            fusion=_positive_mass(probs, task),
            fusion_probs={labels[i]: float(p) for i, p in enumerate(probs)},
        ),
        timings,
    )


def _logits_for(out, task: str) -> torch.Tensor:
    return out.toxicity_logits if task == "toxicity" else out.misinfo_logits


def _softmax(out, task: str, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over temperature-scaled logits.

    Dividing by a positive scalar cannot move an argmax, so this changes the
    probabilities without changing a single prediction — which is the point:
    the reported ablation stays valid while the confidences become usable for
    ranking and for the soft-flag threshold.
    """
    return F.softmax(_logits_for(out, task).float()[0] / temperature, dim=-1)


def _positive_mass(probs: torch.Tensor, task: str) -> float:
    """Probability that an item is *not* the benign class.

    For toxicity that is simply class 1. For misinformation the benign class is
    "true" and both "satire" and "misleading" sit opposite it, so the harm score
    is one minus the true mass rather than any single class.
    """
    if task == "toxicity":
        return float(probs[1])
    return float(1.0 - probs[0])


def _prob(out, task: str, temperature: float = 1.0) -> float:
    return _positive_mass(_softmax(out, task, temperature), task)


def emergent_signal(arms: ArmOutputs, threshold: float = THRESHOLD) -> tuple[bool, float]:
    """Whether harm is visible only in the combination.

    Requires both unimodal arms below threshold *and* the fused arm above it —
    not merely a large delta. A fused score that is higher but still below
    threshold has not changed the decision, and calling that emergent would
    inflate the headline rate with cases no moderator would ever see.
    """
    best_unimodal = max(arms.cv_only, arms.nlp_only)
    delta = arms.fusion - best_unimodal
    is_emergent = (
        best_unimodal < threshold
        and arms.fusion >= threshold
        and delta >= EMERGENT_MARGIN
    )
    return is_emergent, delta


def verdict_for(score: float, threshold: float = THRESHOLD) -> tuple[str, str]:
    """Map a score to a label and a recommended action.

    Three bands, not two. A single cut at threshold would force every borderline
    item into a confident bucket, when the whole point of the review queue is
    that uncertain items go to a person.
    """
    if score >= 0.8:
        return "harmful", "queue_for_review"
    if score >= threshold:
        return "review", "queue_for_review"
    return "benign", "no_action"


def priority_score(score: float, is_emergent: bool) -> float:
    """Queue ordering.

    Emergent items are lifted because they are the ones a single-signal system
    would have missed entirely, so a human seeing them is worth more than a
    human seeing another obvious slur the text classifier already caught.
    """
    return min(1.0, score + (0.12 if is_emergent else 0.0))
