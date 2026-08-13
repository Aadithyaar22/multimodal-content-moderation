"""Evaluation metrics for the report's results chapter.

Metric choice follows PROJECT_CONTEXT.md Sec. 6 rather than defaulting to
accuracy. Accuracy is close to meaningless here: Fakeddit's 3-way collapse is
61% "misleading", so a model that predicts one class forever already scores 61%.
Macro-F1 and per-class recall are what actually distinguish the arms.

Calibration is included because the deployment framing is a *ranked review
queue* for human moderators. A queue is only as good as the ordering, and a
model whose 0.9 confidence is right 60% of the time produces a queue that wastes
moderator attention even at high accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from mcm.data.schema import IGNORE_INDEX


@dataclass
class TaskMetrics:
    """Metrics for one head, computed only over rows that had a label for it."""

    task: str
    n: int
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    auc: float | None = None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    ece: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_task(
    task: str,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
) -> TaskMetrics:
    """Score one head. Rows labelled IGNORE_INDEX are excluded, not counted wrong."""
    applicable = labels != IGNORE_INDEX
    if not applicable.any():
        return TaskMetrics(task=task, n=0)

    logits = logits[applicable]
    labels = labels[applicable]
    probs = _softmax(logits)
    preds = probs.argmax(axis=1)
    n_classes = logits.shape[1]

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_classes)), zero_division=0
    )

    per_class = {
        class_names.get(i, str(i)): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    return TaskMetrics(
        task=task,
        n=int(applicable.sum()),
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(labels, preds, average="weighted", zero_division=0)),
        auc=_safe_auc(labels, probs, n_classes),
        per_class=per_class,
        confusion=confusion_matrix(labels, preds, labels=list(range(n_classes))).tolist(),
        ece=expected_calibration_error(probs, labels),
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _safe_auc(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> float | None:
    """AUC, or None when it is undefined for this slice.

    A split where every row shares one label has no AUC. Returning None makes
    that visible in the results table instead of reporting a fabricated 0.5.
    """
    if len(set(labels.tolist())) < 2:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    Directly relevant to the soft-flag threshold: a moderator queue sorted by a
    miscalibrated score puts the wrong items on top.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def fusion_recall_delta(
    fusion_logits: np.ndarray,
    unimodal_logits: list[np.ndarray],
    labels: np.ndarray,
    positive_class: int = 1,
    threshold: float = 0.5,
) -> dict[str, float]:
    """The report's headline figure (PROJECT_CONTEXT.md Sec. 6).

    Measures recall on exactly the hard cases: harmful samples that *every*
    unimodal arm scored below threshold. Those are the "neither modality alone
    flags it" cases the whole project is premised on, so overall recall would
    dilute the effect with easy samples both arms already catch.

    Reported alongside the subset size, because a large delta over 20 samples is
    not a result.
    """
    applicable = labels != IGNORE_INDEX
    labels = labels[applicable]
    fusion_probs = _softmax(fusion_logits[applicable])[:, positive_class]
    unimodal_probs = [_softmax(u[applicable])[:, positive_class] for u in unimodal_logits]

    is_positive = labels == positive_class
    all_unimodal_missed = np.ones_like(is_positive, dtype=bool)
    for p in unimodal_probs:
        all_unimodal_missed &= p < threshold

    hard = is_positive & all_unimodal_missed
    n_hard = int(hard.sum())
    if n_hard == 0:
        return {"n_hard_cases": 0, "fusion_recall_on_hard": 0.0, "recall_delta": 0.0}

    fusion_caught = int((fusion_probs[hard] >= threshold).sum())
    return {
        "n_hard_cases": n_hard,
        "n_positives": int(is_positive.sum()),
        "hard_fraction_of_positives": float(n_hard / max(1, is_positive.sum())),
        "fusion_recall_on_hard": float(fusion_caught / n_hard),
        # Unimodal recall on this subset is 0 by construction, so the delta is
        # the fusion recall itself. Stated explicitly so the table is readable
        # without re-deriving why.
        "recall_delta": float(fusion_caught / n_hard),
    }
