#!/usr/bin/env python
"""Fit temperature scaling on validation and write it into the checkpoints.

    python scripts/calibrate.py

Why this is needed
------------------
The trained arms are badly overconfident. Measured expected calibration error on
test is 0.28 for the Hateful Memes fusion arm and 0.17 for Fakeddit, against
accuracies of 0.69 and 0.82 — so a "90% confident" prediction is right closer to
60% of the time, and the served probabilities saturate to exactly 1.0. Training
loss reaching 0.0008 while validation plateaued is the cause: the model fits the
training set long after it stops learning anything generalisable, and softmax
confidence runs away with it.

This matters more here than in a plain classifier. The product is a *ranked
review queue*, and the soft-flag threshold in PROJECT_CONTEXT Sec. 6 is a
confidence cut. A moderator working a queue ordered by miscalibrated scores
spends attention in the wrong order regardless of how accurate the argmax is.

Temperature scaling divides the logits by a single scalar fitted to minimise
validation NLL. It is the right tool precisely because it cannot change any
prediction: dividing by a positive constant preserves argmax and preserves
ranking within a head, so accuracy and macro-F1 are untouched and only the
probabilities move. Anything that changed the predictions would invalidate the
ablation table already reported.

Fitted on validation, never on test — a temperature fitted on the split it is
then evaluated on would report a calibration the deployed model does not have.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mcm.config import CHECKPOINT_DIR
from mcm.data.manifest import load_splits
from mcm.data.schema import IGNORE_INDEX
from mcm.serving.inference import _build
from mcm.training.metrics import expected_calibration_error
from mcm.training.trainer import (
    SplitTensors,
    TokenSplitTensors,
    _is_token_arch,
    predict_logits,
)
from mcm.utils import console, get_device

TASK_FOR = {"hateful_memes": "toxicity", "fakeddit": "misinformation"}
LABEL_COL = {"toxicity": "label_toxicity", "misinformation": "label_misinfo_3"}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Minimise validation NLL over a single scalar temperature."""
    keep = labels != IGNORE_INDEX
    z = torch.from_numpy(logits[keep]).float()
    y = torch.from_numpy(labels[keep]).long()

    # Optimised in log space so the temperature cannot go non-positive, which
    # would flip the ordering rather than merely soften it.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["hateful_memes", "fakeddit"])
    ap.add_argument("--arch", nargs="+", default=["cv_only", "nlp_only", "cross_attention"])
    args = ap.parse_args()

    device = get_device()

    for dataset in args.dataset:
        task = TASK_FOR[dataset]
        col = LABEL_COL[task]
        val_frame = load_splits([dataset], "val")
        test_frame = load_splits([dataset], "test")

        for arch in args.arch:
            path = CHECKPOINT_DIR / f"{arch}__{dataset}.pt"
            if not path.exists():
                continue

            blob = torch.load(path, map_location=device, weights_only=False)
            model = _build(arch, blob["config"]).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()

            token_mode = _is_token_arch(arch)
            holder = TokenSplitTensors if token_mode else SplitTensors
            val = holder(val_frame, [dataset], "val", device)
            test = holder(test_frame, [dataset], "test", device)

            val_logits = predict_logits(model, val, token_mode=token_mode)[task]
            test_logits = predict_logits(model, test, token_mode=token_mode)[task]
            val_labels = val_frame[col].to_numpy()
            test_labels = test_frame[col].to_numpy()

            t = fit_temperature(val_logits, val_labels)

            before = _ece(test_logits, test_labels)
            after = _ece(test_logits / t, test_labels)
            acc_before = _accuracy(test_logits, test_labels)
            acc_after = _accuracy(test_logits / t, test_labels)

            blob["temperature"] = t
            torch.save(blob, path)

            console.print(
                f"  {arch:16s} {dataset:14s} T={t:5.2f}  "
                f"ECE {before:.4f} -> {after:.4f}   "
                f"acc {acc_before:.4f} -> {acc_after:.4f}"
            )
            # Accuracy must be identical: a positive scalar divisor cannot move
            # an argmax. If this ever trips, the temperature is not a temperature.
            assert abs(acc_before - acc_after) < 1e-9, "temperature changed a prediction"

    console.print("\n[green]temperatures written into checkpoints[/]")
    return 0


def _ece(logits: np.ndarray, labels: np.ndarray) -> float:
    keep = labels != IGNORE_INDEX
    probs = _softmax(logits[keep])
    return expected_calibration_error(probs, labels[keep])


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    keep = labels != IGNORE_INDEX
    return float((logits[keep].argmax(1) == labels[keep]).mean())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    sys.exit(main())
