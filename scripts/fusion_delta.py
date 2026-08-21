#!/usr/bin/env python
"""The headline figure: recall on cases both unimodal arms missed.

    python scripts/fusion_delta.py --dataset hateful_memes

PROJECT_CONTEXT.md Sec. 6 names this the single most important number in the
report. Overall macro-F1 dilutes the effect with easy samples that either
modality already catches; this measures only the subset the project is actually
about — harmful items where *neither* the vision-only nor the language-only arm
crosses threshold. If cross-modal modelling does anything, it does it here.

All four arms are trained fresh at their recorded best configurations and
evaluated on the identical test frame, so the per-row comparison is exact rather
than an alignment of separately-recorded runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import numpy as np
from scipy import stats

from mcm.data.manifest import load_splits
from mcm.training.metrics import fusion_recall_delta
from mcm.training.trainer import (
    SplitTensors,
    TokenSplitTensors,
    TrainConfig,
    _is_token_arch,
    build_model,
    predict_logits,
    results_dir,
    set_seed,
    train,
)
from mcm.utils import console, get_device

# Best validation configurations found by scripts/sweep_fusion.py.
BEST: dict[str, dict[str, dict]] = {
    "hateful_memes": {
        "cv_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
        "nlp_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
        "late_fusion": {"hidden_dim": 128, "lr": 1e-3, "dropout": 0.1},
        "cross_attention": {
            "d_model": 256,
            "lr": 3e-4,
            "fusion_dropout": 0.3,
            "warmup_epochs": 3,
            "grad_clip": 1.0,
            "weight_decay": 0.05,
        },
    },
    "fakeddit": {
        "cv_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
        "nlp_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
        "late_fusion": {"hidden_dim": 512, "lr": 3e-3, "dropout": 0.5},
        "cross_attention": {
            "d_model": 512,
            "lr": 3e-4,
            "fusion_dropout": 0.3,
            "warmup_epochs": 3,
            "grad_clip": 1.0,
            "weight_decay": 0.05,
        },
    },
}

TASK_FOR = {"hateful_memes": "toxicity", "fakeddit": "misinformation"}
POSITIVE_CLASS = {"toxicity": 1, "misinformation": 2}  # harmful / misleading


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hateful_memes", choices=sorted(BEST))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    dataset = args.dataset
    task = TASK_FOR[dataset]
    device = get_device()

    test_frame = load_splits([dataset], "test")
    labels = test_frame[
        "label_toxicity" if task == "toxicity" else "label_misinfo_3"
    ].to_numpy()

    # The hard subset is defined per seed by that seed's own unimodal arms. A
    # single seed decides this on ~100 items, where a five-item swing looks like
    # a five-point difference, so the whole thing is repeated across seeds.
    per_seed: list[dict] = []
    for seed in args.seeds:
        logits: dict[str, np.ndarray] = {}
        for arm, overrides in BEST[dataset].items():
            console.rule(f"seed {seed} · {arm}")
            cfg = TrainConfig(
                arch=arm,
                datasets=[dataset],
                seed=seed,
                epochs=30 if _is_token_arch(arm) else 60,
                patience=8 if _is_token_arch(arm) else 10,
                batch_size=128,
                **overrides,
            )
            result = train(cfg, save_dir=None)

            holder = TokenSplitTensors if _is_token_arch(arm) else SplitTensors
            data = holder(test_frame, [dataset], "test", device)
            model = _rebuild(cfg, data, device, result)
            logits[arm] = predict_logits(model, data, token_mode=_is_token_arch(arm))[task]

        per_seed.append(measure(task, logits, labels, args.threshold))

    report(dataset, task, per_seed, args.threshold)
    return 0


def measure(task, logits, labels, threshold) -> dict:
    unimodal = [logits["cv_only"], logits["nlp_only"]]
    positive = POSITIVE_CLASS[task]
    return {
        arm: fusion_recall_delta(
            logits[arm], unimodal, labels, positive_class=positive, threshold=threshold
        )
        for arm in ("late_fusion", "cross_attention")
    }


def _rebuild(cfg, data, device, result):
    """Reconstruct the model from the run's best weights.

    train() restores its best-validation state before returning, but does not
    hand the module back, so the arm is rebuilt and reloaded here from the state
    the run settled on.
    """
    set_seed(cfg.seed)
    if _is_token_arch(cfg.arch):
        model = build_model(
            cfg.arch,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.fusion_dropout,
            head_hidden=cfg.hidden_dim,
            head_dropout=cfg.dropout,
        )
    else:
        model = build_model(
            cfg.arch,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            normalize_input=cfg.normalize_input,
        )
    model.load_state_dict(result.best_state)
    return model.to(device).eval()


def report(dataset, task, per_seed: list[dict], threshold) -> None:
    console.rule(f"{dataset}: recall on cases both unimodal arms missed")
    console.print(
        f"  Subset: items labelled '{'harmful' if task == 'toxicity' else 'misleading'}' "
        f"where CV-only AND NLP-only both scored below {threshold}.\n"
        f"  Unimodal recall on this subset is 0 by construction.\n"
    )

    n_hard = [s["late_fusion"]["n_hard_cases"] for s in per_seed]
    console.print(
        f"  hard cases per seed: {n_hard}  (mean {statistics.mean(n_hard):.0f} of "
        f"{per_seed[0]['late_fusion'].get('n_positives', 0)} positives)\n"
    )

    recalls = {}
    for arm in ("late_fusion", "cross_attention"):
        vals = [s[arm]["fusion_recall_on_hard"] for s in per_seed]
        recalls[arm] = vals
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        console.print(f"  {arm:18s} {statistics.mean(vals):.1%} ± {sd:.1%}")

    delta = statistics.mean(recalls["cross_attention"]) - statistics.mean(recalls["late_fusion"])
    console.print(f"\n  Cross-attention - late fusion: {delta:+.1%}")

    if len(per_seed) > 1:
        t_stat, p = stats.ttest_ind(
            recalls["cross_attention"], recalls["late_fusion"], equal_var=False
        )
        console.print(f"  Welch's t-test: t={t_stat:.3f}, p={p:.3f}")
        if p >= 0.05:
            console.print(
                "  [yellow]not separable from seed noise[/] — on this subset neither "
                "fusion arm is measurably better than the other."
            )
        else:
            better = "cross-attention" if delta > 0 else "late fusion"
            console.print(f"  [green]separable from seed noise[/] — {better} is better")
    else:
        console.print("  [yellow]single seed — no error bar; do not interpret[/]")

    path = results_dir() / f"fusion_delta__{dataset}.json"
    path.write_text(
        json.dumps(
            {"threshold": threshold, "per_seed": per_seed, "recalls": recalls}, indent=2
        )
    )
    console.print(f"\n  saved -> {path}")


if __name__ == "__main__":
    sys.exit(main())
