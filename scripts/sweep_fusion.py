#!/usr/bin/env python
"""Hyperparameter search for the cross-attention arm.

    python scripts/sweep_fusion.py --datasets hateful_memes

Why this exists
---------------
The first cross-attention run reused the baselines' hyperparameters and lost to
late fusion. Those settings were chosen for a ~0.4M-parameter MLP head; the
fusion arm has ~14M parameters on 8500 training rows, which is 35x the
parameters-per-sample ratio, and every seed peaked by epoch 16 and then overfit.
Comparing under those conditions tests whether an over-parameterized model with
unsuitable optimizer settings wins, not whether cross-modal attention helps.

So the fusion arm gets a search over the settings that actually matter for it —
capacity, learning rate, regularization — and the winner is then run across
seeds for the ablation table.

**Selection is on validation macro-F1 only. Test is never consulted here.**
Reporting a configuration chosen on test would make the headline number
meaningless, so the sweep does not even read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import product

from mcm.training.trainer import TrainConfig, results_dir, train
from mcm.utils import console

# Deliberately small and targeted: capacity and learning rate are the two knobs
# the diagnosis pointed at, and every extra cell costs ~10 minutes on this machine.
#
# The pooled arms get an equivalent search over their own capacity/regularization
# knobs. Tuning only the proposed architecture and leaving the baselines on
# defaults is the most common way an ablation table flatters its own thesis, so
# every arm is given the same courtesy before anything is compared.
GRIDS: dict[str, dict[str, list]] = {
    "cross_attention": {
        "d_model": [256, 512],
        "lr": [3e-4, 1e-3],
        "fusion_dropout": [0.1, 0.3],
    },
    "late_fusion": {
        "hidden_dim": [128, 256, 512],
        "lr": [3e-4, 1e-3, 3e-3],
        "dropout": [0.1, 0.3, 0.5],
    },
    "cv_only": {
        "hidden_dim": [128, 256, 512],
        "lr": [3e-4, 1e-3, 3e-3],
        "dropout": [0.1, 0.3, 0.5],
    },
    "nlp_only": {
        "hidden_dim": [128, 256, 512],
        "lr": [3e-4, 1e-3, 3e-3],
        "dropout": [0.1, 0.3, 0.5],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="cross_attention", choices=sorted(GRIDS))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1, help="single seed for the search")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    args = ap.parse_args()

    token_arm = args.arch == "cross_attention"
    base = TrainConfig(
        arch=args.arch,
        datasets=args.datasets,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        # Warmup, clipping and heavy decay exist for the transformer arm; forcing
        # them onto the pooled arms would change their recorded results for a
        # reason unrelated to the comparison.
        warmup_epochs=args.warmup_epochs if token_arm else 0,
        grad_clip=args.grad_clip if token_arm else 0.0,
        weight_decay=args.weight_decay if token_arm else 1e-4,
    )

    GRID = GRIDS[args.arch]
    keys = list(GRID)
    trials = []
    for values in product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values, strict=True))
        cfg = replace(base, **overrides)
        console.rule(f"{overrides}")
        result = train(cfg, save_dir=None)  # not saved: search runs are not results
        trials.append(
            {
                **overrides,
                "val_macro_f1": result.best_val_score,
                "best_epoch": result.best_epoch,
                "seconds": round(result.train_seconds, 1),
            }
        )
        console.print(f"  val={result.best_val_score:.4f} @ epoch {result.best_epoch}")

    trials.sort(key=lambda t: -t["val_macro_f1"])
    console.rule(f"{args.arch}: sweep results (ranked by VALIDATION macro-F1)")
    for t in trials:
        params = "  ".join(f"{k}={t[k]}" for k in keys)
        console.print(
            f"  val={t['val_macro_f1']:.4f}  {params}  (epoch {t['best_epoch']}, {t['seconds']}s)"
        )

    out = results_dir() / f"sweep_{args.arch}__{'+'.join(args.datasets)}.json"
    out.write_text(json.dumps(trials, indent=2))
    console.print(f"\n  saved -> {out}")

    best = trials[0]
    flags = " ".join(f"--{k.replace('_', '-')} {best[k]}" for k in keys)
    extra = (
        f" --warmup-epochs {args.warmup_epochs} --grad-clip {args.grad_clip}"
        f" --weight-decay {args.weight_decay}"
        if token_arm
        else ""
    )
    console.print(f"\n[green]best config[/]: {'  '.join(f'{k}={best[k]}' for k in keys)}")
    console.print(
        "  run it across seeds with:\n"
        f"  python scripts/train_baseline.py --arch {args.arch} "
        f"--datasets {' '.join(args.datasets)} --seeds 1 2 3 {flags}{extra}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
