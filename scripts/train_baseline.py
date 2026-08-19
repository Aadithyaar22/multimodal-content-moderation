#!/usr/bin/env python
"""Train one or more ablation arms on cached CLIP features.

    # single arm, single seed
    python scripts/train_baseline.py --arch cv_only --datasets hateful_memes

    # the unimodal floor on both benchmarks, 5 seeds each
    python scripts/train_baseline.py --arch cv_only nlp_only \
        --datasets hateful_memes --seeds 1 2 3 4 5

    # everything available so far
    python scripts/train_baseline.py --all-arms --datasets hateful_memes --seeds 1 2 3

Results are written to reports/results/*.json, one file per run, so the ablation
table can be assembled mechanically instead of transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

from mcm.models.baselines import ARCHITECTURES
from mcm.training.trainer import TrainConfig, results_dir, train
from mcm.utils import console


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", choices=ARCHITECTURES)
    ap.add_argument("--all-arms", action="store_true", help=f"run all of {', '.join(ARCHITECTURES)}")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # Cross-attention only; ignored by the pooled arms.
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--fusion-dropout", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    args = ap.parse_args()

    arches = list(ARCHITECTURES) if args.all_arms else args.arch
    if not arches:
        ap.error("pass --arch NAME [NAME ...] or --all-arms")

    out_dir = results_dir()
    runs = []

    for arch in arches:
        for seed in args.seeds:
            cfg = TrainConfig(
                arch=arch,
                datasets=args.datasets,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                patience=args.patience,
                use_class_weights=not args.no_class_weights,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                n_layers=args.n_layers,
                fusion_dropout=args.fusion_dropout,
                warmup_epochs=args.warmup_epochs,
                grad_clip=args.grad_clip,
            )
            console.rule(f"{arch}  seed={seed}  [{'+'.join(args.datasets)}]")
            runs.append((arch, train(cfg, save_dir=out_dir)))

    summarize(runs, args.seeds)
    return 0


def summarize(runs, seeds) -> None:
    """Aggregate across seeds so the spread is visible, not just the mean."""
    console.rule("summary")
    grouped = defaultdict(lambda: defaultdict(list))
    for arch, result in runs:
        for task, m in result.test.items():
            if m["n"]:
                grouped[arch][task].append(m["macro_f1"])

    console.print(f"  test macro-F1, mean ± sd over {len(seeds)} seed(s)\n")
    for arch, tasks in grouped.items():
        for task, scores in tasks.items():
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            spread = f" ± {sd:.4f}" if len(scores) > 1 else ""
            console.print(f"  {arch:14s} {task:16s} {statistics.mean(scores):.4f}{spread}")

    index = results_dir() / "index.json"
    index.write_text(
        json.dumps(
            {arch: {t: s for t, s in tasks.items()} for arch, tasks in grouped.items()}, indent=2
        )
    )
    console.print(f"\n  aggregate -> {index}")


if __name__ == "__main__":
    sys.exit(main())
