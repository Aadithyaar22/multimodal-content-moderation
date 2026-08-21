#!/usr/bin/env python
"""Train the serving checkpoints at their best validated configurations.

    python scripts/export_models.py
    python scripts/export_models.py --dataset hateful_memes --arch cross_attention

Six checkpoints: the three arms on each benchmark. The API reports per-modality
scores next to the fused one, so the vision-only and language-only heads are not
just ablation artefacts — they are what the interface compares against, and the
emergent-case flag is derived from them at request time.

Each benchmark keeps its own trio because the arms were trained and validated
per benchmark. Serving a Hateful Memes model's misinformation head, which never
saw a misinformation label, would produce a confident number with nothing behind
it.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from mcm.config import CHECKPOINT_DIR
from mcm.training.trainer import TrainConfig, _is_token_arch, train
from mcm.utils import console

# Best validation configurations from scripts/sweep_fusion.py.
BEST: dict[str, dict[str, dict]] = {
    "hateful_memes": {
        "cv_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
        "nlp_only": {"hidden_dim": 256, "lr": 1e-3, "dropout": 0.3},
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=sorted(BEST))
    ap.add_argument("--arch", nargs="+", default=["cv_only", "nlp_only", "cross_attention"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for dataset in args.dataset:
        for arch in args.arch:
            overrides = BEST[dataset][arch]
            cfg = TrainConfig(
                arch=arch,
                datasets=[dataset],
                seed=args.seed,
                epochs=30 if _is_token_arch(arch) else 60,
                patience=8 if _is_token_arch(arch) else 10,
                batch_size=128,
                **overrides,
            )
            console.rule(f"{dataset} · {arch}")
            result = train(cfg, save_dir=None)

            path = CHECKPOINT_DIR / f"{arch}__{dataset}.pt"
            torch.save(
                {
                    "arch": arch,
                    "dataset": dataset,
                    "task": TASK_FOR[dataset],
                    "config": overrides,
                    "state_dict": result.best_state,
                    "val_macro_f1": result.best_val_score,
                    "test": result.test,
                },
                path,
            )
            console.print(f"  saved -> {path}")

            manifest[f"{arch}__{dataset}"] = {
                "path": path.name,
                "task": TASK_FOR[dataset],
                "val_macro_f1": round(result.best_val_score, 4),
                "test_macro_f1": round(
                    result.test[TASK_FOR[dataset]]["macro_f1"], 4
                ),
                **overrides,
            }

    manifest_path = CHECKPOINT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    console.print(f"\n[green]wrote {manifest_path}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
