#!/usr/bin/env python
"""Precompute and cache frozen-CLIP embeddings for every prepared split.

    python scripts/encode_features.py --all
    python scripts/encode_features.py --dataset hateful_memes --batch-size 64

The backbone never trains, so these embeddings are constants. Computing them
once here is what lets every ablation arm train in seconds on cached vectors.

Idempotent: splits that already have a cache are skipped unless --force.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader

from mcm.config import load_data_config, processed_manifest
from mcm.data.datasets import ModerationDataset, make_collate_fn
from mcm.data.features import feature_path, save_features
from mcm.data.manifest import read_manifest
from mcm.models.clip_encoder import FrozenCLIP
from mcm.utils import console, get_device
from mcm.utils.device import empty_cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-encode splits that already have a cache")
    args = ap.parse_args()

    names = list(load_data_config()) if args.all else args.dataset
    if not names:
        ap.error("pass --dataset NAME [NAME ...] or --all")

    device = get_device()
    clip = FrozenCLIP(device=device)
    collate = make_collate_fn(tokenizer=clip.processor.tokenizer, max_length=77)

    total = 0
    for dataset in names:
        for split in ("train", "val", "test"):
            if not processed_manifest(dataset, split).exists():
                continue
            path = feature_path(dataset, split)
            if path.exists() and not args.force:
                console.print(f"[dim]skip {dataset}/{split} (cached; --force to redo)[/]")
                continue

            frame = read_manifest(dataset, split)
            console.rule(f"{dataset}/{split}  n={len(frame)}")
            total += encode_split(
                clip, collate, dataset, split, frame, args.batch_size, args.num_workers, device
            )

    console.print(f"\n[green]encoded {total:,} rows[/]")
    return 0


def encode_split(clip, collate, dataset, split, frame, batch_size, num_workers, device) -> int:
    ds = ModerationDataset(frame, image_processor=clip.image_processor)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # order must match the manifest; uids are saved regardless
        num_workers=num_workers,
        collate_fn=collate,
    )

    uids: list[str] = []
    image_chunks, text_chunks, mask_chunks = [], [], []

    t0 = time.time()
    seen = 0
    for batch in loader:
        feats = clip.encode_pooled(
            pixel_values=batch.pixel_values,
            text_inputs=batch.text_inputs,
        )

        # Text-only rows were collated with a zero image. CLIP still produces an
        # embedding for that blank input, and it is a *consistent* vector, so
        # leaving it in would hand the model a reliable "this row is HateXplain"
        # signal that has nothing to do with content. Zero it out and let
        # image_mask carry the information honestly.
        image_emb = feats.image_emb.clone()
        image_emb[~batch.image_mask.to(image_emb.device)] = 0.0

        uids.extend(batch.uid)
        image_chunks.append(image_emb.cpu())
        text_chunks.append(feats.text_emb.cpu())
        mask_chunks.append(batch.image_mask.cpu())

        seen += len(batch)
        if seen % (batch_size * 20) == 0:
            rate = seen / (time.time() - t0)
            console.print(f"  {seen}/{len(ds)}  ({rate:.0f} rows/s)", highlight=False)

    save_features(
        dataset,
        split,
        uid=uids,
        image_emb=torch.cat(image_chunks),
        text_emb=torch.cat(text_chunks),
        image_mask=torch.cat(mask_chunks),
    )
    empty_cache()
    console.print(f"  done in {time.time() - t0:.1f}s")
    return len(ds)


if __name__ == "__main__":
    sys.exit(main())
