#!/usr/bin/env python
"""Cache token-level CLIP features for the cross-attention arm.

    python scripts/encode_token_features.py --dataset hateful_memes fakeddit

Cross-attention needs the sequences, not the pooled vectors: 50 image patches
(768-d) and 77 text tokens (512-d) per row. That is ~100x the pooled cache, so
these are written as fp16 and read back through a memory map.

Text-only datasets are skipped by default — there is no image for the text to
attend to, so they cannot exercise the mechanism this cache exists to support.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

from mcm.config import processed_manifest
from mcm.data.datasets import ModerationDataset, make_collate_fn
from mcm.data.manifest import read_manifest
from mcm.data.token_features import save_token_features, token_cache_exists
from mcm.models.clip_encoder import TEXT_CONTEXT, FrozenCLIP
from mcm.utils import console, get_device
from mcm.utils.device import empty_cache

MULTIMODAL_DATASETS = ("hateful_memes", "fakeddit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=list(MULTIMODAL_DATASETS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    clip = FrozenCLIP(device=device)
    # Fixed-length padding: every row must be the same width to stack into one
    # memmapped array, unlike the pooled cache which could pad per batch.
    collate = make_collate_fn(tokenizer=None)

    total = 0
    for dataset in args.dataset:
        for split in ("train", "val", "test"):
            if not processed_manifest(dataset, split).exists():
                continue
            if token_cache_exists(dataset, split) and not args.force:
                console.print(f"[dim]skip {dataset}/{split} (cached; --force to redo)[/]")
                continue

            frame = read_manifest(dataset, split)
            console.rule(f"{dataset}/{split}  n={len(frame)}")
            total += encode(clip, collate, dataset, split, frame, args, device)

    console.print(f"\n[green]encoded {total:,} rows[/]")
    return 0


def encode(clip, collate, dataset, split, frame, args, device) -> int:
    ds = ModerationDataset(frame, image_processor=clip.image_processor)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    uids: list[str] = []
    image_chunks, text_chunks, attn_chunks, mask_chunks = [], [], [], []

    t0 = time.time()
    seen = 0
    for batch in loader:
        text_inputs = clip.tokenize(batch.text, max_length=TEXT_CONTEXT)
        feats = clip.encode_tokens(pixel_values=batch.pixel_values, text_inputs=text_inputs)

        image_tokens = feats.image_tokens.cpu().numpy().astype(np.float16)
        # Zero the image stream for rows that have no image, so a blank picture
        # cannot become a consistent learnable signal.
        image_tokens[~batch.image_mask.numpy()] = 0.0

        uids.extend(batch.uid)
        image_chunks.append(image_tokens)
        text_chunks.append(feats.text_tokens.cpu().numpy().astype(np.float16))
        attn_chunks.append(text_inputs["attention_mask"].cpu().numpy().astype(np.uint8))
        mask_chunks.append(batch.image_mask.numpy())

        seen += len(batch)
        if seen % (args.batch_size * 20) == 0:
            console.print(f"  {seen}/{len(ds)}  ({seen / (time.time() - t0):.0f} rows/s)", highlight=False)

    save_token_features(
        dataset,
        split,
        uid=uids,
        image_tokens=np.concatenate(image_chunks),
        text_tokens=np.concatenate(text_chunks),
        text_attention_mask=np.concatenate(attn_chunks),
        image_mask=np.concatenate(mask_chunks),
    )
    empty_cache()
    console.print(f"  done in {time.time() - t0:.1f}s")
    return len(ds)


if __name__ == "__main__":
    sys.exit(main())
