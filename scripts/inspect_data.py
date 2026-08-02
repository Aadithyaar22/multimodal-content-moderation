#!/usr/bin/env python
"""Verify prepared manifests end to end.

    python scripts/inspect_data.py
    python scripts/inspect_data.py --datasets hateful_memes fakeddit --batch-size 16

Loads every available manifest, reports class balance, then builds one real
mixed-dataset batch through the actual collate path and pushes it to MPS. This
is the check that the data layer is genuinely ready for training rather than
merely present on disk.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mcm.config import load_data_config, processed_manifest
from mcm.data.datasets import ModerationDataset, class_weights, make_collate_fn
from mcm.data.manifest import load_splits, read_manifest, resolve_image
from mcm.data.schema import IGNORE_INDEX, label_summary
from mcm.utils import console, device_report, get_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", help="defaults to every prepared dataset")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--check-images", type=int, default=200, help="images to stat per split")
    args = ap.parse_args()

    console.rule("environment")
    for k, v in device_report().items():
        console.print(f"  {k}: {v}")

    names = args.datasets or list(load_data_config())
    available = [n for n in names if processed_manifest(n, "train").exists()]
    skipped = sorted(set(names) - set(available))
    if skipped:
        console.print(f"\n[yellow]not prepared yet, skipping:[/] {', '.join(skipped)}")
    if not available:
        console.print("[red]no prepared datasets found.[/] Run: python scripts/prepare_data.py --all")
        return 1

    totals = {"rows": 0, "images": 0}
    for name in available:
        console.rule(f"{name}")
        for split in ("train", "val", "test"):
            try:
                df = read_manifest(name, split)
            except FileNotFoundError:
                console.print(f"  {split}: [yellow]missing[/]")
                continue

            console.print(f"  [bold]{split}[/]")
            for line in label_summary(df).splitlines():
                console.print(f"    {line}")

            missing = _check_images(df, args.check_images)
            if missing:
                console.print(f"    [red]{missing} of {args.check_images} sampled images missing[/]")

            totals["rows"] += len(df)
            totals["images"] += int(df["has_image"].sum())

        train = read_manifest(name, "train")
        for col in ("label_toxicity", "label_misinfo_3"):
            w = class_weights(train, col)
            if w is not None:
                console.print(f"    {col} loss weights: {[round(x, 3) for x in w.tolist()]}")

    console.rule("mixed batch")
    _smoke_batch(available, args.batch_size)

    console.rule("summary")
    console.print(f"  {totals['rows']:,} rows across {len(available)} dataset(s)")
    console.print(f"  {totals['images']:,} with images, {totals['rows'] - totals['images']:,} text-only")
    return 0


def _check_images(df, n: int) -> int:
    with_img = df[df["has_image"]]
    if with_img.empty:
        return 0
    sample = with_img.sample(min(n, len(with_img)), random_state=0)
    return sum(1 for p in sample["image_path"] if not resolve_image(p).exists())


def _smoke_batch(datasets: list[str], batch_size: int) -> None:
    """Build one real batch through the training data path.

    The batch is drawn evenly from each dataset rather than sampled at random.
    A random draw can easily contain no rows from one source, which would leave
    the sentinel-masking assertion below with nothing to check while still
    printing a pass — a green tick for a check that never ran.
    """
    mixture = load_splits(datasets, "train")
    per_source = max(1, batch_size // len(datasets))
    parts = [
        group.sample(min(len(group), per_source), random_state=0)
        for _, group in mixture.groupby("dataset")
    ]
    stratified = pd.concat(parts).reset_index(drop=True)
    ds = ModerationDataset(stratified)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=make_collate_fn())
    batch = next(iter(loader))

    device = get_device()
    batch.to(device)

    console.print(f"  sources in batch: {sorted(set(batch.dataset))}")
    console.print(f"  pixel_values: {tuple(batch.pixel_values.shape)} on {batch.pixel_values.device}")
    console.print(f"  image_mask: {batch.image_mask.tolist()}")
    console.print(f"  label_toxicity:  {batch.label_toxicity.tolist()}")
    console.print(f"  label_misinfo_3: {batch.label_misinfo_3.tolist()}")

    # The masking contract, exercised on real data: rows whose label is the
    # sentinel must contribute no gradient to that head.
    logits = torch.randn(len(batch), 2, device=device, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(
        logits, batch.label_toxicity, ignore_index=IGNORE_INDEX
    )
    loss.backward()
    ignored = (batch.label_toxicity == IGNORE_INDEX).nonzero().flatten().tolist()
    scored = [i for i in range(len(batch)) if i not in ignored]
    if not ignored:
        verdict = "[yellow]NOT EXERCISED — no ignored rows in batch[/]"
    elif all(torch.count_nonzero(logits.grad[i]) == 0 for i in ignored) and any(
        torch.count_nonzero(logits.grad[i]) > 0 for i in scored
    ):
        verdict = "[green]yes[/]"
    else:
        verdict = "[red]NO[/]"
    console.print(
        f"  toxicity loss={loss.item():.4f}  ignored rows={len(ignored)}/{len(batch)}  "
        f"zero-grad on ignored: {verdict}"
    )


if __name__ == "__main__":
    sys.exit(main())
