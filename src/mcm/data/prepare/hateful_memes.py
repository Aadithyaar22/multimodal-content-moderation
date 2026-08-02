"""Normalize the Hateful Memes Challenge dataset.

This is the dataset the whole fusion thesis rests on: it was built by Meta so
that neither modality alone is offensive — the harm only appears in the
image-text relationship. It supplies the toxicity head's labels and the headline
"fusion vs unimodal recall delta" figure (PROJECT_CONTEXT.md Sec. 6).

Licensing note: the original dataset is released under Meta's research licence
via DrivenData. This pipeline reads a public HuggingFace mirror; anyone using it
is still bound by the original terms, which is recorded in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from mcm.config import DATA_DIR, DatasetSpec, raw_dir
from mcm.data.manifest import write_manifest
from mcm.data.schema import Record, label_summary, make_uid, records_to_frame
from mcm.utils.logging import get_logger

log = get_logger(__name__)

DATASET = "hateful_memes"


def download(spec: DatasetSpec) -> Path:
    """Fetch jsonl split files and the image directory into data/raw."""
    target = raw_dir(DATASET)
    log.info("downloading %s -> %s (images included, this is the slow part)", spec.hf_repo, target)
    snapshot_download(
        repo_id=spec.hf_repo,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["*.jsonl", "img/*", "LICENSE.txt", "README.md"],
        max_workers=8,
    )
    return target


def prepare(spec: DatasetSpec) -> dict[str, Path]:
    root = download(spec)
    opts = spec.options
    split_files = {
        "train": opts.get("train_split_file", "train.jsonl"),
        "val": opts.get("val_split_file", "dev_seen.jsonl"),
        "test": opts.get("test_split_file", "test_seen.jsonl"),
    }

    written: dict[str, Path] = {}
    for split, filename in split_files.items():
        src = root / filename
        if not src.exists():
            raise FileNotFoundError(f"expected split file {src} after download")

        records, missing = [], 0
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)

            # Manifests store paths relative to DATA_DIR so they stay portable.
            img_abs = root / row["img"]
            if not img_abs.exists():
                missing += 1
                continue
            img_rel = img_abs.relative_to(DATA_DIR).as_posix()

            records.append(
                Record(
                    uid=make_uid(DATASET, row["id"]),
                    dataset=DATASET,
                    split=split,
                    text=row["text"],
                    image_path=img_rel,
                    has_image=True,
                    # Hateful Memes carries no misinformation label; the
                    # sentinel keeps that head out of the loss for these rows.
                    label_toxicity=int(row["label"]),
                )
            )

        if missing:
            log.warning("%s/%s: skipped %d rows with no image on disk", DATASET, split, missing)

        df = records_to_frame(records)
        written[split] = write_manifest(df, DATASET, split)
        log.info("%s/%s\n%s", DATASET, split, label_summary(df))

    _warn_on_leakage(written)
    return written


def _warn_on_leakage(written: dict[str, Path]) -> None:
    """Hateful Memes' dev_seen and test_seen overlap in some mirrors.

    A silent train/test overlap would inflate every number in the ablation
    table, so it is checked at build time rather than trusted.
    """
    frames = {s: pd.read_parquet(p, columns=["uid"]) for s, p in written.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in frames and b in frames:
            overlap = set(frames[a]["uid"]) & set(frames[b]["uid"])
            if overlap:
                log.warning(
                    "%s: %d uids appear in BOTH %s and %s — results will be inflated",
                    DATASET,
                    len(overlap),
                    a,
                    b,
                )
