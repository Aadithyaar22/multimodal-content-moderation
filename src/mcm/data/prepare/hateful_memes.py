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
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from mcm.config import DATA_DIR, DatasetSpec, raw_dir
from mcm.data.manifest import write_manifest
from mcm.data.schema import Record, label_summary, make_uid, records_to_frame
from mcm.utils.logging import get_logger

log = get_logger(__name__)

DATASET = "hateful_memes"


def download(spec: DatasetSpec, attempts: int = 5) -> Path:
    """Fetch jsonl split files and the image directory into data/raw.

    This is ~3.3GB across 9664 small files, and long multi-file transfers from
    the Hub fail intermittently — observed here as a mid-download 401 from the
    Xet CAS backend after several thousand files. ``snapshot_download`` is
    resumable, so each retry only fetches what is still missing, and the Xet
    transfer path is disabled on later attempts because the plain HTTP path is
    slower but markedly more reliable for a run this long.
    """
    target = raw_dir(DATASET)
    patterns = ["*.jsonl", "img/*", "LICENSE.txt", "README.md"]

    for attempt in range(1, attempts + 1):
        have = len(list((target / "img").glob("*"))) if (target / "img").exists() else 0
        log.info(
            "downloading %s -> %s (attempt %d/%d, %d images already local)",
            spec.hf_repo,
            target,
            attempt,
            attempts,
            have,
        )
        if attempt > 1:
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        try:
            snapshot_download(
                repo_id=spec.hf_repo,
                repo_type="dataset",
                local_dir=target,
                allow_patterns=patterns,
                max_workers=4 if attempt > 1 else 8,
            )
            return target
        except Exception as e:  # noqa: BLE001
            if attempt == attempts:
                raise
            wait = min(30, 2**attempt)
            log.warning(
                "download attempt %d failed (%s: %s); resuming in %ds",
                attempt,
                type(e).__name__,
                str(e)[:120],
                wait,
            )
            time.sleep(wait)

    return target


def backfill_missing_images(root: Path, split_files: list[str], backfill_repo: str | None) -> int:
    """Fetch split images the primary mirror does not carry.

    The primary mirror ships 9664 images, but 1707 of those belong to the
    ``unseen`` splits and 2043 images referenced by train/dev_seen/test_seen are
    absent — so a naive run silently drops 20% of the dataset, including 20% of
    the test set the headline fusion result is measured on. The expanded mirror
    happens to carry exactly those 2043, so they are backfilled here and the
    dataset reaches its full 10000 samples.

    Returns the number of images fetched.
    """
    if not backfill_repo:
        return 0

    referenced: set[str] = set()
    for filename in split_files:
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                referenced.add(json.loads(line)["img"])

    missing = sorted(rel for rel in referenced if not (root / rel).exists())
    if not missing:
        return 0

    log.warning(
        "%d split images absent from the primary mirror; backfilling from %s",
        len(missing),
        backfill_repo,
    )

    def _grab(rel: str) -> bool:
        try:
            src = hf_hub_download(backfill_repo, rel, repo_type="dataset")
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return True
        except Exception:  # noqa: BLE001
            return False

    fetched = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ok in tqdm(pool.map(_grab, missing), total=len(missing), desc="backfill", unit="img"):
            fetched += bool(ok)

    still_missing = len(missing) - fetched
    log.info("backfilled %d/%d images", fetched, len(missing))
    if still_missing:
        log.warning("%d images unavailable from either mirror; those rows will be dropped", still_missing)
    return fetched


def prepare(spec: DatasetSpec) -> dict[str, Path]:
    root = download(spec)
    opts = spec.options
    split_files = {
        "train": opts.get("train_split_file", "train.jsonl"),
        "val": opts.get("val_split_file", "dev_seen.jsonl"),
        "test": opts.get("test_split_file", "test_seen.jsonl"),
    }

    backfill_missing_images(root, list(split_files.values()), opts.get("backfill_repo"))

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
