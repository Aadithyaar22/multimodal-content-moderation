"""Normalize Fakeddit into the misinformation head's label space.

Two tiers, selected by ``options.tier`` in configs/data.yaml:

``offline``
    A mirror that bundles ~8.6k posts together with their images. No scraping,
    no dead links, runs on a plane. This is the default so the pipeline is
    reproducible for anyone cloning the repo.

``scale``
    The full 794k-row metadata table, sampled and stratified, with images
    fetched from their original URLs. Larger and closer to the published
    benchmark, but a meaningful fraction of 2019-era Reddit links are dead, so
    the realized sample is always smaller than requested.

Both tiers emit the same schema, so switching tiers does not touch model code.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from mcm.config import DATA_DIR, DatasetSpec, raw_dir
from mcm.data.image_fetch import fetch_images
from mcm.data.manifest import write_manifest
from mcm.data.schema import (
    MISINFO_6_TO_3,
    Record,
    label_summary,
    make_uid,
    records_to_frame,
)
from mcm.utils.logging import get_logger

log = get_logger(__name__)

DATASET = "fakeddit"


def prepare(spec: DatasetSpec) -> dict[str, Path]:
    tier = spec.options.get("tier", "offline")
    if tier == "offline":
        return _prepare_offline(spec)
    if tier == "scale":
        return _prepare_scale(spec)
    raise ValueError(f"unknown fakeddit tier {tier!r}; expected 'offline' or 'scale'")


# --------------------------------------------------------------------------- #
# Offline tier
# --------------------------------------------------------------------------- #

def _prepare_offline(spec: DatasetSpec) -> dict[str, Path]:
    repo = spec.options.get("offline_repo", "ams-99/fakeddit_9k")
    root = raw_dir(DATASET)
    img_root = _ensure_offline_images(repo, root)

    # Basename -> path, because the CSV's image_path column points at the
    # original author's Google Drive mount, not at anything on this machine.
    # The basename is a content hash, so it doubles as an image identity key.
    by_name = {p.stem: p for p in img_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    log.info("offline image cache: %d images under %s", len(by_name), img_root)

    split_files = {
        "train": "train_balanced.csv",
        "val": "val_balanced.csv",
        "test": "test_balanced.csv",
    }

    frames: dict[str, pd.DataFrame] = {}
    for split, filename in split_files.items():
        df = pd.read_csv(Path(hf_hub_download(repo, filename, repo_type="dataset")))
        df["image_key"] = df["image_path"].map(lambda p: Path(str(p)).stem)
        frames[split] = df

    leaked = _find_leaked_images(frames)

    written: dict[str, Path] = {}
    for split, df in frames.items():
        # An image reused across splits lets the model memorize it in training
        # and be rewarded for the memory at evaluation. Dropping from the eval
        # side keeps the test sets clean; dropping from train instead would
        # leave the leak in whichever eval split kept it.
        if split != "train" and leaked:
            before = len(df)
            df = df[~df["image_key"].isin(leaked)]
            if before != len(df):
                log.warning(
                    "%s/%s: dropped %d rows whose image also appears in another split",
                    DATASET,
                    split,
                    before - len(df),
                )

        # Genuine duplicates only. The same image under a *different* caption is
        # a distinct post — and image reuse is itself a misinformation signal
        # (PROJECT_CONTEXT.md Sec. 1, Example B), so those rows are kept.
        before = len(df)
        df = df.drop_duplicates(subset=["text", "image_key", "6_way_label"])
        if before != len(df):
            log.info("%s/%s: dropped %d exact duplicate rows", DATASET, split, before - len(df))

        records, missing = [], 0
        for i, row in df.iterrows():
            key = row["image_key"]
            img = by_name.get(key)
            if img is None:
                missing += 1
                continue

            label6 = int(row["6_way_label"])
            records.append(
                Record(
                    # Row index, not image hash: several distinct posts legitimately
                    # share one image, so hashing the image alone collides.
                    uid=make_uid(DATASET, f"{split}_{i}"),
                    dataset=DATASET,
                    split=split,
                    text=str(row["text"]),
                    image_path=img.relative_to(DATA_DIR).as_posix(),
                    has_image=True,
                    # Fakeddit carries no harassment label; the sentinel keeps
                    # the toxicity head out of the loss for these rows.
                    label_misinfo_6=label6,
                    label_misinfo_3=MISINFO_6_TO_3[label6],
                    meta={
                        "source_url": str(row.get("original_url", "")),
                        # Retained so image-reuse analysis (Sec. 6) can group
                        # posts that share a picture.
                        "image_key": key,
                    },
                )
            )

        if missing:
            log.warning("%s/%s: %d rows had no cached image", DATASET, split, missing)

        frame = records_to_frame(records)
        written[split] = write_manifest(frame, DATASET, split)
        log.info("%s/%s\n%s", DATASET, split, label_summary(frame))

    return written


def _find_leaked_images(frames: dict[str, pd.DataFrame]) -> set[str]:
    """Images appearing in more than one split.

    The upstream mirror's own partition has a small amount of overlap. Left in
    place it would inflate every misinformation number in the ablation table, so
    it is detected and reported rather than trusted.
    """
    leaked: set[str] = set()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in frames and b in frames:
            overlap = set(frames[a]["image_key"]) & set(frames[b]["image_key"])
            if overlap:
                log.warning("%s: %d images shared between %s and %s", DATASET, len(overlap), a, b)
                leaked |= overlap
    if leaked:
        log.warning("%s: %d leaked images will be removed from val/test", DATASET, len(leaked))
    return leaked


def _ensure_offline_images(repo: str, root: Path) -> Path:
    """Download and extract the bundled image cache exactly once."""
    img_root = root / "image_cache"
    if img_root.exists() and any(img_root.rglob("*.jpg")):
        return img_root

    log.info("downloading bundled image cache from %s (~500MB, one time)", repo)
    zip_path = hf_hub_download(repo, "image_cache.zip", repo_type="dataset")
    img_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # Guard against path traversal in an archive we did not create.
        for member in zf.infolist():
            dest = (img_root / member.filename).resolve()
            if not str(dest).startswith(str(img_root.resolve())):
                raise ValueError(f"unsafe path in archive: {member.filename}")
        zf.extractall(img_root)
    log.info("extracted image cache -> %s", img_root)
    return img_root


# --------------------------------------------------------------------------- #
# Scale tier
# --------------------------------------------------------------------------- #

def _prepare_scale(spec: DatasetSpec) -> dict[str, Path]:
    opts = spec.options
    seed = int(opts.get("seed", 42))
    sample_size = int(opts.get("sample_size", 40000))
    val_frac = float(opts.get("val_fraction", 0.1))
    workers = int(opts.get("max_download_workers", 16))

    log.info("loading Fakeddit metadata from %s", spec.hf_repo)
    parquet = hf_hub_download(
        spec.hf_repo, "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(parquet)
    df = df[df["hasImage"] & df["image_url"].notna()].copy()
    log.info("%d rows with usable image URLs", len(df))

    # Respect the dataset's own train/test partition rather than reshuffling it,
    # so numbers stay comparable to published Fakeddit baselines.
    train_pool = df[df["general_train"]]
    test_pool = df[df["general_test"]]

    n_test = int(sample_size * float(opts.get("test_fraction", 0.1)))
    n_trainval = sample_size - n_test

    trainval = _stratified_sample(train_pool, n_trainval, seed)
    test = _stratified_sample(test_pool, n_test, seed)

    val = trainval.sample(frac=val_frac, random_state=seed)
    train = trainval.drop(val.index)

    written: dict[str, Path] = {}
    for split, part in (("train", train), ("val", val), ("test", test)):
        written[split] = _materialize_scale_split(part, split, workers)
    return written


def _stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample evenly across the 6-way classes, capped by the rarest class.

    Fakeddit is heavily skewed toward 'true'; sampling it raw would produce a
    misinformation head that mostly learns the prior.
    """
    if n <= 0 or pool.empty:
        return pool.head(0)
    classes = sorted(pool["6_way_label"].unique())
    per_class = max(1, n // len(classes))
    parts = [
        grp.sample(min(len(grp), per_class), random_state=seed)
        for _, grp in pool.groupby("6_way_label")
    ]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.head(n)


def _materialize_scale_split(part: pd.DataFrame, split: str, workers: int) -> Path:
    img_dir = raw_dir(DATASET) / "images"
    items = [(str(r["id"]), str(r["image_url"])) for _, r in part.iterrows()]
    log.info("%s/%s: fetching %d images", DATASET, split, len(items))
    fetched = fetch_images(items, img_dir, max_workers=workers)

    records = []
    for _, row in part.iterrows():
        key = str(row["id"])
        path = fetched.get(key)
        if path is None:
            continue  # dead URL — drop rather than train on a missing modality
        label6 = int(row["6_way_label"])
        records.append(
            Record(
                uid=make_uid(DATASET, key),
                dataset=DATASET,
                split=split,
                text=str(row["text"]),
                image_path=Path(path).relative_to(DATA_DIR).as_posix(),
                has_image=True,
                label_misinfo_6=label6,
                label_misinfo_3=MISINFO_6_TO_3[label6],
                meta={"source_url": str(row["image_url"])},
            )
        )

    frame = records_to_frame(records)
    path = write_manifest(frame, DATASET, split)
    log.info(
        "%s/%s: kept %d/%d rows after dead-link drops\n%s",
        DATASET,
        split,
        len(frame),
        len(part),
        label_summary(frame),
    )
    return path
