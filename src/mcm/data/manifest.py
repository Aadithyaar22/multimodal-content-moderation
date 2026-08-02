"""Reading and writing normalized split manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mcm.config import DATA_DIR, processed_manifest
from mcm.data.schema import validate_frame
from mcm.utils.logging import get_logger

log = get_logger(__name__)


def write_manifest(df: pd.DataFrame, dataset: str, split: str) -> Path:
    df = validate_frame(df)
    path = processed_manifest(dataset, split)
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows) -> %s", f"{dataset}/{split}", len(df), path)
    return path


def read_manifest(dataset: str, split: str) -> pd.DataFrame:
    path = processed_manifest(dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest at {path}. Run: python scripts/prepare_data.py --dataset {dataset}"
        )
    return validate_frame(pd.read_parquet(path))


def load_splits(datasets: list[str], split: str) -> pd.DataFrame:
    """Concatenate one split across several datasets into a single frame.

    This is how mixed-dataset training batches are formed: Hateful Memes
    contributes toxicity labels, Fakeddit contributes misinformation labels, and
    the ignored-label sentinel keeps each head training only on what it should.
    """
    frames = [read_manifest(d, split) for d in datasets]
    merged = pd.concat(frames, ignore_index=True)
    return validate_frame(merged)


def resolve_image(image_path: str) -> Path:
    """Manifest-relative image path -> absolute path on this machine."""
    return DATA_DIR / image_path


def meta_of(row: pd.Series) -> dict:
    """Decode the JSON meta column of a manifest row."""
    raw = row.get("meta") or "{}"
    return json.loads(raw) if isinstance(raw, str) else dict(raw)
