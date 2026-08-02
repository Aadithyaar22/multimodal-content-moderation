"""Per-dataset normalization pipelines.

Each module exposes ``prepare(spec) -> dict[split, Path]`` and is responsible for
turning one source dataset into schema-conformant parquet manifests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mcm.config import DatasetSpec
from mcm.data.prepare import fakeddit, hateful_memes, hatexplain

PREPARERS: dict[str, Callable[[DatasetSpec], dict[str, Path]]] = {
    hateful_memes.DATASET: hateful_memes.prepare,
    fakeddit.DATASET: fakeddit.prepare,
    hatexplain.DATASET: hatexplain.prepare,
}

__all__ = ["PREPARERS", "fakeddit", "hateful_memes", "hatexplain"]
