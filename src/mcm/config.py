"""Project-wide paths and dataset configuration.

Every path in the project resolves through here so that scripts can be run from
any working directory without breaking relative paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/mcm/config.py -> src/mcm -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("MCM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = Path(os.getenv("MCM_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
REPORT_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetSpec:
    """How to obtain and normalize one source dataset."""

    name: str
    hf_repo: str
    kind: str  # "multimodal" | "text"
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


def load_data_config(path: Path | None = None) -> dict[str, DatasetSpec]:
    """Read configs/data.yaml into DatasetSpec objects keyed by dataset name."""
    path = path or CONFIG_DIR / "data.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    specs: dict[str, DatasetSpec] = {}
    for name, cfg in (raw.get("datasets") or {}).items():
        specs[name] = DatasetSpec(
            name=name,
            hf_repo=cfg["hf_repo"],
            kind=cfg.get("kind", "multimodal"),
            enabled=cfg.get("enabled", True),
            options=cfg.get("options", {}) or {},
        )
    return specs


def raw_dir(dataset: str) -> Path:
    d = RAW_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_manifest(dataset: str, split: str) -> Path:
    """Canonical location of a normalized split manifest."""
    d = PROCESSED_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}.parquet"
