"""Cached token-level CLIP features for the cross-attention arm.

The pooled cache (``features.py``) is ~150MB and lives in memory. Token features
are ~100x larger — 50 image patches and 77 text tokens per row — so they are
stored as fp16 ``.npy`` and read through a memory map. Batches are copied to the
device on demand, which keeps resident memory in the hundreds of MB instead of
gigabytes and still avoids re-running CLIP every epoch.

fp16 is safe here because these are frozen inputs, never accumulated into: they
are cast to fp32 on arrival, so no training arithmetic happens at half
precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mcm.config import PROCESSED_DIR
from mcm.utils.logging import get_logger

log = get_logger(__name__)

MODEL_TAG = "clip-vit-b32"


def token_dir(dataset: str, split: str, model_tag: str = MODEL_TAG) -> Path:
    d = PROCESSED_DIR / dataset / f"{split}__tokens__{model_tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class TokenCache:
    """Memory-mapped token features aligned to a manifest's row order."""

    uid: np.ndarray
    image_tokens: np.memmap | np.ndarray  # (N, 50, 768) fp16
    text_tokens: np.memmap | np.ndarray  # (N, 77, 512) fp16
    text_attention_mask: np.ndarray  # (N, 77) uint8
    image_mask: np.ndarray  # (N,) bool
    order: np.ndarray  # row i of the manifest -> row order[i] of the cache

    def __len__(self) -> int:
        return len(self.order)

    def batch(self, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        """Fetch one batch and move it to the device as fp32."""
        rows = self.order[idx]
        return {
            "image_tokens": torch.from_numpy(np.asarray(self.image_tokens[rows])).to(
                device, torch.float32
            ),
            "text_tokens": torch.from_numpy(np.asarray(self.text_tokens[rows])).to(
                device, torch.float32
            ),
            "text_attention_mask": torch.from_numpy(self.text_attention_mask[rows]).to(
                device, torch.long
            ),
            "image_mask": torch.from_numpy(self.image_mask[rows]).to(device, torch.bool),
        }


def save_token_features(
    dataset: str,
    split: str,
    uid: list[str],
    image_tokens: np.ndarray,
    text_tokens: np.ndarray,
    text_attention_mask: np.ndarray,
    image_mask: np.ndarray,
    model_tag: str = MODEL_TAG,
) -> Path:
    d = token_dir(dataset, split, model_tag)
    np.save(d / "image_tokens.npy", image_tokens.astype(np.float16))
    np.save(d / "text_tokens.npy", text_tokens.astype(np.float16))
    np.save(d / "text_attention_mask.npy", text_attention_mask.astype(np.uint8))
    np.save(d / "image_mask.npy", image_mask.astype(bool))
    np.save(d / "uid.npy", np.asarray(uid, dtype=object), allow_pickle=True)

    total_mb = sum(f.stat().st_size for f in d.glob("*.npy")) / 1e6
    log.info("cached %d token features (%.0f MB) -> %s", len(uid), total_mb, d)
    return d


def load_token_features(
    dataset: str,
    split: str,
    frame: pd.DataFrame,
    model_tag: str = MODEL_TAG,
    preload_budget_gb: float = 4.0,
) -> TokenCache:
    """Load a split's token features, aligned to ``frame``.

    Resident in RAM when the split fits within ``preload_budget_gb``, memory
    mapped otherwise. Residency matters far more than it looks: training shuffles
    row order every epoch, so a memmap gets a random gather across the whole file
    on every batch and the run becomes disk-bound — measured at ~1% CPU, which is
    to say not really training at all. The largest split here is 1.2GB, so it
    simply lives in memory.

    Alignment is by uid and stored as an index vector. As with the pooled cache,
    a features/labels mismatch would not crash — it would train on shuffled
    targets — so it is checked rather than assumed.
    """
    d = token_dir(dataset, split, model_tag)
    required = ["image_tokens.npy", "text_tokens.npy", "text_attention_mask.npy", "uid.npy"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"token cache incomplete at {d} (missing {missing}). "
            f"Run: python scripts/encode_token_features.py --dataset {dataset}"
        )

    uid = np.load(d / "uid.npy", allow_pickle=True)
    position = {u: i for i, u in enumerate(uid)}
    wanted = frame["uid"].tolist()

    absent = [u for u in wanted if u not in position]
    if absent:
        raise KeyError(
            f"token cache is missing {len(absent)} uids present in the manifest "
            f"(e.g. {absent[:3]}). Re-run scripts/encode_token_features.py."
        )

    size_gb = sum((d / f).stat().st_size for f in required) / 1e9
    resident = size_gb <= preload_budget_gb
    mode = None if resident else "r"
    log.info(
        "%s/%s token cache: %.2fGB (%s)",
        dataset,
        split,
        size_gb,
        "in memory" if resident else "memory-mapped — training will be I/O bound",
    )

    return TokenCache(
        uid=uid,
        image_tokens=np.load(d / "image_tokens.npy", mmap_mode=mode),
        text_tokens=np.load(d / "text_tokens.npy", mmap_mode=mode),
        text_attention_mask=np.load(d / "text_attention_mask.npy"),
        image_mask=np.load(d / "image_mask.npy"),
        order=np.array([position[u] for u in wanted], dtype=np.int64),
    )


def token_cache_exists(dataset: str, split: str, model_tag: str = MODEL_TAG) -> bool:
    d = token_dir(dataset, split, model_tag)
    return (d / "image_tokens.npy").exists() and (d / "uid.npy").exists()
