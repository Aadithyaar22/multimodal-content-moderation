"""Cached CLIP embeddings.

The backbone is frozen, so every row's embedding is a constant. Computing it
once and reusing it turns each ablation arm's training from a multi-hour GPU-less
grind into seconds of matrix work on cached vectors — which is what makes it
affordable to run every arm across several seeds and report error bars instead of
one lucky number.

The cache stores uids alongside the vectors and every load is realigned against
the manifest by uid. A features/labels misalignment would not crash; it would
quietly train on shuffled targets and produce plausible-looking nonsense, so it
is made structurally impossible rather than left to convention.
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


@dataclass
class FeatureCache:
    """Cached embeddings for one dataset split, aligned row-for-row with its manifest."""

    uid: np.ndarray
    image_emb: torch.Tensor  # (N, 512) float32
    text_emb: torch.Tensor  # (N, 512) float32
    image_mask: torch.Tensor  # (N,) bool

    def __len__(self) -> int:
        return len(self.uid)

    def normalized(self) -> FeatureCache:
        """L2-normalized copy — CLIP's own convention for its projection space."""
        return FeatureCache(
            uid=self.uid,
            image_emb=torch.nn.functional.normalize(self.image_emb, dim=-1),
            text_emb=torch.nn.functional.normalize(self.text_emb, dim=-1),
            image_mask=self.image_mask,
        )


def feature_path(dataset: str, split: str, model_tag: str = "clip-vit-b32") -> Path:
    d = PROCESSED_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}__{model_tag}.npz"


def save_features(
    dataset: str,
    split: str,
    uid: list[str],
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    image_mask: torch.Tensor,
    model_tag: str = "clip-vit-b32",
) -> Path:
    path = feature_path(dataset, split, model_tag)
    np.savez_compressed(
        path,
        uid=np.asarray(uid, dtype=object),
        image_emb=image_emb.cpu().numpy().astype(np.float32),
        text_emb=text_emb.cpu().numpy().astype(np.float32),
        image_mask=image_mask.cpu().numpy().astype(bool),
    )
    log.info("cached %d embeddings -> %s", len(uid), path)
    return path


def load_features(
    dataset: str,
    split: str,
    frame: pd.DataFrame | None = None,
    model_tag: str = "clip-vit-b32",
) -> FeatureCache:
    """Load cached embeddings, realigned to ``frame``'s row order when given.

    Passing the manifest is strongly preferred: it is what guarantees row i of
    the features corresponds to row i of the labels.
    """
    path = feature_path(dataset, split, model_tag)
    if not path.exists():
        raise FileNotFoundError(
            f"no feature cache at {path}. Run: python scripts/encode_features.py --dataset {dataset}"
        )

    blob = np.load(path, allow_pickle=True)
    cache = FeatureCache(
        uid=blob["uid"],
        image_emb=torch.from_numpy(blob["image_emb"]),
        text_emb=torch.from_numpy(blob["text_emb"]),
        image_mask=torch.from_numpy(blob["image_mask"]),
    )

    if frame is not None:
        cache = align_to(cache, frame)
    return cache


def align_to(cache: FeatureCache, frame: pd.DataFrame) -> FeatureCache:
    """Reorder a cache to match a manifest's rows, by uid.

    Raises if the manifest contains uids the cache lacks — training on a subset
    silently would be worse than failing here.
    """
    position = {u: i for i, u in enumerate(cache.uid)}
    wanted = frame["uid"].tolist()

    missing = [u for u in wanted if u not in position]
    if missing:
        raise KeyError(
            f"feature cache is missing {len(missing)} uids present in the manifest "
            f"(e.g. {missing[:3]}). Re-run scripts/encode_features.py for this split."
        )

    idx = torch.tensor([position[u] for u in wanted], dtype=torch.long)
    return FeatureCache(
        uid=np.asarray(wanted, dtype=object),
        image_emb=cache.image_emb[idx],
        text_emb=cache.text_emb[idx],
        image_mask=cache.image_mask[idx],
    )


def load_mixture(
    datasets: list[str], split: str, frame: pd.DataFrame, model_tag: str = "clip-vit-b32"
) -> FeatureCache:
    """Features for a multi-dataset frame, concatenated then aligned to it."""
    parts = [load_features(d, split, frame=None, model_tag=model_tag) for d in datasets]
    merged = FeatureCache(
        uid=np.concatenate([p.uid for p in parts]),
        image_emb=torch.cat([p.image_emb for p in parts]),
        text_emb=torch.cat([p.text_emb for p in parts]),
        image_mask=torch.cat([p.image_mask for p in parts]),
    )
    return align_to(merged, frame)
