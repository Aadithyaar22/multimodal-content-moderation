"""Torch datasets over the normalized manifests.

One Dataset class serves every arm of the ablation study — CV-only, NLP-only,
late fusion, and cross-attention fusion all consume the same batches. The arms
differ in what the *model* attends to, not in how data is loaded, which is what
keeps the comparison in the ablation table honest: no arm gets a different
preprocessing advantage.

Missing modalities are handled explicitly. A text-only record (HateXplain) gets
a zero image plus ``image_mask=False``; the model is expected to consume that
mask rather than silently attending to a black square, which would otherwise
teach the vision branch that "black image" correlates with hate speech.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from mcm.data.manifest import load_splits, read_manifest, resolve_image
from mcm.data.schema import IGNORE_INDEX
from mcm.utils.logging import get_logger

log = get_logger(__name__)

# Some Fakeddit images are truncated mid-file; better to use the partial image
# than to lose the row.
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_SIZE = 224


@dataclass
class Batch:
    """A collated batch. Tensors are on CPU; the training loop moves them."""

    uid: list[str]
    dataset: list[str]
    text: list[str]
    pixel_values: torch.Tensor  # (B, 3, H, W)
    image_mask: torch.Tensor  # (B,) bool — False where the record is text-only
    label_toxicity: torch.Tensor  # (B,) long, IGNORE_INDEX where inapplicable
    label_misinfo_3: torch.Tensor
    label_misinfo_6: torch.Tensor
    text_inputs: dict[str, torch.Tensor] | None = None  # set when a tokenizer is given

    def to(self, device: torch.device) -> Batch:
        self.pixel_values = self.pixel_values.to(device, non_blocking=True)
        self.image_mask = self.image_mask.to(device, non_blocking=True)
        self.label_toxicity = self.label_toxicity.to(device, non_blocking=True)
        self.label_misinfo_3 = self.label_misinfo_3.to(device, non_blocking=True)
        self.label_misinfo_6 = self.label_misinfo_6.to(device, non_blocking=True)
        if self.text_inputs is not None:
            self.text_inputs = {k: v.to(device, non_blocking=True) for k, v in self.text_inputs.items()}
        return self

    def __len__(self) -> int:
        return len(self.uid)


class ModerationDataset(Dataset):
    """Reads a normalized manifest and yields image tensors + raw text.

    Text is kept as strings here and tokenized in the collate function, so a
    batch is padded to its own longest sequence instead of a fixed maximum.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_processor: Any | None = None,
        image_size: int = CLIP_SIZE,
        keep_meta: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_processor = image_processor
        self.image_size = image_size
        self.keep_meta = keep_meta
        self._missing_warned = 0

    @classmethod
    def from_manifest(cls, dataset: str, split: str, **kwargs) -> ModerationDataset:
        return cls(read_manifest(dataset, split), **kwargs)

    @classmethod
    def from_mixture(cls, datasets: list[str], split: str, **kwargs) -> ModerationDataset:
        """Mixed-dataset training: each source trains the heads it has labels for."""
        return cls(load_splits(datasets, split), **kwargs)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        has_image = bool(row["has_image"])
        pixel_values, image_ok = self._load_image(row["image_path"]) if has_image else (None, False)

        item: dict[str, Any] = {
            "uid": row["uid"],
            "dataset": row["dataset"],
            "text": row["text"] or "",
            "pixel_values": pixel_values,
            "image_mask": image_ok,
            "label_toxicity": int(row["label_toxicity"]),
            "label_misinfo_3": int(row["label_misinfo_3"]),
            "label_misinfo_6": int(row["label_misinfo_6"]),
        }
        if self.keep_meta:
            item["meta"] = json.loads(row["meta"] or "{}")
        return item

    def _load_image(self, rel_path: str) -> tuple[torch.Tensor | None, bool]:
        path: Path = resolve_image(rel_path)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.image_processor is not None:
                    out = self.image_processor(images=img, return_tensors="pt")
                    return out["pixel_values"][0], True
                return _fallback_transform(img, self.image_size), True
        except Exception as e:  # noqa: BLE001
            # A corrupt file must not kill a training run hours in. Degrade the
            # row to text-only and let the image mask do its job.
            if self._missing_warned < 10:
                log.warning("unreadable image %s (%s) — treating row as text-only", path, type(e).__name__)
                self._missing_warned += 1
            return None, False


def _fallback_transform(img: Image.Image, size: int) -> torch.Tensor:
    """CLIP-style preprocessing for when no HF processor is supplied."""
    import torchvision.transforms.functional as F

    img = F.resize(img, size, antialias=True)
    img = F.center_crop(img, [size, size])
    tensor = F.to_tensor(img)
    return F.normalize(
        tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )


class Collator:
    """Collates items into a Batch, tokenizing text if a tokenizer is supplied.

    Deliberately a class rather than a closure: macOS spawns dataloader workers
    instead of forking them, so the collate callable must be picklable. A nested
    function is not, and using one fails only once ``num_workers > 0``.
    """

    def __init__(
        self, tokenizer: Any | None = None, max_length: int = 77, image_size: int = CLIP_SIZE
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

    def __call__(self, items: list[dict[str, Any]]) -> Batch:
        texts = [it["text"] for it in items]

        pixels = []
        for it in items:
            pv = it["pixel_values"]
            # Zero tensor, not a black image: paired with image_mask=False it is
            # never attended to, so its actual value is irrelevant — but keeping
            # it zero makes an accidental unmasked read obvious in debugging.
            pixels.append(pv if pv is not None else torch.zeros(3, self.image_size, self.image_size))

        batch = Batch(
            uid=[it["uid"] for it in items],
            dataset=[it["dataset"] for it in items],
            text=texts,
            pixel_values=torch.stack(pixels),
            image_mask=torch.tensor([bool(it["image_mask"]) for it in items], dtype=torch.bool),
            label_toxicity=_label_tensor(items, "label_toxicity"),
            label_misinfo_3=_label_tensor(items, "label_misinfo_3"),
            label_misinfo_6=_label_tensor(items, "label_misinfo_6"),
        )

        if self.tokenizer is not None:
            batch.text_inputs = dict(
                self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            )
        return batch


def make_collate_fn(
    tokenizer: Any | None = None, max_length: int = 77, image_size: int = CLIP_SIZE
) -> Collator:
    """Build a collate function.

    max_length defaults to 77 — CLIP's text encoder context limit. A DistilBERT
    or RoBERTa branch should pass its own larger value.
    """
    return Collator(tokenizer=tokenizer, max_length=max_length, image_size=image_size)


def _label_tensor(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([it.get(key, IGNORE_INDEX) for it in items], dtype=torch.long)


def class_weights(frame: pd.DataFrame, column: str) -> torch.Tensor | None:
    """Inverse-frequency weights over the applicable rows of one label column.

    Class imbalance is a documented concern (report Sec. 3.3) — Hateful Memes
    train is 64/36 and Fakeddit's rarer classes are far worse — so the loss
    needs weighting rather than pretending the priors are flat.
    """
    applicable = frame[frame[column] != IGNORE_INDEX][column]
    if applicable.empty:
        return None
    counts = applicable.value_counts().sort_index()
    n_classes = int(counts.index.max()) + 1
    weights = torch.ones(n_classes)
    total = counts.sum()
    for cls, n in counts.items():
        weights[int(cls)] = total / (len(counts) * n)
    return weights
