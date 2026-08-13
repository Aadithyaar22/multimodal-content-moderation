"""Frozen CLIP ViT-B/32 encoder.

The backbone never trains (PROJECT_CONTEXT.md Sec. 4, decision 1): full CLIP
fine-tuning does not fit the M4's memory budget, and freezing it is also what
keeps the ablation arms comparable — every arm sees identical features, so a
difference in results is attributable to the fusion mechanism rather than to one
arm having learned a better backbone.

Two granularities are exposed:

``pooled``
    The 512-d projected embeddings. Enough for the unimodal and late-fusion
    arms, and small enough to cache for the whole corpus.

``tokens``
    Per-patch (50x768) and per-token (77x512) hidden states. Cross-attention
    needs these — attending between two single pooled vectors is degenerate —
    but they are ~100x larger, so they are computed on demand rather than cached
    wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mcm.utils.device import get_device
from mcm.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
PROJECTION_DIM = 512
VISION_HIDDEN = 768
TEXT_HIDDEN = 512
TEXT_CONTEXT = 77


def _as_embedding(out) -> torch.Tensor:
    """Normalize the return of ``get_*_features`` across transformers versions.

    transformers 4.x returns the projected embedding as a bare tensor; 5.x
    returns a ``BaseModelOutputWithPooling`` carrying it as ``pooler_output``.
    Pinning to either shape would silently break on the other, and the failure
    mode is late and confusing, so both are accepted here.
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"cannot extract an embedding from {type(out).__name__}")
    return pooled


@dataclass
class PooledFeatures:
    image_emb: torch.Tensor  # (B, 512)
    text_emb: torch.Tensor  # (B, 512)


@dataclass
class TokenFeatures:
    image_tokens: torch.Tensor  # (B, 50, 768)
    text_tokens: torch.Tensor  # (B, 77, 512)
    text_attention_mask: torch.Tensor  # (B, 77)


class FrozenCLIP(torch.nn.Module):
    """CLIP with every parameter frozen and eval-mode pinned."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: torch.device | None = None):
        super().__init__()
        self.device_ = device or get_device()
        log.info("loading %s onto %s", model_name, self.device_)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device_)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Stay in eval mode regardless of the parent module's calls.

        Without this, wrapping FrozenCLIP inside a fusion model and calling
        ``.train()`` would re-enable dropout in the backbone, so identical
        inputs would produce different features between arms and silently
        weaken the ablation's controls.
        """
        return super().train(False)

    @torch.no_grad()
    def encode_pooled(
        self,
        pixel_values: torch.Tensor | None = None,
        text_inputs: dict[str, torch.Tensor] | None = None,
        normalize: bool = False,
    ) -> PooledFeatures:
        """Projected 512-d embeddings for one batch.

        normalize applies L2 normalization — CLIP's own contrastive convention.
        It is off by default so the cache stores raw projections and downstream
        models can decide; the choice is then a documented model hyperparameter
        rather than something baked irreversibly into the cache.
        """
        image_emb = None
        if pixel_values is not None:
            image_emb = _as_embedding(
                self.model.get_image_features(pixel_values=pixel_values.to(self.device_))
            )
            if normalize:
                image_emb = F.normalize(image_emb, dim=-1)

        text_emb = None
        if text_inputs is not None:
            text_emb = _as_embedding(
                self.model.get_text_features(
                    **{k: v.to(self.device_) for k, v in text_inputs.items()}
                )
            )
            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)

        return PooledFeatures(image_emb=image_emb, text_emb=text_emb)

    @torch.no_grad()
    def encode_tokens(
        self,
        pixel_values: torch.Tensor,
        text_inputs: dict[str, torch.Tensor],
    ) -> TokenFeatures:
        """Per-patch and per-token hidden states, for cross-attention fusion."""
        vision_out = self.model.vision_model(pixel_values=pixel_values.to(self.device_))
        text_inputs = {k: v.to(self.device_) for k, v in text_inputs.items()}
        text_out = self.model.text_model(**text_inputs)

        return TokenFeatures(
            image_tokens=vision_out.last_hidden_state,
            text_tokens=text_out.last_hidden_state,
            text_attention_mask=text_inputs["attention_mask"],
        )

    def tokenize(self, texts: list[str], max_length: int = TEXT_CONTEXT) -> dict[str, torch.Tensor]:
        return dict(
            self.processor.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        )

    def preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"]

    @property
    def image_processor(self):
        """Handed to ModerationDataset so dataloader workers apply CLIP's own transform."""
        return self.processor.image_processor
