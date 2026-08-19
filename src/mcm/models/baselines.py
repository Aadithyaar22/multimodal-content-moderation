"""Ablation arms operating on cached frozen-CLIP embeddings.

Four arms, all sharing the identical MultiTaskHead so the comparison isolates
what feeds the head rather than the head itself:

``UnimodalModel("image")``  CV-only  — report Ch. 4.4
``UnimodalModel("text")``   NLP-only — report Ch. 5.3
``LateFusionModel``         concat + MLP — the ablation baseline, Ch. 6.1
``CrossAttentionModel``     the proposed architecture — Ch. 6.2 (built later)

The late-fusion arm here concatenates *representations*, which is already more
generous than the score-averaging late fusion described in PROJECT_CONTEXT.md
Sec. 4. That is deliberate: beating the weaker version would prove little, so
the baseline is made as strong as it can be while remaining "no cross-modal
attention".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mcm.models.clip_encoder import PROJECTION_DIM
from mcm.models.heads import HeadOutput, MultiTaskHead


class UnimodalModel(nn.Module):
    """One modality only — the floor the fusion arms have to clear.

    On Hateful Memes this arm is *expected* to do poorly, and that is the point:
    the dataset was constructed so neither modality alone is offensive. A CV-only
    or NLP-only model scoring near chance there is evidence the benchmark is
    doing its job, not evidence of a bug.
    """

    def __init__(
        self,
        modality: str,
        input_dim: int = PROJECTION_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        normalize_input: bool = True,
    ):
        super().__init__()
        if modality not in ("image", "text"):
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
        self.modality = modality
        self.normalize_input = normalize_input
        self.head = MultiTaskHead(input_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> HeadOutput:
        x = image_emb if self.modality == "image" else text_emb
        if self.normalize_input:
            x = nn.functional.normalize(x, dim=-1)
        return self.head(x)


class LateFusionModel(nn.Module):
    """Concatenate the two embeddings, then an MLP. No cross-modal attention.

    This is the control. Both modalities are present, but they only meet after
    each has been compressed to a fixed vector, so the model can learn that
    "toxic text and any image" is bad while remaining structurally unable to
    learn that *this* caption is only harmful *given this* image.

    Missing-image rows are handled by zeroing the image half rather than
    dropping the row, so text-only data still trains the shared trunk.
    """

    def __init__(
        self,
        input_dim: int = PROJECTION_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.project = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> HeadOutput:
        if self.normalize_input:
            image_emb = nn.functional.normalize(image_emb, dim=-1)
            text_emb = nn.functional.normalize(text_emb, dim=-1)

        if image_mask is not None:
            image_emb = image_emb * image_mask.unsqueeze(-1).to(image_emb.dtype)

        fused = self.project(torch.cat([image_emb, text_emb], dim=-1))
        return self.head(fused)


def build_model(arch: str, **kwargs) -> nn.Module:
    """Factory used by the training script so arms are named consistently."""
    if arch == "cv_only":
        return UnimodalModel("image", **kwargs)
    if arch == "nlp_only":
        return UnimodalModel("text", **kwargs)
    if arch == "late_fusion":
        return LateFusionModel(**kwargs)
    if arch == "cross_attention":
        from mcm.models.fusion import CrossAttentionFusion

        return CrossAttentionFusion(**kwargs)
    raise ValueError(f"unknown architecture {arch!r}")


#: Arms that consume pooled 512-d embeddings.
POOLED_ARCHITECTURES = ("cv_only", "nlp_only", "late_fusion")

#: Arms that consume token-level sequences and therefore need the token cache.
TOKEN_ARCHITECTURES = ("cross_attention",)

ARCHITECTURES = POOLED_ARCHITECTURES + TOKEN_ARCHITECTURES
