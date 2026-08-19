"""Cross-attention fusion — the project's core contribution.

Why this and not late fusion
----------------------------
Late fusion combines representations that have each already been compressed to a
single vector. By then the image branch has thrown away *where* it was looking
and the text branch has thrown away *which words* mattered, so a pattern like
"this caption is only threatening given what is in this picture" is no longer
expressible: both branches independently say "benign", and combining two benign
vectors keeps it benign. That is precisely the failure mode Hateful Memes was
constructed to expose.

Cross-attention keeps both sequences alive — 50 image patches and 77 text tokens
— and lets each attend to the other before anything is pooled. The model can
therefore learn that a particular region of the image changes the meaning of a
particular span of the caption.

Design notes
------------
Attention is *bidirectional and parallel*: within a layer, image-attends-to-text
and text-attends-to-image are both computed from the same layer inputs, then
applied. Doing them sequentially would make the second stream see an already
updated first stream, quietly making the architecture asymmetric in a way that
is invisible in the diagram but affects results.

The deepfake branch is deliberately absent here. It reasons about pixel-level
and temporal artifacts that have nothing to do with caption text, so attending
it to language would be modelling a relationship that does not exist. It joins
at score level instead (PROJECT_CONTEXT.md Sec. 4, decision 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mcm.models.clip_encoder import PROJECTION_DIM, TEXT_HIDDEN, VISION_HIDDEN
from mcm.models.heads import HeadOutput, MultiTaskHead


class CrossAttentionBlock(nn.Module):
    """One bidirectional co-attention layer with pre-norm residuals."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.image_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.image_norm_attn = nn.LayerNorm(d_model)
        self.text_norm_attn = nn.LayerNorm(d_model)
        self.image_norm_ffn = nn.LayerNorm(d_model)
        self.text_norm_ffn = nn.LayerNorm(d_model)

        self.image_ffn = _ffn(d_model, ffn_mult, dropout)
        self.text_ffn = _ffn(d_model, ffn_mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image: torch.Tensor,  # (B, N_img, D)
        text: torch.Tensor,  # (B, N_txt, D)
        text_padding_mask: torch.Tensor | None = None,  # (B, N_txt) True where pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both directions read the *same* normalized inputs, so neither stream
        # gets to see the other's update within this layer.
        img_n = self.image_norm_attn(image)
        txt_n = self.text_norm_attn(text)

        img_attended, _ = self.image_attn(
            query=img_n, key=txt_n, value=txt_n, key_padding_mask=text_padding_mask
        )
        # Image keys are never padded — every image contributes all 50 patches —
        # so no key_padding_mask here. Rows that have no image at all are handled
        # by the caller, not by masking every key, which would make the softmax
        # denominator zero and produce NaN.
        txt_attended, _ = self.text_attn(query=txt_n, key=img_n, value=img_n)

        image = image + self.dropout(img_attended)
        text = text + self.dropout(txt_attended)

        image = image + self.dropout(self.image_ffn(self.image_norm_ffn(image)))
        text = text + self.dropout(self.text_ffn(self.text_norm_ffn(text)))
        return image, text


def _ffn(d_model: int, mult: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * mult, d_model),
    )


class CrossAttentionFusion(nn.Module):
    """Frozen CLIP token features -> co-attention -> shared fused embedding -> heads.

    Consumes token-level features (image 50x768, text 77x512) rather than the
    pooled 512-d vectors the other arms use, because attention between two single
    vectors is degenerate — it reduces to a learned scalar gate.

    The head is the same ``MultiTaskHead`` every other arm uses, so any gain here
    is attributable to the fusion mechanism and not to extra classifier capacity.
    """

    def __init__(
        self,
        d_model: int = PROJECTION_DIM,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
        image_dim: int = VISION_HIDDEN,
        text_dim: int = TEXT_HIDDEN,
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )

        self.image_out_norm = nn.LayerNorm(d_model)
        self.text_out_norm = nn.LayerNorm(d_model)
        # Both pooled streams meet here to form the single shared embedding both
        # heads read (PROJECT_CONTEXT.md Sec. 4, decision 3).
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(d_model, hidden_dim=head_hidden, dropout=head_dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,  # (B, 50, 768)
        text_tokens: torch.Tensor,  # (B, 77, 512)
        text_attention_mask: torch.Tensor | None = None,  # (B, 77) 1 = real token
        image_mask: torch.Tensor | None = None,  # (B,) False = no image at all
    ) -> HeadOutput:
        image = self.image_proj(image_tokens)
        text = self.text_proj(text_tokens)

        # nn.MultiheadAttention wants True where a key should be ignored, the
        # inverse of HuggingFace's attention_mask convention.
        text_padding_mask = None
        if text_attention_mask is not None:
            text_padding_mask = text_attention_mask == 0
            # A row whose text is entirely masked leaves the attention softmax
            # with nothing to normalize over, producing NaN that then propagates
            # through every parameter on the backward pass and silently destroys
            # the run. Keep one position visible for such rows; its contribution
            # is discarded anyway by the masked mean at pooling time.
            fully_masked = text_padding_mask.all(dim=1)
            if bool(fully_masked.any()):
                text_padding_mask = text_padding_mask.clone()
                text_padding_mask[fully_masked, 0] = False

        text_before = text
        for block in self.blocks:
            image, text = block(image, text, text_padding_mask=text_padding_mask)

        if image_mask is not None:
            # A text-only row has no image to attend to, so whatever the text
            # stream absorbed from the zero-filled image tokens is noise. Restore
            # its pre-attention state rather than letting a blank image act as a
            # consistent, learnable signal.
            keep = image_mask.view(-1, 1, 1)
            text = torch.where(keep, text, text_before)

        image_pooled = self.image_out_norm(image[:, 0])  # CLIP's vision CLS token
        text_pooled = self.text_out_norm(_masked_mean(text, text_attention_mask))

        if image_mask is not None:
            image_pooled = image_pooled * image_mask.unsqueeze(-1).to(image_pooled.dtype)

        fused = self.fuse(torch.cat([image_pooled, text_pooled], dim=-1))
        return self.head(fused)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over real tokens only; padding must not dilute the representation."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    # clamp guards the degenerate all-padding row against a divide by zero.
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
