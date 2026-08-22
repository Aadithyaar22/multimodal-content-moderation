"""Cross-attention link extraction.

The most direct evidence the system can offer. SHAP and Grad-CAM are post-hoc
approximations that probe the model from outside; these weights are the actual
coefficients the fusion layer used. A link from the word "gift" to a patch in
the doorway is not an estimate of what the model related — it is what it
related.

Links are read from the last block. Earlier layers still hold largely
unmixed representations, so their attention reflects surface similarity between
raw CLIP features more than any learned relationship; by the final layer the
streams have been conditioned on each other and the weights carry the
interaction the head actually consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AttentionLink:
    text_token: str
    image_region: list[float]  # [x0, y0, x1, y1] normalized
    weight: float


def extract_links(
    attentions: list[dict],
    tokens: list[str],
    text_attention_mask: torch.Tensor,
    top_k: int = 4,
    min_weight: float = 0.02,
) -> list[AttentionLink]:
    """Strongest text-token to image-patch links from the final block."""
    if not attentions or not tokens:
        return []

    # (B, N_txt, N_img) -> drop the batch dim.
    weights = attentions[-1]["text_to_image"]
    if weights is None:
        return []
    w = weights[0].detach().float().cpu()

    mask = text_attention_mask[0].detach().cpu()
    n_patches = w.shape[1] - 1  # exclude the vision CLS token
    side = int(math.sqrt(n_patches))
    if side * side != n_patches:
        return []

    # Strip the attention sink before reading any link.
    #
    # Vision transformers develop high-norm "register" patches that every query
    # attends to regardless of content (Darcet et al., 2023). Measured on this
    # model, one patch carried 2.7x uniform attention and was the argmax for
    # seven of eight caption tokens, while its variation *across* tokens was an
    # order of magnitude smaller than that baseline. Reading argmax directly
    # therefore links every word to the same region, which tells a moderator
    # nothing.
    #
    # Subtracting each patch's mean attention across tokens leaves the
    # discriminative part: how much *this* token attends to a patch relative to
    # how much every token does. The raw weights are not wrong, but the question
    # being asked here is which region a particular word relates to, and only
    # the deviation answers it.
    n_real = int(mask.sum())
    body = w[1 : max(2, n_real - 1), 1:]  # real tokens, patches without CLS
    if body.numel() == 0:
        return []
    patch_baseline = body.mean(dim=0)

    links: list[AttentionLink] = []
    # CLIP's tokenizer emits BOS at position 0 and EOS after the content, so
    # positions are offset by one and the sentinels are skipped: neither is a
    # word a moderator can be shown.
    for t_idx in range(1, min(len(tokens) + 1, w.shape[0])):
        if mask[t_idx] == 0:
            continue
        token = tokens[t_idx - 1]

        raw = w[t_idx, 1:]  # skip vision CLS, which has no position
        patch_weights = raw - patch_baseline
        best = int(patch_weights.argmax())
        # The reported weight is the raw attention, not the centred value:
        # centring selects *which* patch, but a negative deviation would be a
        # meaningless number to show next to a link.
        weight = float(raw[best])
        if weight < min_weight:
            continue

        row, col = divmod(best, side)
        links.append(
            AttentionLink(
                text_token=token,
                image_region=[
                    round(col / side, 4),
                    round(row / side, 4),
                    round((col + 1) / side, 4),
                    round((row + 1) / side, 4),
                ],
                weight=round(weight, 4),
            )
        )

    links.sort(key=lambda link: -link.weight)
    return links[:top_k]
