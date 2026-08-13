"""Shared classification heads and the masked multi-task loss.

Every ablation arm — CV-only, NLP-only, late fusion, cross-attention — uses this
identical head module on top of whatever representation it produces. That is
deliberate: if the arms differed in head capacity, a win for cross-attention
could just mean "this arm got a bigger classifier", and the central claim of the
report would not follow from the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcm.data.schema import IGNORE_INDEX

N_TOXICITY_CLASSES = 2
N_MISINFO_CLASSES = 3


@dataclass
class HeadOutput:
    toxicity_logits: torch.Tensor  # (B, 2)
    misinfo_logits: torch.Tensor  # (B, 3)


@dataclass
class LossOutput:
    total: torch.Tensor
    toxicity: torch.Tensor
    misinfo: torch.Tensor
    n_toxicity: int
    n_misinfo: int


class MultiTaskHead(nn.Module):
    """One shared trunk feeding a toxicity head and a misinformation head.

    Both heads read the *same* representation (PROJECT_CONTEXT.md Sec. 4,
    decision 3). Sharing it is what makes the multi-task setup informative about
    task interference rather than just two models in a trenchcoat.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.toxicity = nn.Linear(hidden_dim, N_TOXICITY_CLASSES)
        self.misinfo = nn.Linear(hidden_dim, N_MISINFO_CLASSES)

    def forward(self, x: torch.Tensor) -> HeadOutput:
        h = self.trunk(x)
        return HeadOutput(toxicity_logits=self.toxicity(h), misinfo_logits=self.misinfo(h))


class MaskedMultiTaskLoss(nn.Module):
    """Cross-entropy per head, with inapplicable rows masked out.

    Hateful Memes rows carry no misinformation label and Fakeddit rows carry no
    toxicity label, both marked with IGNORE_INDEX. Each head's loss is computed
    only over the rows that actually have a label for it.

    The subtle failure this guards against: ``F.cross_entropy`` returns NaN when
    *every* target in the batch is the ignore index, because it divides by a zero
    count. A homogeneous batch — entirely Fakeddit, say — would therefore poison
    the whole model with a NaN gradient. Such batches are common here, so each
    head's contribution is skipped outright when it has no labelled rows.
    """

    def __init__(
        self,
        toxicity_weight: torch.Tensor | None = None,
        misinfo_weight: torch.Tensor | None = None,
        task_weights: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self.register_buffer("toxicity_weight", toxicity_weight)
        self.register_buffer("misinfo_weight", misinfo_weight)
        self.task_weights = task_weights

    def forward(
        self,
        out: HeadOutput,
        label_toxicity: torch.Tensor,
        label_misinfo: torch.Tensor,
    ) -> LossOutput:
        device = out.toxicity_logits.device
        zero = torch.zeros((), device=device)

        n_tox = int((label_toxicity != IGNORE_INDEX).sum())
        n_mis = int((label_misinfo != IGNORE_INDEX).sum())

        tox_loss = (
            F.cross_entropy(
                out.toxicity_logits,
                label_toxicity,
                weight=self.toxicity_weight,
                ignore_index=IGNORE_INDEX,
            )
            if n_tox > 0
            else zero
        )
        mis_loss = (
            F.cross_entropy(
                out.misinfo_logits,
                label_misinfo,
                weight=self.misinfo_weight,
                ignore_index=IGNORE_INDEX,
            )
            if n_mis > 0
            else zero
        )

        wt, wm = self.task_weights
        return LossOutput(
            total=wt * tox_loss + wm * mis_loss,
            toxicity=tox_loss.detach(),
            misinfo=mis_loss.detach(),
            n_toxicity=n_tox,
            n_misinfo=n_mis,
        )
