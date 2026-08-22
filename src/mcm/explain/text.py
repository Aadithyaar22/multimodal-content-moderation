"""Token-level attribution for the caption.

Method: leave-one-out occlusion. Each token is removed in turn and the change in
the model's harm score is that token's attribution. Positive means removing it
lowered the score, so it was pushing toward harmful.

On naming
---------
The API field is called `method` and this reports "occlusion", not "shap". They
are not the same thing: SHAP averages a token's marginal contribution over
coalitions of the others, while occlusion measures it once against the full
context. Occlusion is a legitimate and widely-used attribution — it is exactly
what the deletion test in report Ch. 7 measures — but labelling it SHAP would
misdescribe the method in a document that will be examined on its methods.

KernelSHAP is available behind ``method="shap"`` for offline analysis, where its
~100 model evaluations per item are affordable. At request time they are not:
occlusion costs one forward pass per token, which is roughly 15 for a typical
caption, against several hundred for a sampled Shapley estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mcm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TokenAttribution:
    token: str
    score: float


def occlusion_attribution(
    score_fn,
    tokens: list[str],
    baseline: float | None = None,
    max_tokens: int = 60,
) -> list[TokenAttribution]:
    """Attribute a score to each token by removing it.

    score_fn: takes a list of tokens, returns the harm score for that text.
    baseline: score for the full text; computed if not supplied.

    Returns attributions in token order. Positive = pushed toward harmful.
    """
    if not tokens:
        return []

    # Long captions are truncated rather than left to cost one forward pass per
    # token without bound; a 500-word post would otherwise stall the request.
    truncated = tokens[:max_tokens]
    if len(tokens) > max_tokens:
        log.info("attributing first %d of %d tokens", max_tokens, len(tokens))

    base = baseline if baseline is not None else score_fn(truncated)

    out: list[TokenAttribution] = []
    for i in range(len(truncated)):
        without = truncated[:i] + truncated[i + 1 :]
        # A token whose removal lowers the score was holding the score up, so a
        # positive attribution means "pushes toward harmful".
        out.append(TokenAttribution(truncated[i], round(base - score_fn(without), 4)))

    return out


@torch.no_grad()
def build_score_fn(bundle, task: str, image_tokens, image_mask):
    """Score a token list against the fused model, holding the image fixed.

    The image side is encoded once and reused across every occlusion pass. Only
    the text changes, so re-encoding the image per token would multiply the cost
    by the number of tokens for an identical result.
    """
    clip = bundle.clip
    model = bundle.arms[task]["cross_attention"]
    temp = bundle.temperatures.get(task, {}).get("cross_attention", 1.0)

    def score(tokens: list[str]) -> float:
        text = " ".join(tokens)
        text_inputs = {k: v.to(bundle.device) for k, v in clip.tokenize([text]).items()}
        feats = clip.encode_tokens(
            pixel_values=torch.zeros(1, 3, 224, 224, device=bundle.device),
            text_inputs=text_inputs,
        )
        out = model(
            image_tokens=image_tokens,
            text_tokens=feats.text_tokens,
            text_attention_mask=feats.text_attention_mask,
            image_mask=image_mask,
        )
        logits = out.toxicity_logits if task == "toxicity" else out.misinfo_logits
        probs = torch.softmax(logits.float()[0] / temp, dim=-1)
        return float(probs[1]) if task == "toxicity" else float(1.0 - probs[0])

    return score
