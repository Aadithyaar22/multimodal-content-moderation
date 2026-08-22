"""Grad-CAM for the CLIP vision transformer.

Classic Grad-CAM weights convolutional feature maps by the gradient of the
target score. A ViT has no conv stack, so the equivalent is taken over the last
vision block's patch activations: gradient times activation, summed over the
channel dimension, reshaped to the 7x7 patch grid that CLIP ViT-B/32 produces at
224px.

Two details that are easy to get wrong:

The CLS token is dropped before reshaping. It is position 0 of the 50-token
sequence and has no location in the image, so including it either corrupts the
grid or silently shifts every patch by one.

ReLU is applied after summing, not before. Grad-CAM asks which regions push
*toward* the target class; keeping negative contributions would show regions
arguing against it in the same colour, which reads as evidence for exactly the
opposite conclusion.

The backbone stays frozen throughout. Gradients are taken with respect to
activations, never parameters, so nothing here trains anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Region:
    bbox: list[float]  # [x0, y0, x1, y1] normalized, origin top-left
    score: float
    label: str


def grad_cam(
    bundle,
    task: str,
    pixel_values: torch.Tensor,
    text_tokens: torch.Tensor,
    text_attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> np.ndarray | None:
    """Return a 7x7 attention map normalized to [0, 1], or None on failure."""
    clip = bundle.clip
    model = bundle.arms[task]["cross_attention"]

    vision = clip.model.vision_model
    activations: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden.retain_grad()
        activations["hidden"] = hidden

    handle = vision.encoder.layers[-1].register_forward_hook(hook)

    try:
        with torch.enable_grad():
            # The input must require grad for any graph to exist. Every CLIP
            # parameter is frozen, so without this PyTorch builds no autograd
            # graph through the vision tower at all: the activations come back
            # with grad_fn=None, retain_grad() is a silent no-op, and the whole
            # method returns nothing with no error to explain why.
            #
            # This does not train anything. The gradient is taken with respect
            # to activations; the input gradient is a side effect and discarded.
            px = pixel_values.to(bundle.device).detach().requires_grad_(True)

            out = vision(pixel_values=px)
            hidden = activations.get("hidden")
            if hidden is None:
                return None

            image_tokens = out.last_hidden_state
            head_out = model(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                text_attention_mask=text_attention_mask,
                image_mask=image_mask,
            )
            logits = (
                head_out.toxicity_logits if task == "toxicity" else head_out.misinfo_logits
            )
            target = logits[0, logits[0].argmax()]

            model.zero_grad(set_to_none=True)
            clip.model.zero_grad(set_to_none=True)
            target.backward()

            grads = hidden.grad
            if grads is None:
                return None

            # (1, N, D) -> weight each channel by its gradient, sum over channels.
            cam = (grads[0] * hidden[0]).sum(dim=-1)
            cam = torch.relu(cam)

            # Drop CLS: it has no position in the image.
            cam = cam[1:]

            side = int(math.sqrt(cam.numel()))
            if side * side != cam.numel():
                return None

            grid = cam.reshape(side, side).detach().cpu().numpy()
            if grid.max() <= 0:
                return None
            return grid / grid.max()
    finally:
        handle.remove()


def regions_from_cam(
    cam: np.ndarray,
    top_k: int = 2,
    threshold: float = 0.45,
) -> list[Region]:
    """Convert a heatmap into a few labelled boxes.

    Boxes are what a moderator can act on; a continuous heatmap tells them where
    to look but not what the model isolated. Cells above the threshold are
    grouped into connected components and each becomes one box.
    """
    side = cam.shape[0]
    hot = cam >= threshold
    if not hot.any():
        return []

    seen = np.zeros_like(hot, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(side):
        for c in range(side):
            if not hot[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < side and 0 <= nx < side and hot[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(comp)

    scored = []
    for comp in components:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        strength = float(np.mean([cam[y, x] for y, x in comp]))
        scored.append(
            Region(
                bbox=[
                    round(min(xs) / side, 4),
                    round(min(ys) / side, 4),
                    round((max(xs) + 1) / side, 4),
                    round((max(ys) + 1) / side, 4),
                ],
                score=round(strength, 4),
                # Named by position rather than content: the model localises
                # attention, it does not recognise objects, and inventing an
                # object label here would be asserting something not computed.
                label=_describe(
                    (min(xs) + max(xs)) / 2 / side, (min(ys) + max(ys)) / 2 / side
                ),
            )
        )

    scored.sort(key=lambda r: -r.score)
    return scored[:top_k]


def _describe(cx: float, cy: float) -> str:
    vertical = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "centre"
    horizontal = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    if vertical == "centre" and horizontal == "centre":
        return "centre region"
    if vertical == "centre":
        return f"{horizontal} region"
    if horizontal == "centre":
        return f"{vertical} region"
    return f"{vertical} {horizontal} region"
