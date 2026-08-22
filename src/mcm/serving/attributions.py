"""Build the /attributions payload for one stored item.

Re-encodes from the original inputs rather than caching intermediate tensors at
analyze time. Attribution is opened on a minority of items — a moderator looks
closely at the ones they are unsure about — so holding 50x768 image tokens in
memory for every item scored would trade a large, permanent cost against a small
occasional one.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mcm.explain import (
    build_score_fn,
    extract_links,
    grad_cam,
    occlusion_attribution,
    regions_from_cam,
)
from mcm.utils.logging import get_logger

log = get_logger(__name__)


def compute(bundle, record: dict, image=None) -> dict:
    """Return the Attributions shape from docs/api.md.

    Each of the three parts is attempted independently: a failure in one leaves
    the others intact, because a moderator with two of three explanations is far
    better served than one with an error page.
    """
    item_id = record["item_id"]
    task = record["top_head"]
    text = record["input"]["text"] or ""
    started = time.perf_counter()

    if task not in bundle.arms:
        return _empty(item_id)

    device = bundle.device
    clip = bundle.clip
    has_image = image is not None

    pixel_values = (
        clip.preprocess_images([image]) if has_image else torch.zeros(1, 3, 224, 224)
    ).to(device)
    text_inputs = {k: v.to(device) for k, v in clip.tokenize([text]).items()}
    image_mask = torch.tensor([has_image], dtype=torch.bool, device=device)

    with torch.no_grad():
        feats = clip.encode_tokens(pixel_values=pixel_values, text_inputs=text_inputs)

    words = text.split()
    out: dict = {"item_id": item_id, "text": None, "image": None, "cross_attention": None}

    # --- token attribution -------------------------------------------------
    if words:
        try:
            score_fn = build_score_fn(bundle, task, feats.image_tokens, image_mask)
            attributions = occlusion_attribution(score_fn, words)
            out["text"] = {
                # Named for what it is. This is leave-one-out occlusion, not a
                # Shapley estimate, and the report is examined on its methods.
                "method": "occlusion",
                "tokens": [{"token": a.token, "score": a.score} for a in attributions],
            }
        except Exception:
            log.exception("token attribution failed for %s", item_id)

    # --- grad-cam ----------------------------------------------------------
    if has_image:
        try:
            cam = grad_cam(
                bundle,
                task,
                pixel_values,
                feats.text_tokens,
                feats.text_attention_mask,
                image_mask,
            )
            if cam is not None:
                regions = regions_from_cam(cam)
                out["image"] = {
                    "method": "grad-cam",
                    # The overlay is composed client-side from these boxes, so
                    # no heatmap image is rendered or stored server-side.
                    "heatmap_url": "",
                    "regions": [
                        {"bbox": r.bbox, "score": r.score, "label": r.label}
                        for r in regions
                    ],
                    "grid": np.round(cam, 4).tolist(),
                }
        except Exception:
            log.exception("grad-cam failed for %s", item_id)

    # --- cross-attention links --------------------------------------------
    if has_image and words:
        try:
            with torch.no_grad():
                model = bundle.arms[task]["cross_attention"]
                _, attentions = model(
                    image_tokens=feats.image_tokens,
                    text_tokens=feats.text_tokens,
                    text_attention_mask=feats.text_attention_mask,
                    image_mask=image_mask,
                    return_attention=True,
                )
            links = extract_links(attentions, words, feats.text_attention_mask)
            out["cross_attention"] = {
                "available": bool(links),
                "top_links": [
                    {
                        "text_token": link.text_token,
                        "image_region": link.image_region,
                        "weight": link.weight,
                    }
                    for link in links
                ],
            }
        except Exception:
            log.exception("attention extraction failed for %s", item_id)

    if out["cross_attention"] is None:
        out["cross_attention"] = {"available": False, "top_links": []}

    log.info(
        "attributions for %s in %dms", item_id, int((time.perf_counter() - started) * 1000)
    )
    return out


def _empty(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "text": None,
        "image": None,
        "cross_attention": {"available": False, "top_links": []},
    }
