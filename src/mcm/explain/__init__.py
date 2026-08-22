"""Explainability: token attribution, Grad-CAM, and cross-attention links."""

from mcm.explain.attention import AttentionLink, extract_links
from mcm.explain.text import TokenAttribution, build_score_fn, occlusion_attribution
from mcm.explain.vision import Region, grad_cam, regions_from_cam

__all__ = [
    "AttentionLink",
    "Region",
    "TokenAttribution",
    "build_score_fn",
    "extract_links",
    "grad_cam",
    "occlusion_attribution",
    "regions_from_cam",
]
