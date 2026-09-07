#!/usr/bin/env python
"""Deletion-test faithfulness of the explanation layer.

    python scripts/faithfulness_eval.py --datasets hateful_memes
    python scripts/faithfulness_eval.py --datasets hateful_memes fakeddit --n 60 --markdown

PROJECT_CONTEXT.md Sec. 6 names this explicitly: "remove Grad-CAM/SHAP-highlighted
region, verify score drops" — and flags it, alongside the human-agreement study,
as the check most likely to get skipped under time pressure. This is that check,
run for real against the trained cross-attention checkpoints and the actual
Grad-CAM / occlusion code the live explanation endpoint uses (mcm.explain), not
a simulated or hand-picked example.

Method
------
For each sampled test item: mask the region the explanation names as most
important (top Grad-CAM patches for the image, top-occlusion tokens for the
text) and re-score. A faithful explanation should see the score move more from
that masking than from masking an equally-sized *random* region — removing the
part of the input the model claims mattered should hurt more than removing an
arbitrary part of it. Reporting only the top-region drop, with no control,
would not distinguish "the explanation is faithful" from "the model is just
sensitive to losing any 12 patches" — the random-region condition is what makes
this a faithfulness test rather than a sensitivity test. Both conditions run
against the same sampled items, so the two drops are paired, and a paired
t-test says whether the gap between them is separable from noise — the same
standard this project holds its headline ablation result to
(scripts/ablation_table.py).

Masking is done in CLIP's own normalized pixel space (values near the input
distribution's mean are set to 0) rather than by editing and re-preprocessing
the source image — cheaper, and avoids reintroducing JPEG/resize artifacts that
have nothing to do with the region being tested.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field

import torch
from PIL import Image
from scipy import stats

from mcm.config import DATA_DIR
from mcm.data.manifest import read_manifest
from mcm.explain.text import build_score_fn, occlusion_attribution
from mcm.explain.vision import grad_cam
from mcm.serving.inference import ModelBundle, load_bundle
from mcm.training.trainer import results_dir
from mcm.utils import console
from mcm.utils.logging import get_logger

log = get_logger(__name__)

TASK_FOR = {"hateful_memes": "toxicity", "fakeddit": "misinformation"}
PATCH_GRID = 7  # ViT-B/32 at 224px: 224 / 32
MASK_FRACTION = 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["hateful_memes"], choices=list(TASK_FOR))
    ap.add_argument("--n", type=int, default=40, help="items sampled per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table for the report")
    args = ap.parse_args()

    console.rule("loading checkpoints")
    bundle = load_bundle()

    all_results = []
    for dataset in args.datasets:
        task = TASK_FOR[dataset]
        if task not in bundle.arms:
            console.print(f"[yellow]no cross-attention checkpoint for task={task}; skipping {dataset}[/]")
            continue
        result = run_dataset(bundle, dataset, task, n=args.n, seed=args.seed)
        all_results.append(result)
        report(result, markdown=args.markdown)

    out = results_dir() / "faithfulness.json"
    out.write_text(json.dumps([r.to_dict() for r in all_results], indent=2))
    console.print(f"\nsaved -> {out}")
    return 0


@dataclass
class ItemResult:
    uid: str
    baseline: float
    image_top_drop: float | None = None
    image_random_drop: float | None = None
    text_top_drop: float | None = None
    text_random_drop: float | None = None


@dataclass
class DatasetResult:
    dataset: str
    task: str
    n_sampled: int
    n_image_evaluated: int
    n_text_evaluated: int
    seed: int
    mask_fraction: float
    items: list[ItemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "items"}
        d["items"] = [i.__dict__ for i in self.items]
        d["image"] = _summarize([i.image_top_drop for i in self.items], [i.image_random_drop for i in self.items])
        d["text"] = _summarize([i.text_top_drop for i in self.items], [i.text_random_drop for i in self.items])
        return d


def _summarize(top: list[float | None], rand: list[float | None]) -> dict:
    paired = [(t, r) for t, r in zip(top, rand, strict=True) if t is not None and r is not None]
    if not paired:
        return {"n": 0}
    tops = [p[0] for p in paired]
    rands = [p[1] for p in paired]
    d = {
        "n": len(paired),
        "mean_top_deletion_drop": round(statistics.mean(tops), 4),
        "mean_random_deletion_drop": round(statistics.mean(rands), 4),
        "faithfulness_gap": round(statistics.mean(tops) - statistics.mean(rands), 4),
    }
    if len(paired) >= 2 and any(t != r for t, r in paired):
        t_stat, p_val = stats.ttest_rel(tops, rands)
        d["paired_ttest"] = {"t": round(float(t_stat), 3), "p": round(float(p_val), 4)}
    return d


def run_dataset(bundle: ModelBundle, dataset: str, task: str, n: int, seed: int) -> DatasetResult:
    frame = read_manifest(dataset, "test")
    with_image = frame[frame["has_image"]]
    label_col = "label_toxicity" if task == "toxicity" else "label_misinfo_3"
    with_image = with_image[with_image[label_col] != -1]

    rng = random.Random(seed)
    rows = with_image.sample(n=min(n, len(with_image)), random_state=seed).to_dict("records")

    items: list[ItemResult] = []
    n_image = n_text = 0
    for row in rows:
        result = evaluate_item(bundle, task, row, rng)
        if result is None:
            continue
        items.append(result)
        n_image += result.image_top_drop is not None
        n_text += result.text_top_drop is not None

    return DatasetResult(
        dataset=dataset,
        task=task,
        n_sampled=len(rows),
        n_image_evaluated=n_image,
        n_text_evaluated=n_text,
        seed=seed,
        mask_fraction=MASK_FRACTION,
        items=items,
    )


@torch.no_grad()
def _score(bundle: ModelBundle, task: str, pixel_values: torch.Tensor, text_inputs: dict) -> float:
    tokens = bundle.clip.encode_tokens(pixel_values=pixel_values, text_inputs=text_inputs)
    image_mask = torch.tensor([True], device=bundle.device)
    out = bundle.arms[task]["cross_attention"](
        image_tokens=tokens.image_tokens,
        text_tokens=tokens.text_tokens,
        text_attention_mask=tokens.text_attention_mask,
        image_mask=image_mask,
    )
    temp = bundle.temperatures.get(task, {}).get("cross_attention", 1.0)
    logits = out.toxicity_logits if task == "toxicity" else out.misinfo_logits
    probs = torch.softmax(logits.float()[0] / temp, dim=-1)
    return float(probs[1]) if task == "toxicity" else float(1.0 - probs[0])


def _mask_patches(pixel_values: torch.Tensor, patch_indices: list[int]) -> torch.Tensor:
    """Zero (CLIP-normalized-neutral) the given 32px patches of a 224x224 image."""
    masked = pixel_values.clone()
    side = 224 // PATCH_GRID
    for idx in patch_indices:
        r, c = divmod(idx, PATCH_GRID)
        masked[:, :, r * side : (r + 1) * side, c * side : (c + 1) * side] = 0.0
    return masked


@torch.no_grad()
def evaluate_item(bundle: ModelBundle, task: str, row: dict, rng: random.Random) -> ItemResult | None:
    try:
        pil = Image.open(DATA_DIR / row["image_path"]).convert("RGB")
    except Exception:  # noqa: BLE001
        log.warning("could not open image for %s; skipping", row["uid"])
        return None

    pixel_values = bundle.clip.preprocess_images([pil]).to(bundle.device)
    text_inputs = {k: v.to(bundle.device) for k, v in bundle.clip.tokenize([row["text"]]).items()}

    baseline = _score(bundle, task, pixel_values, text_inputs)
    result = ItemResult(uid=row["uid"], baseline=round(baseline, 4))

    # --- image: Grad-CAM top patches vs. random patches ---
    tokens = bundle.clip.encode_tokens(pixel_values=pixel_values, text_inputs=text_inputs)
    image_mask = torch.tensor([True], device=bundle.device)
    cam = grad_cam(
        bundle, task, pixel_values, tokens.text_tokens, tokens.text_attention_mask, image_mask
    )
    if cam is not None:
        n_patches = cam.size
        k = max(1, round(MASK_FRACTION * n_patches))
        ranked = cam.flatten().argsort()[::-1].tolist()
        top_idx = ranked[:k]
        random_idx = rng.sample(range(n_patches), k)

        top_masked = _mask_patches(pixel_values, top_idx)
        rand_masked = _mask_patches(pixel_values, random_idx)
        top_score = _score(bundle, task, top_masked, text_inputs)
        rand_score = _score(bundle, task, rand_masked, text_inputs)

        result.image_top_drop = round(baseline - top_score, 4)
        result.image_random_drop = round(baseline - rand_score, 4)

    # --- text: top-occlusion tokens vs. random tokens ---
    words = row["text"].split()
    if len(words) >= 2:
        score_fn = build_score_fn(bundle, task, tokens.image_tokens, image_mask)
        attributions = occlusion_attribution(score_fn, words, baseline=baseline)
        k = max(1, round(MASK_FRACTION * len(words)))
        ranked = sorted(range(len(words)), key=lambda i: -attributions[i].score)
        top_idx = set(ranked[:k])
        random_idx = set(rng.sample(range(len(words)), k))

        top_kept = [w for i, w in enumerate(words) if i not in top_idx]
        rand_kept = [w for i, w in enumerate(words) if i not in random_idx]
        top_score = score_fn(top_kept)
        rand_score = score_fn(rand_kept)

        result.text_top_drop = round(baseline - top_score, 4)
        result.text_random_drop = round(baseline - rand_score, 4)

    return result


def report(result: DatasetResult, markdown: bool) -> None:
    d = result.to_dict()
    console.rule(f"{result.dataset}  ({result.task} head)  n={result.n_sampled}")

    for modality in ("image", "text"):
        s = d[modality]
        if s.get("n", 0) == 0:
            console.print(f"  {modality}: no evaluable items")
            continue
        gap = s["faithfulness_gap"]
        sig = s.get("paired_ttest")
        sig_str = f"  (paired t={sig['t']}, p={sig['p']})" if sig else ""
        console.print(
            f"  {modality:6s} n={s['n']:3d}  top-deletion drop={s['mean_top_deletion_drop']:+.4f}  "
            f"random-deletion drop={s['mean_random_deletion_drop']:+.4f}  gap={gap:+.4f}{sig_str}"
        )
        if sig and sig["p"] < 0.05 and gap > 0:
            console.print("    [green]explanation removes more score than an equal-sized random region — faithful[/]")
        elif sig:
            console.print("    [yellow]not separable from masking a random region at this sample size[/]")

    if markdown:
        print(f"\n**{result.dataset} ({result.task})**, n={result.n_sampled}\n")
        print("| Modality | n | Top-region drop | Random-region drop | Gap | p |")
        print("|---|---|---|---|---|---|")
        for modality in ("image", "text"):
            s = d[modality]
            if s.get("n", 0) == 0:
                continue
            p = s.get("paired_ttest", {}).get("p", "n/a")
            print(
                f"| {modality} | {s['n']} | {s['mean_top_deletion_drop']:+.4f} | "
                f"{s['mean_random_deletion_drop']:+.4f} | {s['faithfulness_gap']:+.4f} | {p} |"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
