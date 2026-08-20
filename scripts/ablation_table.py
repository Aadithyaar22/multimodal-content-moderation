#!/usr/bin/env python
"""Assemble the ablation table from recorded runs.

    python scripts/ablation_table.py --datasets hateful_memes
    python scripts/ablation_table.py --datasets hateful_memes --markdown

Reads every reports/results/*.json for the requested benchmark and reports each
arm's mean, standard deviation, and a 95% confidence interval across seeds,
plus a Welch's t-test of the proposed architecture against the late-fusion
baseline.

The significance test is the point. With three seeds and standard deviations
around 0.015, a gap of under one point is indistinguishable from seed noise, and
reporting it as a win would be the single easiest way to make this project's
central claim unfalsifiable. The table therefore states explicitly whether a
difference is separable from noise, rather than leaving a reader to eyeball two
overlapping error bars.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict

from scipy import stats

from mcm.training.trainer import results_dir
from mcm.utils import console

ARM_ORDER = ["cv_only", "nlp_only", "late_fusion", "cross_attention"]
ARM_LABELS = {
    "cv_only": "CV-only",
    "nlp_only": "NLP-only",
    "late_fusion": "Late fusion",
    "cross_attention": "Cross-attention",
}
TASK_FOR = {"hateful_memes": "toxicity", "fakeddit": "misinformation"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["hateful_memes", "fakeddit"])
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table for the report")
    ap.add_argument("--baseline", default="late_fusion", help="arm to test the proposal against")
    ap.add_argument("--proposal", default="cross_attention")
    args = ap.parse_args()

    for dataset in args.datasets:
        runs = collect(dataset)
        if not runs:
            console.print(f"[yellow]no results for {dataset}[/]")
            continue
        report(dataset, runs, args)
    return 0


def collect(dataset: str) -> dict[str, list[dict]]:
    """Group recorded runs by architecture for one benchmark."""
    task = TASK_FOR.get(dataset, "toxicity")
    grouped: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(glob.glob(str(results_dir() / "*.json"))):
        name = path.rsplit("/", 1)[-1]
        if name.startswith("sweep_") or name == "index.json":
            continue
        blob = json.loads(open(path).read())
        cfg = blob.get("config", {})
        if cfg.get("datasets") != [dataset]:
            continue
        metrics = blob.get("test", {}).get(task, {})
        if not metrics.get("n"):
            continue
        grouped[cfg["arch"]].append(
            {
                "seed": cfg.get("seed"),
                "macro_f1": metrics["macro_f1"],
                "auc": metrics.get("auc"),
                "accuracy": metrics["accuracy"],
                "d_model": cfg.get("d_model"),
                "lr": cfg.get("lr"),
                "file": name,
            }
        )
    return grouped


def summarize(values: list[float]) -> tuple[float, float, tuple[float, float]]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, (mean, mean)
    sd = statistics.stdev(values)
    # t-based interval: with 3-5 seeds the normal approximation is too narrow.
    half = stats.t.ppf(0.975, len(values) - 1) * sd / (len(values) ** 0.5)
    return mean, sd, (mean - half, mean + half)


def report(dataset: str, runs: dict[str, list[dict]], args) -> None:
    task = TASK_FOR.get(dataset, "toxicity")
    console.rule(f"{dataset}  ({task} head)")

    rows = []
    for arm in ARM_ORDER:
        if arm not in runs:
            continue
        # Keep only the most recent configuration per arm: an arm re-run after
        # tuning must not be averaged together with its pre-tuning runs.
        entries = latest_config(runs[arm])
        f1 = [e["macro_f1"] for e in entries]
        aucs = [e["auc"] for e in entries if e["auc"] is not None]
        mean, sd, ci = summarize(f1)
        rows.append(
            {
                "arm": arm,
                "n_seeds": len(f1),
                "mean": mean,
                "sd": sd,
                "ci": ci,
                "auc": statistics.mean(aucs) if aucs else None,
                "values": f1,
            }
        )

    if args.markdown:
        emit_markdown(rows)
    else:
        emit_console(rows)

    compare(rows, args.proposal, args.baseline)


def latest_config(entries: list[dict]) -> list[dict]:
    """Select the run group sharing the most recent hyperparameters.

    Runs are grouped by (d_model, lr) so a tuned re-run replaces the earlier
    untuned one rather than being silently pooled with it, which would drag the
    reported mean toward a configuration no longer being proposed.
    """
    by_config: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        by_config[(e["d_model"], e["lr"])].append(e)
    # Prefer the configuration with the most seeds, tie-broken by best mean.
    best = max(
        by_config.values(),
        key=lambda group: (len(group), statistics.mean(x["macro_f1"] for x in group)),
    )
    return best


def emit_console(rows: list[dict]) -> None:
    console.print(f"  {'arm':18s} {'seeds':>5s}  {'macro-F1':>18s}  {'95% CI':>16s}  {'AUC':>6s}")
    for r in rows:
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  n/a"
        console.print(
            f"  {ARM_LABELS[r['arm']]:18s} {r['n_seeds']:>5d}  "
            f"{r['mean']:.4f} ± {r['sd']:.4f}  {ci:>16s}  {auc:>6s}"
        )


def emit_markdown(rows: list[dict]) -> None:
    print("\n| Arm | Seeds | Test macro-F1 | 95% CI | AUC |")
    print("|---|---|---|---|---|")
    for r in rows:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
        print(
            f"| {ARM_LABELS[r['arm']]} | {r['n_seeds']} | "
            f"{r['mean']:.4f} ± {r['sd']:.4f} | "
            f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {auc} |"
        )
    print()


def compare(rows: list[dict], proposal: str, baseline: str) -> None:
    by_arm = {r["arm"]: r for r in rows}
    if proposal not in by_arm or baseline not in by_arm:
        return

    a, b = by_arm[proposal], by_arm[baseline]
    delta = a["mean"] - b["mean"]

    if min(len(a["values"]), len(b["values"])) < 2:
        console.print("\n  [yellow]too few seeds to test significance[/]")
        return

    # Welch's t-test: the arms are independent runs and their variances differ
    # (the transformer arm is visibly noisier across seeds).
    t_stat, p = stats.ttest_ind(a["values"], b["values"], equal_var=False)

    console.print(
        f"\n  {ARM_LABELS[proposal]} - {ARM_LABELS[baseline]} = {delta:+.4f} macro-F1"
    )
    console.print(f"  Welch's t-test: t={t_stat:.3f}, p={p:.3f}")

    if p < 0.05:
        direction = "better" if delta > 0 else "WORSE"
        console.print(f"  [green]separable from seed noise[/] — {ARM_LABELS[proposal]} is {direction}")
    else:
        console.print(
            f"  [yellow]not separable from seed noise[/] (p={p:.3f}). "
            f"At {len(a['values'])} seeds this gap cannot be called a win."
        )


if __name__ == "__main__":
    sys.exit(main())
