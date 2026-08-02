"""Normalize HateXplain — the explainability ground truth.

HateXplain is the only dataset here with *human token-level rationales*: for
each harmful post, annotators marked which words carried the harm. That makes it
the reference against which SHAP attributions are scored in Chapter 7, via the
deletion/insertion faithfulness test. Its role is explanation evaluation, not
just another toxicity training set.

It is text-only, so records carry ``has_image=False`` and the collate function
masks the vision side of the cross-attention block.

The upstream JSON lives in the hate-alert/HateXplain GitHub repo. The
HuggingFace dataset of the same name is a loader *script*, which newer
``datasets`` releases refuse to execute, so the raw files are read directly with
an HF mirror as fallback.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import requests

from mcm.config import DatasetSpec, raw_dir
from mcm.data.manifest import write_manifest
from mcm.data.schema import Record, label_summary, make_uid, records_to_frame
from mcm.utils.logging import get_logger

log = get_logger(__name__)

DATASET = "hatexplain"

UPSTREAM = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data"
FALLBACK_REPO = "Yash22CSU192/hatexplain-v2"
FILES = ("dataset.json", "post_id_divisions.json")

#: HateXplain's official split names -> ours.
SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}


def download(spec: DatasetSpec) -> dict[str, Path]:
    root = raw_dir(DATASET)
    paths: dict[str, Path] = {}

    for name in FILES:
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            paths[name] = dest
            continue
        try:
            log.info("downloading %s/%s", UPSTREAM, name)
            resp = requests.get(f"{UPSTREAM}/{name}", timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as e:  # noqa: BLE001
            log.warning("upstream fetch failed (%s); falling back to %s", e, FALLBACK_REPO)
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(FALLBACK_REPO, name, repo_type="dataset")
            dest.write_bytes(Path(src).read_bytes())
        paths[name] = dest

    return paths


def prepare(spec: DatasetSpec) -> dict[str, Path]:
    paths = download(spec)
    data = json.loads(paths["dataset.json"].read_text())
    divisions = json.loads(paths["post_id_divisions.json"].read_text())

    drop_ties = bool(spec.options.get("drop_ties", True))
    offensive_is_harmful = bool(spec.options.get("offensive_is_harmful", True))

    # post_id -> split, from the dataset's own partition. Posts absent from all
    # three divisions are excluded upstream and stay excluded here.
    split_of: dict[str, str] = {}
    for raw_split, ids in divisions.items():
        mapped = SPLIT_MAP.get(raw_split)
        if mapped:
            for pid in ids:
                split_of[pid] = mapped

    buckets: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    ties = 0

    for post_id, entry in data.items():
        split = split_of.get(post_id)
        if split is None:
            continue

        labels = [a["label"] for a in entry["annotators"]]
        majority = _majority(labels)
        if majority is None:
            # Genuine 3-way annotator disagreement. A hard label here would be
            # invented, so these are dropped rather than broken arbitrarily.
            ties += 1
            if drop_ties:
                continue
            majority = labels[0]

        tokens = entry["post_tokens"]
        harmful = majority == "hatespeech" or (offensive_is_harmful and majority == "offensive")

        buckets[split].append(
            Record(
                uid=make_uid(DATASET, post_id),
                dataset=DATASET,
                split=split,
                text=" ".join(tokens),
                has_image=False,
                label_toxicity=int(harmful),
                meta={
                    "tokens": tokens,
                    # Token-level ground truth for the Chapter 7 faithfulness test.
                    "rationale_mask": _consensus_rationale(entry.get("rationales", []), len(tokens)),
                    "fine_label": majority,
                    "targets": sorted(
                        {t for a in entry["annotators"] for t in a.get("target", []) if t != "None"}
                    ),
                },
            )
        )

    if ties:
        log.info(
            "%s: %d posts had no majority label (%s)",
            DATASET,
            ties,
            "dropped" if drop_ties else "kept with first annotator's label",
        )

    written: dict[str, Path] = {}
    for split, records in buckets.items():
        frame = records_to_frame(records)
        written[split] = write_manifest(frame, DATASET, split)
        n_rat = sum(1 for r in records if any(json.loads(r.to_row()["meta"])["rationale_mask"]))
        log.info("%s/%s\n%s\n  with_rationales=%d", DATASET, split, label_summary(frame), n_rat)

    return written


def _majority(labels: list[str]) -> str | None:
    """Strict majority over the 3 annotator labels; None on a tie."""
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def _consensus_rationale(rationales: list[list[int]], n_tokens: int) -> list[int]:
    """Collapse per-annotator token masks into one consensus mask.

    A token counts as rationale if at least half of the annotators who supplied
    a rationale marked it. Union would be too permissive (one annotator can
    highlight a whole sentence) and unanimity too strict to leave much signal.
    """
    valid = [r for r in rationales if len(r) == n_tokens]
    if not valid:
        return [0] * n_tokens
    threshold = len(valid) / 2
    return [int(sum(col) >= threshold) for col in zip(*valid, strict=True)]
