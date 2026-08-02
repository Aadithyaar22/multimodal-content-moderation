"""The canonical record format every dataset normalizes into.

Why this exists
---------------
Hateful Memes carries only a harm label. Fakeddit carries only a misinformation
label. HateXplain is text-only and carries a harm label plus token rationales.
The architecture (PROJECT_CONTEXT.md Sec. 4) puts a toxicity head and a
misinformation head on top of *one shared fused embedding*, so all three have to
flow through the same training loop.

They do that by normalizing into the record below, where a label that does not
apply to a sample is set to ``IGNORE_INDEX`` rather than being dropped or faked.
The loss masks those positions out, so a Hateful Memes batch trains the toxicity
head and the shared trunk while leaving the misinformation head untouched. That
masking is what makes the multi-task setup honest — without it we would either
be inventing labels or training two disjoint models and losing the shared
representation the ablation study is meant to measure.

Text-only records (HateXplain) set ``has_image=False``; the collate function
substitutes a zero image and the model masks the vision side of the
cross-attention block, so a missing modality never silently contributes noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

#: Sentinel for "this label does not apply to this sample".
#: Passed explicitly to ``CrossEntropyLoss(ignore_index=...)`` — note torch's own
#: default is -100, so it must never be relied on implicitly.
IGNORE_INDEX = -1

SPLITS = ("train", "val", "test")

#: Binary harm label shared by Hateful Memes and HateXplain.
TOXICITY_LABELS = {0: "benign", 1: "harmful"}

#: The misinformation head's output space (PROJECT_CONTEXT.md Sec. 4).
MISINFO_3_LABELS = {0: "true", 1: "satire", 2: "misleading"}

#: Fakeddit's native 6-way taxonomy, kept for fine-grained error analysis in
#: Chapter 9 even though the head itself predicts the 3-way collapse above.
MISINFO_6_LABELS = {
    0: "true",
    1: "satire_parody",
    2: "misleading_content",
    3: "imposter_content",
    4: "false_connection",
    5: "manipulated_content",
}

#: How the 6-way taxonomy collapses into the 3-way head.
#: Satire is deliberately kept separate from the other falsehoods: it is
#: not deceptive in intent, and a moderation system that treats parody as
#: misinformation is exactly the false-positive failure mode in Sec. 1.
MISINFO_6_TO_3 = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}

COLUMNS: dict[str, str] = {
    "uid": "string",
    "dataset": "string",
    "split": "string",
    "text": "string",
    "image_path": "string",
    "has_image": "bool",
    "label_toxicity": "int8",
    "label_misinfo_3": "int8",
    "label_misinfo_6": "int8",
    "meta": "string",
}


@dataclass
class Record:
    """One moderation sample in canonical form.

    image_path is stored relative to ``config.DATA_DIR`` so manifests stay
    portable across machines and can be committed as artifacts if ever needed.
    """

    uid: str
    dataset: str
    split: str
    text: str
    image_path: str = ""
    has_image: bool = False
    label_toxicity: int = IGNORE_INDEX
    label_misinfo_3: int = IGNORE_INDEX
    label_misinfo_6: int = IGNORE_INDEX
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["meta"] = json.dumps(row["meta"], ensure_ascii=False)
        return row


def make_uid(dataset: str, original_id: str | int) -> str:
    """Globally unique id, so records stay traceable after datasets are mixed."""
    return f"{dataset}:{original_id}"


def records_to_frame(records: list[Record]) -> pd.DataFrame:
    """Build a schema-conformant DataFrame from records."""
    if not records:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})
    df = pd.DataFrame([r.to_row() for r in records])
    return validate_frame(df)


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column set, dtypes, and label ranges. Raises on violation.

    Every prepare_* pipeline ends with this call, so a malformed manifest fails
    at build time rather than halfway through a training run.
    """
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    df = df[list(COLUMNS)].copy()
    for col, dtype in COLUMNS.items():
        df[col] = df[col].astype(dtype)

    bad_split = set(df["split"].unique()) - set(SPLITS)
    if bad_split:
        raise ValueError(f"unknown split(s): {sorted(bad_split)}; allowed={SPLITS}")

    if df["uid"].duplicated().any():
        dupes = df.loc[df["uid"].duplicated(), "uid"].head(5).tolist()
        raise ValueError(f"duplicate uids, e.g. {dupes}")

    _check_label_range(df, "label_toxicity", TOXICITY_LABELS)
    _check_label_range(df, "label_misinfo_3", MISINFO_3_LABELS)
    _check_label_range(df, "label_misinfo_6", MISINFO_6_LABELS)

    # An image-bearing record without a path would fail at load time, deep inside
    # a dataloader worker where the traceback is useless.
    broken = df["has_image"] & (df["image_path"].fillna("") == "")
    if broken.any():
        raise ValueError(f"{int(broken.sum())} records have has_image=True but no image_path")

    return df.reset_index(drop=True)


def _check_label_range(df: pd.DataFrame, col: str, labels: dict[int, str]) -> None:
    allowed = set(labels) | {IGNORE_INDEX}
    seen = set(df[col].unique().tolist())
    bad = seen - allowed
    if bad:
        raise ValueError(f"{col} has out-of-range values {sorted(bad)}; allowed={sorted(allowed)}")


def label_summary(df: pd.DataFrame) -> str:
    """Human-readable class balance, printed after every prepare run.

    Class imbalance is a documented concern (report Sec. 3.3), so it is surfaced
    at build time rather than discovered during training.
    """
    lines = [f"n={len(df)}  with_image={int(df['has_image'].sum())}"]
    for col, labels in (
        ("label_toxicity", TOXICITY_LABELS),
        ("label_misinfo_3", MISINFO_3_LABELS),
        ("label_misinfo_6", MISINFO_6_LABELS),
    ):
        counts = df[col].value_counts().to_dict()
        applicable = {k: v for k, v in counts.items() if k != IGNORE_INDEX}
        if not applicable:
            continue
        total = sum(applicable.values())
        parts = [
            f"{labels[k]}={applicable[k]} ({applicable[k] / total:.1%})"
            for k in sorted(applicable)
        ]
        lines.append(f"  {col}: " + ", ".join(parts))
    return "\n".join(lines)
