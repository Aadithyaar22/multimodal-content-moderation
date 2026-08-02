"""Schema validation is the guard rail for every downstream training run.

These tests exist because a malformed manifest is otherwise discovered hours
into training, inside a dataloader worker, with a useless traceback.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mcm.data.schema import (
    COLUMNS,
    IGNORE_INDEX,
    MISINFO_3_LABELS,
    MISINFO_6_LABELS,
    MISINFO_6_TO_3,
    Record,
    label_summary,
    make_uid,
    records_to_frame,
    validate_frame,
)


def _record(**overrides) -> Record:
    base = dict(
        uid="ds:1",
        dataset="ds",
        split="train",
        text="hello",
        image_path="raw/ds/1.png",
        has_image=True,
        label_toxicity=1,
    )
    base.update(overrides)
    return Record(**base)


class TestRecordRoundtrip:
    def test_records_to_frame_produces_declared_dtypes(self):
        df = records_to_frame([_record(), _record(uid="ds:2", label_toxicity=0)])
        assert list(df.columns) == list(COLUMNS)
        assert df["has_image"].dtype == bool
        assert df["label_toxicity"].dtype == "int8"
        assert len(df) == 2

    def test_meta_is_serialized_as_json_string(self):
        df = records_to_frame([_record(meta={"tokens": ["a", "b"], "n": 2})])
        assert isinstance(df.loc[0, "meta"], str)
        assert '"tokens"' in df.loc[0, "meta"]

    def test_empty_input_yields_typed_empty_frame(self):
        df = records_to_frame([])
        assert len(df) == 0
        assert list(df.columns) == list(COLUMNS)

    def test_inapplicable_labels_default_to_sentinel(self):
        # A Hateful Memes row carries no misinformation label; it must be the
        # sentinel so the loss masks that head, not 0 which means "true".
        df = records_to_frame([_record()])
        assert df.loc[0, "label_misinfo_3"] == IGNORE_INDEX
        assert df.loc[0, "label_misinfo_6"] == IGNORE_INDEX

    def test_make_uid_namespaces_by_dataset(self):
        assert make_uid("fakeddit", "abc") == "fakeddit:abc"
        assert make_uid("hateful_memes", 42) == "hateful_memes:42"


class TestValidation:
    def test_rejects_duplicate_uids(self):
        with pytest.raises(ValueError, match="duplicate uids"):
            records_to_frame([_record(), _record()])

    def test_rejects_unknown_split(self):
        with pytest.raises(ValueError, match="unknown split"):
            records_to_frame([_record(split="dev")])

    def test_rejects_out_of_range_label(self):
        with pytest.raises(ValueError, match="label_toxicity"):
            records_to_frame([_record(label_toxicity=7)])

    def test_rejects_has_image_without_path(self):
        with pytest.raises(ValueError, match="has_image=True but no image_path"):
            records_to_frame([_record(image_path="")])

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="missing columns"):
            validate_frame(pd.DataFrame({"uid": ["a"], "dataset": ["d"]}))

    def test_text_only_record_is_valid(self):
        df = records_to_frame([_record(image_path="", has_image=False)])
        assert not df.loc[0, "has_image"]

    def test_sentinel_is_always_allowed(self):
        df = records_to_frame(
            [_record(label_toxicity=IGNORE_INDEX, label_misinfo_3=2, label_misinfo_6=5)]
        )
        assert df.loc[0, "label_toxicity"] == IGNORE_INDEX


class TestLabelTaxonomy:
    def test_every_6way_label_maps_into_the_3way_head(self):
        assert set(MISINFO_6_TO_3) == set(MISINFO_6_LABELS)
        assert set(MISINFO_6_TO_3.values()) <= set(MISINFO_3_LABELS)

    def test_satire_stays_distinct_from_misleading(self):
        # Collapsing satire into "misleading" would make parody a moderation
        # target — the exact false-positive failure mode the project is about.
        assert MISINFO_6_TO_3[1] != MISINFO_6_TO_3[2]
        assert MISINFO_3_LABELS[MISINFO_6_TO_3[1]] == "satire"

    def test_true_maps_to_true(self):
        assert MISINFO_3_LABELS[MISINFO_6_TO_3[0]] == "true"

    def test_all_deceptive_classes_collapse_together(self):
        deceptive = {MISINFO_6_TO_3[i] for i in (2, 3, 4, 5)}
        assert deceptive == {2}


class TestLabelSummary:
    def test_reports_only_applicable_heads(self):
        df = records_to_frame([_record(), _record(uid="ds:2", label_toxicity=0)])
        out = label_summary(df)
        assert "label_toxicity" in out
        # No misinformation labels present, so that head is not reported at all.
        assert "label_misinfo_3" not in out

    def test_reports_percentages(self):
        df = records_to_frame(
            [_record(uid=f"ds:{i}", label_toxicity=int(i < 3)) for i in range(4)]
        )
        out = label_summary(df)
        assert "n=4" in out
        assert "75.0%" in out
