"""Regression tests for the Fakeddit dedup and leakage guards.

The upstream mirror's own train/val/test partition shares 25 images across
splits. Left in place, the model memorizes those images during training and is
rewarded for the memory at evaluation, inflating exactly the misinformation
numbers the ablation table reports. These tests pin the fix.
"""

from __future__ import annotations

import pandas as pd

from mcm.data.prepare.fakeddit import _find_leaked_images


def _frame(keys: list[str], texts: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_key": keys,
            "text": texts or [f"post {i}" for i in range(len(keys))],
            "6_way_label": [0] * len(keys),
        }
    )


class TestLeakageDetection:
    def test_detects_train_test_overlap(self):
        frames = {
            "train": _frame(["a", "b", "c"]),
            "val": _frame(["d"]),
            "test": _frame(["c", "e"]),
        }
        assert _find_leaked_images(frames) == {"c"}

    def test_detects_overlap_across_every_split_pair(self):
        frames = {
            "train": _frame(["a", "b"]),
            "val": _frame(["b", "x"]),
            "test": _frame(["x", "z"]),
        }
        # b leaks train->val, x leaks val->test.
        assert _find_leaked_images(frames) == {"b", "x"}

    def test_clean_partition_reports_nothing(self):
        frames = {
            "train": _frame(["a", "b"]),
            "val": _frame(["c"]),
            "test": _frame(["d"]),
        }
        assert _find_leaked_images(frames) == set()

    def test_repeats_within_one_split_are_not_leakage(self):
        # Several posts legitimately share one image; that is only a problem
        # when it spans splits.
        frames = {
            "train": _frame(["a", "a", "a"]),
            "val": _frame(["b"]),
            "test": _frame(["c"]),
        }
        assert _find_leaked_images(frames) == set()


class TestDeduplication:
    def test_shared_image_with_different_captions_is_kept(self):
        """Image reuse under a new caption is a distinct post — and is itself
        the misinformation signal from PROJECT_CONTEXT.md Sec. 1 Example B."""
        df = _frame(["img1", "img1", "img1"], ["astral documents", "a sabbatical", "ask the clerk"])
        deduped = df.drop_duplicates(subset=["text", "image_key", "6_way_label"])
        assert len(deduped) == 3

    def test_exact_duplicate_rows_are_dropped(self):
        df = _frame(["img1", "img1"], ["same caption", "same caption"])
        deduped = df.drop_duplicates(subset=["text", "image_key", "6_way_label"])
        assert len(deduped) == 1
