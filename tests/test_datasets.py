"""Dataloading behaviour, especially around missing modalities.

The mixed-modality path is the subtle one: HateXplain rows have no image and
must arrive with image_mask=False so the model can exclude them from the vision
side of cross-attention. If that mask were ever wrong, the vision branch would
learn that a blank image predicts hate speech, and the fusion result — the
headline number of the whole project — would be quietly corrupted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from mcm.data.datasets import ModerationDataset, class_weights, make_collate_fn
from mcm.data.schema import IGNORE_INDEX, Record, records_to_frame


@pytest.fixture
def image_root(tmp_path, monkeypatch):
    """Point manifest path resolution at a temp dir holding one real image."""
    img_dir = tmp_path / "raw" / "toy"
    img_dir.mkdir(parents=True)
    arr = (np.random.default_rng(0).random((64, 48, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(img_dir / "a.png")

    import mcm.data.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def mixed_frame():
    """One image-bearing toxicity row, one text-only row, one misinfo row."""
    return records_to_frame(
        [
            Record(
                uid="hm:1",
                dataset="hateful_memes",
                split="train",
                text="a meme caption",
                image_path="raw/toy/a.png",
                has_image=True,
                label_toxicity=1,
            ),
            Record(
                uid="hx:1",
                dataset="hatexplain",
                split="train",
                text="a text only post",
                has_image=False,
                label_toxicity=0,
            ),
            Record(
                uid="fd:1",
                dataset="fakeddit",
                split="train",
                text="a reddit title",
                image_path="raw/toy/a.png",
                has_image=True,
                label_misinfo_3=2,
                label_misinfo_6=5,
            ),
        ]
    )


class TestItemLoading:
    def test_image_row_loads_clip_shaped_tensor(self, image_root, mixed_frame):
        item = ModerationDataset(mixed_frame)[0]
        assert item["pixel_values"].shape == (3, 224, 224)
        assert item["image_mask"] is True

    def test_text_only_row_has_no_pixels(self, image_root, mixed_frame):
        item = ModerationDataset(mixed_frame)[1]
        assert item["pixel_values"] is None
        assert item["image_mask"] is False

    def test_unreadable_image_degrades_to_text_only(self, image_root, mixed_frame, tmp_path):
        # A corrupt file must not kill a multi-hour training run.
        bad = mixed_frame.copy()
        bad.loc[0, "image_path"] = "raw/toy/does_not_exist.png"
        item = ModerationDataset(bad)[0]
        assert item["image_mask"] is False
        assert item["pixel_values"] is None

    def test_meta_is_decoded_only_when_requested(self, image_root):
        frame = records_to_frame(
            [
                Record(
                    uid="hx:2",
                    dataset="hatexplain",
                    split="train",
                    text="w x",
                    label_toxicity=1,
                    meta={"rationale_mask": [1, 0]},
                )
            ]
        )
        assert "meta" not in ModerationDataset(frame)[0]
        assert ModerationDataset(frame, keep_meta=True)[0]["meta"]["rationale_mask"] == [1, 0]


class TestCollate:
    def test_batches_mixed_modalities(self, image_root, mixed_frame):
        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn()([ds[i] for i in range(3)])

        assert len(batch) == 3
        assert batch.pixel_values.shape == (3, 3, 224, 224)
        assert batch.image_mask.tolist() == [True, False, True]

    def test_missing_image_is_zero_filled(self, image_root, mixed_frame):
        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn()([ds[i] for i in range(3)])
        # Zero rather than a black image, so an accidental unmasked read is
        # obvious rather than looking like plausible dark pixels.
        assert torch.count_nonzero(batch.pixel_values[1]) == 0
        assert torch.count_nonzero(batch.pixel_values[0]) > 0

    def test_labels_preserve_the_ignore_sentinel(self, image_root, mixed_frame):
        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn()([ds[i] for i in range(3)])

        # Hateful Memes row: toxicity labeled, misinformation ignored.
        assert batch.label_toxicity[0] == 1
        assert batch.label_misinfo_3[0] == IGNORE_INDEX
        # Fakeddit row: the reverse.
        assert batch.label_toxicity[2] == IGNORE_INDEX
        assert batch.label_misinfo_3[2] == 2
        assert batch.label_misinfo_6[2] == 5

    def test_sentinel_survives_cross_entropy(self, image_root, mixed_frame):
        """The masking contract: ignored rows contribute no gradient."""
        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn()([ds[i] for i in range(3)])

        logits = torch.randn(3, 2, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(
            logits, batch.label_toxicity, ignore_index=IGNORE_INDEX
        )
        loss.backward()

        assert torch.isfinite(loss)
        # Row 2 is the Fakeddit row with no toxicity label — zero gradient.
        assert torch.count_nonzero(logits.grad[2]) == 0
        assert torch.count_nonzero(logits.grad[0]) > 0

    def test_tokenizer_output_is_attached(self, image_root, mixed_frame):
        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                n = len(texts)
                return {
                    "input_ids": torch.ones(n, 5, dtype=torch.long),
                    "attention_mask": torch.ones(n, 5, dtype=torch.long),
                }

        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn(tokenizer=FakeTokenizer())([ds[i] for i in range(3)])
        assert batch.text_inputs is not None
        assert batch.text_inputs["input_ids"].shape == (3, 5)

    def test_batch_moves_to_device(self, image_root, mixed_frame):
        ds = ModerationDataset(mixed_frame)
        batch = make_collate_fn()([ds[i] for i in range(3)]).to(torch.device("cpu"))
        assert batch.pixel_values.device.type == "cpu"


class TestClassWeights:
    def test_ignores_sentinel_rows(self, mixed_frame):
        w = class_weights(mixed_frame, "label_toxicity")
        assert w is not None
        assert w.shape == (2,)
        # One benign, one harmful among applicable rows -> balanced.
        assert torch.allclose(w, torch.ones(2))

    def test_upweights_the_minority_class(self):
        frame = records_to_frame(
            [
                Record(uid=f"d:{i}", dataset="d", split="train", text="t", label_toxicity=int(i < 1))
                for i in range(10)
            ]
        )
        w = class_weights(frame, "label_toxicity")
        assert w[1] > w[0]

    def test_returns_none_when_no_labels_apply(self, mixed_frame):
        assert class_weights(mixed_frame, "label_misinfo_6") is not None
        only_toxicity = mixed_frame[mixed_frame["dataset"] != "fakeddit"]
        assert class_weights(only_toxicity, "label_misinfo_3") is None
