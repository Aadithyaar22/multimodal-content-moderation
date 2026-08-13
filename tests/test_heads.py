"""The masked multi-task loss is where a silent bug would be most expensive.

If masking is wrong, the model trains on garbage targets and every number in the
ablation table is wrong in a way that looks plausible. These tests pin the
contract, including the NaN trap that a homogeneous batch would otherwise spring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mcm.data.schema import IGNORE_INDEX, MISINFO_3_LABELS, TOXICITY_LABELS
from mcm.models.baselines import LateFusionModel, UnimodalModel, build_model
from mcm.models.heads import MaskedMultiTaskLoss, MultiTaskHead
from mcm.training.metrics import (
    evaluate_task,
    expected_calibration_error,
    fusion_recall_delta,
)


@pytest.fixture
def head():
    return MultiTaskHead(input_dim=512, hidden_dim=32)


class TestMultiTaskHead:
    def test_output_shapes(self, head):
        out = head(torch.randn(4, 512))
        assert out.toxicity_logits.shape == (4, 2)
        assert out.misinfo_logits.shape == (4, 3)

    def test_both_heads_read_the_same_representation(self, head):
        """Shared trunk is the design (PROJECT_CONTEXT Sec. 4, decision 3)."""
        head.eval()  # dropout would otherwise resample between the two calls
        x = torch.randn(2, 512)
        h = head.trunk(x)
        assert torch.allclose(head.toxicity(h), head(x).toxicity_logits)
        assert torch.allclose(head.misinfo(h), head(x).misinfo_logits)


class TestMaskedLoss:
    def test_all_ignored_does_not_produce_nan(self, head):
        """A batch drawn entirely from one dataset has no labels for the other
        head. Plain cross_entropy returns NaN there and would poison the model."""
        loss_fn = MaskedMultiTaskLoss()
        out = head(torch.randn(4, 512))
        all_ignored = torch.full((4,), IGNORE_INDEX, dtype=torch.long)
        labels = torch.tensor([0, 1, 0, 1])

        result = loss_fn(out, labels, all_ignored)
        assert torch.isfinite(result.total)
        assert result.n_misinfo == 0
        assert float(result.misinfo) == 0.0

    def test_both_heads_ignored_is_finite_zero(self, head):
        loss_fn = MaskedMultiTaskLoss()
        out = head(torch.randn(3, 512))
        ignored = torch.full((3,), IGNORE_INDEX, dtype=torch.long)
        result = loss_fn(out, ignored, ignored)
        assert torch.isfinite(result.total)
        assert float(result.total) == 0.0

    def test_ignored_rows_contribute_no_gradient(self, head):
        loss_fn = MaskedMultiTaskLoss()
        x = torch.randn(4, 512, requires_grad=True)
        out = head(x)
        # Only rows 0 and 2 carry a toxicity label.
        labels = torch.tensor([1, IGNORE_INDEX, 0, IGNORE_INDEX])
        ignored = torch.full((4,), IGNORE_INDEX, dtype=torch.long)

        loss_fn(out, labels, ignored).total.backward()
        assert torch.count_nonzero(x.grad[1]) == 0
        assert torch.count_nonzero(x.grad[3]) == 0
        assert torch.count_nonzero(x.grad[0]) > 0

    def test_counts_reported_reflect_labelled_rows(self, head):
        loss_fn = MaskedMultiTaskLoss()
        out = head(torch.randn(4, 512))
        tox = torch.tensor([1, IGNORE_INDEX, 0, 1])
        mis = torch.tensor([IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX])
        r = loss_fn(out, tox, mis)
        assert r.n_toxicity == 3
        assert r.n_misinfo == 1

    def test_task_weights_scale_contributions(self, head):
        out = head(torch.randn(4, 512))
        tox = torch.tensor([1, 0, 1, 0])
        mis = torch.tensor([2, 1, 0, 2])

        base = MaskedMultiTaskLoss(task_weights=(1.0, 1.0))(out, tox, mis)
        tox_only = MaskedMultiTaskLoss(task_weights=(1.0, 0.0))(out, tox, mis)
        assert float(tox_only.total.detach()) == pytest.approx(float(base.toxicity), abs=1e-5)


class TestModels:
    def test_unimodal_image_ignores_text(self):
        m = UnimodalModel("image").eval()
        img, txt = torch.randn(2, 512), torch.randn(2, 512)
        a = m(image_emb=img, text_emb=txt)
        b = m(image_emb=img, text_emb=torch.randn(2, 512))
        assert torch.allclose(a.toxicity_logits, b.toxicity_logits)

    def test_unimodal_text_ignores_image(self):
        m = UnimodalModel("text").eval()
        img, txt = torch.randn(2, 512), torch.randn(2, 512)
        a = m(image_emb=img, text_emb=txt)
        b = m(image_emb=torch.randn(2, 512), text_emb=txt)
        assert torch.allclose(a.toxicity_logits, b.toxicity_logits)

    def test_late_fusion_uses_both_modalities(self):
        m = LateFusionModel().eval()
        img, txt = torch.randn(2, 512), torch.randn(2, 512)
        base = m(image_emb=img, text_emb=txt).toxicity_logits
        assert not torch.allclose(base, m(image_emb=torch.randn(2, 512), text_emb=txt).toxicity_logits)
        assert not torch.allclose(base, m(image_emb=img, text_emb=torch.randn(2, 512)).toxicity_logits)

    def test_late_fusion_masks_missing_images(self):
        m = LateFusionModel().eval()
        txt = torch.randn(2, 512)
        mask = torch.tensor([False, False])
        a = m(image_emb=torch.randn(2, 512), text_emb=txt, image_mask=mask)
        b = m(image_emb=torch.randn(2, 512), text_emb=txt, image_mask=mask)
        # With the image masked off, differing image inputs must not change output.
        assert torch.allclose(a.toxicity_logits, b.toxicity_logits)

    def test_factory_rejects_unknown_arch(self):
        with pytest.raises(ValueError, match="unknown architecture"):
            build_model("transformer_xl_9000")

    def test_unimodal_rejects_bad_modality(self):
        with pytest.raises(ValueError, match="modality must be"):
            UnimodalModel("audio")


class TestMetrics:
    def test_ignored_rows_excluded_not_counted_wrong(self):
        logits = np.array([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0]])
        labels = np.array([0, 1, IGNORE_INDEX])
        m = evaluate_task("toxicity", logits, labels, TOXICITY_LABELS)
        assert m.n == 2
        assert m.accuracy == 1.0

    def test_returns_empty_when_no_labels_apply(self):
        logits = np.random.randn(3, 3)
        labels = np.full(3, IGNORE_INDEX)
        m = evaluate_task("misinformation", logits, labels, MISINFO_3_LABELS)
        assert m.n == 0
        assert m.macro_f1 == 0.0

    def test_auc_is_none_when_undefined(self):
        """A single-class slice has no AUC; reporting 0.5 would be a fabrication."""
        logits = np.array([[1.0, 2.0], [1.0, 3.0]])
        labels = np.array([1, 1])
        assert evaluate_task("toxicity", logits, labels, TOXICITY_LABELS).auc is None

    def test_perfect_calibration_scores_near_zero(self):
        probs = np.array([[0.5, 0.5]] * 100)
        labels = np.array([0, 1] * 50)
        assert expected_calibration_error(probs, labels) < 0.05

    def test_overconfident_wrong_model_has_high_ece(self):
        probs = np.array([[0.99, 0.01]] * 50)
        labels = np.ones(50, dtype=int)  # always wrong, always certain
        assert expected_calibration_error(probs, labels) > 0.9

    def test_fusion_delta_isolates_cases_both_unimodal_arms_missed(self):
        # 3 positives; unimodal arms miss all of them; fusion catches 2.
        strong, weak = [0.0, 10.0], [10.0, 0.0]
        fusion = np.array([strong, strong, weak])
        uni = np.array([weak, weak, weak])
        labels = np.array([1, 1, 1])

        d = fusion_recall_delta(fusion, [uni, uni], labels)
        assert d["n_hard_cases"] == 3
        assert d["fusion_recall_on_hard"] == pytest.approx(2 / 3)

    def test_fusion_delta_reports_zero_hard_cases_honestly(self):
        strong = [0.0, 10.0]
        logits = np.array([strong, strong])
        labels = np.array([1, 1])
        # Unimodal already catches everything -> no hard subset to measure on.
        d = fusion_recall_delta(logits, [logits], labels)
        assert d["n_hard_cases"] == 0
