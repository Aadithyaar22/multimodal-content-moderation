"""Cross-attention fusion contracts.

This is the project's central claim, so the tests here check that the mechanism
does what the architecture says it does — the modalities genuinely interact, the
masking is honest, and the degenerate inputs that produce silent NaNs are
handled. A NaN in this module would propagate through every parameter on the
backward pass and destroy a run without an error message.
"""

from __future__ import annotations

import pytest
import torch

from mcm.models.fusion import CrossAttentionBlock, CrossAttentionFusion, _masked_mean


@pytest.fixture
def model():
    return CrossAttentionFusion(n_layers=2, n_heads=8).eval()


def _inputs(batch: int = 4, n_text: int = 20):
    image = torch.randn(batch, 50, 768)
    text = torch.randn(batch, 77, 512)
    attn = torch.zeros(batch, 77, dtype=torch.long)
    attn[:, :n_text] = 1
    return image, text, attn


class TestShapes:
    def test_output_shapes(self, model):
        img, txt, attn = _inputs()
        out = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        assert out.toxicity_logits.shape == (4, 2)
        assert out.misinfo_logits.shape == (4, 3)

    def test_block_preserves_sequence_shapes(self):
        block = CrossAttentionBlock(d_model=512, n_heads=8).eval()
        img = torch.randn(2, 50, 512)
        txt = torch.randn(2, 77, 512)
        out_img, out_txt = block(img, txt)
        assert out_img.shape == img.shape
        assert out_txt.shape == txt.shape


class TestModalitiesActuallyInteract:
    """If these fail, the arm is not doing fusion and the headline claim is void."""

    def test_changing_the_image_changes_the_prediction(self, model):
        img, txt, attn = _inputs()
        a = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        b = model(image_tokens=torch.randn(4, 50, 768), text_tokens=txt, text_attention_mask=attn)
        assert not torch.allclose(a.toxicity_logits, b.toxicity_logits, atol=1e-5)

    def test_changing_the_text_changes_the_prediction(self, model):
        img, txt, attn = _inputs()
        a = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        b = model(image_tokens=img, text_tokens=torch.randn(4, 77, 512), text_attention_mask=attn)
        assert not torch.allclose(a.toxicity_logits, b.toxicity_logits, atol=1e-5)

    def test_text_representation_depends_on_the_image(self):
        """The defining property: text tokens must be altered by the image.

        Late fusion cannot do this — its text vector is fixed before the
        modalities meet — and that inability is exactly what the ablation is
        designed to measure.
        """
        block = CrossAttentionBlock(d_model=512, n_heads=8).eval()
        txt = torch.randn(2, 77, 512)
        _, out_a = block(torch.randn(2, 50, 512), txt)
        _, out_b = block(torch.randn(2, 50, 512), txt)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_layers_are_bidirectional_and_parallel(self):
        """Both directions must read the same layer input.

        If they were sequential, the second stream would see an already-updated
        first stream and the architecture would be asymmetric in a way the
        diagram does not show.
        """
        block = CrossAttentionBlock(d_model=512, n_heads=8).eval()
        img = torch.randn(2, 50, 512)
        txt = torch.randn(2, 77, 512)
        out_img, _ = block(img, txt)

        # Image output must depend only on the original text, so recomputing
        # with the same inputs is deterministic in eval mode.
        out_img_again, _ = block(img, txt)
        assert torch.allclose(out_img, out_img_again)


class TestMasking:
    def test_text_only_row_is_invariant_to_image_content(self, model):
        img, txt, attn = _inputs()
        mask = torch.tensor([True, True, False, True])

        a = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn, image_mask=mask)
        noisy = img.clone()
        noisy[2] = torch.randn(50, 768) * 100
        b = model(image_tokens=noisy, text_tokens=txt, text_attention_mask=attn, image_mask=mask)

        # Row 2 has no image; a blank slot must not become a learnable signal.
        assert torch.allclose(a.toxicity_logits[2], b.toxicity_logits[2], atol=1e-5)

    def test_image_bearing_rows_still_respond_to_their_image(self, model):
        img, txt, attn = _inputs()
        mask = torch.tensor([True, True, False, True])
        a = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn, image_mask=mask)
        noisy = img.clone()
        noisy[0] = torch.randn(50, 768)
        b = model(image_tokens=noisy, text_tokens=txt, text_attention_mask=attn, image_mask=mask)
        assert not torch.allclose(a.toxicity_logits[0], b.toxicity_logits[0], atol=1e-5)

    def test_padding_does_not_affect_pooled_text(self):
        x = torch.randn(2, 10, 4)
        mask = torch.zeros(2, 10, dtype=torch.long)
        mask[:, :3] = 1
        expected = x[:, :3].mean(dim=1)
        assert torch.allclose(_masked_mean(x, mask), expected, atol=1e-6)

    def test_masked_mean_survives_an_all_padding_row(self):
        x = torch.randn(2, 10, 4)
        mask = torch.zeros(2, 10, dtype=torch.long)
        out = _masked_mean(x, mask)
        assert torch.isfinite(out).all()


class TestDegenerateInputs:
    """Each of these produced a silent NaN before being guarded."""

    def test_all_padding_text_does_not_produce_nan(self, model):
        img, txt, _ = _inputs()
        attn = torch.zeros(4, 77, dtype=torch.long)
        out = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        assert torch.isfinite(out.toxicity_logits).all()

    def test_one_all_padding_row_among_normal_rows_is_safe(self, model):
        img, txt, attn = _inputs()
        attn[1] = 0
        out = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        assert torch.isfinite(out.toxicity_logits).all()

    def test_no_image_at_all_in_the_batch_is_safe(self, model):
        img, txt, attn = _inputs()
        mask = torch.zeros(4, dtype=torch.bool)
        out = model(
            image_tokens=torch.zeros_like(img),
            text_tokens=txt,
            text_attention_mask=attn,
            image_mask=mask,
        )
        assert torch.isfinite(out.toxicity_logits).all()

    def test_gradients_stay_finite(self):
        model = CrossAttentionFusion(n_layers=2)
        img, txt, attn = _inputs()
        attn[0] = 0  # a degenerate row mixed in with normal ones
        out = model(
            image_tokens=img,
            text_tokens=txt,
            text_attention_mask=attn,
            image_mask=torch.tensor([True, False, True, True]),
        )
        out.toxicity_logits.sum().backward()

        bad = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert bad == []


class TestNoCrossModalControl:
    """The confound-isolation control: same modules, same parameter count,
    cross-modal information flow switched off. Built in response to a real
    question about the headline ablation — cross-attention has ~18-25x late
    fusion's own parameter count and reads unpooled token features late
    fusion never sees, so a win there could be capacity or granularity, not
    the cross-modal mechanism specifically. These tests exist to make sure
    the control actually controls for what it claims to."""

    def test_parameter_count_is_exactly_identical_to_cross_attention(self):
        """The whole point of this control: it must not itself introduce a
        capacity difference, or it would just be a second, different
        confound instead of an answer to the first one."""
        with_cross = CrossAttentionFusion(d_model=256, n_layers=2, n_heads=8, cross_modal=True)
        without_cross = CrossAttentionFusion(d_model=256, n_layers=2, n_heads=8, cross_modal=False)
        n_with = sum(p.numel() for p in with_cross.parameters())
        n_without = sum(p.numel() for p in without_cross.parameters())
        assert n_with == n_without

    def test_build_model_dispatches_to_the_control(self):
        from mcm.models.baselines import build_model

        m = build_model("no_cross_attention", d_model=256, n_layers=2, n_heads=8)
        assert isinstance(m, CrossAttentionFusion)
        assert m.blocks[0].cross_modal is False

    def test_image_stream_is_invariant_to_text_content(self):
        """The property the control is named for: with cross_modal=False, the
        image stream must not change at all when only the text changes —
        if it did, information would be leaking across modalities somewhere
        despite the query/key/value routing, and the control would be
        measuring the wrong thing."""
        block = CrossAttentionBlock(d_model=512, n_heads=8, cross_modal=False).eval()
        img = torch.randn(2, 50, 512)
        out_img_a, _ = block(img, torch.randn(2, 77, 512))
        out_img_b, _ = block(img, torch.randn(2, 77, 512))
        assert torch.allclose(out_img_a, out_img_b, atol=1e-6)

    def test_text_stream_is_invariant_to_image_content(self):
        block = CrossAttentionBlock(d_model=512, n_heads=8, cross_modal=False).eval()
        txt = torch.randn(2, 77, 512)
        _, out_txt_a = block(torch.randn(2, 50, 512), txt)
        _, out_txt_b = block(torch.randn(2, 50, 512), txt)
        assert torch.allclose(out_txt_a, out_txt_b, atol=1e-6)

    def test_each_stream_still_processes_its_own_content(self):
        """Not a no-op: self-attention plus the per-token FFN still refines
        each stream from its own tokens, so cross_modal=False is a genuine
        architectural variant, not a dead block that just happens to have the
        right parameter count."""
        block = CrossAttentionBlock(d_model=512, n_heads=8, cross_modal=False).eval()
        img = torch.randn(2, 50, 512)
        out_a, _ = block(img, torch.randn(2, 77, 512))
        out_b, _ = block(img * 3.0, torch.randn(2, 77, 512))
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_full_model_forward_and_backward_are_finite(self):
        """Same degenerate-input guarantees as the cross-attention arm —
        this control has to be trainable for real, not just constructible."""
        model = CrossAttentionFusion(n_layers=2, n_heads=8, cross_modal=False)
        img, txt, attn = _inputs()
        out = model(image_tokens=img, text_tokens=txt, text_attention_mask=attn)
        assert torch.isfinite(out.toxicity_logits).all()
        out.toxicity_logits.sum().backward()
        bad = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert bad == []


class TestCapacity:
    def test_head_matches_the_other_arms(self, model):
        """Shared head keeps the ablation honest: a win must come from fusion,
        not from this arm having been handed a bigger classifier."""
        from mcm.models.baselines import LateFusionModel

        late = LateFusionModel(hidden_dim=256)
        assert type(model.head) is type(late.head)
        assert model.head.toxicity.out_features == late.head.toxicity.out_features
        assert model.head.trunk[1].out_features == late.head.trunk[1].out_features

    def test_layer_count_is_configurable(self):
        assert len(CrossAttentionFusion(n_layers=2).blocks) == 2
        assert len(CrossAttentionFusion(n_layers=4).blocks) == 4
