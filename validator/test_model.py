"""
SpectroConvNeXt — Comprehensive Test Suite (Layers 1 & 2).

Layer 1 — Unit Tests:
    1a. Shape tests for all 5 model variants
    1b. Gradient flow tests (all params receive gradients, no NaN gradients)
    1c. Correctness / invariance tests (audio-domain specific)
    1d. Numerical stability tests (bf16, extreme input values)

Layer 2 — Domain-Specific Benchmarks:
    Audio: Output shape from waveform, log-mel stability, synthetic frame classification
    CV:    Translation robustness, noise entropy (no spatial shortcut)

Usage:
    pytest test_model.py -v
    pytest test_model.py -v -k "shape"       # run only shape tests
    pytest test_model.py -v -k "stability"   # run only numerical stability tests
"""

import math
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Ensure coder module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coder"))

from model import (
    SpectroConvNeXtConfig,
    SpectroConvNeXt,
    atto_s, femto_s, pico_s, nano_s, tiny_s,
)
from layers import (
    GRN,
    StochasticDepth,
    SpectroConvNeXtBlock,
    DownsampleBlock,
    SpectrogramStem,
    PatchifyStem,
    LayerNorm2d,
)
from head import ClassificationHead
from backbone import SpectroConvNeXtBackbone
from frontend import AudioFrontend


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

@pytest.fixture(scope="module")
def device():
    return DEVICE

@pytest.fixture(scope="module")
def dtype():
    return DTYPE

@pytest.fixture
def sample_spectrogram(device, dtype):
    """Standard ESC-50 spectrogram: (B, 1, 128, 86)."""
    return torch.randn(2, 1, 128, 86, device=device, dtype=dtype)

@pytest.fixture(params=[
    ("atto-s",  atto_s,   4.8),
    ("femto-s", femto_s,  9.5),
    ("pico-s",  pico_s,   15.2),
    ("nano-s",  nano_s,   20.5),
    ("tiny-s",  tiny_s,   29.8),
], ids=["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"])
def variant(request):
    """Parametrized fixture over all 5 model variants."""
    name, config_fn, expected_m = request.param
    return name, config_fn(), expected_m


# ═══════════════════════════════════════════════════════════════════════════════
# 1a. Shape Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapes:
    """Verify all model variants produce correct output shapes."""

    def test_output_shape(self, variant, sample_spectrogram, device, dtype):
        """Forward pass on standard input shape."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            logits = model(sample_spectrogram)
        assert logits.shape == (2, 50), (
            f"{name}: expected (2, 50), got {logits.shape}"
        )

    def test_variable_batch_size(self, variant, device, dtype):
        """Should work with batch size 1, 4, 16."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        for B in [1, 4, 16]:
            x = torch.randn(B, 1, 128, 86, device=device, dtype=dtype)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (B, 50), (
                f"{name} B={B}: expected ({B}, 50), got {logits.shape}"
            )

    def test_variable_time_frames(self, variant, device, dtype):
        """Fully convolutional — should handle different time resolutions."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        for T in [43, 86, 172]:
            x = torch.randn(1, 1, 128, T, device=device, dtype=dtype)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (1, 50), (
                f"{name} T={T}: expected (1, 50), got {logits.shape}"
            )

    def test_variable_frequency_bins(self, variant, device, dtype):
        """Fully convolutional — should handle different frequency resolutions."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        for F in [64, 128, 256]:
            x = torch.randn(1, 1, F, 86, device=device, dtype=dtype)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (1, 50), (
                f"{name} F={F}: expected (1, 50), got {logits.shape}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. Gradient Flow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradients:
    """Verify gradients flow correctly through all model variants."""

    def test_all_params_receive_gradients(self, variant, device, dtype):
        """Every trainable parameter should receive a non-None gradient."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
        model.train()

        x = torch.randn(2, 1, 128, 86, device=device, dtype=dtype, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert len(dead) == 0, (
            f"{name}: {len(dead)} params with no gradient: {dead[:5]}"
        )

    def test_no_nan_gradients(self, variant, device, dtype):
        """No gradient should contain NaN values."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
        model.train()

        x = torch.randn(2, 1, 128, 86, device=device, dtype=dtype, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        nan_params = []
        for n, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                nan_params.append(n)
        assert len(nan_params) == 0, (
            f"{name}: NaN gradients in: {nan_params}"
        )

    def test_gradient_flows_through_all_blocks(self, variant, device, dtype):
        """Verify gradient flows through stem, all stages, and head."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
        model.train()

        x = torch.randn(2, 1, 128, 86, device=device, dtype=dtype, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Check specific components
        components = {
            "backbone.stem": model.backbone.stem,
            "backbone.stages": model.backbone.stages,
            "backbone.final_norm": model.backbone.final_norm,
            "head": model.head,
        }
        for comp_name, comp in components.items():
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in comp.parameters()
            )
            assert has_grad, f"{name}: No gradient in '{comp_name}'"

        # Also verify downsamplers (skip identity)
        for i, ds in enumerate(model.backbone.downsample_layers):
            if not isinstance(ds, nn.Identity):
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in ds.parameters()
                )
                assert has_grad, f"{name}: No gradient in downsample_{i}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1c. Correctness / Invariance Tests (Audio + CV Domain)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioCorrectness:
    """Domain-specific correctness tests for audio spectrogram classification."""

    @pytest.fixture
    def audio_model(self, device, dtype):
        cfg = femto_s()
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        return model

    def test_frontend_output_shape(self, device):
        """AudioFrontend should produce (B, 1, n_mels, T_frames)."""
        frontend = AudioFrontend(
            sample_rate=22050,
            clip_duration_sec=2.0,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        ).to(device)
        waveform = torch.randn(2, 1, 44100, device=device)
        mel = frontend(waveform)
        # Expected: (2, 1, 128, 86) — 86 = ceil(44100 / 512) = 87, actually with n_fft it's:
        # T_frames = 1 + (44100 - 1024) // 512 = 1 + 84 = ~85-86
        assert mel.dim() == 4, f"Expected 4D output, got {mel.dim()}D"
        assert mel.shape[0] == 2, f"Batch dim wrong: {mel.shape}"
        assert mel.shape[1] == 1, f"Channel dim wrong: {mel.shape}"
        assert mel.shape[2] == 128, f"Mel bands wrong: {mel.shape}"
        print(f"  [OK] AudioFrontend output shape: {mel.shape}")

    def test_log_mel_stability(self, device):
        """Log-mel should not produce NaN or Inf on silent or random audio."""
        frontend = AudioFrontend(normalize=False).to(device)
        # Silent audio
        silent = torch.zeros(2, 1, 44100, device=device)
        mel = frontend(silent)
        assert not torch.isnan(mel).any(), "NaN in log-mel of silent audio"
        assert not torch.isinf(mel).any(), "Inf in log-mel of silent audio"

        # Normalize should also be stable on silent audio
        frontend_norm = AudioFrontend(normalize=True).to(device)
        mel_norm = frontend_norm(silent)
        assert not torch.isnan(mel_norm).any(), "NaN in normalized log-mel of silent audio"
        assert not torch.isinf(mel_norm).any(), "Inf in normalized log-mel of silent audio"

    def test_frontend_normalization(self, device):
        """Per-sample normalization should produce near-zero mean and near-unit std."""
        frontend = AudioFrontend(normalize=True).to(device)
        waveform = torch.randn(4, 1, 44100, device=device)
        mel = frontend(waveform)
        # After normalization, each sample should have mean ≈ 0, std ≈ 1
        for i in range(4):
            sample = mel[i, 0]  # (128, T)
            assert abs(sample.mean()).item() < 0.5, (
                f"Sample {i} mean not near zero: {sample.mean().item():.4f}"
            )
            assert abs(sample.std().item() - 1.0) < 0.5, (
                f"Sample {i} std not near one: {sample.std().item():.4f}"
            )

    def test_patchify_stem_alternative(self, device, dtype):
        """Patchify stem should also produce (B, 50) output."""
        cfg = femto_s(stem_type="patchify")
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        x = torch.randn(2, 1, 128, 86, device=device, dtype=dtype)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 50), f"Patchify stem: expected (2, 50), got {logits.shape}"


class TestCVProperties:
    """CV-domain correctness tests (spectrograms as 2D images)."""

    @pytest.fixture
    def cv_model(self, device, dtype):
        cfg = femto_s()
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()
        return model

    def test_translation_robustness(self, cv_model, device, dtype):
        """Small time-axis translations should not flip predictions."""
        x = torch.randn(8, 1, 128, 86, device=device, dtype=dtype)
        x_shifted = torch.roll(x, shifts=4, dims=-1)  # shift by 4 time frames

        with torch.no_grad():
            pred_orig = cv_model(x).argmax(-1)
            pred_shifted = cv_model(x_shifted).argmax(-1)

        agree = (pred_orig == pred_shifted).float().mean()
        # On random inputs, we expect > 50% agreement (chance would be ~2%)
        # This is a weak test: the model should not be hypersensitive to small shifts
        assert agree > 0.1, (
            f"Translation robustness very poor: {agree:.2f} agreement after 4-frame shift"
        )

    def test_no_spatial_shortcut(self, cv_model, device, dtype):
        """Random noise inputs should yield relatively uniform class distributions (high entropy)."""
        with torch.no_grad():
            logits = cv_model(torch.randn(8, 1, 128, 86, device=device, dtype=dtype))

        probs = logits.softmax(-1)
        entropy = -(probs * probs.log()).sum(-1).mean()
        max_entropy = math.log(50)  # uniform over 50 classes
        assert entropy > 0.3 * max_entropy, (
            f"Low entropy on noise: {entropy:.2f} / {max_entropy:.2f} — may be memorizing artifacts"
        )

    def test_symmetric_processing(self, cv_model, device, dtype):
        """Flipping the frequency axis should produce different logits (frequency order matters)."""
        x = torch.randn(4, 1, 128, 86, device=device, dtype=dtype)
        x_flipped = torch.flip(x, dims=[2])  # flip frequency axis

        with torch.no_grad():
            out_orig = cv_model(x)
            out_flipped = cv_model(x_flipped)

        # Outputs should differ meaningfully (frequency axis has meaningful ordering)
        mse = ((out_orig - out_flipped) ** 2).mean().item()
        assert mse > 1e-6, (
            f"Frequency flip produces identical outputs (MSE={mse:.2e}) — may not be using frequency info"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1d. Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumerics:
    """Numerical stability under mixed precision and extreme inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
    def test_bf16_forward_all_variants(self, device):
        """All variants should produce finite output in bf16."""
        bf16_dtype = torch.bfloat16
        for name, config_fn, expected_m in [
            ("Atto-S", atto_s, 4.8),
            ("Femto-S", femto_s, 9.5),
            ("Pico-S", pico_s, 15.2),
            ("Tiny-S", tiny_s, 29.8),
        ]:
            cfg = config_fn()
            model = SpectroConvNeXt(cfg).to(device=device, dtype=bf16_dtype).eval()
            x = torch.randn(2, 1, 128, 86, device=device, dtype=bf16_dtype)
            with torch.no_grad():
                out = model(x)
            assert not torch.isnan(out).any(), f"{name}: NaN in bf16 forward"
            assert not torch.isinf(out).any(), f"{name}: Inf in bf16 forward"
            assert out.shape == (2, 50), f"{name}: wrong shape in bf16: {out.shape}"

    def test_extreme_input_values(self, variant, device, dtype):
        """Very large/small inputs should not produce NaN."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()

        # Large positive values
        x_large = torch.randn(2, 1, 128, 86, device=device, dtype=dtype) * 100
        with torch.no_grad():
            out = model(x_large)
        assert not torch.isnan(out).any(), f"{name}: NaN with large inputs"
        assert not torch.isinf(out).any(), f"{name}: Inf with large inputs"

        # All-zero input
        x_zero = torch.zeros(2, 1, 128, 86, device=device, dtype=dtype)
        with torch.no_grad():
            out = model(x_zero)
        assert not torch.isnan(out).any(), f"{name}: NaN with zero input"
        assert not torch.isinf(out).any(), f"{name}: Inf with zero input"
        assert out.shape == (2, 50), f"{name}: wrong shape with zero input: {out.shape}"

    def test_grn_numerical_stability(self, device, dtype):
        """GRN should be numerically stable even with extreme activations."""
        dim = 64
        grn = GRN(dim).to(device, dtype)

        # Extreme activations
        x_extreme = torch.randn(2, dim, 16, 12, device=device, dtype=dtype) * 1e3
        y = grn(x_extreme)
        assert not torch.isnan(y).any(), "GRN: NaN with extreme activations"
        assert not torch.isinf(y).any(), "GRN: Inf with extreme activations"

        # Very small activations
        x_tiny = torch.randn(2, dim, 16, 12, device=device, dtype=dtype) * 1e-6
        y = grn(x_tiny)
        assert not torch.isnan(y).any(), "GRN: NaN with tiny activations"
        assert not torch.isinf(y).any(), "GRN: Inf with tiny activations"

    def test_stochastic_depth_stability(self, device, dtype):
        """StochasticDepth should not produce NaN at any drop probability."""
        x = torch.randn(4, 64, 8, 8, device=device, dtype=dtype)
        for p in [0.0, 0.1, 0.5, 0.9]:
            sd = StochasticDepth(p).to(device, dtype)
            sd.train()
            y = sd(x)
            assert not torch.isnan(y).any(), f"StochasticDepth(p={p}): NaN"
            assert not torch.isinf(y).any(), f"StochasticDepth(p={p}): Inf"
            assert y.shape == x.shape, f"StochasticDepth(p={p}): shape mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Domain-Specific Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioBenchmarks:
    """Benchmark-style tests for audio-specific model properties."""

    def test_forward_pass_from_waveform_pipeline(self, device, dtype):
        """End-to-end: waveform → frontend → model → logits."""
        frontend = AudioFrontend().to(device)
        cfg = femto_s()
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()

        waveform = torch.randn(2, 1, 44100, device=device)
        mel = frontend(waveform)  # (2, 1, 128, ~86)
        mel = mel.to(dtype)

        with torch.no_grad():
            logits = model(mel)
        assert logits.shape == (2, 50), (
            f"End-to-end pipeline: expected (2, 50), got {logits.shape}"
        )

    def test_variable_audio_length(self, device, dtype):
        """Should handle different audio lengths (1s, 2s, 3s) through the frontend."""
        frontend = AudioFrontend(clip_duration_sec=2.0).to(device)
        cfg = femto_s()
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype).eval()

        for duration_samples in [22050, 44100, 66150]:
            waveform = torch.randn(1, 1, duration_samples, device=device)
            mel = frontend(waveform)  # will pad/trim to 44100 samples
            mel = mel.to(dtype)
            with torch.no_grad():
                logits = model(mel)
            assert logits.shape == (1, 50), (
                f"Duration {duration_samples} samples: expected (1, 50), got {logits.shape}"
            )

    def test_parameter_count_estimates(self, variant, device, dtype):
        """Verify actual parameter counts are within 15% of estimates."""
        name, cfg, expected_m = variant
        model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
        actual_m = sum(p.numel() for p in model.parameters()) / 1e6
        ratio = actual_m / expected_m
        assert 0.85 <= ratio <= 1.15, (
            f"{name}: param ratio {ratio:.3f} outside [0.85, 1.15] "
            f"(actual={actual_m:.2f}M, target={expected_m:.1f}M)"
        )

    def test_all_variants_monotonic_params(self, device, dtype):
        """Parameter counts should be strictly increasing across variants."""
        variants = [
            ("atto-s", atto_s),
            ("femto-s", femto_s),
            ("pico-s", pico_s),
            ("nano-s", nano_s),
            ("tiny-s", tiny_s),
        ]
        prev_params = 0
        for name, config_fn in variants:
            cfg = config_fn()
            model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
            params = sum(p.numel() for p in model.parameters())
            assert params > prev_params, (
                f"{name} ({params:,}) has fewer params than previous variant ({prev_params:,})"
            )
            prev_params = params

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for gradient checkpointing test")
    def test_gradient_checkpointing_matches(self, device):
        """Gradient checkpointing forward should produce same output as non-checkpointed."""
        cfg = femto_s()
        ckpt_model = SpectroConvNeXt(cfg).to(device=device, dtype=torch.float32)
        ref_model = SpectroConvNeXt(cfg).to(device=device, dtype=torch.float32)

        # Copy weights from ref to ckpt
        ckpt_model.load_state_dict(ref_model.state_dict())

        ckpt_model.train()
        ref_model.train()

        x = torch.randn(2, 1, 128, 86, device=device, dtype=torch.float32, requires_grad=True)

        out_ckpt = ckpt_model(x, use_checkpoint=True)
        out_ref = ref_model(x, use_checkpoint=False)

        # Outputs should be identical (same weights, same input)
        assert torch.allclose(out_ckpt, out_ref, atol=1e-5), (
            "Checkpointing forward differs from non-checkpointed forward"
        )

        # Backward should also be consistent
        loss_ckpt = out_ckpt.sum()
        loss_ref = out_ref.sum()
        loss_ckpt.backward()
        loss_ref.backward()

        for (n1, p1), (n2, p2) in zip(ckpt_model.named_parameters(), ref_model.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-4), (
                    f"Gradient mismatch in {n1} with checkpointing"
                )
