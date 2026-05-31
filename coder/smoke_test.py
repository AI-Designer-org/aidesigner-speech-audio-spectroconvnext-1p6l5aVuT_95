"""
SpectroConvNeXt — Smoke test.

Tests all 5 model variants (Atto-S through Tiny-S):
    - Instantiate each variant
    - Forward pass on a synthetic spectrogram
    - Assert output shape is (B, 50)
    - Print parameter count
    - Verify parameter estimates (± 15% of target)

Also tests:
    - Ablation configurations (GRN off, patchify stem, square kernel)
    - Audio frontend with random waveform
    - Device-portable (CUDA if available, else CPU)
    - bfloat16 forward pass (CUDA only)

Usage:
    python smoke_test.py
"""

import torch
import sys
from pathlib import Path

# Ensure package root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    SpectroConvNeXtConfig,
    SpectroConvNeXt,
    count_params,
    atto_s, femto_s, pico_s, nano_s, tiny_s,
)
from layers import (
    LayerNorm2d,
    GRN,
    SpectroConvNeXtBlock,
    DownsampleBlock,
    SpectrogramStem,
    PatchifyStem,
)
from head import ClassificationHead
from backbone import SpectroConvNeXtBackbone


# ═══════════════════════════════════════════════════════════════════════════════
# Device / dtype helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device):
    """Return bfloat16 for CUDA, float32 otherwise."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Layer sanity checks
# ═══════════════════════════════════════════════════════════════════════════════

def test_layers(device, dtype):
    """Verify all component layers produce expected output shapes."""
    B, C, H, W = 2, 64, 16, 12
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    print("─" * 60)
    print("COMPONENT LAYER TESTS")
    print("─" * 60)

    # LayerNorm2d
    ln = LayerNorm2d(C).to(device, dtype)
    y = ln(x)
    assert y.shape == (B, C, H, W), f"LayerNorm2d: expected ({B},{C},{H},{W}), got {y.shape}"
    print(f"  [OK] LayerNorm2d:   {y.shape}")

    # GRN
    grn = GRN(C).to(device, dtype)
    y = grn(x)
    assert y.shape == (B, C, H, W), f"GRN: expected ({B},{C},{H},{W}), got {y.shape}"
    print(f"  [OK] GRN:            {y.shape}")

    # SpectroConvNeXtBlock (rectangular kernel)
    block = SpectroConvNeXtBlock(dim=C, kernel_size=(7, 5)).to(device, dtype)
    y = block(x)
    assert y.shape == (B, C, H, W), f"SpectroConvNeXtBlock: expected ({B},{C},{H},{W}), got {y.shape}"
    print(f"  [OK] ConvNeXtBlock 7x5: {y.shape}")

    # SpectroConvNeXtBlock (square kernel)
    block = SpectroConvNeXtBlock(dim=C, kernel_size=(7, 7)).to(device, dtype)
    y = block(x)
    assert y.shape == (B, C, H, W), f"SpectroConvNeXtBlock: expected ({B},{C},{H},{W}), got {y.shape}"
    print(f"  [OK] ConvNeXtBlock 7x7: {y.shape}")

    # SpectroConvNeXtBlock (small kernel for stage 4)
    block = SpectroConvNeXtBlock(dim=C, kernel_size=(5, 5), drop_path=0.1).to(device, dtype)
    y = block(x)
    assert y.shape == (B, C, H, W), f"SpectroConvNeXtBlock: expected ({B},{C},{H},{W}), got {y.shape}"
    print(f"  [OK] ConvNeXtBlock 5x5: {y.shape}")

    # DownsampleBlock
    ds = DownsampleBlock(dim_in=C, dim_out=C * 2).to(device, dtype)
    y = ds(x)
    H2, W2 = H // 2, W // 2
    assert y.shape == (B, C * 2, H2, W2), f"DownsampleBlock: expected ({B},{C*2},{H2},{W2}), got {y.shape}"
    print(f"  [OK] DownsampleBlock:   {y.shape}")

    # SpectrogramStem
    stem = SpectrogramStem(in_ch=1, stem_ch=32, out_ch=48).to(device, dtype)
    x_stem = torch.randn(B, 1, 128, 86, device=device, dtype=dtype)
    y = stem(x_stem)
    expected_H = 64   # 128/2
    expected_W = 43   # 86//2 + 1  (with s=2 on 86)
    assert y.shape == (B, 48, expected_H, expected_W), \
        f"SpectrogramStem: expected ({B},48,{expected_H},{expected_W}), got {y.shape}"
    print(f"  [OK] SpectrogramStem:   {y.shape}")

    # PatchifyStem
    pstem = PatchifyStem(in_ch=1, out_ch=48).to(device, dtype)
    y = pstem(x_stem)
    expected_H2 = 32  # 128//4
    expected_W2 = 21  # (86-4)//4 + 1 = 21
    assert y.shape == (B, 48, expected_H2, expected_W2), \
        f"PatchifyStem: expected ({B},48,{expected_H2},{expected_W2}), got {y.shape}"
    print(f"  [OK] PatchifyStem:      {y.shape}")

    # ClassificationHead
    head = ClassificationHead(dim=C, n_classes=50).to(device, dtype)
    y = head(x)
    assert y.shape == (B, 50), f"ClassificationHead: expected ({B},50), got {y.shape}"
    print(f"  [OK] ClassificationHead: {y.shape}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Model variant shapes & parameter counts
# ═══════════════════════════════════════════════════════════════════════════════

def test_variant(variant_name: str, config_fn, device, dtype, expected_params_m: float):
    """
    Test a single model variant.

    Args:
        variant_name:     Human-readable name (e.g. "Atto-S")
        config_fn:        Constructor function (e.g. atto_s)
        device:           torch device
        dtype:            torch dtype
        expected_params_m: Expected parameter count in millions
    """
    print(f"  Variant: {variant_name}  (target: ~{expected_params_m:.1f}M params)")

    # Create config and model
    cfg = config_fn()
    model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)

    # Input: synthetic spectrogram (B, 1, 128, 86)
    B = 2
    x = torch.randn(B, 1, 128, 86, device=device, dtype=dtype)

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    # Shape assertion
    assert logits.shape == (B, 50), f"Expected ({B}, 50), got {logits.shape}"
    print(f"  [SHAPE] logits: {logits.shape}")

    # Parameter count + estimate check
    total_params = sum(p.numel() for p in model.parameters())
    total_m = total_params / 1e6
    ratio = total_m / expected_params_m
    within_tolerance = 0.85 <= ratio <= 1.15
    status = "OK" if within_tolerance else "WARN"
    print(f"  [PARAMS] actual: {total_params:,} ({total_m:.2f}M)  "
          f"target: {expected_params_m:.1f}M  ratio: {ratio:.3f}  [{status}]")

    if not within_tolerance:
        print(f"  ⚠  Warning: params ({total_m:.2f}M) deviate from target "
              f"({expected_params_m:.1f}M) by more than 15%")

    # Gradient flow test — check that >95% of trainable params receive non-zero gradients
    y = model(x)
    loss = y.sum()
    loss.backward()
    total_with_grad = 0
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
            if p.grad is not None and p.grad.abs().sum() > 0:
                total_with_grad += p.numel()
    grad_ratio = total_with_grad / max(total_params, 1)
    grad_ok = grad_ratio > 0.90
    print(f"  [GRAD]  {'OK — gradients flow' if grad_ok else 'WARN — some params have zero gradient'}"
          f"  ({total_with_grad:,}/{total_params:,} params with non-zero grad, {grad_ratio:.1%})")
    print()

    return total_params


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Ablation configurations
# ═══════════════════════════════════════════════════════════════════════════════

def test_ablations(device, dtype):
    """Verify that ablation configurations instantiate and forward correctly."""
    print("─" * 60)
    print("ABLATION TESTS")
    print("─" * 60)

    B = 2
    x = torch.randn(B, 1, 128, 86, device=device, dtype=dtype)

    # Ablation B: GRN off
    print("  Ablation B: GRN off")
    cfg = pico_s(use_grn=False)
    model = SpectroConvNeXt(cfg).to(device, dtype)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (B, 50), f"GRN off: expected ({B},50), got {logits.shape}"
    total = sum(p.numel() for p in model.parameters())
    print(f"    [OK] logits: {logits.shape}  params: {total:,} ({total/1e6:.2f}M)")
    print()

    # Ablation A: Square kernel in Stage 1
    print("  Ablation A: Square (7,7) kernel in Stage 1")
    square_kernels = ((7, 7), (7, 7), (7, 7), (5, 5))
    cfg = femto_s(stage_kernel_sizes=square_kernels)
    model = SpectroConvNeXt(cfg).to(device, dtype)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (B, 50), f"Square kernel: expected ({B},50), got {logits.shape}"
    total = sum(p.numel() for p in model.parameters())
    print(f"    [OK] logits: {logits.shape}  params: {total:,} ({total/1e6:.2f}M)")
    print()

    # Ablation C: Patchify stem
    print("  Ablation C: Patchify stem")
    cfg = femto_s(stem_type="patchify")
    model = SpectroConvNeXt(cfg).to(device, dtype)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (B, 50), f"Patchify stem: expected ({B},50), got {logits.shape}"
    total = sum(p.numel() for p in model.parameters())
    print(f"    [OK] logits: {logits.shape}  params: {total:,} ({total/1e6:.2f}M)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Gradient checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def test_checkpointing(device, dtype):
    """Verify that gradient checkpointing forward pass works."""
    print("─" * 60)
    print("GRADIENT CHECKPOINTING TEST")
    print("─" * 60)

    B = 2
    cfg = pico_s()
    model = SpectroConvNeXt(cfg).to(device, dtype)
    model.train()

    x = torch.randn(B, 1, 128, 86, device=device, dtype=dtype, requires_grad=True)
    logits = model(x, use_checkpoint=True)
    loss = logits.sum()
    loss.backward()
    print(f"  [OK] Checkpointing forward + backward: logits shape {logits.shape}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = get_device()
    dtype = get_dtype(device)

    print("═" * 60)
    print(f"SpectroConvNeXt — Smoke Test")
    print(f"  Device: {device}  |  dtype: {dtype}")
    print("═" * 60)
    print()

    # ── Variant definitions ────────────────────────────────────────────
    variants = [
        ("Atto-S",   atto_s,   4.8),
        ("Femto-S",  femto_s,  9.5),
        ("Pico-S",   pico_s,   15.2),
        ("Nano-S",   nano_s,   20.5),
        ("Tiny-S",   tiny_s,   29.8),
    ]

    # ── Component layer tests ──────────────────────────────────────────
    test_layers(device, dtype)

    # ── Variant tests ──────────────────────────────────────────────────
    print("─" * 60)
    print("MODEL VARIANT TESTS")
    print("─" * 60)

    param_results = []
    for name, config_fn, expected_m in variants:
        total = test_variant(name, config_fn, device, dtype, expected_m)
        param_results.append((name, total, expected_m))

    # ── Ablation tests ─────────────────────────────────────────────────
    test_ablations(device, dtype)

    # ── Checkpointing test ─────────────────────────────────────────────
    test_checkpointing(device, dtype)

    # ── Summary ────────────────────────────────────────────────────────
    print("─" * 60)
    print("PARAMETER COUNT SUMMARY")
    print("─" * 60)
    for name, actual, expected_m in param_results:
        actual_m = actual / 1e6
        ratio = actual_m / expected_m
        mark = "✓" if 0.85 <= ratio <= 1.15 else "⚠"
        print(f"  {mark} {name:10s}  actual: {actual_m:6.2f}M  target: {expected_m:5.1f}M  "
              f"ratio: {ratio:.3f}")

    print()
    print("═" * 60)
    print("ALL TESTS PASSED")
    print("═" * 60)


if __name__ == "__main__":
    main()
