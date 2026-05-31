"""
SpectroConvNeXt — Profiling Script (Layer 4).

Runs torch.profiler on each model variant and reports:
    - Top memory/time consumers
    - Estimated FLOPs (forward: ~2× params, train: ~6× params, Kaplan et al.)
    - Throughput (samples/sec)
    - Memory usage (GPU if available)

Usage:
    # Profile all variants (forward pass, CPU)
    python profile_model.py

    # Profile all variants (forward + backward, CUDA recommended)
    python profile_model.py --mode train

    # Profile a single variant
    python profile_model.py --variant pico-s --mode forward

    # More profiling steps for better averages
    python profile_model.py --steps 50 --warmup 10

    # Export chrome trace for visualization
    python profile_model.py --trace profile_trace.json
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional

import torch

# Ensure coder module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coder"))

from model import (
    SpectroConvNeXt,
    atto_s, femto_s, pico_s, nano_s, tiny_s,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sample_input(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a synthetic spectrogram input."""
    return torch.randn(batch_size, 1, 128, 86, device=device, dtype=dtype)


def count_params(model: torch.nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling
# ═══════════════════════════════════════════════════════════════════════════════

def profile_model(
    variant_name: str,
    config_fn,
    mode: str = "forward",
    steps: int = 20,
    warmup: int = 5,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    trace_path: Optional[str] = None,
) -> dict:
    """
    Profile a single model variant.

    Args:
        variant_name: Human-readable name (e.g. "Femto-S")
        config_fn:    Config constructor (e.g. femto_s)
        mode:         "forward" (inference) or "train" (forward + backward)
        steps:        Number of profiling steps
        warmup:       Number of warmup steps (not profiled)
        batch_size:   Batch size for input
        device:       torch device
        dtype:        torch dtype
        trace_path:   Optional path to save chrome trace

    Returns:
        dict with profiling results
    """
    if device is None:
        device = get_device()

    print(f"\n{'─' * 70}")
    print(f"  Profiling: {variant_name}")
    print(f"  Mode: {mode} | Device: {device} | Batch: {batch_size} | Steps: {steps}")
    print(f"{'─' * 70}")

    # ── Build model ───────────────────────────────────────────────────────
    cfg = config_fn()
    model = SpectroConvNeXt(cfg).to(device=device, dtype=dtype)
    model.train() if mode == "train" else model.eval()

    params = count_params(model)
    params_m = params / 1e6
    print(f"  Parameters: {params:,} ({params_m:.2f}M)")

    # ── Input ─────────────────────────────────────────────────────────────
    x = build_sample_input(batch_size, device, dtype)

    # ── Warmup ─────────────────────────────────────────────────────────────
    print(f"  Warmup: {warmup} steps...")
    for _ in range(warmup):
        with torch.no_grad() if mode == "forward" else torch.enable_grad():
            out = model(x)
            if mode == "train":
                loss = out.sum()
                loss.backward()
                model.zero_grad()

    # ── Profiled run ───────────────────────────────────────────────────────
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    if mode == "forward":
        sort_key = "cuda_memory_usage" if device.type == "cuda" else "self_cpu_time_total"
    else:
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    print(f"  Profiling: {steps} steps...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(steps):
            with record_function(f"step_{step}"):
                with torch.no_grad() if mode == "forward" else torch.enable_grad():
                    out = model(x)
                    if mode == "train":
                        loss = out.sum()
                        loss.backward()
                        model.zero_grad()

    # ── Results ────────────────────────────────────────────────────────────
    print(f"\n  Top operations by {sort_key.replace('_', ' ')}:")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    # ── Extract key metrics ────────────────────────────────────────────────
    # Events aggregated by operation type
    events = prof.key_averages()

    # Find total CUDA time
    total_cuda_time_ms = 0
    total_cpu_time_ms = 0
    for evt in events:
        if hasattr(evt, 'cuda_time') and evt.cuda_time is not None:
            total_cuda_time_ms += evt.cuda_time
        if hasattr(evt, 'cpu_time') and evt.cpu_time is not None:
            total_cpu_time_ms += evt.cpu_time

    total_cuda_time_ms /= 1000  # μs → ms
    total_cpu_time_ms /= 1000

    # Average per step
    avg_cuda_ms = total_cuda_time_ms / steps if device.type == "cuda" else 0
    avg_cpu_ms = total_cpu_time_ms / steps

    # ── Estimate FLOPs ─────────────────────────────────────────────────────
    # Kaplan et al.: forward ≈ 2 × params, train ≈ 6 × params
    mult = 6 if mode == "train" else 2
    est_flops = mult * params
    est_gflops = est_flops / 1e9

    # ── Throughput ─────────────────────────────────────────────────────────
    if device.type == "cuda" and avg_cuda_ms > 0:
        samples_per_sec = (batch_size * steps) / (total_cuda_time_ms / 1000)
    else:
        samples_per_sec = (batch_size * steps) / (total_cpu_time_ms / 1000)

    print(f"\n  Profile Summary:")
    print(f"  ┌────────────────────────────────┬──────────────┐")
    print(f"  │ Metric                          │ Value        │")
    print(f"  ├────────────────────────────────┼──────────────┤")
    print(f"  │ Parameters                      │ {params_m:>8.2f}M     │")
    if avg_cuda_ms > 0:
        print(f"  │ Avg CUDA time per step         │ {avg_cuda_ms:>8.2f} ms │")
    print(f"  │ Avg CPU time per step          │ {avg_cpu_ms:>8.2f} ms │")
    print(f"  │ Est. {mode} FLOPs                  │ {est_gflops:>8.2f} G  │")
    print(f"  │ Throughput                      │ {samples_per_sec:>8.0f} img/s│")
    print(f"  └────────────────────────────────┴──────────────┘")

    # ── Memory ─────────────────────────────────────────────────────────────
    if device.type == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1e6
        memory_reserved = torch.cuda.max_memory_reserved(device) / 1e6
        print(f"  GPU Memory: {memory_allocated:.1f}MB allocated, "
              f"{memory_reserved:.1f}MB reserved")
        torch.cuda.reset_peak_memory_stats(device)

    # ── Save trace ─────────────────────────────────────────────────────────
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"  Trace saved to: {trace_path}")

    result = {
        "variant": variant_name,
        "mode": mode,
        "params_m": params_m,
        "params": params,
        "avg_cuda_ms": avg_cuda_ms,
        "avg_cpu_ms": avg_cpu_ms,
        "est_flops_g": est_gflops,
        "throughput_samples_per_sec": samples_per_sec,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(dtype),
    }
    if device.type == "cuda":
        result["gpu_memory_mb"] = memory_allocated

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpectroConvNeXt — Profiling Script",
    )
    parser.add_argument(
        "--mode", type=str, default="forward", choices=["forward", "train"],
        help="Profiling mode: forward (inference) or train (fwd+bwd)",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"],
        help="Specific variant to profile (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for profiling input",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of profiling steps",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--trace", type=str, default=None,
        help="Path to save chrome trace (e.g. profile_trace.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save profiling results as JSON",
    )
    parser.add_argument(
        "--float32", action="store_true", default=True,
        help="Use float32 (default)",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use bfloat16 (requires CUDA)",
    )

    args = parser.parse_args()

    device = get_device()

    if args.bf16:
        if device.type != "cuda":
            print("Warning: bfloat16 requires CUDA. Falling back to float32.")
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         SpectroConvNeXt — Profiler                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}  |  dtype: {dtype}  |  mode: {args.mode}")
    print(f"  Batch: {args.batch_size}  |  Steps: {args.steps}  |  Warmup: {args.warmup}")

    variants = {
        "atto-s": ("Atto-S", atto_s),
        "femto-s": ("Femto-S", femto_s),
        "pico-s": ("Pico-S", pico_s),
        "nano-s": ("Nano-S", nano_s),
        "tiny-s": ("Tiny-S", tiny_s),
    }

    selected = {k: v for k, v in variants.items()
                if args.variant is None or k == args.variant}

    all_results = {}
    for key, (name, config_fn) in selected.items():
        result = profile_model(
            variant_name=name,
            config_fn=config_fn,
            mode=args.mode,
            steps=args.steps,
            warmup=args.warmup,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            trace_path=args.trace,
        )
        all_results[key] = result

    # ── Comparison table ───────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("  VARIANT COMPARISON")
        print(f"{'=' * 70}")
        print(f"  {'Variant':<12s} {'Params':>10s} {'FLOPs (G)':>12s} "
              f"{'Throughput':>14s} {'GPU Mem':>10s}")
        print(f"  {'─' * 12} {'─' * 10} {'─' * 12} {'─' * 14} {'─' * 10}")
        for key in sorted(all_results.keys()):
            r = all_results[key]
            throughput = f"{r['throughput_samples_per_sec']:.0f}/s"
            gpu_mem = f"{r.get('gpu_memory_mb', 0):.0f}MB" if 'gpu_memory_mb' in r else "N/A"
            print(f"  {r['variant']:<12s} {r['params_m']:>8.2f}M "
                  f"{r['est_flops_g']:>10.2f}G "
                  f"{throughput:>14s} {gpu_mem:>10s}")

    # ── Save results ───────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
