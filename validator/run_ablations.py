"""
SpectroConvNeXt — Ablation Runner (Layer 3).

Runs all 6 named ablations from the architecture document. Each ablation is a
single-field ModelConfig change.  Reports parameter counts, and if a training
function is provided, also reports the evaluation metric.

Usage:
    # Quick check: report parameter count deltas only
    python run_ablations.py

    # Full run: train and evaluate each ablation (requires training script)
    python run_ablations.py --train /path/to/train.py --eval /path/to/eval.py

    # Run a subset of ablations
    python run_ablations.py --only A,B,C
"""

import argparse
import sys
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure coder module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coder"))

from model import (
    SpectroConvNeXtConfig,
    SpectroConvNeXt,
    atto_s, femto_s, pico_s, nano_s, tiny_s,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each ablation is a dict with:
#   name:        Short identifier
#   description: Human-readable description
#   configs:     Dict of {config_label: (variant_fn, overrides_dict)}
#   hypothesis:  What claim is being tested
#   scales:      Which model scales to test at
#   expected_delta: Expected metric change direction

ABLATIONS = {
    "A": {
        "name": "rect_vs_square_s1_kernel",
        "description": "Rectangular (7,5) → Square (7,7) Stage 1 kernel",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated": (femto_s, {"stage_kernel_sizes": ((7, 7), (7, 7), (7, 7), (5, 5))}),
        },
        "hypothesis": "Rectangular kernels benefit at all scales (claim C2)",
        "scales": ["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"],
        "expected_delta": "baseline ≥ ablated + 0.5 pp",
        "architect_field": "stage_kernel_sizes[0]",
    },
    "B": {
        "name": "grn_on_vs_off",
        "description": "GRN on vs off",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated": (femto_s, {"use_grn": False}),
        },
        "hypothesis": "GRN transfers to spectrograms (claim C3)",
        "scales": ["atto-s", "pico-s", "tiny-s"],
        "expected_delta": "baseline ≥ ablated + 0.3 pp",
        "architect_field": "use_grn",
    },
    "C": {
        "name": "spectrogram_vs_patchify_stem",
        "description": "Two-layer spectrogram stem vs 4×4 patchify stem",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated": (femto_s, {"stem_type": "patchify"}),
        },
        "hypothesis": "Frequency preservation helps (design decision D1)",
        "scales": ["femto-s"],
        "expected_delta": "baseline ≥ ablated + 0.5 pp",
        "architect_field": "stem_type",
    },
    "D": {
        "name": "spec_augment",
        "description": "With and without SpecAugment",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated": (femto_s, {"augmentation": None}),  # augmented below via config override
        },
        "hypothesis": "SpecAugment helps on ESC-50",
        "scales": ["femto-s"],
        "expected_delta": "baseline ≥ ablated + 1.0 pp",
        "architect_field": "augmentation.spec_augment_num_masks",
        "note": "Ablation D requires setting augmentation config; handled in _build_config",
    },
    "E": {
        "name": "drop_path_rate",
        "description": "DropPath rate halved/doubled",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated_zero": (femto_s, {"drop_path_rate": 0.0}),
            "ablated_double": (femto_s, {"drop_path_rate": 0.5}),
        },
        "hypothesis": "DropPath prevents overfitting at larger scales",
        "scales": ["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"],
        "expected_delta": "0.0 DPR → overfitting; 0.5 DPR → underfitting",
        "architect_field": "drop_path_rate",
    },
    "F": {
        "name": "stage4_kernel_5x5_vs_7x7",
        "description": "Stage 4 kernel (5,5) vs (7,7)",
        "configs": {
            "baseline": (femto_s, {}),
            "ablated": (femto_s, {"stage_kernel_sizes": ((7, 5), (7, 7), (7, 7), (7, 7))}),
        },
        "hypothesis": "5×5 is sufficient when spatial dims are 8×6",
        "scales": ["femto-s"],
        "expected_delta": "difference < 0.1 pp",
        "architect_field": "stage_kernel_sizes[3]",
    },
}


def _build_config(config_spec: Tuple, scale: str) -> SpectroConvNeXtConfig:
    """
    Build a config from a (variant_fn, overrides) spec.

    For ablation D (SpecAugment), we handle the augmentation sub-config override
    since it's not a top-level field.
    """
    variant_fn, overrides = config_spec

    # Special handling for Ablation D: disable SpecAugment
    if overrides is not None and "augmentation" in overrides and overrides["augmentation"] is None:
        from model import AugmentationConfig
        # Create base config with variant defaults then override
        base_cfg = variant_fn()
        aug = replace(base_cfg.augmentation, spec_augment_num_masks=0)
        overrides_clean = {k: v for k, v in overrides.items() if k != "augmentation"}
        return replace(base_cfg, augmentation=aug, **overrides_clean)

    return variant_fn(**overrides) if overrides else variant_fn()


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter Count Report
# ═══════════════════════════════════════════════════════════════════════════════

def get_param_info(cfg: SpectroConvNeXtConfig) -> Dict:
    """
    Count parameters for a given config.

    Returns dict with param count and human-readable size.
    """
    model = SpectroConvNeXt(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_m": round(total / 1e6, 3),
        "trainable_m": round(trainable / 1e6, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_param_check(
    ablation_id: str,
    ablation_def: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Check parameter counts for all config variants in an ablation.

    Args:
        ablation_id: Ablation identifier (A-F)
        ablation_def: Ablation definition dict
        verbose: Whether to print results

    Returns:
        Dict of {scale: {config_label: param_info_dict}}
    """
    results = {}
    for scale in ablation_def["scales"]:
        scale_results = {}
        for label, config_spec in ablation_def["configs"].items():
            cfg = _build_config(config_spec, scale)
            # Ensure the scale is set
            if cfg.variant != scale:
                cfg = replace(cfg, variant=scale)
                cfg.__post_init__()
            param_info = get_param_info(cfg)
            scale_results[label] = param_info
        results[scale] = scale_results

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Ablation {ablation_id}: {ablation_def['name']}")
        print(f"  {ablation_def['description']}")
        print(f"  Hypothesis: {ablation_def['hypothesis']}")
        print(f"  Expected: {ablation_def['expected_delta']}")
        print(f"{'=' * 70}")
        for scale, scale_results in results.items():
            print(f"\n  Scale: {scale}")
            for label, info in scale_results.items():
                print(f"    {label:20s}  {info['total_m']:8.3f}M params  "
                      f"({info['total']:,})")
        # For ablation E, report relative differences
        if ablation_id == "E":
            for scale, scale_results in results.items():
                baseline = scale_results.get("baseline", {}).get("total_m", 0)
                zero = scale_results.get("ablated_zero", {}).get("total_m", 0)
                double = scale_results.get("ablated_double", {}).get("total_m", 0)
                print(f"    {'-' * 50}")
                print(f"    Same params expected (only drop prob changes): "
                      f"baseline={baseline}M, zero={zero}M, double={double}M")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Training-Aware Ablation Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablations_with_training(
    train_fn: Callable,
    eval_fn: Callable,
    dry_run: bool = False,
    only: Optional[List[str]] = None,
) -> Dict:
    """
    Run ablations with actual training and evaluation.

    Args:
        train_fn: Callable(model, cfg, variant_label) that trains the model
        eval_fn:  Callable(model, cfg, variant_label) → float metric
        dry_run:  If True, skip training (just report configs)
        only:     List of ablation IDs to run (A-F). None = run all.

    Returns:
        Dict of {ablation_id: {scale: {config_label: metric_value}}}
    """
    results = {}

    ablation_ids = list(ABLATIONS.keys())
    if only:
        ablation_ids = [a for a in ablation_ids if a in only]

    for ablation_id in ablation_ids:
        ablation_def = ABLATIONS[ablation_id]
        ablation_results = {}

        for scale in ablation_def["scales"]:
            scale_results = {}
            for label, config_spec in ablation_def["configs"].items():
                cfg = _build_config(config_spec, scale)
                # Ensure the scale is set
                if cfg.variant != scale:
                    cfg = replace(cfg, variant=scale)
                    cfg.__post_init__()

                if not dry_run:
                    print(f"  Training: ablation={ablation_id} scale={scale} label={label}")
                    model = SpectroConvNeXt(cfg)
                    train_fn(model, cfg, f"{ablation_id}_{scale}_{label}")
                    metric = eval_fn(model, cfg, f"{ablation_id}_{scale}_{label}")
                else:
                    metric = 0.0
                    model = SpectroConvNeXt(cfg)
                    param_m = sum(p.numel() for p in model.parameters()) / 1e6
                    print(f"  [DRY RUN] ablation={ablation_id} scale={scale} "
                          f"label={label} params={param_m:.2f}M")

                scale_results[label] = {
                    "metric": metric,
                    "params_m": param_m if dry_run else None,
                }

            ablation_results[scale] = scale_results

        results[ablation_id] = ablation_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def format_ablation_report(results: Dict) -> str:
    """Format ablation results as a markdown table."""
    lines = [
        "# Ablation Results",
        "",
        "| ID | Ablation | Scale | Config | Params (M) | Metric | Δ vs Baseline |",
        "|---|----------|-------|--------|------------|--------|---------------|",
    ]

    for ablation_id in sorted(results.keys()):
        ablation_def = ABLATIONS[ablation_id]
        for scale in sorted(results[ablation_id].keys()):
            scale_data = results[ablation_id][scale]
            baseline_val = None
            for label, data in sorted(scale_data.items()):
                if label == "baseline":
                    baseline_val = data.get("metric", 0)
                    delta = "—"
                else:
                    if baseline_val is not None:
                        delta = f"{data.get('metric', 0) - baseline_val:+.4f}"
                    else:
                        delta = "N/A"

                lines.append(
                    f"| {ablation_id} | {ablation_def['name']} "
                    f"| {scale} | {label} "
                    f"| {data.get('params_m', '?'):.3f} "
                    f"| {data.get('metric', '?'):.4f} "
                    f"| {delta} |"
                )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpectroConvNeXt Ablation Runner",
    )
    parser.add_argument(
        "--train", type=str, default=None,
        help="Path to training script/module (optional — without it, only param counts are reported)",
    )
    parser.add_argument(
        "--eval", type=str, default=None,
        help="Path to evaluation script/module (required if --train is provided)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated list of ablation IDs to run (e.g. A,B,C)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report configs without training",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save ablation report (JSON)",
    )

    args = parser.parse_args()

    only = args.only.split(",") if args.only else None

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         SpectroConvNeXt — Ablation Runner                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.train and args.eval:
        # Import training/eval functions
        sys.path.insert(0, str(Path(args.train).resolve().parent))
        train_module = __import__(Path(args.train).stem)
        eval_module = __import__(Path(args.eval).stem)

        results = run_ablations_with_training(
            train_fn=train_module.train,
            eval_fn=eval_module.evaluate,
            dry_run=args.dry_run,
            only=only,
        )

        report = format_ablation_report(results)
        print("\n" + report)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
    else:
        # Parameter-count-only mode (no training required)
        print("\nNo training script provided. Running parameter-count checks only.\n")
        all_results = {}
        ablation_ids = list(ABLATIONS.keys())
        if only:
            ablation_ids = [a for a in ablation_ids if a in only]

        for ablation_id in ablation_ids:
            all_results[ablation_id] = run_ablation_param_check(
                ablation_id, ABLATIONS[ablation_id], verbose=True
            )

        # Summary
        print(f"\n{'=' * 70}")
        print("  SUMMARY: Parameter Count Consistency")
        print(f"{'=' * 70}")
        all_ok = True
        for ablation_id in ablation_ids:
            ablation_def = ABLATIONS[ablation_id]
            for scale in ablation_def["scales"]:
                scale_results = all_results[ablation_id].get(scale, {})
                for label, info in scale_results.items():
                    if label == "baseline":
                        expected = info["total_m"]
                    else:
                        actual = info["total_m"]
                        # Ablations should have comparable params to baseline
                        # (GRN off is slightly fewer; kernel changes are same)
                        if actual > expected * 1.05 or actual < expected * 0.90:
                            print(f"  ⚠  {ablation_id} {scale} {label}: "
                                  f"{actual:.3f}M vs baseline {expected:.3f}M params "
                                  f"({'more' if actual > expected else 'fewer'} than expected)")
                            all_ok = False
        if all_ok:
            print("  ✓ All ablation parameter counts consistent")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
