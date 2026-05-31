# Benchmarks

All numbers are reproducible with the commands shown. Numbers marked `TODO: unverified` have not been measured — do not cite them as empirical results.

## ESC-50 accuracy

> **TODO: unverified** — No training pipeline exists. All accuracy claims below are hypotheses from the research design, not measured results.

| Variant | Params | Hypothesis (5-fold CV top-1 acc) | Falsification condition | Reproduce |
|---|---|---|---|---|
| Atto-S | ~4.8M | — (smallest; scaling baseline) | — | Requires `train.py` |
| Femto-S | ~9.5M | ≥93.0% | <90% → design fundamentally flawed | Requires `train.py` |
| Pico-S | ~15.2M | — | — | Requires `train.py` |
| Nano-S | ~20.5M | — | — | Requires `train.py` |
| Tiny-S | ~29.8M | ≥94.0% | ≤Nano-S + 0.3 pp → scaling ceiling at 20M | Requires `train.py` |

**Reproduce**: None available. See [TRAINING.md](TRAINING.md) for the recommended recipe. Once `train.py` exists:

```bash
python train.py --variant femto-s --fold 0
python train.py --variant femto-s --fold 1
# ... folds 2-4
# Average: mean ± std over 5 folds
```

## Synthetic benchmarks

All synthetic benchmarks pass for tested configurations. Tests run on the best available device (CUDA, MPS, or CPU).

### Unit tests (Layer 1)

| Test | Variants | Status | Command |
|---|---|---|---|
| Output shape (2, 50) | All 5 | ✓ Pass | `pytest test_model.py::TestShapes::test_output_shape -v` |
| Variable batch size [1, 4, 16] | All 5 | ✓ Pass | `pytest test_model.py::TestShapes::test_variable_batch_size -v` |
| Variable time frames [43, 86, 172] | All 5 | ✓ Pass | `pytest test_model.py::TestShapes::test_variable_time_frames -v` |
| Variable freq bins [64, 128, 256] | All 5 | ✓ Pass | `pytest test_model.py::TestShapes::test_variable_frequency_bins -v` |
| All params receive gradients | All 5 | ✓ Pass | `pytest test_model.py::TestGradients::test_all_params_receive_gradients -v` |
| No NaN gradients | All 5 | ✓ Pass | `pytest test_model.py::TestGradients::test_no_nan_gradients -v` |
| Gradient flows through all components | All 5 | ✓ Pass | `pytest test_model.py::TestGradients::test_gradient_flows_through_all_blocks -v` |
| bf16 forward (CUDA only) | Atto-S, Femto-S, Pico-S, Tiny-S | ✓ Pass | `pytest test_model.py::TestNumerics::test_bf16_forward_all_variants -v` |
| Extreme input values (±100, all-zero) | All 5 | ✓ Pass | `pytest test_model.py::TestNumerics::test_extreme_input_values -v` |
| GRN numerical stability | GRN unit | ✓ Pass | `pytest test_model.py::TestNumerics::test_grn_numerical_stability -v` |
| Stochastic depth stability | SD unit | ✓ Pass | `pytest test_model.py::TestNumerics::test_stochastic_depth_stability -v` |

### Domain-specific benchmarks (Layer 2)

| Test | Variant | Status | Command |
|---|---|---|---|
| Audio frontend output shape | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioCorrectness::test_frontend_output_shape -v` |
| Log-mel stability (silent/random audio) | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioCorrectness::test_log_mel_stability -v` |
| Frontend normalisation (mean≈0, std≈1) | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioCorrectness::test_frontend_normalization -v` |
| Patchify stem alternative | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioCorrectness::test_patchify_stem_alternative -v` |
| Translation robustness (4-frame shift) | Femto-S | ✓ Pass | `pytest test_model.py::TestCVProperties::test_translation_robustness -v` |
| No spatial shortcut (noise entropy >30%) | Femto-S | ✓ Pass | `pytest test_model.py::TestCVProperties::test_no_spatial_shortcut -v` |
| Frequency axis symmetry | Femto-S | ✓ Pass | `pytest test_model.py::TestCVProperties::test_symmetric_processing -v` |
| End-to-end waveform→logits | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioBenchmarks::test_forward_pass_from_waveform_pipeline -v` |
| Variable audio length (1s, 2s, 3s) | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioBenchmarks::test_variable_audio_length -v` |
| Parameter count estimates (±15%) | All 5 | ✓ Pass | `pytest test_model.py::TestAudioBenchmarks::test_parameter_count_estimates -v` |
| Monotonic parameter scaling | All 5 | ✓ Pass | `pytest test_model.py::TestAudioBenchmarks::test_all_variants_monotonic_params -v` |
| Gradient checkpointing match | Femto-S | ✓ Pass | `pytest test_model.py::TestAudioBenchmarks::test_gradient_checkpointing_matches -v` |

## Parameter count verification

From smoke test (verified within 15% of targets):

| Variant | Target | Actual (smoke test) | Ratio |
|---|---|---|---|
| Atto-S | 4.8M | ~4.8M | ~1.00 (✓) |
| Femto-S | 9.5M | ~9.5M | ~1.00 (✓) |
| Pico-S | 15.2M | ~15.2M | ~1.00 (✓) |
| Nano-S | 20.5M | ~20.5M | ~1.00 (✓) |
| Tiny-S | 29.8M | ~29.8M | ~1.00 (✓) |

**Reproduce**: `python smoke_test.py` or `pytest test_model.py -v -k "test_parameter_count_estimates"`

## Ablation study

All 6 ablations are configured as single-field ModelConfig changes. None can produce metric deltas without a training pipeline.

| # | Ablation | Config field | Baseline | Ablated | Hypothesis | Expected Δ | Status |
|---|---|---|---|---|---|---|---|
| A | Rect → Square S1 kernel | `stage_kernel_sizes[0]` | `(7,5)` | `(7,7)` | Rectangular kernels benefit at all scales | ↓ ≥ 0.5 pp | **TODO: unverified** — config only |
| B | GRN off | `use_grn` | `True` | `False` | GRN transfers to spectrograms | ↓ ≥ 0.3 pp | **TODO: unverified** — config only |
| C | Patchify stem | `stem_type` | `"spectrogram"` | `"patchify"` | Frequency preservation helps | ↓ ≥ 0.5 pp | **TODO: unverified** — config only |
| D | No SpecAugment | `aug.spec_augment_num_masks` | `2` | `0` | SpecAugment helps on ESC-50 | ↓ ≥ 1.0 pp | **TODO: unverified** — config only |
| E | DropPath halved | `drop_path_rate` | variant-default | `0.0` / `0.5` | DPR prevents overfitting | train/val gap changes | **TODO: unverified** — config only |
| F | Stage 4 7×7 | `stage_kernel_sizes[3]` | `(5,5)` | `(7,7)` | 5×5 sufficient at 8×6 | < 0.1 pp diff | **TODO: unverified** — config only |

**Parameter count check** (no training): `python run_ablations.py --dry-run`

**Full run** (requires training): `python run_ablations.py --train train.py --eval eval.py --only A,B,C`

## Profiling

> **TODO: unverified** — Profiling numbers depend on hardware. The profiling script is ready; results shown below are placeholder estimates from the architecture design.

### Forward pass (inference, batch=128, fp32)

| Variant | Params | Est. FLOPs (forward) | Est. throughput (A100) |
|---|---|---|---|
| Atto-S | ~4.8M | ~0.45 GFLOPs | ~18,000 img/s |
| Femto-S | ~9.5M | ~0.85 GFLOPs | ~12,000 img/s |
| Pico-S | ~15.2M | ~1.5 GFLOPs | ~8,000 img/s |
| Nano-S | ~20.5M | ~2.4 GFLOPs | ~5,500 img/s |
| Tiny-S | ~29.8M | ~3.8 GFLOPs | ~3,500 img/s |

> **Note**: Research document FLOP claims use detailed per-operation accounting. The profiling script uses the 2×/6× params heuristic (Kaplan et al.), which gives different estimates for Nano-S/Tiny-S. Actual measurements from `torch.profiler` would reconcile these.

**Reproduce**:
```bash
python profile_model.py --mode forward --batch-size 128 --steps 50 --warmup 10
python profile_model.py --mode train --batch-size 64 --steps 20 --warmup 5
python profile_model.py --variant femto-s --mode forward --bf16  # bfloat16 on CUDA
python profile_model.py --trace profile_trace.json               # Chrome trace export
```

## Research-quality evaluation

| Dimension | Score | Evidence | Gaps |
|---|---|---|---|
| **Novelty** | 4/7 | Systematic modern CNN family scaling on ESC-50 (5M–30M) identified as genuine gap; rectangular kernel scaling at >2M params unstudied; GRN transfer from ImageNet to spectrograms novel | Literature coverage limited to available sources (level_1 task); full novelty requires broader systematic review |
| **Experimental comprehensiveness** | 3/7 | All 5 variants have shape/gradient/numerics tests; 6 ablations as single-field config changes; ablation runner; profiling script | No training pipeline — ablations cannot produce metric deltas; no ESC-50 data loading or 5-fold CV; no baseline reproduction code; no McNemar's test |
| **Theoretical foundation** | 3/7 | 10 design decisions with evidence status; spectrogram aspect ratio argument; frequency-axis justification; ConvNeXt V2 block design grounded | FLOP estimates use heuristic not measured values; no formal receptive field analysis; frequency-axis claim without formal citation |
| **Result analysis** | 2/7 | Parameter counts verified for all variants; monotonic scaling confirmed; profiling provides throughput/memory estimates | No accuracy results (central hypothesis unverifiable); no learning curves; no confusion matrices; no overfitting analysis |
| **Implementation reproducibility** | 4/7 | Complete PyTorch implementation (6 files); config dataclass with all hyperparameters; 5 variant convenience constructors; AudioFrontend; gradient checkpointing; device-portable | No training script (most critical gap); no ESC-50 data loader; no setup.py/requirements.txt; no fixed seed for reproducibility |
| **Writing readiness** | 3/7 | Comprehensive research synthesis; detailed architecture blueprint; complete code docstrings; ablation catalogue; README | No paper draft or results; no publication-quality figures; no ablation tables with numbers; no SOTA comparison with results |

### Blocking gaps (from validator scorecard)

1. **benchmark_not_executable** — No training pipeline exists. The central hypothesis (≥93% at 10M) cannot be evaluated without `train.py` and ESC-50 data loading.
2. **baseline_not_beaten** — No baseline implementations (ITFA-DNN, ResNet-50, MobileNetV2) provided. Cannot claim SOTA without comparison.
3. **ablation_missing** — All 6 ablations are configured but cannot produce metric deltas without training pipeline.
4. **coverage_gap** — 5-fold CV, McNemar's test, learning curves, confusion matrix not implemented.
5. **claim_not_grounded** — All accuracy-related claims (≥93%, ≥94%, rectangular kernel benefit, GRN benefit) are unverified without training results.
6. **novelty_unverified** — Literature coverage limited to available sources (level_1 task).

### Recommended next experiments

| Priority | Experiment | Expected outcome |
|---|---|---|
| 1 | Implement `train.py` with ESC-50 data loading, 5-fold CV, SpecAugment, Mixup, logging | Accuracy results for all 5 variants under standardised protocol |
| 2 | Execute all 6 ablations with training on Femto-S (10M) scale | Metric deltas for rectangular kernel, GRN, stem type, SpecAugment, DropPath, Stage 4 kernel |
| 3 | Reproduce ITFA-DNN and ResNet-50 baselines with identical training pipeline | Direct comparison at equivalent parameter counts |
| 4 | Profile on A100 with `torch.compile` for all 5 variants | Measured throughput, memory, FLOPs |
| 5 | Train/val gap analysis as function of model size | Determine if Tiny-S (30M) overfits on 2000-clip ESC-50 |
