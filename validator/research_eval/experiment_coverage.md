# Experiment Coverage: SpectroConvNeXt

## Legend
- **[✓] Implemented**: Test or benchmark exists and is executable
- **[~] Partial**: Configured/specified but requires external component (training pipeline)
- **[✗] Missing**: Not implemented, not configured
- **[—] Out of scope**: Explicitly excluded from this phase

---

## Baseline Requirements (from Research Contract)

| # | Baseline | Status | Location | Notes |
|---|---|---|---|---|
| B1 | ConvNeXt V2 Femto (stock, 4×4 patchify stem) | [~] Config available | `stem_type="patchify"` in `femto_s()` | No training results |
| B2 | ConvNeXt V2 Femto (spectrogram stem, square 7×7 kernels) | [~] Config available | `stage_kernel_sizes` override in ablation A | No training results |
| B3 | ITFA-DNN (Chen & Peng, 2025) | [✗] Missing | — | External reproduction needed |
| B4 | MobileNetV2 + SPA (SoundMLR, 2025) | [✗] Missing | — | External reproduction needed |
| B5 | ResNet-50 (He et al., 2016) | [✗] Missing | — | External reproduction needed |
| B6 | Lightweight CRNN + CoordAtt (2026) | [✗] Missing | — | External reproduction needed |
| B7 | SpectroConvNeXt Femto-S (self-baseline) | [✓] Implemented | `model.py`, `test_model.py` | Architecture and shape verified |

**Coverage**: 1/7 baselines implemented, 2/7 config-available, 4/7 missing.

---

## Evaluation Requirements (from Research Contract)

| Requirement | Status | Location | Notes |
|---|---|---|---|
| ESC-50 dataset, 2000 clips, 50 classes | [✗] Missing | — | No data loader |
| 5-fold cross-validation (stratified) | [✗] Missing | — | Needs training pipeline |
| Top-1 accuracy (mean ± std over 5 folds) | [✗] Missing | — | Needs training pipeline |
| Per-class accuracy, confusion matrix | [✗] Missing | — | Needs training pipeline |
| Parameter count verification | [✓] Implemented | `test_model.py::TestAudioBenchmarks::test_parameter_count_estimates` | All 5 variants within 15% |
| Inference throughput (samples/sec) | [✓] Implemented | `profile_model.py` | All variants, batch=128, forward/train modes |
| McNemar's test (α=0.05) | [✗] Missing | — | Needs multiple trained models |

---

## Ablation Requirements (from Architect)

| Ablation | Config Field | Baseline | Ablated | Status | Location |
|---|---|---|---|---|---|
| A: Rect → Square S1 kernel | `stage_kernel_sizes[0]` | `(7,5)` | `(7,7)` | [~] Config implemented | `run_ablations.py` (Ablation A) |
| B: GRN off | `use_grn` | `True` | `False` | [~] Config implemented | `run_ablations.py` (Ablation B) |
| C: Patchify stem | `stem_type` | `"spectrogram"` | `"patchify"` | [~] Config implemented | `run_ablations.py` (Ablation C) |
| D: No SpecAugment | `aug.spec_augment_num_masks` | `2` | `0` | [~] Config implemented | `run_ablations.py` (Ablation D) |
| E: DropPath halved/doubled | `drop_path_rate` | variant-default | `0.0` / `0.5` | [~] Config implemented | `run_ablations.py` (Ablation E) |
| F: Stage 4 7×7 | `stage_kernel_sizes[3]` | `(5,5)` | `(7,7)` | [~] Config implemented | `run_ablations.py` (Ablation F) |

**Coverage**: 6/6 ablations configured as single-field config changes. All support
`--dry-run` for parameter-count verification. 0/6 can produce metric deltas without
a training pipeline.

---

## Synthetic Benchmarks (Layer 2)

| Benchmark | Status | Location | Notes |
|---|---|---|---|
| Output shape from waveform (end-to-end) | [✓] | `test_model.py::TestAudioBenchmarks::test_forward_pass_from_waveform_pipeline` | waveform → frontend → model → logits |
| Log-mel stability (silent/random audio) | [✓] | `test_model.py::TestAudioCorrectness::test_log_mel_stability` | No NaN/Inf on silent audio |
| Frontend normalization | [✓] | `test_model.py::TestAudioCorrectness::test_frontend_normalization` | Mean≈0, std≈1 after normalization |
| Variable audio length | [✓] | `test_model.py::TestAudioBenchmarks::test_variable_audio_length` | 1s, 2s, 3s clips |
| Translation robustness | [✓] | `test_model.py::TestCVProperties::test_translation_robustness` | 4-frame time shift |
| No spatial shortcut (noise entropy) | [✓] | `test_model.py::TestCVProperties::test_no_spatial_shortcut` | Entropy > 30% of max on noise |
| Frequency axis symmetry | [✓] | `test_model.py::TestCVProperties::test_symmetric_processing` | Frequency flip changes output |
| Gradient checkpointing correctness | [✓] | `test_model.py::TestAudioBenchmarks::test_gradient_checkpointing_matches` | Forward + backward match non-checkpointed |

**Coverage**: 8/8 synthetic benchmarks implemented and executable.

---

## Unit Tests (Layer 1)

| Test Category | Status | Location |
|---|---|---|
| Shape: Output shape (all 5 variants) | [✓] | `test_model.py::TestShapes::test_output_shape` |
| Shape: Variable batch size | [✓] | `test_model.py::TestShapes::test_variable_batch_size` |
| Shape: Variable time frames | [✓] | `test_model.py::TestShapes::test_variable_time_frames` |
| Shape: Variable frequency bins | [✓] | `test_model.py::TestShapes::test_variable_frequency_bins` |
| Gradient: All params receive gradients | [✓] | `test_model.py::TestGradients::test_all_params_receive_gradients` |
| Gradient: No NaN gradients | [✓] | `test_model.py::TestGradients::test_no_nan_gradients` |
| Gradient: Flows through all components | [✓] | `test_model.py::TestGradients::test_gradient_flows_through_all_blocks` |
| Numerics: bf16 forward (all variants) | [✓] | `test_model.py::TestNumerics::test_bf16_forward_all_variants` |
| Numerics: Extreme input values | [✓] | `test_model.py::TestNumerics::test_extreme_input_values` |
| Numerics: GRN numerical stability | [✓] | `test_model.py::TestNumerics::test_grn_numerical_stability` |
| Numerics: Stochastic depth stability | [✓] | `test_model.py::TestNumerics::test_stochastic_depth_stability` |

**Coverage**: 11/11 unit test categories implemented.

---

## Profiling (Layer 4)

| Artifact | Status | Location |
|---|---|---|
| torch.profiler script | [✓] | `profile_model.py` |
| Forward mode (inference) | [✓] | `--mode forward` |
| Train mode (forward + backward) | [✓] | `--mode train` |
| All 5 variants | [✓] | Default: all; `--variant` for single |
| FLOP estimate (2×/6× params heuristic) | [✓] | Kaplan et al. scaling law |
| Throughput (samples/sec) | [✓] | Computed from profiler timing |
| GPU memory tracking | [✓] | `torch.cuda.max_memory_*` |
| Chrome trace export | [✓] | `--trace <path>.json` |
| bfloat16 support | [✓] | `--bf16` flag |

---

## Summary

| Category | Implemented | Partial | Missing | Total |
|---|---|---|---|---|
| Baselines | 1 | 2 | 4 | 7 |
| Evaluation requirements | 2 | 0 | 4 | 6 |
| Ablations | 0 | 6 | 0 | 6 |
| Synthetic benchmarks | 8 | 0 | 0 | 8 |
| Unit tests | 11 | 0 | 0 | 11 |
| Profiling | 9 | 0 | 0 | 9 |
| **Total** | **31** | **8** | **8** | **47** |

**Overall coverage**: 31/47 (66%) implemented, 8/47 (17%) partial, 8/47 (17%) missing.

### Key coverage gaps (blocking):
1. No training pipeline → all accuracy claims unverifiable
2. No baseline implementations → no SOTA comparison possible
3. No ESC-50 data loader → can't run standardized 5-fold CV
4. No McNemar's test → no statistical significance assessment
5. No learning curves or validation metrics → no overfitting analysis

### Can the benchmark suite distinguish from a trivial baseline?
**Yes, partially.** The suite can verify:
- Parameter counts match targets (±15%)
- All variants produce correct output shapes (any input resolution)
- Gradients flow through all parameters
- Numerical stability under bf16 and extreme values
- GRN and stochastic depth numerical behavior
- Translation robustness and noise entropy characteristics
- End-to-end pipeline from waveform to logits

**But cannot** distinguish accuracy on ESC-50 from chance without a training pipeline.
The fundamental "does it work on the actual task?" question is unanswered.
