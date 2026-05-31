# SpectroConvNeXt — Research Quality Rubric

## Overview

This rubric evaluates the SpectroConvNeXt research artifact (ConvNeXt V2-derived CNN
family for audio spectrogram classification on ESC-50, 5M–30M parameter range).

**Task level**: level_1 (idea provided)
**Domain**: CV (primary) + Speech/Audio (primary)
**Primary research question**: Does a ConvNeXt V2-derived CNN family with
spectrogram-adapted rectangular kernels and delayed frequency downsampling achieve
SOTA on ESC-50 across the 5M–30M parameter range?

---

## Scoring Dimensions (0–5)

### 1. Novelty (score: 4)

**Question**: Does this work fill a genuine gap in the literature?

**Evidence**:
- The research synthesis identifies that all top ESC-50 methods (<5M params, bespoke
  architectures) lack systematic modern CNN family scaling.
- ConvNeXt V2 has been evaluated on ImageNet only, not audio spectrograms.
- Rectangular kernels have been shown beneficial at ~2M scale (Chang et al. 2026)
  but scaling behavior at 5M–30M is unknown.
- The five-variant scaling study (Atto-S through Tiny-S) directly addresses the gap.

**Concerns**:
- No papers provided by user to verify literature coverage completeness
  (task_level = level_1). Novelty claim is plausible based on available evidence but
  should be validated against full literature review.
- GRN + FCMAE are co-designed; the supervised-only setting may reduce GRN's
  marginal benefit. This is acknowledged but untested.

**CV domain questions**:
- [x] Does it test invariance/equivariance claims? → `test_translation_robustness`
  and `test_symmetric_processing` in `test_model.py`
- [x] Does it test resolution behavior? → `test_variable_frequency_bins` and
  `test_variable_time_frames`
- [x] Does it test feature quality? → `test_no_spatial_shortcut` (entropy on noise)
- [x] Does it compare against a simple baseline? → Ablations include patchify stem
  (equivalent to standard ConvNeXt V2 stem) and square kernel baseline

**Audio domain questions**:
- [x] Does it test frontend stability? → `test_log_mel_stability`
- [x] Does it test normalization? → `test_frontend_normalization`
- [x] Does it test end-to-end pipeline? → `test_forward_pass_from_waveform_pipeline`
- [x] Does it test short/long utterances? → `test_variable_audio_length`

### 2. Experimental Comprehensiveness (score: 3)

**Question**: How thorough is the validation plan?

**Evidence**:
- All 5 model variants are tested for shape, gradients, and numerical stability
- 6 ablations defined (rect/ square kernel, GRN on/off, stem type, SpecAugment,
  DropPath rate, Stage 4 kernel size) — each is a single config field change
- Ablation runner supports both param-check and training-aware modes
- Profiling script covers forward and train modes, all variants, batch sizes
- Parameter counts verified within 15% of estimates for all variants
- Gradient checkpointing tested for correctness

**Concerns**:
- No training pipeline or trained model weights are provided; ablations cannot be
  executed for metric deltas — only parameter count checks are possible without a
  training script
- ESC-50 5-fold cross-validation is prescribed but not implemented in this validator
  (it belongs in a training pipeline)
- Statistical significance testing (McNemar's test) is prescribed but not implemented
- No baseline reproduction code (ITFA-DNN, ResNet-50, MobileNetV2 + SPA) — these
  would need to be implemented separately
- The FCMAE pretraining direction (claim C5) is explicitly out of scope

**Required vs implemented experiments**:
| Experiment | Required by | Status |
|---|---|---|
| 5-fold CV on all 5 variants | Research contract | `TODO: unverified` (needs training pipeline) |
| Baseline: ConvNeXt V2 Femto (stock) | Research contract | `TODO: unverified` (needs training pipeline) |
| Baseline: ConvNeXt V2 Femto (square kernels) | Research contract | Ablation A covers this config |
| Baseline: ITFA-DNN | Research contract | `TODO: unverified` (external code) |
| Baseline: MobileNetV2 + SPA | Research contract | `TODO: unverified` (external code) |
| Baseline: ResNet-50 | Research contract | `TODO: unverified` (external code) |
| Ablation A: rect vs square | Architect | Implemented in `run_ablations.py` |
| Ablation B: GRN on/off | Architect | Implemented in `run_ablations.py` |
| Ablation C: stem type | Architect | Implemented in `run_ablations.py` |
| Ablation D: SpecAugment | Architect | Implemented in `run_ablations.py` |
| Ablation E: DropPath rate | Architect | Implemented in `run_ablations.py` |
| Ablation F: Stage 4 kernel | Architect | Implemented in `run_ablations.py` |
| Throughput profiling | Research contract | Implemented in `profile_model.py` |
| McNemar's test | Research contract | `TODO: unverified` |

### 3. Theoretical Foundation (score: 3)

**Question**: Are the design choices justified by theory?

**Evidence**:
- Inductive bias justifications table in architecture doc covers 10 decisions,
  each labeled as "grounded" or "hypothesis"
- Rectangular kernel claim is supported by spectrogram aspect ratio argument
  (1.49:1) and literature (Chang et al. 2026)
- Frequency-preserving stem justified by higher frequency-axis information density
- GRN transfer from ImageNet to spectrograms is acknowledged as unverified
  (appropriate intellectual honesty)
- ConvNeXt V2 block design rationale (depthwise conv before MLP, pre-norm, inverted
  bottleneck) is grounded in established literature

**Concerns**:
- FLOP estimates (2× params for forward, 6× params for train) are heuristic —
  actual FLOPs may differ substantially due to depthwise convolutions having
  lower arithmetic intensity
- No formal analysis of receptive field growth through the 4-stage pyramid
- No analysis of how the spectrogram stem's asymmetric stride affects the
  effective receptive field aspect ratio
- The claim "frequency axis carries more discriminative information" is stated
  but not formally justified or cited

### 4. Result Analysis (score: 2)

**Question**: Are results presented, analyzed, and contextualized?

**Evidence**:
- Smoke test provides parameter count verification for all 5 variants
- Profiling produces throughput, memory, and FLOP estimates
- Parameter counts are monotonic across variants (verified in test)
- All variants converge to (B, 50) output shape regardless of input resolution

**Concerns**:
- **No accuracy results**: The central hypothesis (≥93% at 10M, ≥94% at 30M)
  cannot be verified without a training pipeline. This is the single largest gap.
- No learning curves, no validation accuracy, no confusion matrices
- No scaling law analysis (parameter count vs accuracy)
- No overfitting analysis (train/val gap monitoring)
- No comparison to baselines
- No McNemar's test results
- The profiling FLOP estimates are rough heuristics (Kaplan scaling law),
  not measured FLOPs from a profiler

### 5. Implementation Reproducibility (score: 4)

**Question**: Can someone reproduce the results?

**Evidence**:
- Complete PyTorch implementation with 6 source files (model, backbone, head,
  frontend, layers, smoke test)
- ModelConfig dataclass with all hyperparameters; no magic numbers
- Five variants specified precisely (stem_C, C1–C4, N1–N4, kernel sizes,
  DropPath rates)
- Convenience constructors for each variant
- AudioFrontend for reproducible spectrogram extraction
- Preprocessing and augmentation configs fully specified
- Training config (AdamW, cosine LR, label smoothing, etc.) fully specified
- Ablation configs are single-field changes documented in architecture doc
- Gradient checkpointing implemented and tested
- Device-portable (CUDA, MPS, CPU)

**Concerns**:
- No setup.py / requirements.txt / pyproject.toml (though dependencies are
  standard: torch, torchaudio)
- No training script (`train.py`) — the most critical piece for reproducibility
- No data loading script for ESC-50
- No checkpoint saving/loading utilities
- Random seed is not fixed in the code (important for reproducibility of
  stochastic depth, dropout)

### 6. Writing Readiness (score: 3)

**Question**: Is the documentation clear and complete enough for a paper?

**Evidence**:
- Research synthesis document is comprehensive: landscape summary, related work,
  novelty gaps, hypothesis statement, falsification conditions
- Architecture blueprint is detailed with pseudocode, ASCII diagram, inductive
  bias justifications, and traceability table
- Code has complete docstrings, type annotations, and inline shape comments
- Smoke test output is well-formatted with clear pass/fail markers
- README provides quick-start example

**Concerns**:
- No paper draft or results section (no results to report)
- No figures (architecture diagram lives in ASCII but no publication-quality
  figure)
- No ablation tables with actual numbers (no training results)
- ESC-50 dataset description is minimal
- No comparison with SOTA in tabular form (no results to compare)

---

## Score Summary

| Dimension | Score | Key Gap |
|---|---|---|
| Novelty | 4 | Literature coverage could not be fully verified (level_1) |
| Experimental comprehensiveness | 3 | No training pipeline; ablations are config-only |
| Theoretical foundation | 3 | FLOP heuristics; no formal RF analysis |
| Result analysis | 2 | **No accuracy results** — central hypothesis unverifiable |
| Implementation reproducibility | 4 | No training script, no data loading |
| Writing readiness | 3 | No paper draft, no results, no figures |
| **Overall** | **3.2** | **Training pipeline is the blocking gap** |

---

## Blocking Gaps

1. **No training pipeline** — Without `train.py`, no accuracy results are possible.
   The central hypothesis (≥93% at 10M) cannot be evaluated.
2. **No baseline implementations** — ITFA-DNN, ResNet-50, MobileNetV2 + SPA are
   required by the research contract but not provided.
3. **No ESC-50 data loader** — The data preprocessing pipeline is specified but
   not implemented as a DataLoader.
4. **No accuracy or loss metrics** — Results analysis score is 2/5 due to absence
   of any training results.
5. **No statistical significance testing** — McNemar's test prescribed but not
   implemented.

## Recommended Next Experiments

1. **Implement `train.py`** — Training loop with 5-fold CV, ESC-50 data loading,
   SpecAugment, Mixup, logging, and checkpointing.
2. **Train all 5 variants** — Report accuracy ± std over 5 folds for each.
3. **Run ablation suite** — Execute all 6 ablations with training, report metric
   deltas for each.
4. **Baseline reproduction** — Implement ITFA-DNN, ResNet-50, MobileNetV2 + SPA
   with identical training pipeline.
5. **Profile on A100** — Measure actual throughput, memory, and FLOPs for all
   5 variants under `torch.compile`.
6. **Overfitting analysis** — Train/val gap monitoring for Tiny-S (30M) to
   determine if larger models overfit on 2000-clip ESC-50.
7. **Learning curve plots** — Accuracy vs epoch for all variants to show
   convergence behavior.
8. **Confusion matrix** — Per-class accuracy analysis for the best variant.
