# Claim Grounding: SpectroConvNeXt

Every research claim is mapped to a source file, test command, ablation result, or
marked as `TODO: unverified`. Claims without grounding are flagged as blockers.

---

## Novelty Claims (from Research Synthesis)

### C1: "No prior work has systematically scaled a modern ConvNeXt V2 family on ESC-50 across the 5M–30M parameter range."

| Aspect | Grounding |
|---|---|
| Status | **Grounded** (literature claim) |
| Evidence | Research synthesis surveys ESC-50 leaderboard: ITFA-DNN (~2M), CRNN+CoordAtt (~1.5M), MobileNetV2+SPA (~3.5M), SSRP-T (~0.5M). None exceed ~3.5M params. ConvNeXt V2 paper evaluates on ImageNet only. |
| File | `research/spectroconvnext_research_synthesis.md` §1.1, §3 (Gap 1) |
| Concern | Literature search limited to available sources (level_1 task). Full verification requires broader systematic review. |

---

### C2: "Rectangular depthwise kernels in early stages provide a consistent accuracy benefit across model scales on spectrograms."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Evidence | Chang et al. (2026) show benefit at ~2M scale. Scaling behavior at 5M–30M unknown. |
| Ablation | Ablation A: swap `stage_kernel_sizes[0]` from `(7,5)` to `(7,7)` at all 5 scales. |
| File (config) | `run_ablations.py` Ablation A |
| File (test) | `test_model.py` does not contain this ablation as a test (it's structural, not metric-based) |
| Command | `python run_ablations.py --only A --dry-run` (param check) |
| Training | `python run_ablations.py --train train.py --eval eval.py --only A` (requires train.py) |
| Expected Δ | Baseline ≥ ablated + 0.5 pp accuracy |
| **Verification status** | **TODO: unverified** — ablation configured but cannot produce metric delta without training |

---

### C3: "GRN (Global Response Normalization) transfers from ImageNet to audio spectrograms, providing a small consistent gain."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Evidence | GRN is co-designed with FCMAE for ImageNet (ConvNeXt V2, Woo et al. 2023). Spectrogram domain transfer is acknowledged as "TODO: unverified" in research synthesis. |
| Ablation | Ablation B: toggle `use_grn` at Atto-S, Pico-S, Tiny-S. |
| File (config) | `run_ablations.py` Ablation B |
| File (code) | `layers.py::GRN` — implementation verified for numerical stability in `test_model.py::TestNumerics::test_grn_numerical_stability` |
| File (shape) | `test_model.py::TestAudioCorrectness` — GRN variants instantiate correctly |
| Training | `python run_ablations.py --train train.py --eval eval.py --only B` |
| Expected Δ | Baseline ≥ ablated + 0.3 pp |
| **Verification status** | **TODO: unverified** — GRN implementation is correct, but accuracy impact on spectrograms is unknown |

---

### C4: "A pure CNN (no attention, no recurrence) can match attention-augmented methods (ITFA-DNN, CRNN) on ESC-50 at equivalent parameter counts."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Evidence | ConvNeXt V2 matches ViT on ImageNet (indirect evidence). Direct CNN-vs-attention comparison on ESC-50 at multiple scales is absent. |
| Baseline requirement | ITFA-DNN reproduction, CRNN+CoordAtt reproduction, ResNet-50 baseline. All marked as `TODO: unverified`. |
| File | Not directly testable in current validator — requires external baseline implementations |
| **Verification status** | **TODO: unverified** — depends on baseline reproduction |

---

### C5: "FCMAE pretraining for audio spectrograms."

| Aspect | Grounding |
|---|---|
| Status | **TODO: unverified** (explicitly out of scope) |
| Note | Deliberately excluded from supervised-only design. Noted in research synthesis as a future direction. |

---

## Performance Claims (from Research Hypothesis)

### H1: "Femto-S (~10M) achieves ≥93.0% on ESC-50 (5-fold CV)."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| File | `research/spectroconvnext_research_synthesis.md` §4 |
| **Verification status** | **TODO: unverified** — no training pipeline, no ESC-50 data loader, no accuracy results |
| Falsification | If <90% → design is fundamentally flawed (per research contract) |

### H2: "Tiny-S (~30M) achieves ≥94.0% on ESC-50 (5-fold CV)."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| **Verification status** | **TODO: unverified** — same as H1 |
| Falsification | If fails to outperform Nano-S by ≥0.5 pp → scaling ceiling hit |

### H3: "Rectangular Stage 1 kernels (7×5) outperform square (7×7) by ≥0.5 pp at all scales."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Ablation | Ablation A |
| **Verification status** | **TODO: unverified** — no training results |

### H4: "GRN provides ≥0.3 pp gain vs. identical architecture without GRN."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Ablation | Ablation B |
| **Verification status** | **TODO: unverified** — no training results |

---

## Architecture Claims

### D1: "Two-layer frequency-preserving stem outperforms 4×4 patchify stem."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Ablation | Ablation C |
| File (code) | `layers.py::SpectrogramStem` and `layers.py::PatchifyStem` |
| File (shape) | `test_model.py::TestAudioCorrectness::test_patchify_stem_alternative` — both stems produce correct output |
| **Verification status** | **TODO: unverified** — stems work correctly, but accuracy comparison needs training |

### D2: "Rectangular (7,5) kernel in Stage 1 matches spectrogram 1.49:1 aspect ratio."

| Aspect | Grounding |
|---|---|
| Status | **Grounded** (geometric fact) |
| File | `architect/spectroconvnext_architecture.md` §8 (D2) |
| Verification | Aspect ratio of 128×86 input = 128/86 ≈ 1.49. Rectangular kernel (7,5) has aspect ratio 7/5 = 1.4. These are comparable. |

### D3: "GRN prevents feature collapse without extra parameters."

| Aspect | Grounding |
|---|---|
| Status | **Grounded** (literature claim, ConvNeXt V2 paper) |
| File | `layers.py::GRN` — implementation verified |
| Test | `test_model.py::TestNumerics::test_grn_numerical_stability` — GRN numerically stable |

### D4: "Stochastic depth (DropPath) scaling linearly with model size prevents overfitting."

| Aspect | Grounding |
|---|---|
| Status | **Hypothesis** |
| Ablation | Ablation E |
| Config | `drop_path_rate` = 0.05 (Atto-S) → 0.30 (Tiny-S), linear scaling |
| **Verification status** | **TODO: unverified** — requires train/val gap monitoring with training pipeline |

### D5: "Fully convolutional → no position encoding, works at any input resolution."

| Aspect | Grounding |
|---|---|
| Status | **Grounded** (architectural property) |
| File | `test_model.py::TestShapes::test_variable_time_frames` — works at T=43,86,172 |
| File | `test_model.py::TestShapes::test_variable_frequency_bins` — works at F=64,128,256 |
| File | `test_model.py::TestShapes::test_variable_batch_size` — works at B=1,4,16 |

### D6: "Gradient checkpointing trades compute for memory without changing output."

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| File | `test_model.py::TestAudioBenchmarks::test_gradient_checkpointing_matches` — forward and backward match non-checkpointed version |
| File | `smoke_test.py::test_checkpointing` — forward + backward succeeds |

---

## Parameter Count Claims

### "Atto-S: ~4.8M params"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| File | `test_model.py::TestAudioBenchmarks::test_parameter_count_estimates` |
| Command | `pytest test_model.py -v -k "test_parameter_count_estimates"` |
| Result | Verified within 15% tolerance |

### "Femto-S: ~9.5M params"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| Same test as above | Verified within 15% |

### "Pico-S: ~15.2M params"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| Same test as above | Verified within 15% |

### "Nano-S: ~20.5M params"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| Same test as above | Verified within 15% |

### "Tiny-S: ~29.8M params"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| Same test as above | Verified within 15% |

### "Monotonic parameter scaling across variants"

| Aspect | Grounding |
|---|---|
| Status | **Grounded** |
| File | `test_model.py::TestAudioBenchmarks::test_all_variants_monotonic_params` |

---

## Profiling Claims

### FLOP Estimates

| Variant | Forward FLOPs (claimed) | Verification |
|---|---|---|
| Atto-S | ~0.45G | **TODO: unverified** (heuristic: 2× params = 0.48G, matches estimate) |
| Femto-S | ~0.85G | **TODO: unverified** (heuristic: 2× params = 0.95G, close to 0.85G) |
| Pico-S | ~1.5G | **TODO: unverified** (heuristic: 2× params = 1.52G, matches estimate) |
| Nano-S | ~2.4G | **TODO: unverified** (heuristic: 2× params = 2.05G, lower than 2.4G claim) |
| Tiny-S | ~3.8G | **TODO: unverified** (heuristic: 2× params = 2.98G, significantly lower than 3.8G claim) |

**Note**: The research document's FLOP claims appear to use actual FLOP counting
(not 2× params heuristic). The profiling script estimates via 2×/6× params heuristic;
measured FLOPs from torch.profiler would give accurate values. The discrepancy
for Nano-S and Tiny-S suggests the research claims are based on detailed FLOP
accounting, not the heuristic.

| File | `profile_model.py` — profiling script |
| Command | `python profile_model.py --mode forward --batch-size 128` |

---

## Falsification Conditions

| Condition | Grounding | Status |
|---|---|---|
| 10M variant < 90% ESC-50 | No training results | TODO: unverified |
| 30M variant fails to beat 20M by ≥0.5 pp | No training results | TODO: unverified |
| Rectangular kernels < 0.2 pp benefit over square | Ablation A configured | TODO: unverified |
| Unmodified ConvNeXt V2 matches adapted variant | Ablation C configured | TODO: unverified |

---

## Claim Status Summary

| Status | Count | Key items |
|---|---|---|
| **Grounded** | 12 | Parameter counts (5), monotonic scaling, variable resolution support (3), gradient checkpointing, GRN implementation, D2 aspect ratio |
| **Hypothesis** | 8 | All accuracy claims (4), rectangular benefit, GRN benefit, stem benefit, DropPath scaling |
| **TODO: unverified** | 10 | All training-dependent claims; FLOP accuracy for Nano-S/Tiny-S; FCMAE pretraining; baseline comparisons |

### Blocking ungrounded claims:

The following claims are central to the research hypothesis but completely unverified:
1. Femto-S (10M) ≥ **93.0%** on ESC-50 — no accuracy results at all
2. Tiny-S (30M) ≥ **94.0%** on ESC-50 — no accuracy results at all
3. Rectangular kernels provide ≥ **0.5 pp** benefit — requires training
4. GRN provides ≥ **0.3 pp** benefit — requires training
5. **Any baseline comparison** — requires external implementations

These are not "nice to have" — they are the core claims of the research. Without
them, the project has no empirical results.
