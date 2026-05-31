# Changelog

## [0.1.0] — 2026-05-31
### Added
- Initial implementation of SpectroConvNeXt, a ConvNeXt V2-derived CNN family for audio spectrogram classification (ESC-50).
- Five model variants: Atto-S (~4.8M), Femto-S (~9.5M), Pico-S (~15.2M), Nano-S (~20.5M), Tiny-S (~29.8M).
- Frequency-preserving two-layer stem (spectrogram-adapted, asymmetric stride) with PatchifyStem as ablation alternative.
- SpectroConvNeXtBlock: depthwise conv → LayerNorm → inverted bottleneck FFN → GRN → stochastic depth + residual.
- GRN (Global Response Normalization) with bf16-safe float32 casting.
- ClassificationHead: GAP → LayerNorm → Linear.
- AudioFrontend: waveform → log-mel spectrogram with per-sample normalisation.
- Gradient checkpointing support via `forward_with_checkpoint`.
- Unit test suite (11 categories): shape tests for all 5 variants, variable input resolutions, gradient flow, numerical stability (bf16, extreme values), GRN stability, stochastic depth stability.
- Domain-specific benchmarks (8): audio frontend output shape, log-mel stability, normalisation correctness, end-to-end waveform→logits pipeline, variable audio length, translation robustness, noise entropy, frequency axis symmetry, gradient checkpointing correctness.
- Parameter count verification (all 5 variants within 15% of targets, monotonic scaling).
- Profiling script (`profile_model.py`): forward/train modes, all variants, batch size sweep, GPU memory tracking, Chrome trace export, bfloat16 support.
- Ablation runner (`run_ablations.py`): 6 ablations (A–F) configured as single-field config changes with dry-run and training-aware modes.
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API.
- Research-quality evaluation artefacts: scorecard, claim grounding matrix, experiment coverage matrix.
