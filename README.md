> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# SpectroConvNeXt

A ConvNeXt V2-derived CNN family with spectrogram-adapted rectangular kernels for environmental sound classification on ESC-50, spanning five model variants from 5M to 30M parameters.

SpectroConvNeXt addresses the gap in systematic modern CNN scaling on ESC-50, where prior methods are all <5M parameters and use ad-hoc architectures. It applies four key adaptations to ConvNeXt V2 for audio spectrograms: a frequency-preserving two-layer stem, rectangular depthwise kernels in Stage 1 (matching the 1.49:1 spectrogram aspect ratio), Global Response Normalization in every block, and DropPath regularisation scaled with model size.

> **All accuracy-related claims are TODO: unverified** — see [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md#research-quality-evaluation). A training pipeline is required to evaluate the central hypotheses (≥93% at 10M, ≥94% at 30M).

## Highlights

- **Five cleanly scaled variants (5M–30M)** — Atto-S (~4.8M), Femto-S (~9.5M), Pico-S (~15.2M), Nano-S (~20.5M), Tiny-S (~29.8M); see [ARCHITECTURE.md#7-variant-scaling-table](documenter/docs/ARCHITECTURE.md#7-variant-scaling-table)
- **Spectrogram-native adaptations** — frequency-preserving two-layer stem, rectangular (7,5) depthwise kernel in Stage 1, asymmetric stride; see [ARCHITECTURE.md#4-architecture-overview](documenter/docs/ARCHITECTURE.md#4-architecture-overview)
- **GRN in every block** — Global Response Normalization (ConvNeXt V2) for inter-channel competition without extra parameters; see [ARCHITECTURE.md#54-global-response-normalization-grn](documenter/docs/ARCHITECTURE.md#54-global-response-normalization-grn)
- **Parameter counts verified** — all five variants within 15% of targets (grounded by validator); see [BENCHMARKS.md#synthetic-benchmarks](documenter/docs/BENCHMARKS.md#synthetic-benchmarks)
- **Fully convolutional, any input resolution** — no position encoding; operates at any frequency/time resolution (validated at F=64,128,256 and T=43,86,172)

## Quick start

```bash
pip install torch torchaudio
cd coder/
python smoke_test.py        # all 5 variants + ablations
pytest test_model.py -v     # 30+ tests: shape, gradient, numerics, audio pipeline
python run_ablations.py     # parameter-count check for all 6 ablations
```

## Repository layout

```
coder/                      # Implementation
    model.py                # Full model: SpectroConvNeXt + config dataclasses
    backbone.py             # Backbone: stem + 4 stages + downsamplers
    layers.py               # Building blocks: GRN, StochasticDepth, SpectroConvNeXtBlock, stems
    head.py                 # Classification head: GAP → LayerNorm → Linear
    frontend.py             # AudioFrontend: waveform → log-mel spectrogram
    smoke_test.py           # Smoke test: variants, ablations, gradient flow
validator/                  # Tests and benchmarks
    test_model.py           # Unit tests (shape, gradient, CV, audio, numerics)
    profile_model.py        # torch.profiler throughput and memory profiling
    run_ablations.py        # Ablation runner (6 ablations, single-config-field changes)
    research_eval/          # Research-quality evaluation artifacts
        scorecard.json      # Numerical scores and gap analysis
        claim_grounding.md  # Trace each claim to source/verification
        experiment_coverage.md  # Coverage matrix of all required experiments
architect/                  # Architecture design
    spectroconvnext_architecture.md  # Full blueprint with pseudocode and ASCII diagram
    spectroconvnext_config.py        # Standalone ModelConfig dataclass
research/                   # Research synthesis
    spectroconvnext_research_synthesis.md  # Landscape, gaps, hypothesis, lifecycle contract
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — design motivation, inductive biases, tensor shape evolution, design decisions
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — environment setup, recommended hyperparameters, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — test results, ablation catalogue, profiling, research-quality evaluation
- [docs/API.md](documenter/docs/API.md) — module-level API reference with shape contracts

## Citation

```bibtex
@misc{spectroconvnext-2026,
  title  = {SpectroConvNeXt: A Modern CNN Family for Audio Spectrogram Classification},
  author = {TODO},
  year   = {2026},
  note   = {Generated via ml-designer pipeline}
}
```
