# SpectroConvNeXt — Architecture Deliverables

## Files

| File | Description |
|---|---|
| `spectroconvnext_config.py` | ModelConfig dataclass with full hyperparameter surface, variant auto-config, and convenience constructors |
| `spectroconvnext_architecture.md` | Complete architecture blueprint: pseudocode, ASCII diagram, inductive bias traces, ablation catalogue, risk flags, baseline/evaluation contracts |

## Quick Start

```python
from spectroconvnext_config import femto_s, pico_s

# 10M variant
config = femto_s()

# 15M variant with ablation: remove GRN
config = pico_s(use_grn=False)

# Ablation: rectangular → square kernel in Stage 1
config = pico_s(stage_kernel_sizes=[(7,7), (7,7), (7,7), (5,5)])
```

## Variant Summary

| Variant | Params | Stem C | Channels | FLOPs | DropPath |
|---|---|---|---|---|---|
| Atto-S  | ~4.8M  | 32  | [48, 96, 192, 384]  | 0.45G | 0.05 |
| Femto-S | ~9.5M  | 40  | [64, 128, 256, 512]  | 0.85G | 0.10 |
| Pico-S  | ~15.2M | 48  | [80, 160, 320, 640]  | 1.5G  | 0.15 |
| Nano-S  | ~20.5M | 56  | [96, 192, 384, 768]  | 2.4G  | 0.20 |
| Tiny-S  | ~29.8M | 64  | [112, 224, 448, 896] | 3.8G  | 0.30 |

## Novel Design Elements

1. **Frequency-preserving stem** — two conv layers with asymmetric stride `[(1,2), (2,1)]` instead of 4×4 patchify
2. **Rectangular (7,5) depthwise kernel** in Stage 1 — matches spectrogram 1.49:1 aspect ratio
3. **GRN** in every block — prevents feature collapse without extra parameters
4. **5×5 kernel** in Stage 4 — accommodates 8×6 spatial resolution
5. **DropPath scaled with model size** — 0.05 (5M) → 0.30 (30M) to combat overfitting on 2000-clip ESC-50

## Key Hypotheses

- Femto-S (10M): ≥93.0% on ESC-50 (5-fold CV)
- Tiny-S (30M): ≥94.0% on ESC-50
- Rectangular kernels provide ≥0.5 pp benefit over square at all scales
- GRN provides ≥0.3 pp benefit over identical arch without GRN

## Ablations (6 total)

See Section 12 of the architecture doc. Each is a single `ModelConfig` field change.
