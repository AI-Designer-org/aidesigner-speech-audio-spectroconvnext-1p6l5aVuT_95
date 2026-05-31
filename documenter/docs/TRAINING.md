# Training & Reproduction

> **No training pipeline is available yet.** This document describes the recommended training recipe from the architecture design. All accuracy claims in this document are `TODO: unverified` until a `train.py` script and ESC-50 data loader are implemented and results are obtained.

## Environment

Recommended environment (based on the ConvNeXt V2 training recipe and validated component compatibility):

- **Python**: 3.10+
- **PyTorch**: 2.0+ (2.3+ recommended for `torch.compile`)
- **CUDA**: 11.8+ (12.1 recommended), tested on A100 40GB / 80GB
- **torchaudio**: 2.0+ (for `AudioFrontend` — MelSpectrogram transform)
- **Other**: `torchvision` (for Mixup), `scikit-learn` (for stratified KFold, McNemar's test)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio torchvision scikit-learn
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The codebase is device-portable: CUDA, MPS (Apple Silicon), and CPU are all supported. For training, an A100 or similar GPU with ≥16GB VRAM is recommended for batch sizes ≥64 with larger variants.

## Default hyperparameters

### Model configuration (Femto-S shown)

| Field | Default | Rationale |
|---|---|---|
| `variant` | `"femto-s"` | ~10M params — central hypothesis target |
| `stem_type` | `"spectrogram"` | Frequency-preserving two-layer stem (hypothesis — ablated) |
| `use_grn` | `True` | Global Response Normalization (hypothesis — ablated) |
| `expand_ratio` | `4` | Standard inverted bottleneck expansion |
| `detection` | `drop_path_rate` | Auto-scaled by variant (0.05→0.30) |
| `stage_kernel_sizes` | `((7,5), (7,7), (7,7), (5,5))` | Rectangular in Stage 1 for spectrogram aspect ratio |

### Data preprocessing

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 22050 Hz | Balances resolution and compute; standard for ESC-50 |
| Clip length | 2.0 s (44100 samples) | Standard ESC-50 clip duration |
| n_fft | 1024 | Frequency resolution ~21.5 Hz |
| hop_length | 512 | ~43% overlap, ~86 time frames |
| n_mels | 128 | Standard mel bands for ESC-50 |
| Log offset | 1e-6 | Prevents log(0) on silent frames |
| Normalization | Per-sample mean/std | Standardises dynamic range across clips |

### Augmentation

| Parameter | Value | Rationale |
|---|---|---|
| SpecAugment freq mask | F=8 | Masks 8 consecutive mel bands |
| SpecAugment time mask | T=8 | Masks 8 consecutive time frames |
| SpecAugment num masks | 2 | Standard choice for ESC-50 |
| Mixup alpha | 0.2 | Soft mix; stronger than 0.1 but not as aggressive as 0.5 |
| Random crop | 128×80 → resize to 128×86 | Frequency-preserving crop + resize back |

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Peak learning rate | 1e-3 | Linear warmup over 10 epochs, cosine decay to 1e-5 |
| Batch size | 128 | Gradient accumulation if GPU memory limited |
| Weight decay | 0.05 | Excluded from bias/norm parameters |
| Epochs | 300 | Full convergence on ESC-50 with augmentation |
| Warmup epochs | 10 | Linear warmup from 0 to peak LR |
| Label smoothing | 0.1 | Softens targets; helps generalisation on small dataset |
| EMA decay | 0.9998 | Exponential moving average of weights |
| Loss | Cross-entropy | With label smoothing |
| Precision | bf16 mixed | Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` — GRN handles bf16 safely via internal float32 casting |
| Gradient clipping | 1.0 | Global norm (recommended but not tested) |

### AdamW parameter groups

The standard ConvNeXt parameter grouping strategy:
- **No weight decay on**: bias parameters, LayerNorm weight/bias
- **Weight decay on**: all conv weight parameters
- **Learning rate**: same LR for all groups (no layer-wise decay by default; `layer_decay` config field available if needed)

### DropPath rates per variant

| Variant | DropPath rate | Rationale |
|---|---|---|
| Atto-S | 0.05 | Lightest regularisation — small model, low overfitting risk |
| Femto-S | 0.10 | Light regularisation |
| Pico-S | 0.15 | Moderate regularisation |
| Nano-S | 0.20 | Moderate-heavy regularisation |
| Tiny-S | 0.30 | Heaviest regularisation — 30M params on 2000 clips |

These DropPath rates scale linearly with block depth within each variant: the first block gets rate 0, the final block gets the full rate.

### Domain-specific training additions

- **Audioset**: Data augmentation pipeline (SpecAugment, Mixup, random crop) is recommended before normalisation.
- **5-fold cross-validation**: ESC-50 standard protocol. Use `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Report mean ± std top-1 accuracy.
- **Evaluation**: Use EMA weights for validation (higher accuracy). Apply Softmax to logits for accuracy computation.

## Expected behavior

| Training phase | Expected observation | Interpretation |
|---|---|---|
| Epoch 0-10 (warmup) | Loss decreases from ~3.9 (random) to ~2.5-3.0 | Learning rate ramp-up; model learning basic patterns |
| Epoch 10-100 | Loss decreases to ~1.0-1.5; accuracy rises to 80-90% | Main learning phase |
| Epoch 100-300 | Loss slowly decreases to ~0.5-1.0; accuracy plateaus | Fine-tuning; diminishing returns |
| Train/val gap | <5 pp for Atto-S; may widen to >10 pp for Tiny-S | Overfitting expected at larger model sizes |

> **TODO: unverified** — No reference training run exists. The above is extrapolated from ConvNeXt V2 ImageNet training behaviour and ESC-50 literature.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16 float-sensitive op in depthwise conv | Verify GRN uses internal float32 (it does — see `layers.py` line 97). If NaN persists, check depthwise conv for numerical issues. |
| Train acc high (>95%), val acc low (<80%) | Overfitting — model memorises 2000 training clips | Increase DropPath rate, increase weight decay, reduce epochs, or add stronger augmentation. |
| Train acc stagnates at <50% | Learning rate too low; gradient flow blocked | Verify gradients flow (pytest `test_gradient_flows_through_all_blocks`). If OK, increase LR or reduce weight decay. |
| Loss decreases but accuracy does not | Label smoothing is masking overfitting; model is confident on wrong classes | Reduce label smoothing (0.05) or disable for the last 50 epochs. |
| Femto-S accuracy <90% | Hypothesis falsified (per research contract: 10M < 90% → design flawed) | Re-evaluate architecture. Check data pipeline first (SpecAugment, normalisation). If data pipeline is correct, the ConvNet design may need attention mechanisms (coordinate attention, etc.). |
| Tiny-S accuracy within 0.3 pp of Nano-S | Scaling ceiling hit at ~20M | Report as a finding. The 30M variant may be impractical for 2000-clip ESC-50. |
| Validation accuracy oscillates (±3-5 pp) | Learning rate too high; insufficient warmup | Reduce LR to 3e-4, increase warmup to 20 epochs. |
| Training extremely slow | Depthwise convolution memory bandwidth bottleneck | Enable `torch.compile` (fuses depthwise conv kernels). If using A100, ensure cuDNN backend is optimised (`torch.backends.cudnn.benchmark=True`). |
| Gradient checkpoint OOM | Checkpoint memory saving insufficient for batch size | Reduce batch size or disable checkpointing for early layers. |

### Known implementation risks (from architecture doc)

| Risk | Severity | Mitigation | Status |
|---|---|---|---|
| GRN numerical instability in bf16 | Low | Internal float32 casting in `GRN.forward()` | **Implemented** — see `layers.py` line 96-108 |
| Depthwise conv memory-bandwidth-bound | Medium | `torch.compile`, cuDNN benchmark | Available but untested |
| Tiny-S overfitting on 2000 clips | Medium-High | DropPath=0.3, label smoothing, Mixup, SpecAugment | Configurable but untested |
| Rectangular (7,5) kernel perf cliff with `torch.compile` | Low-Medium | Fall back to (7,7) if compilation issues | Untested |
