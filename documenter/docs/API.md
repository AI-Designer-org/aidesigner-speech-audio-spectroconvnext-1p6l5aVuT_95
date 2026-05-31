# API Reference

## `model.py` — Top-level model and configuration

### `count_params(model)`
Print total and trainable parameter counts for a model.
- **Args:** `model (nn.Module)` — any PyTorch module
- **Returns:** `None` (prints to stdout)

### `class PreprocessingConfig`
Audio preprocessing / spectrogram extraction parameters.
- **Fields:**
  - `sample_rate (int=22050)` — target sample rate in Hz
  - `clip_duration_sec (float=2.0)` — clip duration in seconds
  - `n_fft (int=1024)` — FFT size
  - `hop_length (int=512)` — hop length in samples
  - `n_mels (int=128)` — number of mel filterbanks
  - `power (float=2.0)` — power of the spectrogram
  - `f_min (int=0)` — minimum frequency in Hz
  - `f_max (Optional[int]=None)` — maximum frequency (None = Nyquist)
  - `log_offset (float=1e-6)` — small constant before log to avoid log(0)
  - `normalize (bool=True)` — per-sample mean/std normalisation

### `class AugmentationConfig`
Data augmentation parameters (applied in data pipeline).
- **Fields:**
  - `spec_augment_freq_mask (int=8)` — frequency mask size
  - `spec_augment_time_mask (int=8)` — time mask size
  - `spec_augment_num_masks (int=2)` — number of SpecAugment masks
  - `mixup_alpha (float=0.2)` — Mixup interpolation parameter
  - `random_crop_size (Tuple[int,int]=(128,80))` — (freq, time) crop size
  - `random_crop_resize (Tuple[int,int]=(128,86))` — resize back to

### `class TrainingConfig`
Training hyperparameters.
- **Fields:**
  - `optimizer (str="adamw")` — optimizer name
  - `learning_rate (float=1e-3)` — peak learning rate
  - `lr_min (float=1e-5)` — minimum learning rate (cosine decay floor)
  - `lr_schedule (str="cosine")` — learning rate schedule
  - `weight_decay (float=0.05)` — AdamW weight decay
  - `beta1 (float=0.9)` — Adam beta1
  - `beta2 (float=0.999)` — Adam beta2
  - `batch_size (int=128)` — batch size
  - `epochs (int=300)` — number of training epochs
  - `warmup_epochs (int=10)` — linear warmup epochs
  - `label_smoothing (float=0.1)` — label smoothing factor
  - `ema_decay (float=0.9998)` — EMA decay rate

### `class SpectroConvNeXtConfig`
Model family configuration. Variant auto-derives stem channels, stage channels, block counts, and default drop-path rate.

**Fields:**
- `variant (Literal["atto-s","femto-s","pico-s","nano-s","tiny-s"]="femto-s")` — model variant
- `stem_channels (int=40)` — channels after first conv (auto-derived from variant)
- `stage_channels (Tuple[int,...]=(64,128,256,512))` — C1–C4 (auto-derived)
- `stage_blocks (Tuple[int,...]=(2,2,6,2))` — N1–N4 (auto-derived)
- `stage_kernel_sizes (Tuple[Tuple[int,int],...]=((7,5),(7,7),(7,7),(5,5)))` — kernel per stage
- `expand_ratio (int=4)` — inverted bottleneck expansion
- `use_grn (bool=True)` — Global Response Normalisation
- `norm_eps (float=1e-6)` — norm epsilon
- `drop_path_rate (float=0.15)` — stochastic depth (auto-scaled by variant)
- `dropout (float=0.0)` — dropout probability
- `stem_type (Literal["spectrogram","patchify"]="spectrogram")` — stem design
- `n_classes (int=50)` — output classes
- `in_channels (int=1)` — input channels (mono mel-spectrogram)
- `preprocessing (PreprocessingConfig)` — sub-config for audio preprocessing
- `augmentation (AugmentationConfig)` — sub-config for augmentation
- `training (TrainingConfig)` — sub-config for training hyperparameters

**Properties:**
- `n_stages` — number of stages (always 4)
- `total_blocks` — sum of blocks across all stages
- `param_estimate_m` — human-readable param estimate string

### `atto_s(**overrides) -> SpectroConvNeXtConfig`
Convenience constructor for ~5M variant.

### `femto_s(**overrides) -> SpectroConvNeXtConfig`
Convenience constructor for ~10M variant.

### `pico_s(**overrides) -> SpectroConvNeXtConfig`
Convenience constructor for ~15M variant.

### `nano_s(**overrides) -> SpectroConvNeXtConfig`
Convenience constructor for ~20M variant.

### `tiny_s(**overrides) -> SpectroConvNeXtConfig`
Convenience constructor for ~30M variant.

### `class SpectroConvNeXt(nn.Module)`
Full model: backbone + classification head.

**Constructor:** `SpectroConvNeXt(config: SpectroConvNeXtConfig)`

**Methods:**
- `forward(x: Tensor, use_checkpoint: bool=False) -> Tensor`
  - **Input:** `(B, 1, H, W)` — log-mel spectrogram, typically `(B, 1, 128, ~86)`
  - **Returns:** `(B, n_classes)` — class logits (no built-in Softmax)

**Shape invariants:**
- Batch B ≥ 1; H and W can vary (fully convolutional)
- dtype in {float32, bfloat16}; bfloat16 recommended for training (CUDA)

---

## `layers.py` — Building blocks

### `class LayerNorm2d(nn.Module)`
Channel-wise LayerNorm for convolutional feature maps. Permutes `(B, C, H, W)` → `(B, H, W, C)`, applies `F.layer_norm`, permutes back.

**Constructor:** `LayerNorm2d(dim: int, eps: float=1e-6)`
- **Args:** `dim` — number of channels (C); `eps` — epsilon for numerical stability

**Forward:** `(B, C, H, W) → (B, C, H, W)`

### `class GRN(nn.Module)`
Global Response Normalisation (ConvNeXt V2, Woo et al. 2023).

For each channel i: `gx_i = ||X_i||₂` (spatial L2 norm), then `nx_i = gx_i / mean(gx) + eps`, then output `= γ · X_i · nx_i + β`.

**Constructor:** `GRN(dim: int, eps: float=1e-6, gamma_init: float=1.0, beta_init: float=0.0)`

**Forward:** `(B, C, H, W) → (B, C, H, W)`

**Numerical safety:** Internal float32 computation; casts back to original dtype.

### `class StochasticDepth(nn.Module)`
DropPath / Stochastic Depth. Randomly drops entire samples with probability `drop_prob` during training.

**Constructor:** `StochasticDepth(drop_prob: float=0.0)`

**Forward:** `(B, ...) → (B, ...)` — same shape, stochastically dropped

### `class SpectroConvNeXtBlock(nn.Module)`
Core novel operator. Architecture: `LayerNorm → Depthwise Conv2D → LayerNorm → 1×1 expand (4×) → GELU → 1×1 project → GRN → DropPath + residual`.

**Constructor:**
`SpectroConvNeXtBlock(dim: int, kernel_size: Tuple[int,int], expand_ratio: int=4, drop_path: float=0.0, use_grn: bool=True)`

**Forward:** `(B, C, H, W) → (B, C, H, W)` — same shape, residual NOT yet added (handled internally)

**Methods:**
- `forward_with_checkpoint(x: Tensor) -> Tensor` — gradient checkpointing wrapper

### `class DownsampleBlock(nn.Module)`
Spatial downsampling between stages. `LayerNorm → Conv2D(kernel=2, stride=2)`.

**Constructor:** `DownsampleBlock(dim_in: int, dim_out: int)`

**Forward:** `(B, C_in, H, W) → (B, C_out, H//2, W//2)`

### `class SpectrogramStem(nn.Module)`
Two-layer frequency-preserving stem: Layer 1 stride (1,2) preserves freq, halves time; Layer 2 stride (2,1) halves freq, preserves time.

**Constructor:** `SpectrogramStem(in_ch: int, stem_ch: int, out_ch: int)`

**Forward:** `(B, in_ch, 128, ~86) → (B, out_ch, 64, ~43)`

### `class PatchifyStem(nn.Module)`
Original ConvNeXt V2 patchify stem: 4×4 convolution with stride 4. Used for Ablation C.

**Constructor:** `PatchifyStem(in_ch: int, out_ch: int)`

**Forward:** `(B, in_ch, 128, ~86) → (B, out_ch, 32, ~21)`

---

## `backbone.py` — Feature extractor

### `class BaseOperator(ABC, nn.Module)`
Abstract base class for core feature extractors. Subclasses must implement `forward(x: Tensor) -> Tensor`.

### `class SpectroConvNeXtBackbone(BaseOperator)`
Feature extractor: stem + 4 stages of SpectroConvNeXt blocks + interleaved downsamplers + final norm.

**Constructor:**
`SpectroConvNeXtBackbone(stem_type="spectrogram", in_ch=1, stem_channels=40, stage_channels=(64,128,256,512), stage_blocks=(2,2,6,2), stage_kernel_sizes=((7,5),(7,7),(7,7),(5,5)), expand_ratio=4, use_grn=True, drop_path_rate=0.15)`

**Forward:** `(B, 1, H, W) → (B, C4, H_out, W_out)` — feature map at 1/16 resolution

**Methods:**
- `get_block_drop_probs() -> List[float]` — return DropPath probabilities per block (for logging)

---

## `head.py` — Classification head

### `class ClassificationHead(nn.Module)`
Global Average Pooling → LayerNorm → Dropout → Linear → logits.

**Constructor:** `ClassificationHead(dim: int, n_classes: int=50, dropout: float=0.0)`

**Forward:** `(B, C4, H_out, W_out) → (B, n_classes)` — class logits

---

## `frontend.py` — Audio feature extraction

### `class AudioFrontend(nn.Module)`
Audio feature extraction: waveform → normalised log-mel spectrogram.

Pipeline: resample → trim/pad to 2.0s → STFT → Mel filterbank → log compression → per-sample mean/std normalisation → add channel dim.

**Constructor:**
`AudioFrontend(sample_rate=22050, clip_duration_sec=2.0, n_fft=1024, hop_length=512, n_mels=128, power=2.0, f_min=0, f_max=None, log_offset=1e-6, normalize=True)`

**Forward:** `(B, 1, T_samples) → (B, 1, n_mels, T_frames)`
- Input: raw audio signals in [-1, 1] range, sample_rate must match constructor arg
- Output: typical shape `(B, 1, 128, 86)` at 22050 Hz

**Methods:**
- `get_num_frames(num_samples=None) -> int` — compute time frames given sample count

---

## `validator/profile_model.py` — Profiling script

### `profile_model(variant_name, config_fn, mode="forward", steps=20, warmup=5, batch_size=128, device=None, dtype=float32, trace_path=None) -> dict`
Profile a single model variant using `torch.profiler`.

**Returns:** dict with keys: `variant`, `mode`, `params_m`, `params`, `avg_cuda_ms`, `avg_cpu_ms`, `est_flops_g`, `throughput_samples_per_sec`, `batch_size`, `device`, `dtype`, `gpu_memory_mb` (if CUDA).

**CLI usage:**
```bash
python profile_model.py --mode forward --batch-size 128 --steps 50
python profile_model.py --variant femto-s --mode train --bf16
python profile_model.py --trace trace.json --output results.json
```

---

## `validator/run_ablations.py` — Ablation runner

All 6 ablations (A–F) are defined in the `ABLATIONS` dict. Each is a single-field ModelConfig change.

**CLI usage:**
```bash
python run_ablations.py                                # parameter-count check only
python run_ablations.py --only A,B,C --dry-run         # specific ablations, no training
python run_ablations.py --train train.py --eval eval.py  # full training run
```
