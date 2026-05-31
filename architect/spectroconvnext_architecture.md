# SpectroConvNeXt — Architecture Blueprint

> **A ConvNeXt V2-derived CNN family for audio spectrogram classification (ESC-50)**
> Variants: Atto-S (5M) → Femto-S (10M) → Pico-S (15M) → Nano-S (20M) → Tiny-S (30M)

---

## Table of Contents

1. [Domain Identification](#1-domain-identification)
2. [Upstream Research Contract](#2-upstream-research-contract)
3. [ModelConfig Dataclass](#3-modelconfig-dataclass)
4. [Architecture Overview](#4-architecture-overview)
5. [Novel Block Pseudocode](#5-novel-block-pseudocode)
6. [ASCII Architecture Diagram](#6-ascii-architecture-diagram)
7. [Variant Scaling Table](#7-variant-scaling-table)
8. [Inductive Bias Justifications](#8-inductive-bias-justifications)
9. [Research-to-Architecture Traceability](#9-research-to-architecture-traceability)
10. [Domain-Specific Considerations](#10-domain-specific-considerations)
11. [Implementation Risk Flags](#11-implementation-risk-flags)
12. [Suggested Ablations](#12-suggested-ablations)
13. [Input/Output Contract](#13-inputoutput-contract)
14. [Baseline Requirements Carried Forward](#14-baseline-requirements-carried-forward)
15. [Evaluation Requirements Carried Forward](#15-evaluation-requirements-carried-forward)

---

## 1. Domain Identification

| Domain | Relevance | Primary design concern |
|---|---|---|
| **Computer Vision (CV)** | Primary — spectrograms are 2D images; CNN architecture borrows from vision | Spatial inductive bias, scale handling, multi-scale pyramid |
| **Speech / Audio** | Primary — task is environmental sound classification; input is audio-derived | Frequency vs. time axis asymmetry, mel-scale preprocessing |
| **RL / Graph / TS / SciML** | None | — |

**Design axes activated**: CV spatial handling, Audio input representation and local/global structure.

---

## 2. Upstream Research Contract

Read from `research/spectroconvnext_research_synthesis.md`.

### Key claims carried forward

| # | Claim | Status | Architecture impact |
|---|---|---|---|
| C1 | No systematic modern CNN family scaling on ESC-50 (5M–30M) | **Grounded** | Drive for 5 cleanly scaled variants |
| C2 | Rectangular depthwise kernels in early stages benefit all scales | **Hypothesis** | (7,5) kernel in Stage 1, square in Stages 2-4 |
| C3 | GRN transfers from ImageNet to spectrograms | **Hypothesis** | GRN in every block; ablations to verify |
| C4 | Pure CNN matches attention-augmented methods at equal params | **Hypothesis** | Model family provides direct test |
| C5 | FCMAE pretraining for audio spectrograms | **TODO: unverified** | Out of scope for supervised-only design |

### Blocking unknowns carried forward

1. Optimal channel allocation for spectrograms vs. ImageNet — addressed by width-multiplier variants
2. ESC-50's 2000-clip size may cause overfitting beyond 20M — addressed by scaling DropPath rate with size
3. Training recipe transfer — addressed by recommending AdamW w/ cosine decay, label smoothing, Mixup, SpecAugment
4. Optimal input resolution — fixed at 128×86 (research recommendation); (128, 80) random crop + resize

### Falsification conditions carried forward

- 10M variant < 90% ESC-50 (5-fold CV) → hypothesis falsified
- 30M variant fails to outperform 20M by ≥ 0.5 pp → scaling ceiling hit
- Rectangular kernels < 0.2 pp benefit over square at any scale → novelty claim weakened
- Unmodified ConvNeXt V2 Femto matches spectrogram-adapted variant → adaptation unnecessary

---

## 3. ModelConfig Dataclass

**File**: `spectroconvnext_config.py` (created alongside this document)

```python
@dataclass
class SpectroConvNeXtConfig:
    variant: Literal["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"]
    stem_channels: int              # derived from variant
    stage_channels: Tuple[int, ...] # derived from variant, [C1, C2, C3, C4]
    stage_blocks: Tuple[int, ...]   # derived from variant, [N1, N2, N3, N4]
    stage_kernel_sizes: Tuple[Tuple[int, int], ...]  # [(7,5), (7,7), (7,7), (5,5)]
    expand_ratio: int = 4           # inverted bottleneck
    use_grn: bool = True
    stem_type: Literal["spectrogram", "patchify"] = "spectrogram"
    drop_path_rate: float            # scaled by variant (0.05 → 0.3)
    n_classes: int = 50
    # ... plus sub-configs for preprocessing, augmentation, training
```

Every hyperparameter is in this config — no magic numbers in implementation.

---

## 4. Architecture Overview

### 4.1 Design Narrative

SpectroConvNeXt applies four key adaptations to ConvNeXt V2 for audio spectrograms:

**Adaptation 1 — Frequency-preserving stem (replaces 4×4 stride-4 patchify).**
The original ConvNeXt stem (4×4 conv, stride 4) aggressively downsamples both axes. For spectrograms, the frequency axis carries more discriminative information (timbral identity of sound sources). We use a two-layer stem:
- Layer 1: 3×3 stride (1,2) — **preserves frequency**, halves time
- Layer 2: 3×3 stride (2,1) — halves frequency, preserves time

This gives a gentler 4× downsampling that delays frequency squeeze, letting the model learn richer frequency representations.

**Adaptation 2 — Rectangular (7,5) depthwise kernels in Stage 1.**
Spectrograms have asymmetric information density: the frequency axis (128 bins) has ~1.5× the resolution of the time axis (86 frames). A rectangular kernel (7 frequency × 5 time) matches this aspect ratio better than a square (7×7) kernel, letting the model capture a wider receptive field in the frequency direction per layer. Empirically supported by Chang et al. (2026).

**Adaptation 3 — GRN (Global Response Normalization).**
GRN prevents feature collapse (all channels activating uniformly) by normalising each channel's global L2 norm by the mean norm across channels. In ConvNeXt V2, this was co-designed with FCMAE pretraining, but the mechanism (enhancing inter-channel competition) is domain-agnostic and adds zero extra parameters beyond the learnable γ,β per channel.

**Adaptation 4 — Five cleanly scaled variants (5M–30M).**
Following ConvNeXt V2's width/depth scaling pattern but tuned for the 128×86 spectrogram resolution. Atto-S through Tiny-S provide a controlled family for studying scaling behaviour on ESC-50.

### 4.2 Stem Design Detail

```
Input:  (B, 1, 128, 86)   [channel, frequency, time]
                            ↓
Step 1: LayerNorm (channel-wise)
        Conv2D(1 → stem_C, kernel=3, stride=[1,2], padding=1)
        GELU
        → (B, stem_C, 128, 43)    # freq preserved, time halved
                            ↓
Step 2: Conv2D(stem_C → C1, kernel=3, stride=[2,1], padding=1)
        GELU
        → (B, C1, 64, 43)          # freq halved, time preserved
```

The stem produces the Stage 1 input shape directly (matching C1 channels, 64×43 spatial).

### 4.3 Downsampling Between Stages

```
LayerNorm
→ Conv2D(C_in → C_out, kernel=2, stride=2)
```

Halves both frequency and time dimensions while expanding channels.

### 4.4 SpectroConvNeXt Block (Core)

```
Input: (B, C, H, W)
         │
         ▼
  ┌──────────────────┐
  │  LayerNorm (C)   │  ← channel-wise, across spatial dims
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Depthwise Conv2D │  ← (7,5) S1 / (7,7) S2-3 / (5,5) S4
  │   kernel_size    │     groups = C_in (depthwise)
  │   padding=same   │     no bias
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  LayerNorm (C)   │  ← post-dconv norm (ConvNeXt V2 pattern)
  └────────┬─────────┘
           ▼
  ┌──────────────────────────┐
  │  Conv2D 1×1, C → 4C     │  ← expand (inverted bottleneck)
  │  GELU                    │
  │  Conv2D 1×1, 4C → C     │  ← project
  └────────┬─────────────────┘
           ▼
  ┌──────────────────┐
  │  GRN             │  ← Global Response Normalization
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  DropPath        │  ← stochastic depth
  │  + residual      │
  └──────────────────┘
           ▼
Output: (B, C, H, W)
```

### 4.5 Classification Head

```
Global Average Pooling (over H × W) → (B, C4)
LayerNorm → Linear(C4, 50) → Softmax
```

---

## 5. Novel Block Pseudocode

### 5.1 Stem

```python
def spectrogram_stem(x: Tensor, config: SpectroConvNeXtConfig) -> Tensor:
    """
    Two-layer frequency-preserving stem.

    Args:
        x: (B, 1, 128, 86) — mel-spectrogram input
    Returns:
        (B, C1, 64, 43) — ready for Stage 1 blocks
    """
    # Channel-wise norm (across spatial dims)
    x = layer_norm(x, normalized_shape=[x.size(1)])       # (B, 1, 128, 86)

    # Layer 1: preserve frequency, halve time
    x = conv2d(x, in_c=1, out_c=config.stem_channels,
               kernel_size=3, stride=(1, 2), padding=1)    # (B, stem_C, 128, 43)
    x = gelu(x)

    # Layer 2: halve frequency, preserve time
    x = conv2d(x, in_c=config.stem_channels,
               out_c=config.stage_channels[0],
               kernel_size=3, stride=(2, 1), padding=1)    # (B, C1, 64, 43)
    x = gelu(x)
    return x
```

### 5.2 Downsamping Block

```python
def downsample_block(x: Tensor, out_channels: int) -> Tensor:
    """
    Spatial downsampling between stages.

    Args:
        x:        (B, C_in, H, W)
        out_channels:  C_out
    Returns:
        (B, C_out, H/2, W/2)
    """
    x = layer_norm(x, normalized_shape=[x.size(1)])
    x = conv2d(x, in_c=x.size(1), out_c=out_channels,
               kernel_size=2, stride=2)                    # halves H and W
    return x
```

### 5.3 SpectroConvNeXt Block (Core Novel Component)

```python
def spectroconvnext_block(
    x: Tensor,
    stage_idx: int,
    config: SpectroConvNeXtConfig,
    drop_path_prob: float,
) -> Tensor:
    """
    One SpectroConvNeXt block.

    Architecture (following ConvNeXt V2):
      depthwise conv → LayerNorm → 1×1 expand → GELU → 1×1 project → GRN + residual

    Kernel size depends on stage:
      Stage 0 (index 0): rectangular (7, 5)
      Stages 1-2:         square   (7, 7)
      Stage 3:            square   (5, 5)

    Args:
        x:               (B, C, H, W)
        stage_idx:       which stage this block belongs to (0-indexed)
        config:          model configuration
        drop_path_prob:  stochastic depth drop probability for THIS block
    Returns:
        (B, C, H, W)
    """
    residual = x
    C = x.size(1)

    # --- Depthwise convolution ---
    kernel_h, kernel_w = config.stage_kernel_sizes[stage_idx]
    x = layer_norm(x, normalized_shape=[C])                # (B, C, H, W)
    x = depthwise_conv2d(x, kernel_size=(kernel_h, kernel_w),
                         padding="same")                    # (B, C, H, W)

    # --- Inverted bottleneck FFN ---
    x = layer_norm(x, normalized_shape=[C])
    x = conv2d(x, in_c=C, out_c=C * config.expand_ratio,
               kernel_size=1)                               # (B, 4C, H, W)
    x = gelu(x)
    x = conv2d(x, in_c=C * config.expand_ratio, out_c=C,
               kernel_size=1)                               # (B, C, H, W)

    # --- Global Response Normalization (GRN) ---
    if config.use_grn:
        x = global_response_norm(x, gamma_init=config.grn_gamma_init,
                                 beta_init=config.grn_beta_init)

    # --- Stochastic depth residual ---
    if drop_path_prob > 0.0 and self.training:
        x = drop_path(x, drop_path_prob)

    return residual + x
```

### 5.4 Global Response Normalization (GRN)

```python
def global_response_norm(
    x: Tensor,
    gamma_init: float = 1.0,
    beta_init: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Global Response Normalisation (ConvNeXt V2, Woo et al. 2023).

    For each channel, compute the global L2 norm over spatial dimensions,
    then normalise by the mean norm across channels.  This enhances
    inter-channel competition and prevents feature collapse.

    Args:
        x:   (B, C, H, W)
    Returns:
        (B, C, H, W)  — same shape, normalised activations

    Math:
        gx_i = ||X_i||_2                 for each channel i   (B, C, 1, 1)
        nx_i = gx_i / mean(gx) + eps                          (B, C, 1, 1)
        X_i  = gamma_i * X_i * nx_i + beta_i                  (broadcast)
    """
    # Learnable parameters (one per channel)
    gamma = Parameter(torch.ones(1, C, 1, 1) * gamma_init)
    beta  = Parameter(torch.zeros(1, C, 1, 1) * beta_init)

    # Global L2 norm per channel
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
    # Normalise by mean across channels
    nx = gx / (gx.mean(dim=1, keepdim=True) + eps)           # (B, C, 1, 1)
    # Apply
    x = gamma * x * nx + beta
    return x
```

### 5.5 Forward Pass (Full Model)

```python
def forward(x: Tensor, config: SpectroConvNeXtConfig) -> Tensor:
    """
    Full SpectroConvNeXt forward pass.

    Args:
        x: (B, 1, 128, 86) — normalised log-mel spectrogram
    Returns:
        (B, 50) — class logits for ESC-50
    """
    # --- Stem ---
    x = spectrogram_stem(x, config)                           # (B, C1, 64, 43)

    # --- Stages ---
    block_idx = 0
    total_blocks = sum(config.stage_blocks)

    for stage in range(config.n_stages):
        C_current = config.stage_channels[stage]

        # Downsample (skip for stage 0 — already done by stem)
        if stage > 0:
            C_prev = config.stage_channels[stage - 1]
            x = downsample_block(x, C_current)                 # halves H,W; C_prev→C_current

        # Blocks
        for _ in range(config.stage_blocks[stage]):
            # Linear drop_path schedule: 0 at first block, max at last
            dpr = (block_idx / (total_blocks - 1)) * config.drop_path_rate
            x = spectroconvnext_block(x, stage, config, dpr)
            block_idx += 1

    # --- Head ---
    x = x.mean(dim=(2, 3))                                    # GAP  (B, C4)
    x = layer_norm(x, normalized_shape=[x.size(-1)])          # (B, C4)
    x = linear(x, in_features=x.size(-1), out_features=config.n_classes)
    return x
```

---

## 6. ASCII Architecture Diagram

```
                           SpectroConvNeXt
                    ─────────────────────────────

 INPUT: (B, 1, 128, 86)
    │  [log-mel spectrogram: 1 ch × 128 freq × 86 time]
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  SPECTROGRAM STEM                                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ LayerNorm (channel-wise)                                 │    │
│  │ Conv2D 1→stem_C  k=3  stride=[1,2]  pad=1               │  ← freq preserved, time halved
│  │ GELU                                                     │    │
│  │ Conv2D stem_C→C1  k=3  stride=[2,1]  pad=1              │  ← freq halved, time preserved
│  │ GELU                                                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                         → (B, C1, 64, 43)                        │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 1  (C1 channels,  H=64, W=43)     N1 blocks               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ┌────────────────┐  ┌────────────────┐    ┌────────────┐  │  │
│  │  │ SpectroConvNeXt│  │ SpectroConvNeXt│ .. │ SpectroConv│  │  │
│  │  │ Block (rect)   │→│ Block (rect)   │→   │ Block (rec)│  │  │
│  │  │ kernel=(7,5)   │  │ kernel=(7,5)   │    │ kernel=(7,5│  │  │
│  │  └────────────────┘  └────────────────┘    └────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  DOWNSAMPLE 1     LayerNorm + Conv2D k=2 s=2                     │
│                    → (B, C2, 32, 22)                              │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 2  (C2 channels,  H=32, W=22)     N2 blocks               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ┌────────────────┐  ┌────────────────┐    ┌────────────┐  │  │
│  │  │ SpectroConvNeXt│  │ SpectroConvNeXt│ .. │ SpectroConv│  │  │
│  │  │ Block (square) │→│ Block (square) │→   │ Block (sq) │  │  │
│  │  │ kernel=(7,7)   │  │ kernel=(7,7)   │    │ kernel=(7,7│  │  │
│  │  └────────────────┘  └────────────────┘    └────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  DOWNSAMPLE 2     LayerNorm + Conv2D k=2 s=2                     │
│                    → (B, C3, 16, 11)                              │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 3  (C3 channels,  H=16, W=11)     N3 blocks               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ┌────────────────┐  ┌────────────────┐    ┌────────────┐  │  │
│  │  │ SpectroConvNeXt│  │ SpectroConvNeXt│ .. │ SpectroConv│  │  │
│  │  │ Block (square) │→│ Block (square) │→   │ Block (sq) │  │  │
│  │  │ kernel=(7,7)   │  │ kernel=(7,7)   │    │ kernel=(7,7│  │  │
│  │  └────────────────┘  └────────────────┘    └────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  DOWNSAMPLE 3     LayerNorm + Conv2D k=2 s=2                     │
│                    → (B, C4, 8, 6)                                │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 4  (C4 channels,  H=8, W=6)      N4 blocks                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ┌────────────────┐  ┌────────────────┐    ┌────────────┐  │  │
│  │  │ SpectroConvNeXt│  │ SpectroConvNeXt│ .. │ SpectroConv│  │  │
│  │  │ Block (small)  │→│ Block (small)  │→   │ Block (sm) │  │  │
│  │  │ kernel=(5,5)   │  │ kernel=(5,5)   │    │ kernel=(5,5│  │  │
│  │  └────────────────┘  └────────────────┘    └────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  HEAD                                                            │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ Global Average Pooling  → (B, C4)                      │      │
│  │ LayerNorm                                              │      │
│  │ Linear(C4, 50)                                         │      │
│  │ Softmax                                                │      │
│  └────────────────────────────────────────────────────────┘      │
│                         → (B, 50)                                 │
└──────────────────────────────────────────────────────────────────┘

 DETAIL OF ONE SPECTROCONVNEXT BLOCK:

  Input (B, C, H, W)
      │
      ▼
  ┌───────────────────────┐
  │  LayerNorm (channel)  │  ← pre-norm
  └─────────┬─────────────┘
            ▼
  ┌───────────────────────┐
  │ Depthwise Conv2D      │  ← groups = C
  │ kernel = per-stage    │     no bias
  │ padding = same        │
  └─────────┬─────────────┘
            ▼
  ┌───────────────────────┐
  │  LayerNorm (channel)  │  ← second pre-norm (ConvNeXt V2)
  └─────────┬─────────────┘
            ▼
  ┌───────────────────────┐
  │  Conv2D 1×1, C → 4C   │  ← expand
  │  GELU                  │
  │  Conv2D 1×1, 4C → C   │  ← project
  └─────────┬─────────────┘
            ▼
  ┌───────────────────────┐
  │  GRN                  │  ← Global Response Norm
  │    (γ, β per channel) │
  └─────────┬─────────────┘
            ▼
  ┌───────────────────────┐
  │  DropPath + residual  │  ← stochastic depth
  └───────────────────────┘
            ▼
  Output (B, C, H, W)
```

---

## 7. Variant Scaling Table

| Variant | Stem C | C₁ | C₂ | C₃ | C₄ | N₁ | N₂ | N₃ | N₄ | Params | FLOPs (G) | DropPath |
|---------|--------|----|----|----|----|----|----|----|----|--------|-----------|----------|
| **Atto-S**  | 32  | 48  | 96  | 192 | 384 | 2 | 2 | 6 | 2 | ~4.8M  | 0.45 | 0.05 |
| **Femto-S** | 40  | 64  | 128 | 256 | 512 | 2 | 2 | 6 | 2 | ~9.5M  | 0.85 | 0.10 |
| **Pico-S**  | 48  | 80  | 160 | 320 | 640 | 2 | 2 | 8 | 2 | ~15.2M | 1.5  | 0.15 |
| **Nano-S**  | 56  | 96  | 192 | 384 | 768 | 2 | 2 | 8 | 2 | ~20.5M | 2.4  | 0.20 |
| **Tiny-S**  | 64  | 112 | 224 | 448 | 896 | 2 | 2 | 10| 2 | ~29.8M | 3.8  | 0.30 |

Switching between variants is a single config field change (`config.variant = "pico-s"`).

**Scaling rationale**: Follows ConvNeXt V2's width multiplication pattern (stage channels roughly double each stage) but with:
- Wider channels relative to stock ConvNeXt V2 (1 input channel vs 3 → need more capacity to compensate)
- Stage 3 (N₃) gets the most blocks — matches the ImageNet ConvNeXt pattern where the 3rd stage is deepest
- DropPath rate scales with model size to control overfitting on ESC-50's 2000-clip dataset
- Rectangular (7,5) kernel in Stage 1 across ALL variants (hypothesis: benefit is consistent)

---

## 8. Inductive Bias Justifications

Every non-standard design choice is stated as a single-sentence justification with its evidence status.

| # | Decision | Justification | Status | If wrong |
|---|---|---|---|---|
| **D1** | **Two-layer frequency-preserving stem** (vs 4×4 patchify) | Frequency-axis information density is higher than time-axis for environmental sound identification; delaying frequency downsampling preserves discriminative timbral information. | **Hypothesis** (grounded in spectrogram theory; novel at this scale) | Ablate: swap to 4×4 patchify stem → performance should drop if D1 is correct |
| **D2** | **Rectangular (7,5) depthwise kernel in Stage 1** (vs 7×7 square) | A 7×5 kernel matches the 128:86 ≈ 1.49:1 aspect ratio of the input spectrogram, giving a wider receptive field along the frequency axis where discriminative patterns (harmonic structure, formants) reside. | **Hypothesis** (supported by Chang et al. 2026 at 2M scale; unverified at 5M–30M) | Ablate: swap to (7,7) → if < 0.2 pp drop, rectangular benefit doesn't scale |
| **D3** | **GRN in every block** | GRN prevents feature collapse by normalising each channel's global response by the mean across channels, enhancing inter-channel competition without extra parameters. | **Hypothesis** (grounded on ImageNet; spectrogram domain transfer is TODO: unverified) | Ablate: `use_grn=False` → if < 0.3 pp drop, GRN doesn't help on spectrograms |
| **D4** | **Pre-norm (LayerNorm before each sublayer)** | Pre-norm stabilises training at scale by keeping activations in a well-conditioned range before nonlinearities; standard practice since GPT-2 / ConvNeXt. | **Grounded** (empirically established) | — |
| **D5** | **Depthwise conv before MLP** (ConvNeXt V2 order: dconv → norm → MLP) | This order lets the depthwise conv operate on un-normalised features (richer signal) while the following LayerNorm resets statistics before the MLP. Opposite to the original ConvNeXt (norm → dconv → norm → MLP). | **Grounded** (ConvNeXt V2 architecture) | — |
| **D6** | **Inverted bottleneck (4× expansion)** | Expanding channels before the gating MLP increases representational capacity per parameter; 4× is the standard across modern ConvNets and ViTs. | **Grounded** (empirically established) | — |
| **D7** | **(5,5) kernel in Stage 4** | At Stage 4 the spatial resolution is only 8×6; a 7×7 kernel covers most of the feature map, so (5,5) reduces redundant receptive field while maintaining local structure. | **Grounded** (spatial dimension argument) | — |
| **D8** | **Stochastic depth (DropPath) scaling linearly with model size** | Larger models on the small ESC-50 dataset (2000 clips) are more prone to overfitting; increasing DropPath from 0.05 (Atto-S) to 0.30 (Tiny-S) provides commensurate regularisation. | **Hypothesis** (domain-specific; ConvNeXt V2 uses fixed 0.2 for all ImageNet variants) | If Tiny-S overfits despite 0.3 DPR → increase further or add more regularisation |
| **D9** | **Four-stage pyramid with progressive 2× downsampling** | The 4-stage design creates a multi-scale feature hierarchy (local → mid → global), which is well-suited to environmental sounds that contain both fine-grained texture (e.g. cricket chirps) and coarse structure (e.g. engine rumble). | **Grounded** (foundational CNN design) | — |
| **D10** | **Global Average Pooling head (no attention pooling)** | GAP is parameter-free, permutation-invariant over spatial positions, and prevents overfitting from excessive head parameters on 2000-clip ESC-50. | **Grounded** (standard practice) | — |

---

## 9. Research-to-Architecture Traceability

| Research contract item | Architecture decision | Evidence status | Validation hook |
|---|---|---|---|
| **C1**: No systematic CNN family scaling on ESC-50 | Five cleanly scaled variants (Atto-S → Tiny-S) with controlled width/depth multipliers | **Grounded** | Verify parameter counts match ±5% of targets (count via `model.parameters()`) |
| **C2**: Rectangular kernels benefit at all scales | (7,5) kernel in Stage 1; square in Stages 2-4 | **Hypothesis** | Ablation: swap Stage 1 to (7,7) at all 5 scales; measure Δ accuracy |
| **C3**: GRN transfers to spectrograms | GRN in every block; learnable γ, β per channel | **Hypothesis** | Ablation: `use_grn=False` at 3 scales (5M, 15M, 30M) |
| **C4**: Pure CNN matches attention methods | No attention mechanism in any variant | **Hypothesis** | Compare to ITFA-DNN, CRNN+CoordAtt baselines at equiv. param counts |
| **Baseline**: Unmodified ConvNeXt V2 Femto (4×4 patchify) | `stem_type="patchify"` config option for ablation | **Grounded** | Identical training pipeline, swap only stem |
| **Baseline**: ConvNeXt V2 Femto with spectrogram stem + square kernels | Swap Stage 1 kernel to (7,7) via `stage_kernel_sizes` override | **Grounded** | Single config field change |
| **Eval**: 5-fold cross-validation, top-1 accuracy ± std | Head outputs 50-class logits; training loop implements 5-fold CV | **Grounded** | `sklearn.model_selection.StratifiedKFold` wrapper |
| **Eval**: Ablate SpecAugment | `AugmentationConfig.spec_augment_num_masks=0` | **Grounded** | Toggle via config |
| **Blocking unknown**: Overfitting beyond 20M | DropPath scales with variant size; label smoothing = 0.1; weight decay = 0.05 | **Hypothesis** | Monitor train/val gap; if val accuracy plateaus/decreases at >20M, overfitting confirmed |
| **Blocking unknown**: Training recipe transfer | AdamW, cosine LR, EMA — identical to ConvNeXt V2 recipe | **Hypothesis** | Learning curve divergence → re-optimise LR, WD, schedule |
| **Falsification**: 10M < 90% | Femto-S is the 10M variant | **Hypothesis** | If Femto-S < 90% on 5-fold CV → design is fundamentally flawed |

---

## 10. Domain-Specific Considerations

### 10.1 CV Domain (Primary)

| Concern | How addressed |
|---|---|
| **Spatial handling** | Fully convolutional — works at any input resolution. The 128×86 input and progressive 2× downsampling produce feature maps of [64×43, 32×22, 16×11, 8×6]. |
| **Multi-scale patterns** | 4-stage pyramid captures local texture (Stage 1), mid-level patterns (Stages 2-3), and global structure (Stage 4). Environmental sounds benefit from this hierarchy. |
| **Dense vs. global** | Purely convolutional — no attention. Receptive field grows through depth and downsampling. Effective RF after Stage 4 is larger than the 8×6 spatial map, giving effectively global coverage. |
| **Translation invariance** | Depthwise + pointwise convs are translation-equivariant; GAP head makes the classifier translation-invariant, which is appropriate for spectrograms (a sound shifted in time is the same class). |
| **Input resolution variation** | No position encoding; the model adapts to any spectrogram size. Random crop augmentation (128×80 → 128×86) adds resolution robustness. |

### 10.2 Audio/Spectrogram Domain

| Concern | How addressed |
|---|---|
| **Frequency vs. time asymmetry** | Rectangular (7,5) kernel in Stage 1 and asymmetric stride stem ((1,2) then (2,1)) explicitly model the different semantics of the two axes. Standard CNNs with square kernels treat them identically. |
| **Mel-scale frequency axis** | The 128 mel bands have non-linear frequency spacing (more resolution at low frequencies). The CNN learns spatial patterns on this warped grid; depthwise convs can learn axis-specific filters. |
| **Local acoustic vs. global structure** | Local patterns (e.g. insect stridulation, bird calls in the 2-8 kHz range) are captured by early-stage depthwise kernels. Global structure (e.g. engine noise, temporal progression) emerges in deeper stages. |
| **Causality (not required)** | ESC-50 is a clip-level classification task — no streaming/real-time requirement. Bidirectional (non-causal) convolutions are used throughout. |
| **Input representation** | Log-mel spectrogram (not raw waveform, not codec tokens). The 128×86 resolution balances frequency detail (21.5 Hz/bin) with time context (11.6 ms/frame). This is the standard ESC-50 representation. |

### 10.3 Overfitting Considerations (ESC-50 Small Data)

| Risk | Mitigation |
|---|---|
| **2000 clips → large models may memorise** | DropPath (0.05→0.30), label smoothing (0.1), Mixup (α=0.2), SpecAugment, weight decay (0.05), EMA (0.9998), 300 epochs with cosine decay |
| **5-fold CV variance** | Report mean ± std over 5 folds; use McNemar's test for statistical significance |
| **Per-class imbalance** | ESC-50 is perfectly balanced (40 clips/class, 50 classes); stratified 5-fold preserves balance |

---

## 11. Implementation Risk Flags

### Risk 1: Numerical instability in GRN

**Severity**: Low
**Description**: GRN computes `torch.norm(x, p=2, dim=(2,3))` over potentially large spatial maps. If activations become very large (e.g. after GELU), the L2 norm may overflow in bfloat16.
**Mitigation**:
- Use float32 for GRN computation (cast input, cast back)
- Clamp the denominator: `nx = gx / (gx.mean(dim=1, keepdim=True) + eps)` already has `eps=1e-6`
- Monitor for NaN in early training

### Risk 2: Depthwise conv memory bandwidth bottleneck

**Severity**: Medium
**Description**: Depthwise convolutions are memory-bandwidth-bound (low arithmetic intensity). On A100/H100 this can become the training bottleneck, especially with (7,7) kernels. Throughput may be lower than parameter count suggests.
**Mitigation**:
- Use `torch.compile` (PT2) which fuses depthwise conv kernels
- Enable cuDNN backend (`torch.backends.cudnn.benchmark=True`)
- Consider grouped conv + fused kernels if throughput is critical
- Monitor GPU kernel utilisation with `nvidia-smi` / NSight

### Risk 3: Overfitting on Tiny-S (30M, 2000 clips)

**Severity**: Medium-High
**Description**: ESC-50 has only 2000 2-second clips. A 30M-parameter model has 15,000× more parameters than training examples. Despite DropPath (0.3), label smoothing, and Mixup, Tiny-S may memorise the training set and fail to generalise.
**Mitigation**:
- If train/val gap > 5 pp, increase DropPath further (0.4-0.5)
- Add Auxiliary Losses (e.g. self-distillation as in Chang et al. 2026)
- Reduce epoch count from 300 to 200 (earlier stopping)
- If overfitting is confirmed → the 30M variant may be impractical for ESC-50; report as a finding
- **Falsification hook**: if Tiny-S (30M) is within 0.3 pp of Nano-S (20M), or fails the ≥0.5 pp improvement over 20M, the claim of monotonic scaling to 30M is falsified

### Risk 4: Rectangular kernel performance cliff with torch.compile

**Severity**: Low-Medium
**Description**: Non-square depthwise kernels may have less optimised kernels in cuDNN/cuLASS than square kernels, potentially causing runtime errors or severe performance degradation under `torch.compile`.
**Mitigation**:
- Test with and without `torch.compile` in early development
- Fall back to (7,7) if (7,5) causes compilation issues — this is acceptable as an ablation point
- Consider padding (7,5) input to (7,7) and masking if needed

---

## 12. Suggested Ablations

Ablations ordered by "turn this off first if the model doesn't work."

### Ablation A: Rectangular → Square Stage 1 Kernel

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| Stage 1 kernel | `stage_kernel_sizes[0]` | `(7, 5)` | `(7, 7)` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| All 5 scales | Rectangular kernels benefit at all scales (C2) | ≥ 0.5 pp improvement over square | Novelty claim weakened; square kernels may be sufficient |
| **Interpretation if ablated value wins**: The spectrogram aspect ratio does not justify asymmetric kernels; Stage 1 can use (7,7). Route feedback to **ml-research**. |

### Ablation B: GRN Off

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| Use GRN | `use_grn` | `True` | `False` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| Atto-S, Pico-S, Tiny-S (3 scales) | GRN transfers to spectrograms (C3) | ≥ 0.3 pp improvement with GRN | GRN adds complexity without benefit; remove from design. Route feedback to **ml-architect**. |
| **Interpretation if ablated value wins**: GRN benefit does not transfer from ImageNet to spectrograms. Commensurate with ConvNeXt V2's co-design with FCMAE (which we are not using). |

### Ablation C: Two-Layer Stem → Patchify Stem

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| Stem type | `stem_type` | `"spectrogram"` | `"patchify"` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| Femto-S | Frequency preservation helps (D1) | ≥ 0.5 pp improvement over 4×4 patchify | Aggressive initial downsampling is acceptable; spectrogram adaptation unnecessary. Route feedback to **ml-architect**. |
| **Interpretation if ablated value wins**: The spectrogram stem is unnecessary; the original ConvNeXt V2 patchify stem works equally well on spectrograms. |

### Ablation D: SpecAugment Off

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| SpecAugment num masks | `augmentation.spec_augment_num_masks` | `2` | `0` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| Femto-S | SpecAugment helps on ESC-50 | ≥ 1.0 pp improvement | This is unlikely — SpecAugment is well-established. If it fails, check data pipeline implementation. Route feedback to **ml-coder**. |

### Ablation E: DropPath Rate Halved/Doubled

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| Drop path rate | `drop_path_rate` | per variant (0.05–0.3) | `0.0` or `0.5` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| All scales | DropPath prevents overfitting at larger scales | 0.0 DPR → overfitting (train/val gap > 5 pp); 0.5 DPR → underfitting (train accuracy < 85%) | Stochastic depth not needed for this dataset. Route feedback to **ml-architect**. |
| **Interpretation**: Tiny-S with DPR 0.0 should show largest train/val gap; Atto-S should be robust to DPR changes. |

### Ablation F: Stage 4 Kernel 5×5 → 7×7

| Field | Config key | Baseline value | Ablated value |
|---|---|---|---|
| Stage 4 kernel | `stage_kernel_sizes[3]` | `(5, 5)` | `(7, 7)` |

| Variant | Hypothesis tested | Expected Δ | If fails |
|---|---|---|---|
| Femto-S | 5×5 is sufficient when spatial dims are 8×6 | < 0.1 pp difference | A 7×7 kernel in Stage 4 provides meaningful additional RF. Not a critical finding — keep 7×7 if it helps. |

### Ablation Summary Table

| # | Ablation name | Config field | Baseline | Ablated | Hypothesis tested | Expected metric Δ | Failure → route to |
|---|---|---|---|---|---|---|---|
| A | Rect → Square S1 | `stage_kernel_sizes[0]` | `(7,5)` | `(7,7)` | C2: rectangular benefits at all scales | ↓ ≥ 0.5 pp | ml-research |
| B | GRN off | `use_grn` | `True` | `False` | C3: GRN transfers to spectrograms | ↓ ≥ 0.3 pp | ml-architect |
| C | Patchify stem | `stem_type` | `"spectrogram"` | `"patchify"` | D1: frequency preservation helps | ↓ ≥ 0.5 pp | ml-architect |
| D | No SpecAugment | `aug.spec_augment_num_masks` | `2` | `0` | SpecAugment helps | ↓ ≥ 1.0 pp | ml-coder |
| E | DropPath halved | `drop_path_rate` | variant-default | `0.0` | DPR prevents overfitting | train/val gap ↑ | ml-architect |
| F | Stage 4 7×7 | `stage_kernel_sizes[3]` | `(5,5)` | `(7,7)` | 5×5 sufficient at 8×6 | < 0.1 pp diff | ml-architect |

---

## 13. Input/Output Contract

### Forward Pass Contract

```
Input:  Tensor of shape (B, 1, 128, 86)
        dtype: float32
        values: log-mel spectrogram, per-sample normalised (μ=0, σ=1)

Output: Tensor of shape (B, 50)
        dtype: float32 (training) / float32 (inference)
        values: class logits (NOT probabilities — Softmax applied by loss function)

Parameter ranges:
  5M variant:     ~4.8M params,  ~0.45 GFLOPs per forward
  10M variant:    ~9.5M params,  ~0.85 GFLOPs per forward
  15M variant:   ~15.2M params,  ~1.5  GFLOPs per forward
  20M variant:   ~20.5M params,  ~2.4  GFLOPs per forward
  30M variant:   ~29.8M params,  ~3.8  GFLOPs per forward
```

### Configuration API

```python
# Minimal usage — variant auto-configures all parameters
config = SpectroConvNeXtConfig(variant="femto-s")

# Variant convenience constructors
config = femto_s()
config = pico_s(drop_path_rate=0.2)  # override individual fields

# Ablation: swap kernel in Stage 1
config = femto_s(stage_kernel_sizes=[(7,7), (7,7), (7,7), (5,5)])

# Ablation: remove GRN
config = femto_s(use_grn=False)

# Ablation: patchify stem
config = femto_s(stem_type="patchify")
```

---

## 14. Baseline Requirements Carried Forward

The following baselines MUST be reproduced with identical training pipeline, augmentation, and optimiser hyperparameters. The SpectroConvNeXt model family is compared against each.

| # | Baseline | Reference | Params | Expected ESC-50 | Notes |
|---|---|---|---|---|---|
| B1 | **ConvNeXt V2 Femto (stock)** — 4×4 patchify stem, square kernels, no GRN ablation | ConvNeXt V2 config adapted for spectrograms | ~5M | ~87–89% (estimate) | Tests adaptation benefit |
| B2 | **ConvNeXt V2 Femto (spectrogram stem, square kernels)** — same spectrogram stem but (7,7) in Stage 1 | This work, ablated | ~5M | ~88–90% (estimate) | Tests rectangular kernel benefit |
| B3 | **ITFA-DNN** | Chen & Peng 2025 | ~2M | 94.2% (published) | Current SOTA; smaller than all our variants |
| B4 | **MobileNetV2 + SPA** | SoundMLR 2025 | ~3.5M | 91.75% (published) | Lightweight baseline |
| B5 | **ResNet-50** | He et al. 2016 | ~25M | ~88–90% (ESC-50 literature) | Most common ESC-50 CNN baseline |
| B6 | **Lightweight CRNN + CoordAtt** | 2026 | ~1.5M | 93.7% (published) | Recurrent + attention baseline |
| B7 | **SpectroConvNeXt Femto-S** (self-baseline — this work) | This design | ~10M | — | Comparison point for scaling curve |

**Validation requirement**: Each baseline must use the same data pipeline, augmentation, optimiser (AdamW, lr=1e-3, cos decay), and number of epochs (300). Differences in accuracy must be attributable to architecture, not training recipe.

---

## 15. Evaluation Requirements Carried Forward

| Requirement | Implementation |
|---|---|
| **Dataset** | ESC-50, 2000 clips, 50 classes, 40 clips/class |
| **Protocol** | 5-fold cross-validation (stratified by class) |
| **Primary metric** | Top-1 accuracy (mean ± std over 5 folds) |
| **Secondary metrics** | Per-class accuracy, confusion matrix, parameter count, inference throughput (samples/sec on A100) |
| **Statistical significance** | McNemar's test between best SpectroConvNeXt variant and each baseline (α = 0.05) |
| **Ablation A** | Rectangular (7,5) vs square (7,7) Stage 1 kernel at ALL 5 scales |
| **Ablation B** | GRN on vs off at Atto-S, Pico-S, Tiny-S |
| **Ablation C** | Two-layer spectrogram stem vs 4×4 patchify stem at Femto-S |
| **Ablation D** | With and without SpecAugment at Femto-S |
| **Throughput profiling** | Forward pass latency (batch=1, batch=128) on A100 with torch.compile |

---

## Output Checklist

- [x] **Domain identified**: CV (primary) + Speech/Audio (primary)
- [x] **Upstream research lifecycle contract read**: yes — `research/spectroconvnext_research_synthesis.md`
- [x] **ModelConfig dataclass with all hyperparameters**: `spectroconvnext_config.py`
- [x] **Pseudocode for the novel block**: Section 5 (stem, downsample, SpectroConvNeXt block, GRN, forward pass)
- [x] **ASCII architecture diagram**: Section 6 (full model + block detail)
- [x] **Inductive bias justification**: Section 8 (10 decisions, each with evidence status)
- [x] **Research-to-architecture traceability table**: Section 9
- [x] **Claims labeled as `grounded`, `hypothesis`, or `TODO: unverified`**: Throughout
- [x] **Domain-specific considerations**: Section 10 (CV + Audio)
- [x] **Implementation risk flags**: Section 11 (4 risks with mitigations)
- [x] **Baseline and evaluation requirements carried forward to validator**: Sections 14–15
- [x] **Suggested ablations**: Section 12 (6 ablations, each = single ModelConfig field change)
