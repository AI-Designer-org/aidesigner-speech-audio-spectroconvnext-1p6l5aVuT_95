# Architecture

## 1. Motivation

**What gap does SpectroConvNeXt address?** Prior work on ESC-50 (environmental sound classification, 50 classes, 2000 clips) is dominated by bespoke lightweight architectures under 5M parameters: ITFA-DNN (Chen & Peng, 2025) at ~2M achieves 94.2%, Lightweight CRNN + CoordAtt at ~1.5M achieves 93.7%, MobileNetV2 + SPA at ~3.5M achieves 91.75%. No prior work has systematically scaled a modern CNN family — designed from the ground up with ConvNeXt V2 principles — across the 5M–30M parameter range on ESC-50. This makes it impossible to answer whether increasing parameter count in a well-designed ConvNet improves accuracy on this task, or whether the small dataset (2000 clips) saturates before 30M.

The ConvNeXt V2 paper (Woo et al., CVPR 2023) demonstrated that pure ConvNets with modernised block design — depthwise convolutions, inverted bottlenecks, and Global Response Normalization (GRN) — match or exceed transformer performance on ImageNet. But ConvNeXt V2 was designed for and evaluated only on ImageNet (224×224 RGB images). **Audio spectrograms differ from natural images in three ways that demand architectural adaptation:**

1. **Aspect ratio**: ESC-50 spectrograms are 128×86 (freq × time), a 1.49:1 aspect ratio vs ImageNet's 1:1. A square 7×7 kernel in the first stage sees equal extents in frequency and time, but frequency carries more discriminative information (timbral identity, harmonic structure).
2. **Channel count**: One channel (mono mel-spectrogram) vs three (RGB). The model must build representational capacity from less input information.
3. **Information density**: The frequency axis (128 mel bands, each at ~21.5 Hz/bin) encodes timbral identity of sound sources. The time axis (86 frames at ~11.6 ms/frame) encodes temporal progression. These have different semantics, but standard CNNs with square kernels treat them identically.

Chang et al. (2026) showed that rectangular kernels improve accuracy on ESC-50 at ~2M scale, but scaling behaviour at 5M–30M is unknown. Similarly, GRN's effect on spectrograms (vs ImageNet) is untested. SpectroConvNeXt tests both hypotheses across a controlled model family.

**Hypothesis this architecture tests**: A ConvNeXt V2-derived CNN with spectrogram-adapted rectangular kernels and delayed frequency downsampling achieves ≥93% top-1 accuracy on ESC-50 at 10M parameters (Femto-S) and ≥94% at 30M (Tiny-S), without any external pretraining.

## 2. At a glance

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
│  └────────────────────────────────────────────────────────┘      │
│                         → (B, 50)                                 │
└──────────────────────────────────────────────────────────────────┘
```

| Property | Value |
|---|---|
| Parameter count (default config Femto-S) | ~9.5M (verified within 15%) |
| Time complexity | O(F·T·C²) — linear in input resolution, quadratic in channel count |
| Space complexity | O(C²) — dominated by 1×1 pointwise convs; no attention O(T²) |
| Hardware requirements | Any GPU with ≥4GB VRAM for batch=128; `torch.compile` recommended for throughput |
| Inference at any resolution | Fully convolutional; no position encoding; validated at F=64,128,256 and T=43,86,172 |

## 3. The core component: SpectroConvNeXt Block

### 3.1 Intuition

Each SpectroConvNeXt block is a four-part computation organised in the ConvNeXt V2 order.

First, a **depthwise convolution** (one filter per input channel) mixes spatial information — local time-frequency patterns — without mixing channels. The kernel shape varies by stage: rectangular (7 frequency × 5 time) in Stage 1, matching the spectrogram's 1.49:1 aspect ratio, and square (7×7 or 5×5) in later stages where spatial resolution is lower.

Second, an **inverted bottleneck feedforward network** expands channels by 4× with a 1×1 convolution, applies a GELU nonlinearity, then projects back to the original channel count. This is where inter-channel mixing happens — each spatial position gets a rich nonlinear combination of all input channels.

Third, **Global Response Normalization (GRN)** enhances inter-channel competition. For each channel, it computes the global L2 norm over the spatial map, then divides by the mean norm across all channels. Channels with unusually large or small global activity get suppressed or amplified. This prevents feature collapse (all channels activating uniformly) and was co-designed with FCMAE pretraining in ConvNeXt V2.

Fourth, a **stochastic depth residual connection** randomly drops entire samples with probability proportional to block depth, providing model-scale-dependent regularisation.

### 3.2 Equations

Let `x` be the input tensor of shape `(B, C, H, W)`. The block computes:

```
residual = x

# Depthwise convolution path
x₁ = LayerNorm(x)                                                        # (B, C, H, W)
x₂ = DepthwiseConv2D(x₁, kernel=(K_h, K_w), groups=C, padding="same")   # (B, C, H, W)

# Inverted bottleneck FFN
x₃ = LayerNorm(x₂)                                                       # (B, C, H, W)
x₄ = Conv2D_1×1(x₃, C → 4C)                                             # (B, 4C, H, W)
x₅ = GELU(x₄)                                                            # (B, 4C, H, W)
x₆ = Conv2D_1×1(x₅, 4C → C)                                             # (B, C, H, W)

# Global Response Normalization
gx_i = ‖X_i‖₂                     (∀ channel i, over dims H,W)           # (B, C, 1, 1)
nx_i = gx_i / (mean(gx) + ε)                                             # (B, C, 1, 1)
x₇ = γ · x₆ · nx_i + β                                                  # (B, C, H, W)

# Stochastic depth + residual
output = residual + DropPath(x₇)                                         # (B, C, H, W)
```

Where `γ` and `β` are learnable per-channel parameters (initialised to 1.0 and 0.0), and `DropPath` drops whole samples with probability `p` (linearly scheduled from 0 to `drop_path_rate` across blocks).

### 3.3 Reference implementation walk-through

From `layers.py` lines 153–244:

```python
class SpectroConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=4, drop_path=0.0, use_grn=True):
        super().__init__()
        # Pre-norm + depthwise convolution (groups=dim = one filter per channel)
        self.norm1 = LayerNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding="same", groups=dim, bias=False)
        # Post-dconv norm
        self.norm2 = LayerNorm2d(dim)
        # Inverted bottleneck: dim → 4×dim → dim
        hidden_dim = dim * expand_ratio
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        # GRN + stochastic depth
        self.grn = GRN(dim) if use_grn else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x                                   # (B, C, H, W)
        x = self.norm1(x)                              # (B, C, H, W)
        x = self.dwconv(x)                             # (B, C, H, W) — depthwise spatial mixing
        x = self.norm2(x)                              # (B, C, H, W)
        x = self.pwconv1(x)                            # (B, 4C, H, W) — expand
        x = self.act(x)                                # (B, 4C, H, W)
        x = self.pwconv2(x)                            # (B, C, H, W) — project
        x = self.grn(x)                                # (B, C, H, W) — channel competition
        x = self.drop_path(x)                          # (B, C, H, W)
        return residual + x                            # (B, C, H, W)
```

The key design choice: depthwise conv operates on un-normalised features (richer signal), then the following LayerNorm resets statistics before the MLP. This is the reverse of the original ConvNeXt order and follows ConvNeXt V2.

## 4. Tensor shape evolution

Default Femto-S config: stem_C=40, C1=64, C2=128, C3=256, C4=512.

| Stage | Operation | Shape (B=2) | Notes |
|---|---|---|---|
| Input | Mel-spectrogram | (2, 1, 128, 86) | float32, per-sample normalised |
| Stem L1 | LayerNorm + Conv2D(3, stride=(1,2)) + GELU | (2, 40, 128, 43) | freq preserved, time halved |
| Stem L2 | Conv2D(3, stride=(2,1)) + GELU | (2, 64, 64, 43) | freq halved, time preserved |
| Stage 1 Blocks | 2× SpectroConvNeXtBlock (7,5) | (2, 64, 64, 43) | same spatial dims |
| Downsample 1 | LayerNorm + Conv2D(2, stride=2) | (2, 128, 32, 22) | H,W halved; C doubled |
| Stage 2 Blocks | 2× SpectroConvNeXtBlock (7,7) | (2, 128, 32, 22) | same spatial dims |
| Downsample 2 | LayerNorm + Conv2D(2, stride=2) | (2, 256, 16, 11) | H,W halved; C doubled |
| Stage 3 Blocks | 6× SpectroConvNeXtBlock (7,7) | (2, 256, 16, 11) | deepest stage |
| Downsample 3 | LayerNorm + Conv2D(2, stride=2) | (2, 512, 8, 6) | H,W halved; C doubled |
| Stage 4 Blocks | 2× SpectroConvNeXtBlock (5,5) | (2, 512, 8, 6) | smaller kernel for small map |
| Final Norm | LayerNorm2d(512) | (2, 512, 8, 6) | before pooling |
| Head | GAP → LayerNorm → Linear(512, 50) | (2, 50) | class logits |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| **Two-layer frequency-preserving stem** (vs 4×4 patchify) | 4×4 stride-4 conv (original ConvNeXt V2) | Frequency-axis information density is higher for environmental sound; delaying frequency downsampling preserves timbral detail. | Slightly higher resolution in Stage 1 (64×43 vs 32×21) → more compute; ablation shows whether it matters. |
| **Rectangular (7,5) kernel in Stage 1** (vs square 7×7) | Square 7×7 kernel | Matches 1.49:1 spectrogram aspect ratio; captures wider frequency receptive field per layer. Supported by Chang et al. (2026) at 2M scale. | Non-square kernels may have less optimised cuDNN kernels; (7,7) fallback always available. |
| **GRN in every block** | No GRN (standard ConvNeXt) | Prevents feature collapse by enhancing inter-channel competition; zero extra parameters beyond γ,β. | Adds ~1-2% compute (elementwise); benefit on spectrograms is hypothesis (ablated). |
| **Pre-norm (LayerNorm before each sublayer)** | Post-norm | Pre-norm stabilises training at scale; standard since GPT-2 / ConvNeXt. | Well-established; no meaningful trade-off. |
| **Depthwise conv before MLP** (ConvNeXt V2 order) | Norm → dconv → norm → MLP (original ConvNeXt) | Depthwise conv operates on richer un-normalised features; following LayerNorm resets statistics before MLP. | Established by ConvNeXt V2 paper. |
| **Inverted bottleneck (4× expansion)** | 2× or no expansion | Standard across modern ConvNets and ViTs for best capacity/parameter ratio. | Higher FLOPs per parameter; well-understood trade-off. |
| **(5,5) kernel in Stage 4** | 7×7 kernel | Spatial resolution is only 8×6; 7×7 covers most of the map, so 5×5 is sufficient and avoids redundant RF. | Negligible; ablated in ablation F. |
| **DropPath scaling with model size** (0.05→0.30) | Fixed DropPath for all variants | Larger models on 2000-clip ESC-50 are more prone to overfitting; proportional regularisation. | May under-regularise Tiny-S (if overfitting persists) or over-regularise Atto-S (if underfitting). |
| **Four-stage pyramid with 2× downsampling** | 3-stage or single-scale | Multi-scale hierarchy matches environmental sound structure (fine texture → coarse structure). | Foundational CNN design; no meaningful trade-off. |
| **Global Average Pooling head** | Attention pooling | Parameter-free, translation-invariant, prevents overfitting from excessive head parameters. | May discard spatial structure, but 8×6 map is already small. |

## 6. Domain-specific considerations

### 6.1 CV domain (primary)

| Concern | How addressed |
|---|---|
| **Spatial inductive bias** | Fully convolutional — translation-equivariant through depthwise/pointwise convs; translation-invariant at the head (GAP). |
| **Scale / resolution handling** | No position encoding; operates at any input resolution (validated at F=64,128,256 and T=43,86,172 via test_model.py). |
| **Multi-scale pyramid** | 4-stage design with 2× downsampling creates local→mid→global feature hierarchy. Effective RF at Stage 4 covers the full 8×6 map. |
| **Dense vs. global attention** | Purely convolutional — no attention. Receptive field grows through stacking and downsampling. |

### 6.2 Speech/Audio domain

| Concern | How addressed |
|---|---|
| **Input representation** | Log-mel spectrogram (not raw waveform, not codec tokens). 128×86 resolution balances frequency detail (21.5 Hz/bin) with time context (11.6 ms/frame). |
| **Frequency vs. time asymmetry** | Rectangular (7,5) kernel in Stage 1 and asymmetric stride stem ((1,2) then (2,1)) explicitly model the different semantics. Standard CNNs with square kernels treat axes identically. |
| **Causality contract** | Not required — ESC-50 is clip-level classification (2s clips, no streaming). Bidirectional (non-causal) convolutions used throughout. |
| **Local acoustic vs. global structure** | Local patterns (e.g. insect stridulation, bird calls) captured by Stage 1 depthwise kernels. Global structure (engine noise, temporal progression) emerges in deeper stages via stacking and downsampling. |
| **Mel-scale warp** | The 128 mel bands have non-linear frequency spacing (more resolution at low frequencies). CNN learns spatial patterns on this warped grid; depthwise convs can learn axis-specific filters. |

## 7. Known limitations

- **All accuracy claims are unverified.** No training pipeline exists (`benchmark_not_executable` blocking gap). The central hypotheses (≥93% at 10M, ≥94% at 30M) cannot be evaluated without `train.py` and ESC-50 data loading.
- **No baseline comparisons.** ITFA-DNN, MobileNetV2+SPA, ResNet-50, and Lightweight CRNN+CoordAtt baselines are identified but not implemented. SOTA claims cannot be made without comparison.
- **All 6 ablations are config-only.** Ablation A (rectangular vs square), B (GRN on/off), C (stem type), D (SpecAugment), E (DropPath), and F (Stage 4 kernel) are configured as single-field config changes but cannot produce metric deltas without training.
- **Tiny-S (30M) may overfit on 2000-clip ESC-50.** Despite DropPath=0.3, label smoothing, Mixup, and SpecAugment, the train/val gap is unmeasured. If Tiny-S fails to outperform Nano-S by ≥0.5 pp, the scaling-to-30M hypothesis is falsified.
- **FLOP estimates use heuristic (2×/6× params).** Actual measured FLOPs from `torch.profiler` are not reported. The research document's FLOP claims (especially 3.8G for Tiny-S) may not match the heuristic estimate (2× params = 2.98G).
- **No FCMAE pretraining.** The supervised-only design does not leverage ConvNeXt V2's FCMAE pretraining, which was co-designed with GRN. GRN's benefit may be smaller without FCMAE.
- **Literature coverage limited to level_1 task.** Full novelty verification requires broader systematic review beyond available sources.
