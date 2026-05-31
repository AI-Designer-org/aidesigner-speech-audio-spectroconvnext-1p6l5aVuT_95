# SpectroConvNeXt: Modern CNN Family for Audio Spectrogram Classification

## Domain Identification

| Domain | Relevance |
|---|---|
| **Computer Vision (CV)** | Primary — spectrograms as 2D images; CNN architecture design borrows from vision |
| **Speech / Audio** | Primary — task is environmental sound classification; input is audio-derived |
| **GenAI (Self-Supervised)** | Secondary — FCMAE/self-supervised pretraining could apply |

---

## 1. Landscape Summary

### 1.1 ESC-50 Benchmark State of the Art (2025–2026)

| Method | Accuracy | Params | Year | Key Innovation |
|--------|----------|--------|------|----------------|
| **ITFA-DNN** (Chen & Peng) | **94.2%** | ~2M (lightweight) | 2025 | Two-stream interactive time-frequency attention + depthwise separable conv |
| **Lightweight CRNN + CoordAtt** | **93.7%** | ~1.5M | 2026 | Bi-GRU + coordinate attention, 28% fewer params |
| **MobileNetV2 + SPA + SoundMLR** | **91.75%** | ~3.5M | 2025 | Metric learning hybrid loss + Spectral Pooling Attention |
| **Rectangular Kernels + Self-Distill** | **89.3%** | ~2M | 2026 | Rectangular conv + dilated conv + self-distilled soft labels |
| **SSRP-T** (Top-K pooling) | **80.69%** | ~0.5M | 2025 | Sparse Salient Region Pooling |

**Key observations:**
- **ESC-50 ceiling is ~94–95%** with pure supervised learning. Further gains likely require large-scale pretraining (AudioSet) or self-supervision.
- **Attention mechanisms (coordinate, spectral pooling, time-frequency)** are the dominant trend.
- **Lightweight models** dominate the leaderboard — no one has systematically scaled a modern CNN family from 5M to 30M on ESC-50.
- **Feature stacking** (Log-Mel + chroma + spectral contrast + MFCC + GTCC) yields strong results with CNNs and avoids AST's need for AudioSet pretraining.

### 1.2 Modern CNN Design (ConvNeXt V2)

ConvNeXt V2 (Woo et al., 2023) demonstrated that pure ConvNets can match or exceed transformer performance when:
1. **Modernized block design**: 7×7 depthwise conv → LayerNorm → 1×1 conv (4× expansion) → GELU → 1×1 conv (projection) → **GRN** → residual
2. **GRN (Global Response Normalization)**: Prevents feature collapse in self-supervised settings; enhances inter-channel competition without extra parameters
3. **FCMAE**: Fully Convolutional Masked Autoencoder co-designs architecture + pretraining
4. **Model scaling variants** provide a clean family from 3.7M (Atto) to 660M (Huge)

| Variant | Channels | Blocks | Params | ImageNet Top-1 |
|---------|----------|--------|--------|----------------|
| Atto | [40, 80, 160, 320] | [2, 2, 6, 2] | 3.7M | 76.7% |
| Femto | [48, 96, 192, 384] | [2, 2, 6, 2] | 5.2M | 78.5% |
| Pico | [64, 128, 256, 512] | [2, 2, 6, 2] | 9.1M | 80.3% |
| Nano | [80, 160, 320, 640] | [2, 2, 8, 2] | 15.6M | 81.9% |
| Tiny | [96, 192, 384, 768] | [2, 2, 6, 2] | 28.6M | 83.0% |

### 1.3 Audio Spectrogram Preprocessing (ESC-50 Best Practice)

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 22050 Hz | Balances resolution and compute |
| Clip length | 2 seconds (44100 samples) | Standard for ESC-50 |
| n_fft | 1024 | Frequency resolution ~21.5 Hz |
| hop_length | 512 | ~43% overlap, ~86 time frames |
| n_mels | 128 | Standard mel bands |
| Input shape | 1 × 128 × 86 | (channels, freq, time) |
| Normalization | Log-mel (power_to_db) + per-sample mean/std | Standard |
| Augmentation | SpecAugment (time/freq masking), Mixup, random crop | De-facto standard |

---

## 2. Proposed Architecture: SpectroConvNeXt

### 2.1 Design Principles

1. **Spectrogram-native adaptations**
   - Frequency and time axes have *different semantics* — rectangular kernels in early stages
   - Delayed frequency downsampling preserves discriminative timbral information
   - Initial stem designed to preserve frequency resolution

2. **Modern ConvNeXt V2 block as core operator**
   - 7×7 or rectangular depthwise convolution (group=channel)
   - LayerNorm → 1×1 conv (4× expansion) → GELU → 1×1 conv (projection) → GRN
   - GRN prevents feature collapse and enhances channel competition

3. **Family scaling (5M–30M params)**
   - Five variants precisely targeting the 5M–30M range
   - Width multiplier and depth scaling following ConvNeXt V2 patterns

4. **Regularization**
   - Stochastic Depth (DropPath)
   - Label smoothing
   - Mixup + SpecAugment
   - Weight decay scheduling

### 2.2 Architecture Details

#### Stem Design

```
Input: 1 × 128 × 86 (ch × freq × time)

Stem:
  LayerNorm (channel-wise)
  Conv2D(3×3, stride=[1,2], padding=1) → GELU   # freq preserved, time halved → 1 × 128 × 43
  Conv2D(3×3, stride=[2,1], padding=1) → GELU   # freq halved, time preserved → C₁ × 64 × 43
```

This two-layer stem is gentler than ConvNeXt V2's 4×4 stride-4 patchify, preserving frequency resolution in the first layer and time resolution in the second.

#### Downsampling Blocks

Between stages, use:
```
LayerNorm → Conv2D(2×2, stride=2)  # Halves both freq and time
```

#### Stage Configuration

| Stage | Input Size | Downsample | Operation |
|-------|-----------|------------|-----------|
| 1 | C₁ × 64 × 43 | — | N₁ blocks (rectangular 7×5 dconv) |
| 2 | C₂ × 32 × 22 | 2× stride | N₂ blocks (square 7×7 dconv) |
| 3 | C₃ × 16 × 11 | 2× stride | N₃ blocks (square 7×7 dconv) |
| 4 | C₄ × 8 × 6 | 2× stride | N₄ blocks (square 5×5 dconv) |

#### Head

```
Global Average Pooling (over freq × time) → 1 × C₄
LayerNorm → Linear(C₄, 50) → Softmax
```

### 2.3 Model Family Variants

| Variant | Target | Stem C | Stage Channels [C₁–C₄] | Blocks [N₁–N₄] | Expected Params |
|---------|--------|--------|----------------------|-----------------|-----------------|
| Atto-S | ~5M | 32 | [48, 96, 192, 384] | [2, 2, 6, 2] | ~4.8M |
| Femto-S | ~10M | 40 | [64, 128, 256, 512] | [2, 2, 6, 2] | ~9.5M |
| Pico-S | ~15M | 48 | [80, 160, 320, 640] | [2, 2, 8, 2] | ~15.2M |
| Nano-S | ~20M | 56 | [96, 192, 384, 768] | [2, 2, 8, 2] | ~20.5M |
| Tiny-S | ~30M | 64 | [112, 224, 448, 896] | [2, 2, 10, 2] | ~29.8M |

The "-S" suffix denotes **Spectrogram-adapted**. The key differences from stock ConvNeXt V2:
1. Rectangular 7×5 depthwise conv in Stage 1 (vs 7×7)
2. Two-layer frequency-preserving stem (vs 4×4 patchify)
3. Slightly wider channels in final stages to compensate for 1 input channel vs 3
4. 5×5 depthwise conv in Stage 4 (spatial dims are small)

### 2.4 Complexity Properties

| Variant | Fwd FLOPs (per 128×86 input) | Params | Throughput (imgs/s, A100 est.) |
|---------|------|--------|-------------------------------|
| Atto-S | ~0.45G | ~4.8M | ~18,000 |
| Femto-S | ~0.85G | ~9.5M | ~12,000 |
| Pico-S | ~1.5G | ~15.2M | ~8,000 |
| Nano-S | ~2.4G | ~20.5M | ~5,500 |
| Tiny-S | ~3.8G | ~29.8M | ~3,500 |

**Complexity analysis by axis:**

| Axis | Assessment |
|------|-----------|
| **Time complexity** | O(F·T·C²) where F=freq bins, T=time frames, C=channels. Linear in input resolution, quadratic in channel count. |
| **Space complexity** | O(C²) parameters per block (dominated by 1×1 pointwise convs). No attention O(T²) pathology. |
| **Parallelism** | Fully convolutional — embarrassingly parallel across batch. No sequential token mixing. |
| **Hardware fit** | Depthwise convs are memory-bandwidth-bound; pointwise convs are compute-bound. Fused kernels (e.g., `torch.compile`, TensorRT) help significantly. GRN is a lightweight elementwise operation. |
| **Length generalization** | Fully convolutional — no position encoding. Works at any time/freq resolution. Padding mode matters for very small inputs. |

### 2.5 Expressiveness

| Capability | Assessment |
|------------|-----------|
| **Local time-frequency patterns** | Strong — depthwise 7×7 / 7×5 kernels capture local texture |
| **Global frequency patterns** | Moderate — relies on stacking and downsampling. Coordinate attention or global pooling can supplement. |
| **Temporal progression** | Moderate — convs are translation-invariant; no explicit temporal ordering beyond 2D position. |
| **Multi-scale patterns** | Strong — 4-stage pyramid with progressive downsampling |
| **Out-of-distribution robustness** | Moderate — pure CNNs generalize better than ViTs on distribution shift but worse than self-supervised ViTs (DINO). |

---

## 3. Novelty Gaps

### Gap 1: No systematic modern CNN family scaling on ESC-50
- **What**: All top ESC-50 results use ad-hoc architectures (CRNN, MobileNetV2, custom lightweight nets). No one has scaled a ConvNeXt V2 family from 5M→30M on ESC-50.
- **Partially addressed by**: Gong et al. (2021, PSLA) and Chen et al. (2025, ITFA-DNN) use ResNet/CNN backbones but don't provide a scaled family.
- **What remains missing**: Ablation of whether the ConvNeXt V2 design choices (GRN, inverted bottleneck, depthwise conv) transfer to audio spectrograms, and at what scale.

### Gap 2: Rectangular kernels for spectrograms not studied at scale
- **What**: Chang et al. (2026) showed rectangular kernels help on ESC-50, but only at small scale (<2M params). It's unknown whether this benefit persists at 10M–30M.
- **Partially addressed by**: Chang et al. (Pattern Analysis and Applications, 2026).
- **What remains missing**: A controlled study at multiple parameter scales.

### Gap 3: No supervised-only modern ConvNet SOTA on ESC-50
- **What**: Top methods (ITFA-DNN, Lightweight CRNN) use attention mechanisms and recurrent layers, not pure ConvNet blocks. The question "Can a modern ConvNet match attention-based models on ESC-50 without pretraining?" is open.
- **Partially addressed by**: ConvNeXt V2 on ImageNet shows pure ConvNets match ViT — but audio spectrograms are a different domain.
- **What remains missing**: Direct comparison of modern ConvNeXt V2 vs. PSA/AST/CRNN at equivalent parameter counts.

### Gap 4: FCMAE pretraining for audio spectrograms
- **What**: ConvNeXt V2's FCMAE (masked autoencoder) was designed for images. Its applicability to spectrograms is unexplored.
- **Partially addressed by**: MAE-Audio (Baade et al., 2022) applied ViT-MAE to audio; SSAST (Gong et al., 2022) applied self-supervised AST. Neither uses a ConvNet.
- **What remains missing**: Whether FCMAE + GRN co-design transfers to the audio domain, and at what scale.

---

## 4. Recommended Direction

### Hypothesis

> **A ConvNeXt V2-derived CNN family with spectrogram-adapted rectangular kernels in early stages and delayed frequency downsampling achieves ≥93% top-1 accuracy on ESC-50 at 10M parameters (Femto-S variant), and ≥94% at 30M (Tiny-S variant), without any external pretraining (AudioSet, ImageNet).**

### Justification

1. The SOTA ceiling on ESC-50 is ~94.2% (ITFA-DNN, ~2M params). A scaled modern CNN should match or exceed this.
2. ConvNeXt V2's block design (depthwise conv + inverted bottleneck + GRN) is proven on ImageNet. Spectrograms share local texture structure with images, so the inductive bias transfers.
3. Rectangular kernels (Chang et al., 2026) and delayed frequency downsampling exploit the asymmetric information density of spectrograms (frequency axis is more discriminative).
4. The 5M–30M range spans the gap between current lightweight ESC-50 models and full-scale architectures, providing a controlled study.

### Expected Observable Behavior

- Femto-S (~10M): ≥93.0% on ESC-50 (5-fold cross-validation)
- Tiny-S (~30M): ≥94.0% on ESC-50 (5-fold cross-validation)
- Scaling trend: monotonic improvement with parameter count, with diminishing returns past 20M
- Rectangular stage 1 kernels (7×5) outperform square (7×7) by ≥0.5 pp at all scales
- GRN provides ≥0.3 pp gain vs. identical architecture without GRN

### Falsification Condition

The hypothesis is falsified if:
1. The 10M variant achieves <90% on ESC-50 (5-fold CV), OR
2. The 30M variant fails to outperform the 20M variant by ≥0.5 pp, OR
3. Rectangular kernels provide no benefit (<0.2 pp) over square kernels at any scale, OR
4. A simpler baseline (e.g., standard ConvNeXt V2 Femto with 4×4 stem, unmodified) matches or exceeds the adapted variant.

---

## 5. Research Lifecycle Contract

```yaml
task_level: level_1
domain: CV, Speech/Audio
research_question: >
  Does a ConvNeXt V2-derived CNN family with spectrogram-adapted rectangular
  kernels and delayed frequency downsampling achieve SOTA on ESC-50 across
  the 5M–30M parameter range?
novelty_claims:
  - claim: >
      No prior work has systematically scaled a modern ConvNeXt V2 family on
      ESC-50 across the 5M–30M parameter range.
    status: grounded
    evidence:
      - ESC-50 leaderboard methods (ITFA-DNN, CRNN+CoordAtt, MobileNetV2+SPA)
        are all <5M params, bespoke architectures.
      - ConvNeXt V2 paper evaluates on ImageNet only, not audio spectrograms.
  - claim: >
      Rectangular depthwise kernels in early stages provide a consistent
      accuracy benefit across model scales on spectrograms.
    status: hypothesis
    evidence:
      - Chang et al. (2026) show benefit at ~2M scale; scaling behavior unknown.
      - Domain literature supports frequency-axis information density asymmetry.
  - claim: >
      GRN (Global Response Normalization) transfers from ImageNet to audio
      spectrograms, providing a small consistent gain.
    status: hypothesis
    evidence:
      - GRN is co-designed with FCMAE for ImageNet; spectrogram domain transfer
        is "TODO: unverified".
  - claim: >
      A pure CNN (no attention, no recurrence) can match attention-augmented
      methods (ITFA-DNN, CRNN) on ESC-50 at equivalent parameter counts.
    status: hypothesis
    evidence:
      - ConvNeXt V2 matches ViT on ImageNet (indirect evidence).
      - Direct CNN-vs-attention comparison on ESC-50 at multiple scales is absent.
known_related_work:
  - work: ITFA-DNN (Chen & Peng, 2025)
    covers: >
      Two-stream time-frequency attention with depthwise conv; achieves 94.2%
      on ESC-50 with ~2M params.
    leaves_open: >
      Does not scale beyond 2M; relies on explicit attention branches rather
      than pure ConvNet design.
  - work: ConvNeXt V2 (Woo et al., 2023)
    covers: >
      Modern CNN design (GRN, inverted bottleneck, depthwise conv), FCMAE
      pretraining, clean scaling family from 3.7M to 660M.
    leaves_open: >
      ImageNet-only evaluation; no audio/spectrogram domain adaptation;
      rectangular kernels not studied.
  - work: Rectangular Kernels + Self-Distillation (Chang et al., 2026)
    covers: >
      Shows rectangular conv kernels improve ESC-50 accuracy at small scale.
    leaves_open: >
      Limited to ~2M params; only 2-layer CNN, not modern block design;
      no GRN or scaling study.
  - work: Lightweight CRNN + Coordinate Attention (2026)
    covers: >
      Bi-GRU + coord attention achieves 93.7% on ESC-50.
    leaves_open: >
      Uses recurrence (sequential, harder to parallelize); scaling beyond 2M
      not studied.
baseline_requirements:
  - Unmodified ConvNeXt V2 Femto with 4×4 patchify stem (image-domain baseline)
  - Unmodified ConvNeXt V2 Femto with 2-layer spectrogram stem but square 7×7
    kernels (ablates rectangular kernel benefit)
  - ITFA-DNN reproduced at its published ~2M scale
  - MobileNetV2 + SPA (SoundMLR) baseline
  - Standard ResNet-50 baseline (most common ESC-50 CNN baseline)
  - Each baseline trained with identical data pipeline, augmentation, and
    optimizer hyperparameters
evaluation_requirements:
  - Dataset: ESC-50, 5-fold cross-validation (standard protocol)
  - Metric: Top-1 accuracy (mean ± std over 5 folds)
  - Secondary: Per-class accuracy, confusion matrix, parameter count, inference
    throughput (samples/sec on A100)
  - Ablations:
    a. Rectangular (7×5) vs square (7×7) Stage 1 kernels at each scale
    b. GRN on vs off at each scale
    c. Two-layer spectrogram stem vs 4×4 patchify stem at each scale
    d. With and without SpecAugment
  - Statistical significance: McNemar's test between best variant and each
    baseline at α=0.05
blocking_unknowns:
  - Whether the optimal channel/stage allocation for ImageNet transfers to
    audio spectrograms (different resolution, single channel, different aspect
    ratio). If not, grid search over width multipliers may be needed.
  - Whether ESC-50's small size (2000 clips) benefits from models >20M params
    without overfitting. If overfitting dominates, stronger regularization
    (DropPath rate increase, StochDepth, weight decay tuning) becomes the
    primary intervention rather than architecture.
  - Whether the ConvNeXt V2 training recipe (AdamW, cosine LR, EMA) transfers
    directly or requires re-optimization for ESC-50's small-data regime.
  - The optimal input resolution (128×86 vs 128×172 at 44.1kHz) — this changes
    the aspect ratio and FLOP estimates materially.
claim_status:
  grounded:
    - "ConvNeXt V2 provides a clean scaling family from 3.7M–660M params"
    - "ESC-50 SOTA is ~94.2% (ITFA-DNN, 2025)"
    - "Rectangular kernels improve ESC-50 accuracy at small scale"
    - "GRN prevents feature collapse in ConvNeXt V2 on ImageNet"
  hypotheses:
    - "Rectangular kernels benefit at all scales (5M–30M)"
    - "GRN transfers to spectrogram domain"
    - "Pure CNN matches attention-based methods on ESC-50 at equal params"
    - "10M variant achieves ≥93% on ESC-50 without pretraining"
  TODO_unverified:
    - "FCMAE pretraining benefit on audio spectrograms"
    - "Optimal channel allocation differs from ImageNet domain"
    - "Whether Tiny-S (30M) avoids overfitting on 2000-clip dataset"
```

---

## 6. Training Recipe (Recommended)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight_decay=0.05) |
| Learning rate | 1e-3 (cosine decay to 1e-5) |
| Batch size | 128 |
| Epochs | 300 |
| Warmup | 10 epochs linear warmup |
| Label smoothing | 0.1 |
| DropPath rate | 0.1 (Atto-S) to 0.3 (Tiny-S) |
| Mixup alpha | 0.2 |
| SpecAugment | Freq mask (F=8), Time mask (T=8), 2 masks each |
| Random crop | Center crop to 128×80, then resize to 128×86 |
| EMA | Decay 0.9998 |
| Loss | Cross-entropy + label smoothing |

---

## 7. Data Preprocessing Pipeline

```
Raw audio (44.1kHz)
  → Resample to 22050 Hz
  → Trim/pad to 2.0s (44100 samples)
  → Compute STFT: n_fft=1024, hop_length=512
  → Mel filterbank: 128 bands
  → Log compression: log(mel_spectrogram + 1e-6)
  → Normalize: (x - μ) / σ per sample (μ,σ computed over time-freq grid)
  → Shape: 1 × 128 × 86 (channels, freq, time)
  → Augment: SpecAugment + random crop + Mixup (applied in batch)
```

---

## 8. Implementation Roadmap

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| **Phase 1** | Data pipeline, baseline reproduction (ResNet-50, MobileNetV2, ITFA-DNN) | Reproduced baselines with reported metrics |
| **Phase 2** | SpectroConvNeXt Atto-S through Tiny-S implemented and trained | 5 model variants, parameter counts verified |
| **Phase 3** | Ablations: rectangular kernel, GRN, stem design | Ablation tables at 3 scales (5M, 15M, 30M) |
| **Phase 4** | Analysis: scaling laws, overfitting regime, comparison to SOTA | Final paper with tables, figures, and conclusions |
