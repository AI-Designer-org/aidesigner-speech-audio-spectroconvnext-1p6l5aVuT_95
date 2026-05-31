"""
SpectroConvNeXt — ModelConfig dataclass.

A ConvNeXt V2-derived CNN family for audio spectrogram classification (ESC-50),
with spectrogram-adapted rectangular kernels and delayed frequency downsampling.

Five variants target the 5M–30M parameter range.

Reference: Woo et al. "ConvNeXt V2" (CVPR 2023)
           Chang et al. "Rectangular Kernels + Self-Distillation" (Pattern Analysis & Applications, 2026)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass
class PreprocessingConfig:
    """Audio preprocessing / spectrogram extraction parameters."""
    sample_rate: int = 22050
    clip_duration_sec: float = 2.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    power: float = 2.0
    f_min: int = 0
    f_max: Optional[int] = None
    log_offset: float = 1e-6
    normalize: bool = True  # per-sample mean/std normalisation


@dataclass
class AugmentationConfig:
    """Data augmentation parameters (applied in the data pipeline)."""
    spec_augment_freq_mask: int = 8
    spec_augment_time_mask: int = 8
    spec_augment_num_masks: int = 2
    mixup_alpha: float = 0.2
    random_crop_size: Tuple[int, int] = (128, 80)  # (freq, time) after crop
    random_crop_resize: Tuple[int, int] = (128, 86)  # resize back to


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    lr_min: float = 1e-5
    lr_schedule: str = "cosine"
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 128
    epochs: int = 300
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    ema_decay: float = 0.9998
    n_classes: int = 50


@dataclass
class SpectroConvNeXtConfig:
    """
    SpectroConvNeXt model family configuration.

    Switching variants is a single-field change::

        config = SpectroConvNeXtConfig(variant="femto-s")

    The stage channels, block counts, and stem width are derived
    automatically from the variant name.

    Input shape:  (B, 1, 128, 86)   (channel, frequency, time)
    Output shape: (B, 50)
    """

    # ---------------------------
    # Variant selection
    # ---------------------------
    variant: Literal["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"] = "femto-s"

    # ---------------------------
    # Architecture — derived from variant (users should not set these directly)
    # ---------------------------
    stem_channels: int = 40          # C after first conv layer
    stage_channels: Tuple[int, ...] = (64, 128, 256, 512)    # C1–C4
    stage_blocks: Tuple[int, ...] = (2, 2, 6, 2)             # N1–N4

    # ---------------------------
    # Kernel sizes per stage
    # ---------------------------
    # Stage 0 uses rectangular (7, 5) to match spectrogram aspect ratio.
    # Stages 1–2 use square (7, 7).
    # Stage 3 uses (5, 5) because spatial dims are small (8 × 6).
    stage_kernel_sizes: Tuple[Tuple[int, int], ...] = (
        (7, 5),   # Stage 1 — rectangular: wider in frequency, narrower in time
        (7, 7),   # Stage 2 — square
        (7, 7),   # Stage 3 — square
        (5, 5),   # Stage 4 — smaller to accommodate 8 × 6 activation map
    )

    # ---------------------------
    # Block internals
    # ---------------------------
    expand_ratio: int = 4            # inverted bottleneck expansion factor
    use_grn: bool = True             # Global Response Normalisation
    grn_gamma_init: float = 1.0      # GRN learnable scale initialisation
    grn_beta_init: float = 0.0       # GRN learnable bias initialisation
    norm_eps: float = 1e-6
    activation: str = "gelu"

    # ---------------------------
    # Regularisation
    # ---------------------------
    drop_path_rate: float = 0.15     # stochastic depth survival rate; increase with model size
    dropout: float = 0.0             # no dropout in ConvNeXt by default
    layer_decay: Optional[float] = None  # layer-wise LR decay (optional, disabled if None)

    # ---------------------------
    # Stem design
    # ---------------------------
    # "spectrogram" = two-layer stem (preserves freq resolution first, then halves freq)
    # "patchify"    = 4×4 stride-4 stem (original ConvNeXt V2, for ablation)
    stem_type: Literal["spectrogram", "patchify"] = "spectrogram"

    # ---------------------------
    # Head
    # ---------------------------
    head_pooling: str = "global_avg"  # "global_avg" | "attention_pool"
    n_classes: int = 50               # ESC-50

    # ---------------------------
    # Data type
    # ---------------------------
    dtype: str = "float32"

    # ---------------------------
    # Derived helpers (computed in __post_init__)
    # ---------------------------
    _param_estimate: Optional[float] = None  # filled by post-init

    def __post_init__(self):
        variant_map = {
            "atto-s":  dict(stem=32, channels=(48, 96, 192, 384), blocks=(2, 2, 6, 2), params=4.8),
            "femto-s": dict(stem=40, channels=(64, 128, 256, 512), blocks=(2, 2, 6, 2), params=9.5),
            "pico-s":  dict(stem=48, channels=(80, 160, 320, 640), blocks=(2, 2, 8, 2), params=15.2),
            "nano-s":  dict(stem=56, channels=(96, 192, 384, 768), blocks=(2, 2, 8, 2), params=20.5),
            "tiny-s":  dict(stem=64, channels=(112, 224, 448, 896), blocks=(2, 2, 10, 2), params=29.8),
        }
        if self.variant in variant_map:
            v = variant_map[self.variant]
            self.stem_channels = v["stem"]
            self.stage_channels = v["channels"]
            self.stage_blocks = v["blocks"]
            self._param_estimate = v["params"]
        else:
            # Allow manual override; trust user-supplied values
            self._param_estimate = None

        # Scale drop path rate with variant size (larger models need more regularisation)
        if self.variant == "atto-s":
            default_dpr = 0.05
        elif self.variant == "femto-s":
            default_dpr = 0.1
        elif self.variant == "pico-s":
            default_dpr = 0.15
        elif self.variant == "nano-s":
            default_dpr = 0.2
        elif self.variant == "tiny-s":
            default_dpr = 0.3
        else:
            default_dpr = self.drop_path_rate

        # Only override if user has NOT explicitly set a non-default value
        # We use a sentinel: if drop_path_rate is still the class default (0.15)
        # OR if it's being set during __post_init__ for the first time
        if self.drop_path_rate == 0.15:  # class default
            self.drop_path_rate = default_dpr

    @property
    def n_stages(self) -> int:
        return len(self.stage_channels)

    @property
    def total_blocks(self) -> int:
        return sum(self.stage_blocks)

    def param_count_str(self) -> str:
        if self._param_estimate is not None:
            return f"~{self._param_estimate}M"
        return "custom (no estimate)"


# ==============================================================
# Convenience constructors for each variant
# ==============================================================

def atto_s(**overrides) -> SpectroConvNeXtConfig:
    """~5M parameter variant."""
    return SpectroConvNeXtConfig(variant="atto-s", **overrides)


def femto_s(**overrides) -> SpectroConvNeXtConfig:
    """~10M parameter variant."""
    return SpectroConvNeXtConfig(variant="femto-s", **overrides)


def pico_s(**overrides) -> SpectroConvNeXtConfig:
    """~15M parameter variant."""
    return SpectroConvNeXtConfig(variant="pico-s", **overrides)


def nano_s(**overrides) -> SpectroConvNeXtConfig:
    """~20M parameter variant."""
    return SpectroConvNeXtConfig(variant="nano-s", **overrides)


def tiny_s(**overrides) -> SpectroConvNeXtConfig:
    """~30M parameter variant."""
    return SpectroConvNeXtConfig(variant="tiny-s", **overrides)
