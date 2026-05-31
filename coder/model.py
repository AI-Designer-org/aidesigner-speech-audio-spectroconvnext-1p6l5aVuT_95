"""
SpectroConvNeXt — A ConvNeXt V2-derived CNN family for audio spectrogram classification.

Designed for ESC-50 (environmental sound classification, 50 classes).
Five variants target the 5M–30M parameter range:
    Atto-S   (~5M)
    Femto-S  (~10M)
    Pico-S   (~15M)
    Nano-S   (~20M)
    Tiny-S   (~30M)

Key adaptations from ConvNeXt V2 for spectrograms:
    1. Frequency-preserving two-layer stem (vs 4×4 patchify)
    2. Rectangular (7,5) depthwise kernels in Stage 1
    3. GRN in every block
    4. DropPath scaled with model size (0.05 → 0.30)

Usage:
    >>> from model import SpectroConvNeXtConfig, SpectroConvNeXt
    >>> cfg = SpectroConvNeXtConfig(variant="femto-s")
    >>> model = SpectroConvNeXt(cfg)
    >>> x = torch.randn(2, 1, 128, 86)
    >>> logits = model(x)   # (2, 50)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from backbone import SpectroConvNeXtBackbone
from head import ClassificationHead


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Parameter count
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> None:
    """
    Print total and trainable parameter counts for a model.

    Usage:
        count_params(model)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,} | Trainable: {trainable:,}  ({total/1e6:.2f}M)")


# ═══════════════════════════════════════════════════════════════════════════════
# ModelConfig
# ═══════════════════════════════════════════════════════════════════════════════

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
    random_crop_size: Tuple[int, int] = (128, 80)     # (freq, time) after crop
    random_crop_resize: Tuple[int, int] = (128, 86)    # resize back to


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


# ──────────────────────────────────────────────────────────────────────────────

# Variant database: maps variant name → architecture parameters
_VARIANT_MAP = {
    "atto-s":  dict(stem=32, channels=(48, 96, 192, 384),  blocks=(2, 2, 6, 2),  params=4.8,  dpr=0.05),
    "femto-s": dict(stem=40, channels=(64, 128, 256, 512), blocks=(2, 2, 6, 2),  params=9.5,  dpr=0.10),
    "pico-s":  dict(stem=48, channels=(80, 160, 320, 640), blocks=(2, 2, 8, 2),  params=15.2, dpr=0.15),
    "nano-s":  dict(stem=56, channels=(96, 192, 384, 768), blocks=(2, 2, 8, 2),  params=20.5, dpr=0.20),
    "tiny-s":  dict(stem=64, channels=(112, 224, 448, 896), blocks=(2, 2, 10, 2), params=29.8, dpr=0.30),
}


@dataclass
class SpectroConvNeXtConfig:
    """
    SpectroConvNeXt model family configuration.

    Switching variants is a single-field change::

        config = SpectroConvNeXtConfig(variant="femto-s")

    All stage channels, block counts, stem width, and default drop-path rate
    are derived automatically from the variant name.

    Input shape:  (B, 1, 128, ~86)   (channel, frequency, time)
    Output shape: (B, 50)           (class logits)
    """

    # ── Variant selection ────────────────────────────────────────────────
    variant: Literal["atto-s", "femto-s", "pico-s", "nano-s", "tiny-s"] = "femto-s"

    # ── Architecture — derived from variant ──────────────────────────────
    stem_channels: int = 40               # C after first conv layer (stem_C)
    stage_channels: Tuple[int, ...] = (64, 128, 256, 512)   # C1–C4
    stage_blocks: Tuple[int, ...] = (2, 2, 6, 2)            # N1–N4

    # ── Kernel sizes per stage ──────────────────────────────────────────
    # Stage 1: rectangular (7,5) — matches spectrogram ~1.49:1 aspect ratio
    # Stages 2–3: square (7,7)
    # Stage 4: (5,5) — spatial dims are small
    stage_kernel_sizes: Tuple[Tuple[int, int], ...] = (
        (7, 5),   # Stage 1
        (7, 7),   # Stage 2
        (7, 7),   # Stage 3
        (5, 5),   # Stage 4
    )

    # ── Block internals ─────────────────────────────────────────────────
    expand_ratio: int = 4                 # inverted bottleneck expansion factor
    use_grn: bool = True                  # Global Response Normalisation
    norm_eps: float = 1e-6

    # ── Regularisation ──────────────────────────────────────────────────
    drop_path_rate: float = 0.15          # stochastic depth; auto-scaled from variant
    dropout: float = 0.0                  # no dropout in ConvNeXt by default

    # ── Stem design ─────────────────────────────────────────────────────
    # "spectrogram" = two-layer stem (freq-preserving)
    # "patchify"    = 4×4 stride-4 stem (original ConvNeXt V2, for ablation)
    stem_type: Literal["spectrogram", "patchify"] = "spectrogram"

    # ── Head ────────────────────────────────────────────────────────────
    n_classes: int = 50                   # ESC-50

    # ── Input channels ──────────────────────────────────────────────────
    in_channels: int = 1                  # mono mel-spectrogram

    # ── Sub-configs ─────────────────────────────────────────────────────
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ── Internal ────────────────────────────────────────────────────────
    _param_estimate: Optional[float] = None

    def __post_init__(self):
        if self.variant in _VARIANT_MAP:
            v = _VARIANT_MAP[self.variant]
            self.stem_channels = v["stem"]
            self.stage_channels = v["channels"]
            self.stage_blocks = v["blocks"]
            self._param_estimate = v["params"]
            # Override drop_path_rate only if user hasn't explicitly changed it
            if self.drop_path_rate == 0.15:  # class default → apply variant default
                self.drop_path_rate = v["dpr"]
        else:
            self._param_estimate = None

    @property
    def n_stages(self) -> int:
        return len(self.stage_channels)

    @property
    def total_blocks(self) -> int:
        return sum(self.stage_blocks)

    @property
    def param_estimate_m(self) -> str:
        if self._param_estimate is not None:
            return f"~{self._param_estimate}M"
        return "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class SpectroConvNeXt(nn.Module):
    """
    SpectroConvNeXt: modern CNN for audio spectrogram classification.

    Composes a backone (stem + 4 ConvNeXt stages) with a classification head
    (GAP → LayerNorm → Linear).

    Usage::

        cfg = SpectroConvNeXtConfig(variant="femto-s")
        model = SpectroConvNeXt(cfg)
        logits = model(spectrogram_batch)   # (B, 50)

    Args:
        config: Model configuration dataclass
    """

    def __init__(self, config: SpectroConvNeXtConfig):
        super().__init__()
        self.config = config

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = SpectroConvNeXtBackbone(
            stem_type=config.stem_type,
            in_ch=config.in_channels,
            stem_channels=config.stem_channels,
            stage_channels=config.stage_channels,
            stage_blocks=config.stage_blocks,
            stage_kernel_sizes=config.stage_kernel_sizes,
            expand_ratio=config.expand_ratio,
            use_grn=config.use_grn,
            drop_path_rate=config.drop_path_rate,
        )

        # ── Head ─────────────────────────────────────────────────────────
        C4 = config.stage_channels[-1]
        self.head = ClassificationHead(
            dim=C4,
            n_classes=config.n_classes,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            x: (B, 1, H, W) — log-mel spectrogram, typically (B, 1, 128, ~86)
            use_checkpoint: If True, use gradient checkpointing on backbone blocks
        Returns:
            (B, n_classes) — class logits
        """
        # ── Feature extraction ───────────────────────────────────────────
        x = self.backbone(x, use_checkpoint=use_checkpoint)            # (B, C4, ~8, ~5)

        # ── Classification head ──────────────────────────────────────────
        x = self.head(x)                                               # (B, 50)

        return x
