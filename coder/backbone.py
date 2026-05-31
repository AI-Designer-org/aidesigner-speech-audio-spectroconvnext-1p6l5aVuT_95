"""
SpectroConvNeXt — Backbone feature extractor.

Implements the BaseOperator abstract class.  The backbone comprises:
    - A spectrogram-adapted stem (or patchify stem for ablation)
    - Four stages of SpectroConvNeXt blocks
    - Downsampling layers between stages

Input:  (B, 1, 128, ~86)   — log-mel spectrogram
Output: (B, C4, ~8, ~5)    — feature map for downstream head
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from layers import (
    LayerNorm2d,
    SpectroConvNeXtBlock,
    DownsampleBlock,
    SpectrogramStem,
    PatchifyStem,
)


class BaseOperator(ABC, nn.Module):
    """
    Abstract base class for the core feature extractor.

    Subclasses must implement forward() that maps (B, C, H, W) → (B, C_out, H_out, W_out).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C_out, H_out, W_out)"""
        pass


class SpectroConvNeXtBackbone(BaseOperator):
    """
    SpectroConvNeXt feature extractor (backbone).

    Stacks a stem + 4 stages of SpectroConvNeXt blocks with interleaved
    downsampling, producing a multi-scale feature hierarchy.

    Args:
        stem_type:         "spectrogram" (two-layer) or "patchify" (4×4 stride-4)
        in_ch:             Number of input channels (1 for mono mel-spectrogram)
        stem_channels:     Channels after first conv layer (stem_C)
        stage_channels:    Tuple of (C1, C2, C3, C4)
        stage_blocks:      Tuple of (N1, N2, N3, N4) — blocks per stage
        stage_kernel_sizes: Tuple of kernel sizes per stage
        expand_ratio:      Inverted bottleneck expansion factor
        use_grn:           Whether to include GRN in blocks
        drop_path_rate:    Max stochastic depth rate (linearly scheduled across blocks)
    """

    def __init__(
        self,
        stem_type: str = "spectrogram",
        in_ch: int = 1,
        stem_channels: int = 40,
        stage_channels: Tuple[int, ...] = (64, 128, 256, 512),
        stage_blocks: Tuple[int, ...] = (2, 2, 6, 2),
        stage_kernel_sizes: Tuple[Tuple[int, int], ...] = (
            (7, 5), (7, 7), (7, 7), (5, 5),
        ),
        expand_ratio: int = 4,
        use_grn: bool = True,
        drop_path_rate: float = 0.15,
    ):
        super().__init__()

        n_stages = len(stage_channels)
        assert n_stages == 4, "SpectroConvNeXt requires exactly 4 stages"
        assert len(stage_blocks) == n_stages
        assert len(stage_kernel_sizes) == n_stages

        # ── Stem ─────────────────────────────────────────────────────────
        if stem_type == "spectrogram":
            self.stem = SpectrogramStem(in_ch, stem_channels, stage_channels[0])
            # (B, 1, 128, ~86) → (B, C1, 64, ~43)
        elif stem_type == "patchify":
            self.stem = PatchifyStem(in_ch, stage_channels[0])
            # (B, 1, 128, ~86) → (B, C1, 32, ~21)
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}")

        # ── Stages + Downsamplers ────────────────────────────────────────
        total_blocks = sum(stage_blocks)
        block_idx = 0

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Keep track of block drop probabilities for external access (logging etc.)
        self._block_drop_probs: List[float] = []

        for stage in range(n_stages):
            C_in = stage_channels[stage]
            kernel_size = stage_kernel_sizes[stage]
            n_blocks = stage_blocks[stage]

            # Downsampler (skip for stage 0 — already done by stem)
            if stage > 0:
                C_prev = stage_channels[stage - 1]
                self.downsample_layers.append(DownsampleBlock(C_prev, C_in))
                # (B, C_prev, H, W) → (B, C_in, H//2, W//2)
            else:
                # Placeholder for stage 0 (no downsample before stage 1)
                self.downsample_layers.append(nn.Identity())

            # Blocks for this stage
            stage_blocks_list = nn.ModuleList()
            for _ in range(n_blocks):
                # Linear drop-path schedule: 0 at first block, max at last
                dpr = (block_idx / max(total_blocks - 1, 1)) * drop_path_rate
                self._block_drop_probs.append(dpr)

                stage_blocks_list.append(
                    SpectroConvNeXtBlock(
                        dim=C_in,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        drop_path=dpr,
                        use_grn=use_grn,
                    )
                )
                block_idx += 1

            self.stages.append(stage_blocks_list)

        # ── Final norm ───────────────────────────────────────────────────
        self.final_norm = LayerNorm2d(stage_channels[-1])

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) — input spectrogram, typically (B, 1, 128, ~86)
            use_checkpoint: If True, use gradient checkpointing on each block
        Returns:
            (B, C4, H_out, W_out) — feature map at 1/16 resolution
        """
        # ── Stem ─────────────────────────────────────────────────────────
        x = self.stem(x)                                                                  # (B, C1, H//2, W//2) or (B, C1, H//4, W//4)

        # ── Stages ───────────────────────────────────────────────────────
        for stage_idx in range(len(self.stages)):
            # Downsample (identity for stage 0)
            x = self.downsample_layers[stage_idx](x)                                      # (B, C_stage, H_out, W_out)

            # Blocks in this stage
            for block in self.stages[stage_idx]:
                if use_checkpoint and self.training:
                    x = block.forward_with_checkpoint(x)                                   # (B, C_stage, H, W)
                else:
                    x = block(x)                                                           # (B, C_stage, H, W)

        # ── Final norm ───────────────────────────────────────────────────
        x = self.final_norm(x)                                                             # (B, C4, H_out, W_out)

        return x

    def get_block_drop_probs(self) -> List[float]:
        """Return the list of DropPath probabilities per block (for logging)."""
        return self._block_drop_probs
