"""
SpectroConvNeXt — Building block layers.

Components:
    - LayerNorm2d: Channel-wise LayerNorm for convolutional feature maps
    - GRN: Global Response Normalization (ConvNeXt V2)
    - StochasticDepth: DropPath for stochastic depth regularization
    - SpectroConvNeXtBlock: Core novel operator (dconv → LN → FFN → GRN → residual)
    - DownsampleBlock: Spatial downsampling between stages
    - SpectrogramStem: Two-layer frequency-preserving stem
    - PatchifyStem: Standard 4×4 stride-4 stem (for ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# LayerNorm2d
# ═══════════════════════════════════════════════════════════════════════════════

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for convolutional feature maps.

    Normalises each spatial position's C-dimensional feature vector,
    following ConvNeXt convention.  This is equivalent to:
        x: (B, C, H, W) → permute(0, 2, 3, 1) → LayerNorm(C) → permute back

    Args:
        dim: Number of channels (C)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — same shape, normalised
        """
        B, C, H, W = x.shape                                    # batch, channels, height, width
        x = x.permute(0, 2, 3, 1).contiguous()                   # (B, H, W, C)
        x = F.layer_norm(x, normalized_shape=(C,),
                         weight=self.weight, bias=self.bias,
                         eps=self.eps)                           # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()                   # (B, C, H, W)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Global Response Normalization  (ConvNeXt V2, Woo et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

class GRN(nn.Module):
    """
    Global Response Normalisation.

    For each channel, computes the global L2 norm over spatial dimensions,
    then normalises by the mean norm across channels.  Enhances inter-channel
    competition and prevents feature collapse.

    Math:
        gx_i = ||X_i||_2                 for each channel i   (B, C, 1, 1)
        nx_i = gx_i / mean(gx) + eps                          (B, C, 1, 1)
        X_i  = gamma_i * X_i * nx_i + beta_i                  (broadcast)

    Args:
        dim: Number of channels (C)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6, gamma_init: float = 1.0, beta_init: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), gamma_init))  # learnable scale
        self.beta = nn.Parameter(torch.full((dim,), beta_init))    # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — normalised activations, same shape

        Dtype invariant:
            Statistics computed in float32 for bf16 safety; output cast back
            to input dtype.
        """
        # bf16/fp16 safety: compute statistics in float32
        orig_dtype = x.dtype
        x = x.float()                                      # (B, C, H, W) float32

        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)   # (B, C, 1, 1)  L2 norm per channel
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps) # (B, C, 1, 1)  normalise by mean across channels

        gamma = self.gamma[None, :, None, None]             # (1, C, 1, 1)
        beta = self.beta[None, :, None, None]               # (1, C, 1, 1)

        x = gamma * x * nx + beta                           # (B, C, H, W)

        # Cast back to original dtype
        return x.to(orig_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Stochastic Depth (DropPath)
# ═══════════════════════════════════════════════════════════════════════════════

class StochasticDepth(nn.Module):
    """
    DropPath / Stochastic Depth.

    Randomly drops entire samples in the batch with probability `drop_prob`
    during training.  Keeps the residual branch alive with probability
    (1 - drop_prob).

    Args:
        drop_prob: Probability of dropping a sample (0 = no dropout)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ...)
        Returns:
            (B, ...) — same shape, with stochastic dropout applied
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        B = x.shape[0]
        shape = (B,) + (1,) * (x.ndim - 1)                  # broadcast over spatial dims
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()                               # 1 with prob keep_prob, 0 with prob drop_prob
        return x.div(keep_prob) * random_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# SpectroConvNeXt Block  (Core Novel Operator)
# ═══════════════════════════════════════════════════════════════════════════════

class SpectroConvNeXtBlock(nn.Module):
    """
    One SpectroConvNeXt block.

    Architecture (ConvNeXt V2 order):
        LayerNorm → Depthwise Conv2D → LayerNorm → 1×1 expand (4×) → GELU
        → 1×1 project → GRN → DropPath + residual

    Kernel size depends on stage:
        Stage 0 (index 0): rectangular  (7, 5) — matches spectrogram 1.49:1 aspect ratio
        Stages 1–2:        square       (7, 7)
        Stage 3:           square       (5, 5) — spatial dims are small

    Args:
        dim:           Number of input/output channels
        kernel_size:   (K_h, K_w) for the depthwise convolution
        expand_ratio:  Inverted bottleneck expansion factor (default: 4)
        drop_path:     Stochastic depth drop probability for this block
        use_grn:       Whether to include GRN (default: True)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: Tuple[int, int],
        expand_ratio: int = 4,
        drop_path: float = 0.0,
        use_grn: bool = True,
    ):
        super().__init__()
        # Pre-norm + depthwise convolution
        self.norm1 = LayerNorm2d(dim)                                                             # channel-wise norm
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),  # "same" padding
            groups=dim,                                           # depthwise: one filter per channel
            bias=False,
        )                                                                                          # (B, dim, H, W) → (B, dim, H, W)

        # Post-dconv norm
        self.norm2 = LayerNorm2d(dim)                                                             # (B, dim, H, W)

        # Inverted bottleneck FFN
        hidden_dim = dim * expand_ratio
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)                                  # (B, dim, H, W) → (B, 4*dim, H, W)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)                                  # (B, 4*dim, H, W) → (B, dim, H, W)

        # Global Response Normalization
        self.grn = GRN(dim) if use_grn else nn.Identity()

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — input feature map
        Returns:
            (B, C, H, W) — residual added, same shape

        Shape invariants:
            - dtype in {float32, bfloat16}; bfloat16 compatible via GRN float32 casting
        """
        residual = x                                                                              # (B, C, H, W)

        # Depthwise convolution path
        x = self.norm1(x)                                                                          # (B, C, H, W)
        x = self.dwconv(x)                                                                         # (B, C, H, W)

        # Inverted bottleneck FFN
        x = self.norm2(x)                                                                          # (B, C, H, W)
        x = self.pwconv1(x)                                                                        # (B, 4C, H, W)
        x = self.act(x)
        x = self.pwconv2(x)                                                                        # (B, C, H, W)

        # GRN + stochastic depth + residual
        x = self.grn(x)                                                                            # (B, C, H, W)
        x = self.drop_path(x)                                                                      # (B, C, H, W)

        return residual + x                                                                        # (B, C, H, W)

    def forward_with_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient checkpointing wrapper.
        Use during training to trade compute for memory.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — residual added, same shape as input
        """
        return torch.utils.checkpoint.checkpoint(self.forward, x, use_reentrant=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Downsample Block  (Between Stages)
# ═══════════════════════════════════════════════════════════════════════════════

class DownsampleBlock(nn.Module):
    """
    Spatial downsampling between stages.

    LayerNorm → Conv2D(kernel=2, stride=2)

    Halves both spatial dimensions and expands channels from C_in to C_out.

    Args:
        dim_in:  Number of input channels
        dim_out: Number of output channels
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.norm = LayerNorm2d(dim_in)                                                           # channel-wise norm
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2, bias=False)               # (B, C_in, H, W) → (B, C_out, H/2, W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H//2, W//2)
        """
        x = self.norm(x)                                                                           # (B, C_in, H, W)
        x = self.conv(x)                                                                           # (B, C_out, H//2, W//2)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Spectrogram Stem  (Frequency-preserving, two-layer)
# ═══════════════════════════════════════════════════════════════════════════════

class SpectrogramStem(nn.Module):
    """
    Two-layer frequency-preserving stem for spectrogram inputs.

    Design rationale (cf. ConvNeXt's 4×4 stride-4 patchify):
        - Layer 1: stride (1, 2) — preserves frequency resolution, halves time
        - Layer 2: stride (2, 1) — halves frequency, preserves time

    This delays frequency downsampling to preserve discriminative timbral
    information in the frequency axis.

    Input:  (B, in_ch, 128, ~86)   — log-mel spectrogram
    Output: (B, out_ch, 64, ~43)

    Args:
        in_ch:       Number of input channels (1 for mono mel-spectrogram)
        stem_ch:     Number of channels after first conv layer
        out_ch:      Number of channels after second conv layer (= C1 = stage_channels[0])
    """

    def __init__(self, in_ch: int, stem_ch: int, out_ch: int):
        super().__init__()
        self.norm = LayerNorm2d(in_ch)                                                             # channel-wise norm

        # Layer 1: preserve frequency, halve time
        self.conv1 = nn.Conv2d(
            in_ch, stem_ch,
            kernel_size=3, stride=(1, 2), padding=1, bias=False,
        )                                                                                          # (B, 1, 128, 86) → (B, stem_C, 128, 43)
        self.act1 = nn.GELU()

        # Layer 2: halve frequency, preserve time
        self.conv2 = nn.Conv2d(
            stem_ch, out_ch,
            kernel_size=3, stride=(2, 1), padding=1, bias=False,
        )                                                                                          # (B, stem_C, 128, 43) → (B, C1, 64, 43)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) — typically (B, 1, 128, ~86)
        Returns:
            (B, C_out, H/2, ~W/2) — typically (B, C1, 64, ~43)
        """
        x = self.norm(x)                                                                           # (B, C_in, H, W)
        x = self.conv1(x)                                                                          # (B, stem_C, H, W//2 + 1)
        x = self.act1(x)
        x = self.conv2(x)                                                                          # (B, C1, H//2, W//2 + 1)
        x = self.act2(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Patchify Stem  (Original ConvNeXt V2, for ablation)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchifyStem(nn.Module):
    """
    Original ConvNeXt V2 patchify stem: 4×4 convolution with stride 4.

    Used for Ablation C (see architecture doc) — tests whether the
    frequency-preserving stem outperforms aggressive initial downsampling.

    Input:  (B, in_ch, 128, ~86)
    Output: (B, out_ch, 32, ~21)

    Args:
        in_ch:  Number of input channels (1 for mono mel-spectrogram)
        out_ch: Number of output channels (= C1 = stage_channels[0])
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = LayerNorm2d(in_ch)                                                             # channel-wise norm
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=4, bias=False)                  # (B, 1, 128, 86) → (B, C1, 32, 21)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) — typically (B, 1, 128, ~86)
        Returns:
            (B, C_out, H//4, W//4) — typically (B, C1, 32, ~21)
        """
        x = self.norm(x)                                                                           # (B, C_in, H, W)
        x = self.conv(x)                                                                           # (B, C1, H//4, W//4)
        x = self.act(x)
        return x
