"""
SpectroConvNeXt — Classification Head.

Global Average Pooling → LayerNorm → Linear → logits

Input:  (B, C, H, W)   — feature map from backbone
Output: (B, n_classes) — class logits
"""

import torch
import torch.nn as nn

from layers import LayerNorm2d


class ClassificationHead(nn.Module):
    """
    Task-specific classification head for ESC-50.

    Architecture:
        Global Average Pooling (over H × W)
        → LayerNorm (channel-wise on the pooled vector)
        → Linear(C, n_classes)
        → (logits — no built-in Softmax; loss function applies it)

    Args:
        dim:       Number of input channels from backbone (C4)
        n_classes: Number of output classes (50 for ESC-50)
        dropout:   Dropout probability before the linear layer (default: 0.0)
    """

    def __init__(self, dim: int, n_classes: int = 50, dropout: float = 0.0):
        super().__init__()
        # Global Average Pooling — parameter-free, translation-invariant
        self.gap = nn.AdaptiveAvgPool2d(1)                      # (B, dim, H, W) → (B, dim, 1, 1)

        # Post-pooling normalization + classifier
        self.norm = nn.LayerNorm(dim)                           # (B, dim) → (B, dim)  — feature norm
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(dim, n_classes)                 # (B, dim) → (B, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C4, H_out, W_out) — feature map from backbone
        Returns:
            (B, n_classes) — class logits (not probabilities)
        """
        x = self.gap(x)                                          # (B, C4, 1, 1)
        x = x.flatten(1)                                         # (B, C4)
        x = self.norm(x)                                         # (B, C4)
        x = self.dropout(x)                                      # (B, C4)
        x = self.linear(x)                                       # (B, 50)
        return x
