"""
CBAM: Convolutional Block Attention Module (Woo et al., 2018).

Applied after each EfficientNet stage output, before it's passed to the FPN.
CBAM has two sequential sub-modules:

1. Channel Attention: "WHAT is meaningful" -- uses both avg-pool and max-pool
   descriptors of the feature map, passed through a shared MLP, to produce a
   per-channel attention weight. E.g. learns to boost channels that respond to
   hemorrhage-like textures.

2. Spatial Attention: "WHERE is meaningful" -- pools across the channel axis
   (avg + max) to build a 2D map, convolves it, and produces a per-pixel
   attention weight. This is what lets GradCAM-style visualizations later show
   the model focusing on lesion regions rather than the image border/background.

Channel attention is applied first, then spatial -- this ordering (channel
-> spatial) is what the original CBAM paper found works best.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = self.sigmoid(self.conv(concat))          # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


if __name__ == "__main__":
    cbam = CBAM(channels=160)
    dummy = torch.randn(2, 160, 24, 24)
    out = cbam(dummy)
    print("CBAM input:", dummy.shape, "-> output:", out.shape)
