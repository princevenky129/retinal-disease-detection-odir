"""
Bridge module: adapts FPN's multi-scale CNN feature maps into a single input
tensor that Swin-B's patch embedding stage can consume.

The problem: FPN outputs 4 feature maps of DIFFERENT spatial resolutions
(P3..P6), all with 256 channels. Swin-B's patch embedding expects a single
image-like tensor, e.g. (B, 3, 224, 224) if used from scratch, or a single
feature map if we're feeding it mid-pipeline features instead of raw pixels.

Approach taken here (documented explicitly since this is a genuine design
decision, not a standard off-the-shelf recipe):
1. Resize every FPN level to a common spatial resolution (P3's resolution,
   the largest/finest one) via bilinear upsampling.
2. Concatenate along the channel dimension: 4 levels x 256ch = 1024 channels.
3. 1x1 conv to project 1024 -> 3 channels, so the fused multi-scale feature
   map can be fed into Swin-B's standard patch embedding layer (which expects
   3-channel input) while still benefiting from ImageNet-pretrained weights.
4. Resize to Swin-B's expected input resolution (224x224 by default for
   swin_base_patch4_window7_224).

This keeps the FPN's multi-scale reasoning intact while letting us reuse an
off-the-shelf pretrained Swin-B without modifying its patch embedding layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNToSwinBridge(nn.Module):
    def __init__(self, fpn_channels: int = 256, num_levels: int = 4,
                 swin_input_resolution: int = 224):
        super().__init__()
        self.swin_input_resolution = swin_input_resolution
        self.project = nn.Sequential(
            nn.Conv2d(fpn_channels * num_levels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def forward(self, fpn_outputs):
        """
        Args:
            fpn_outputs: list of 4 tensors [P3, P4, P5, P6], each
                (B, fpn_channels, H_i, W_i) with decreasing spatial size.
        Returns:
            (B, 3, swin_input_resolution, swin_input_resolution) tensor,
            ready to feed into Swin-B's patch embedding.
        """
        target_size = fpn_outputs[0].shape[-2:]  # finest level's resolution
        resized = [fpn_outputs[0]] + [
            F.interpolate(level, size=target_size, mode="bilinear", align_corners=False)
            for level in fpn_outputs[1:]
        ]
        fused = torch.cat(resized, dim=1)              # (B, 256*4, H, W)
        projected = self.project(fused)                  # (B, 3, H, W)
        out = F.interpolate(
            projected, size=(self.swin_input_resolution, self.swin_input_resolution),
            mode="bilinear", align_corners=False,
        )
        return out


if __name__ == "__main__":
    bridge = FPNToSwinBridge()
    dummy_fpn = [
        torch.randn(2, 256, 96, 96),
        torch.randn(2, 256, 48, 48),
        torch.randn(2, 256, 24, 24),
        torch.randn(2, 256, 12, 12),
    ]
    out = bridge(dummy_fpn)
    print("Bridge output:", out.shape)
