"""
Feature Pyramid Network (FPN) neck.

Takes the CBAM-refined backbone features (varying channel counts, decreasing
spatial resolution) and produces 4 feature maps that all have the SAME channel
count (256) but different spatial resolutions (1/4, 1/8, 1/16, 1/32 of input).

Why this matters for ODIR-5K specifically: microaneurysms (early diabetic
retinopathy) are tiny, while large hemorrhages or an enlarged optic cup
(glaucoma) are comparatively huge. A single-scale feature map forces the
model to pick one "zoom level" to reason at. FPN's top-down pathway +
lateral connections let fine-grained (high-res, shallow) features be
enriched with semantic (low-res, deep) context, so both lesion types are
detectable.

Standard FPN recipe:
1. Lateral 1x1 convs project each backbone stage to `out_channels`.
2. Top-down pathway: upsample the coarsest map and add it to the next lateral
   (nearest-neighbor upsampling + element-wise add).
3. 3x3 conv ("smoothing") on each merged map to reduce aliasing from upsampling.
4. An extra level (P_extra) is derived by max-pooling the coarsest output,
   giving us 4 total output levels from 3 input backbone stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels: int = 256):
        """
        Args:
            in_channels_list: channel counts of the backbone stages, ordered
                shallow -> deep, e.g. [56, 160, 448].
            out_channels: unified channel count for every FPN output level.
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        # Extra level: derived via stride-2 max pool on the deepest FPN output,
        # giving us a 4th ("P6"-style) level without needing a 4th backbone stage.
        self.extra_pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, features):
        """
        Args:
            features: list of backbone feature maps, shallow -> deep,
                e.g. [C3 (56ch), C4 (160ch), C5 (448ch)].
        Returns:
            List of 4 feature maps, all with `out_channels` channels, ordered
            shallow -> deep: [P3, P4, P5, P6].
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: start from deepest, propagate to shallower levels.
        merged = [laterals[-1]]
        for lateral in reversed(laterals[:-1]):
            upsampled = F.interpolate(merged[-1], size=lateral.shape[-2:], mode="nearest")
            merged.append(lateral + upsampled)
        merged = merged[::-1]  # back to shallow -> deep order

        outputs = [smooth(m) for smooth, m in zip(self.smooth_convs, merged)]

        # 4th extra level from the deepest output.
        extra_level = self.extra_pool(outputs[-1])
        outputs.append(extra_level)

        return outputs  # [P3, P4, P5, P6]


if __name__ == "__main__":
    fpn = FPN(in_channels_list=[56, 160, 448], out_channels=256)
    dummy_feats = [
        torch.randn(2, 56, 48, 48),
        torch.randn(2, 160, 24, 24),
        torch.randn(2, 448, 12, 12),
    ]
    outs = fpn(dummy_feats)
    for i, o in enumerate(outs):
        print(f"P{i + 3}: {tuple(o.shape)}")
