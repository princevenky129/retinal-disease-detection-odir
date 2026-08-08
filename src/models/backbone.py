"""
EfficientNet-B4 backbone wrapper (via timm), used as a multi-scale feature
extractor. We don't use the full classification network -- only its
intermediate feature maps, which get fed into CBAM -> FPN.

timm's `features_only=True` mode returns a list of feature maps at
progressively downsampled resolutions. For EfficientNet-B4 the stage output
channels are approximately:
    stage 1 (stride 4):  24
    stage 2 (stride 8):  32
    stage 3 (stride 16): 56
    stage 4 (stride 16): 160
    stage 5 (stride 32): 448

We pull 3 stages (matching config.model.backbone.out_stage_channels) that feed
the FPN's 3 lateral inputs; the FPN itself derives a 4th (extra downsampled)
level internally -- see src/models/fpn.py.
"""

import timm
import torch
import torch.nn as nn


class EfficientNetB4Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_indices=(2, 3, 4)):
        """
        Args:
            pretrained: load ImageNet weights.
            out_indices: which timm feature stages to return. Default (2,3,4)
                corresponds to strides (16, 16, 32) i.e. the deeper, semantically
                richer stages -- appropriate for disease/lesion classification
                (as opposed to low-level edge stages 0,1 which are more useful
                for detection tasks needing precise localization).
        """
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) input image batch.
        Returns:
            List of feature maps, e.g. [C3, C4, C5], each (B, C_i, H_i, W_i).
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Quick shape sanity check -- run `python -m src.models.backbone` from repo root.
    model = EfficientNetB4Backbone()
    dummy = torch.randn(2, 3, 380, 380)
    feats = model(dummy)
    print("Backbone output channels:", model.out_channels)
    for i, f in enumerate(feats):
        print(f"  stage {i}: {tuple(f.shape)}")
