"""
HybridRetinalModel: assembles all pieces into the final end-to-end network.

Pipeline:
  image
    -> EfficientNet-B4 backbone (3 intermediate stages)
    -> CBAM applied to EACH stage independently (channel+spatial refinement)
    -> FPN neck (fuses the 3 CBAM-refined stages into 4 multi-scale levels)
    -> Bridge (fuses the 4 FPN levels into a single 3-channel "image")
    -> Swin-B encoder (pretrained, shifted-window global attention)
    -> Classification head (GAP already done by Swin -> Dropout -> FC -> FC)
    -> 8 raw logits (sigmoid applied outside the model, at loss/inference time)

Build/test each submodule in isolation first (run each file's __main__ block)
before trusting this assembly -- see Phase 4 in the project plan.
"""

import torch
import torch.nn as nn
import yaml

from src.models.backbone import EfficientNetB4Backbone
from src.models.cbam import CBAM
from src.models.fpn import FPN
from src.models.bridge import FPNToSwinBridge
from src.models.swin_encoder import SwinBEncoder
from src.models.head import ClassificationHead


class HybridRetinalModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        # 1. Backbone
        self.backbone = EfficientNetB4Backbone(pretrained=m["backbone"]["pretrained"])
        stage_channels = self.backbone.out_channels  # e.g. [56, 160, 448]

        # 2. CBAM -- one instance per backbone stage (channel counts differ per stage)
        self.cbam_blocks = nn.ModuleList([
            CBAM(
                channels=ch,
                reduction_ratio=m["cbam"]["reduction_ratio"],
                spatial_kernel_size=m["cbam"]["spatial_kernel_size"],
            )
            for ch in stage_channels
        ])

        # 3. FPN neck
        self.fpn = FPN(
            in_channels_list=stage_channels,
            out_channels=m["fpn"]["out_channels"],
        )

        # 4. Bridge (FPN multi-scale -> Swin-compatible single tensor)
        self.bridge = FPNToSwinBridge(
            fpn_channels=m["fpn"]["out_channels"],
            num_levels=m["fpn"]["num_levels"],
            swin_input_resolution=m["swin"]["input_resolution"],
        )

        # 5. Swin-B encoder
        self.swin = SwinBEncoder(
            pretrained=m["swin"]["pretrained"],
            model_name=m["swin"]["name"],
        )

        # 6. Classification head
        self.head = ClassificationHead(
            in_features=self.swin.out_features,
            hidden_dim=m["head"]["hidden_dim"],
            num_classes=m["head"]["num_classes"],
            dropout=m["head"]["dropout"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) preprocessed fundus image batch.
        Returns:
            (B, num_classes) raw logits.
        """
        stage_feats = self.backbone(x)                                  # list of 3 feature maps
        refined_feats = [cbam(f) for cbam, f in zip(self.cbam_blocks, stage_feats)]
        fpn_feats = self.fpn(refined_feats)                              # list of 4 feature maps
        bridged = self.bridge(fpn_feats)                                 # (B, 3, 224, 224)
        embedding = self.swin(bridged)                                   # (B, 1024)
        logits = self.head(embedding)                                    # (B, 8)
        return logits


def build_model_from_config(config_path: str = "config/config.yaml") -> HybridRetinalModel:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return HybridRetinalModel(cfg)


if __name__ == "__main__":
    # End-to-end shape sanity check -- run `python -m src.models.model` from repo root.
    model = build_model_from_config()
    dummy = torch.randn(2, 3, 380, 380)
    logits = model(dummy)
    print("Final model output (logits):", logits.shape)  # expect (2, 8)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,} | Trainable: {n_trainable:,}")
