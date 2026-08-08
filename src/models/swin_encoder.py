"""
Swin-B Transformer encoder wrapper (via timm).

Consumes the bridge module's output (B, 3, 224, 224) -- a fused multi-scale
representation projected back into "image" format -- and runs it through a
pretrained Swin-B, using shifted-window self-attention (window_size=7) to
build global relationships between regions. This is what lets the model
reason about multi-label co-occurrence (e.g. "this patient's Diabetes-looking
region AND their separately-located AMD-looking region are both present").

We use `num_classes=0` in timm to get the pooled feature embedding directly
(shape (B, 1024) for Swin-B), rather than timm's own classification head,
since our own head.py does the final 8-class projection.
"""

import timm
import torch
import torch.nn as nn


class SwinBEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, model_name: str = "swin_base_patch4_window7_224"):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # return pooled features, not class logits
        )
        self.out_features = self.encoder.num_features  # 1024 for Swin-B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) tensor from the bridge module.
        Returns:
            (B, out_features) pooled embedding.
        """
        return self.encoder(x)


if __name__ == "__main__":
    encoder = SwinBEncoder()
    dummy = torch.randn(2, 3, 224, 224)
    out = encoder(dummy)
    print("Swin-B output:", out.shape, "| out_features:", encoder.out_features)
