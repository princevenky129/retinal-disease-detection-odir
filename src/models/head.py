"""
Classification head: GAP -> Dropout(0.3) -> FC(256) -> FC(8).

Note: Swin's timm wrapper (num_classes=0) already returns a globally-pooled
embedding, so "GAP" here is effectively already done by swin_encoder.py.
This head starts from that pooled (B, 1024) vector.

Output: raw logits, shape (B, 8). Sigmoid is applied OUTSIDE the model
(in the loss function during training, and explicitly at inference time) --
never inside forward() -- so that Asymmetric Loss (which expects logits) and
mixed-precision training both work correctly.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = 1024, hidden_dim: int = 256,
                 num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, num_classes) raw logits


if __name__ == "__main__":
    head = ClassificationHead()
    dummy = torch.randn(2, 1024)
    out = head(dummy)
    print("Head output (logits):", out.shape)
