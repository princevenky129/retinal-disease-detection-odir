"""
Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021).

Why this instead of plain BCEWithLogitsLoss:
In multi-label problems, for any given sample most of the 8 labels are
NEGATIVE (e.g. a patient with only Diabetes has 7 negative labels and 1
positive). Standard BCE treats all these easy negatives equally, and their
combined gradient can drown out the signal from rare positive classes
(Hypertension has ~94 samples vs ~1135 for Normal/Diabetes).

Asymmetric Loss fixes this with two separate focusing parameters:
- gamma_neg (focus on negatives): down-weights EASY negatives (confidently
  correct predictions) via (1-p)^gamma_neg-style focal weighting, so abundant
  easy negatives stop dominating the gradient.
- gamma_pos (focus on positives): typically kept low (often 0-1) so positive
  gradients -- especially from rare classes -- are NOT down-weighted.
- clip: a probability margin subtracted from negative predictions before
  computing their loss, which further suppresses easy-negative gradients
  ("asymmetric probability shifting").

This directly targets the exact failure mode Hypertension would otherwise
suffer from: being statistically drowned out by the negative gradient from
1000+ Normal/Diabetes samples where Hypertension=0.
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1,
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw model outputs (pre-sigmoid).
            targets: (B, num_classes) binary ground-truth labels (0/1 floats).
        Returns:
            scalar loss (if reduction='mean'/'sum') or (B, num_classes) if 'none'.
        """
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric probability shifting: give negatives a "free margin" of
        # `clip` before they start contributing loss, further suppressing
        # easy-negative gradients.
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        # Asymmetric focusing: different gamma for positive vs negative terms.
        with torch.no_grad():
            pt_pos = probs_pos * targets
            pt_neg = probs_neg * (1 - targets)
            focusing_weight = torch.pow(
                1 - pt_pos - pt_neg,
                self.gamma_pos * targets + self.gamma_neg * (1 - targets),
            )

        loss = -(loss_pos + loss_neg) * focusing_weight

        if self.reduction == "mean":
            return loss.sum(dim=1).mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    criterion = AsymmetricLoss()
    logits = torch.randn(4, 8, requires_grad=True)
    targets = torch.randint(0, 2, (4, 8)).float()
    loss = criterion(logits, targets)
    print("Asymmetric loss value:", loss.item())
    loss.backward()
    print("Backward pass OK, grad norm:", logits.grad.norm().item())
