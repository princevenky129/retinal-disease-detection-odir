"""
Cosine Annealing with Warm Restarts scheduler.

Why this over a flat/step schedule: the periodic LR restarts help the model
escape sharp local minima during training on a relatively small (~10K image)
dataset, and the smooth cosine decay within each cycle tends to produce better
final convergence than step decay for transformer-containing architectures
like Swin-B.
"""

import torch


def build_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg["scheduler"]
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg["T_0"],
        T_mult=sched_cfg["T_mult"],
        eta_min=float(sched_cfg["eta_min"]),
    )
