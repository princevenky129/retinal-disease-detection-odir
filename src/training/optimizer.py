"""AdamW optimizer setup, per config.optimizer."""

import torch


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg["betas"]),
    )
