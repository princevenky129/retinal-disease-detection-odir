"""
Training entry point.

Usage:
    python scripts/run_training.py

Prerequisite: run scripts/prepare_data.py first to generate
data/processed/{train,val,test}.csv and the CLAHE cache.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import torch

from src.data.dataset import ODIRDataset
from src.data.augmentation import get_train_transforms, get_eval_transforms
from src.models.model import HybridRetinalModel
from src.losses.asymmetric_loss import AsymmetricLoss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.train import train_model
from src.data.sampler import build_weighted_sampler


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    image_size = cfg["data"]["image_size"]
    mean = cfg["augmentation"]["normalize_mean"]
    std = cfg["augmentation"]["normalize_std"]

    train_transform = get_train_transforms(
        image_size, mean, std,
        rotation_degrees=cfg["augmentation"]["rotation_degrees"],
        brightness=cfg["augmentation"]["color_jitter"]["brightness"],
        contrast=cfg["augmentation"]["color_jitter"]["contrast"],
        saturation=cfg["augmentation"]["color_jitter"]["saturation"],
        hue=cfg["augmentation"]["color_jitter"]["hue"],
        scale_range=tuple(cfg["augmentation"].get("scale_range", (0.9, 1.1))),
    )
    eval_transform = get_eval_transforms(image_size, mean, std)

    train_dataset = ODIRDataset(
        csv_path=f"{cfg['data']['processed_dir']}/train.csv",
        images_dir=cfg["data"]["raw_images_dir"],
        transform=train_transform,
        pairing_mode=cfg["data"]["pairing_mode"],
        use_clahe_cache=cfg["data"]["use_clahe_cache"],
        clahe_cache_dir=cfg["data"]["clahe_cache_dir"],
    )
    val_dataset = ODIRDataset(
        csv_path=f"{cfg['data']['processed_dir']}/val.csv",
        images_dir=cfg["data"]["raw_images_dir"],
        transform=eval_transform,
        pairing_mode=cfg["data"]["pairing_mode"],
        use_clahe_cache=cfg["data"]["use_clahe_cache"],
        clahe_cache_dir=cfg["data"]["clahe_cache_dir"],
    )

    model = HybridRetinalModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = AsymmetricLoss(
        gamma_neg=cfg["loss"]["gamma_neg"],
        gamma_pos=cfg["loss"]["gamma_pos"],
        clip=cfg["loss"]["clip"],
        eps=float(cfg["loss"]["eps"]),
    )

    train_sampler = build_weighted_sampler(train_dataset.df)
    train_model(model, train_dataset, val_dataset, optimizer, scheduler, criterion, cfg, train_sampler=train_sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
