"""
Test-set evaluation entry point.

Usage:
    python scripts/run_evaluation.py
"""

import argparse
import yaml
import torch

from src.data.dataset import ODIRDataset
from src.data.augmentation import get_eval_transforms
from src.models.model import HybridRetinalModel
from src.evaluation.evaluate import run_evaluation


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    image_size = cfg["data"]["image_size"]
    mean = cfg["augmentation"]["normalize_mean"]
    std = cfg["augmentation"]["normalize_std"]
    eval_transform = get_eval_transforms(image_size, mean, std)

    test_dataset = ODIRDataset(
        csv_path=f"{cfg['data']['processed_dir']}/test.csv",
        images_dir=cfg["data"]["raw_images_dir"],
        transform=eval_transform,
        pairing_mode=cfg["data"]["pairing_mode"],
        use_clahe_cache=cfg["data"]["use_clahe_cache"],
        clahe_cache_dir=cfg["data"]["clahe_cache_dir"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRetinalModel(cfg)
    checkpoint = torch.load(cfg["training"]["checkpoint"]["save_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_macro_f1={checkpoint['val_macro_f1']:.4f})")

    run_evaluation(model, test_dataset, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
