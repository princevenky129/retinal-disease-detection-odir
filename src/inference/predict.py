"""
Single or paired-image inference, used by both scripts/ and app/streamlit_app.py.
"""

import torch
import yaml
import numpy as np

from src.models.model import HybridRetinalModel
from src.data.preprocessing import load_and_preprocess
from src.data.augmentation import get_eval_transforms

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {
    "N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract",
    "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Other",
}


def load_model_for_inference(config_path: str = "config/config.yaml", device=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRetinalModel(cfg)

    checkpoint_path = cfg["inference"]["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, cfg, device


@torch.no_grad()
def predict_single_image(model, cfg, device, image_path: str) -> dict:
    """Run inference on one fundus image, return {class: probability} dict."""
    image_size = cfg["data"]["image_size"]
    mean = cfg["augmentation"]["normalize_mean"]
    std = cfg["augmentation"]["normalize_std"]
    per_class_thresholds = cfg["inference"].get("per_class_thresholds")
    flat_threshold = cfg["inference"]["threshold"]

    raw_image = load_and_preprocess(image_path, target_size=image_size)
    transform = get_eval_transforms(image_size, mean, std)
    tensor = transform(image=raw_image)["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = {}
    for c, p in zip(CLASSES, probs):
        threshold = per_class_thresholds[c] if per_class_thresholds else flat_threshold
        predictions[CLASS_NAMES[c]] = {
            "probability": float(p),
            "positive": bool(p >= threshold),
        }
    return predictions


@torch.no_grad()
def predict_paired_images(model, cfg, device, left_image_path: str, right_image_path: str) -> dict:
    """
    Run inference on a left+right eye pair using the model's native paired
    fusion (both eyes -> shared backbone -> averaged embedding -> ONE
    patient-level prediction), matching how the model was trained and how
    ODIR-5K's ground-truth labels were actually assigned.
    """
    image_size = cfg["data"]["image_size"]
    mean = cfg["augmentation"]["normalize_mean"]
    std = cfg["augmentation"]["normalize_std"]
    per_class_thresholds = cfg["inference"].get("per_class_thresholds")
    flat_threshold = cfg["inference"]["threshold"]

    left_raw = load_and_preprocess(left_image_path, target_size=image_size)
    right_raw = load_and_preprocess(right_image_path, target_size=image_size)
    transform = get_eval_transforms(image_size, mean, std)

    left_t = transform(image=left_raw)["image"]
    right_t = transform(image=right_raw)["image"]
    paired_tensor = torch.stack([left_t, right_t], dim=0).unsqueeze(0).to(device)  # (1, 2, 3, H, W)

    logits = model(paired_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = {}
    for c, p in zip(CLASSES, probs):
        threshold = per_class_thresholds[c] if per_class_thresholds else flat_threshold
        predictions[CLASS_NAMES[c]] = {
            "probability": float(p),
            "positive": bool(p >= threshold),
        }
    return predictions
