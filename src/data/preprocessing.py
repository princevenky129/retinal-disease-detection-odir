"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.

Why we need this:
ODIR-5K images come from 3 different camera sources (Canon, Zeiss, Kowa),
each with different exposure/contrast characteristics. CLAHE normalizes local
contrast so the model sees a more consistent input distribution regardless of
which camera captured the image, instead of learning camera-specific quirks.

We apply CLAHE on the L-channel of LAB color space (not directly on RGB),
because doing it on L preserves color information (important for distinguishing
e.g. hemorrhages) while still equalizing brightness/contrast.
"""

import cv2
import numpy as np
import os


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to a single RGB image.

    Args:
        image: RGB image as a numpy array, shape (H, W, 3), dtype uint8.
        clip_limit: Threshold for contrast limiting (higher = more contrast, more noise risk).
        tile_grid_size: Size of the grid for the adaptive histogram equalization.

    Returns:
        RGB image (uint8) with CLAHE applied to its luminance channel.
    """
    if image is None:
        raise ValueError("apply_clahe received a None image.")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb_eq


def load_and_preprocess(image_path: str, target_size: int = 380, clip_limit: float = 2.0) -> np.ndarray:
    """
    Load an image from disk, resize, and apply CLAHE.

    Args:
        image_path: Path to the fundus image file.
        target_size: Square resolution to resize to (matches EfficientNet-B4 input).
        clip_limit: CLAHE clip limit.

    Returns:
        Preprocessed RGB image (uint8), shape (target_size, target_size, 3).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    rgb_clahe = apply_clahe(rgb, clip_limit=clip_limit)
    return rgb_clahe


def build_clahe_cache(raw_images_dir: str, cache_dir: str, target_size: int = 380, clip_limit: float = 2.0):
    """
    Precompute CLAHE-enhanced versions of every image in raw_images_dir and save
    them to cache_dir. Speeds up training by avoiding repeated CLAHE computation
    every epoch (CLAHE is deterministic, so this is safe to cache).

    Run this once via scripts/prepare_data.py before training.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filenames = [f for f in os.listdir(raw_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in filenames:
        src_path = os.path.join(raw_images_dir, fname)
        dst_path = os.path.join(cache_dir, fname)
        if os.path.exists(dst_path):
            continue  # already cached
        try:
            processed = load_and_preprocess(src_path, target_size=target_size, clip_limit=clip_limit)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, processed_bgr)
        except FileNotFoundError:
            print(f"[WARN] Skipping missing/corrupt image: {fname}")

    print(f"CLAHE cache built: {len(os.listdir(cache_dir))} images in {cache_dir}")
