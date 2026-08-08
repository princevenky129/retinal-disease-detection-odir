"""
ODIRDataset: PyTorch Dataset for ODIR-5K.

Supports two modes (set via config.data.pairing_mode):
- "individual": each row in the processed CSV is ONE eye image (left or right)
  with its own 8-dim label vector. Simpler, ~7000 samples. Recommended starting
  point for a first working pipeline.
- "paired": each row is ONE patient with BOTH left and right images loaded
  together and stacked/concatenated, matching how ODIR-5K labels were actually
  assigned (per-patient, not per-eye). ~3500 samples, more faithful, slightly
  more complex to feed into the model (needs a fusion strategy).

Expects processed/{train,val,test}.csv to already exist (built by
scripts/prepare_data.py via src/data/split.py) with columns:
  - "filename" (individual mode) OR "left_filename","right_filename" (paired mode)
  - one column per class in CLASSES, values 0/1
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_and_preprocess

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


class ODIRDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, transform=None,
                 pairing_mode: str = "individual", classes=CLASSES,
                 use_clahe_cache: bool = True, clahe_cache_dir: str = None):
        """
        Args:
            csv_path: path to processed train/val/test.csv
            images_dir: directory containing the (raw or CLAHE-cached) images
            transform: an Albumentations transform (see src/data/augmentation.py)
            pairing_mode: "individual" or "paired"
            classes: ordered list of the 8 class codes
            use_clahe_cache: if True, read pre-CLAHE'd images directly from
                clahe_cache_dir (fast). If False, apply CLAHE on the fly (slow
                but useful for debugging/visual comparison).
            clahe_cache_dir: required if use_clahe_cache=True
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self.pairing_mode = pairing_mode
        self.classes = classes
        self.use_clahe_cache = use_clahe_cache
        self.clahe_cache_dir = clahe_cache_dir

        if use_clahe_cache and clahe_cache_dir is None:
            raise ValueError("clahe_cache_dir must be set when use_clahe_cache=True")

        expected_cols = (
            ["filename"] if pairing_mode == "individual" else ["left_filename", "right_filename"]
        )
        for col in expected_cols + self.classes:
            if col not in self.df.columns:
                raise ValueError(f"Expected column '{col}' not found in {csv_path}")

    def __len__(self):
        return len(self.df)

    def _load_image(self, filename: str) -> np.ndarray:
        source_dir = self.clahe_cache_dir if self.use_clahe_cache else self.images_dir
        path = os.path.join(source_dir, filename)
        if self.use_clahe_cache:
            import cv2
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError(f"Missing cached image: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            return load_and_preprocess(path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row[self.classes].values.astype(np.float32))

        if self.pairing_mode == "individual":
            image = self._load_image(row["filename"])
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, label

        else:  # paired
            left_image = self._load_image(row["left_filename"])
            right_image = self._load_image(row["right_filename"])
            if self.transform:
                left_image = self.transform(image=left_image)["image"]
                right_image = self.transform(image=right_image)["image"]
            # Stack along a new "eye" dimension: (2, C, H, W).
            # The model's forward pass / bridge module decides how to fuse
            # left+right (e.g. separate backbone passes then feature averaging).
            images = torch.stack([left_image, right_image], dim=0)
            return images, label
