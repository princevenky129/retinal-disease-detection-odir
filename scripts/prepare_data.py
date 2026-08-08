"""
Builds data/processed/{train,val,test}.csv from the raw ODIR-5K annotation
CSV, and (optionally) pre-computes the CLAHE image cache.

Usage:
    python scripts/prepare_data.py

Expects:
    data/raw/ODIR-5K_Training_Images/           (raw images)
    data/raw/ODIR-5K_Training_Annotations.csv   (raw patient-level annotations)
"""

import argparse
import os
import yaml
import pandas as pd

from src.data.split import stratified_split
from src.data.preprocessing import build_clahe_cache

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def expand_patients_to_individual_eyes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ODIR-5K's per-patient rows into per-eye rows for "individual" mode.

    NOTE: Adjust the raw column names below (`left_col`, `right_col`, etc.) to
    match your actual downloaded ODIR-5K_Training_Annotations.csv header --
    Kaggle mirrors of this dataset sometimes use slightly different column
    names (e.g. "Left-Fundus" vs "left_fundus"). Run `raw_df.columns.tolist()`
    first in notebooks/01_eda.ipynb to confirm, then adjust here.
    """
    left_col, right_col = "Left-Fundus", "Right-Fundus"

    left_rows = raw_df[[left_col] + CLASSES].rename(columns={left_col: "filename"})
    right_rows = raw_df[[right_col] + CLASSES].rename(columns={right_col: "filename"})

    individual_df = pd.concat([left_rows, right_rows], axis=0, ignore_index=True)
    individual_df = individual_df.dropna(subset=["filename"]).reset_index(drop=True)
    return individual_df


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_annotations = pd.read_csv(cfg["data"]["raw_annotations_csv"])
    print(f"Loaded raw annotations: {len(raw_annotations)} patients, "
          f"columns: {raw_annotations.columns.tolist()}")

    if cfg["data"]["pairing_mode"] == "individual":
        df = expand_patients_to_individual_eyes(raw_annotations)
    else:
        df = raw_annotations.rename(
            columns={"Left-Fundus": "left_filename", "Right-Fundus": "right_filename"}
        )

    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=cfg["data"]["split"]["train_ratio"],
        val_ratio=cfg["data"]["split"]["val_ratio"],
        test_ratio=cfg["data"]["split"]["test_ratio"],
        seed=cfg["project"]["seed"],
    )

    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    train_df.to_csv(os.path.join(cfg["data"]["processed_dir"], "train.csv"), index=False)
    val_df.to_csv(os.path.join(cfg["data"]["processed_dir"], "val.csv"), index=False)
    test_df.to_csv(os.path.join(cfg["data"]["processed_dir"], "test.csv"), index=False)
    print("Saved train.csv / val.csv / test.csv to", cfg["data"]["processed_dir"])

    if cfg["data"]["use_clahe_cache"]:
        print("Building CLAHE cache (this can take a while the first time)...")
        build_clahe_cache(
            raw_images_dir=cfg["data"]["raw_images_dir"],
            cache_dir=cfg["data"]["clahe_cache_dir"],
            target_size=cfg["data"]["image_size"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
