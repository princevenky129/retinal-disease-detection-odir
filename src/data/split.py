"""
Stratified train/val/test split for multi-label data.

Plain sklearn train_test_split's `stratify` argument only works for single-label
targets. For multi-label data (a patient can be Diabetes AND AMD at once) we
approximate stratification by grouping on a composite label signature.

Approach:
1. Build a single "combo" string per row from the 8 binary labels
   (e.g. "10000000" for Normal-only, "01000010" for Diabetes+Myopia).
2. Stratify on that combo string. This keeps rare label *combinations* together
   across splits, not just rare individual classes.
3. Fallback: any combo that appears fewer than 2 times (can't be split into
   train/val/test with stratification) is grouped into train only, with a
   warning -- this mainly affects rare multi-disease co-occurrences.

This is simpler than iterative stratification (e.g. skmultilearn) but is
sufficient and easy to explain in a viva. If class balance across splits looks
off in 01_eda.ipynb, swap this for skmultilearn's IterativeStratification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def build_label_combo(df: pd.DataFrame, classes=CLASSES) -> pd.Series:
    return df[classes].astype(int).astype(str).agg("".join, axis=1)


def stratified_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15,
                      test_ratio: float = 0.15, seed: int = 42, classes=CLASSES):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    df = df.copy()
    df["_combo"] = build_label_combo(df, classes)

    combo_counts = df["_combo"].value_counts()
    rare_combos = combo_counts[combo_counts < 2].index
    if len(rare_combos) > 0:
        print(f"[WARN] {len(rare_combos)} label combinations appear <2 times; "
              f"they will all be placed in the training set.")

    df_rare = df[df["_combo"].isin(rare_combos)]
    df_common = df[~df["_combo"].isin(rare_combos)]

    # First split off test set
    train_val_df, test_df = train_test_split(
        df_common, test_size=test_ratio, stratify=df_common["_combo"], random_state=seed
    )

    # Then split remaining into train/val
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df["_combo"], random_state=seed
    )

    # Rare combos go entirely into train
    train_df = pd.concat([train_df, df_rare], axis=0)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.drop(columns=["_combo"], inplace=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df
