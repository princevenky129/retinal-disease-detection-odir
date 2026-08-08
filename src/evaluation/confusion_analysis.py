"""
Multi-label confusion pattern analysis.

Standard confusion matrices assume single-label problems. For multi-label
data we instead build:
1. A co-occurrence matrix of PREDICTED classes (how often the model predicts
   class i and class j together) vs. the TRUE co-occurrence matrix -- lets us
   see if the model over/under-predicts certain disease combinations.
2. Per-class "confusion partners": for each class, which OTHER class is most
   often (wrongly) predicted alongside it, or most often missed when this
   class is the true label.
"""

import numpy as np
import pandas as pd

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def co_occurrence_matrix(binary_labels: np.ndarray, classes=CLASSES) -> pd.DataFrame:
    """binary_labels: (N, num_classes) binary array (either true labels or thresholded predictions)."""
    co_matrix = binary_labels.T @ binary_labels  # (num_classes, num_classes)
    return pd.DataFrame(co_matrix, index=classes, columns=classes)


def compare_true_vs_predicted_cooccurrence(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES):
    true_co = co_occurrence_matrix(y_true, classes)
    pred_co = co_occurrence_matrix(y_pred, classes)
    diff = pred_co - true_co
    return true_co, pred_co, diff


def per_class_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES) -> pd.DataFrame:
    """
    For each class, report:
    - false_negatives: true=1, pred=0 (missed diagnoses -- most clinically costly)
    - false_positives: true=0, pred=1 (over-diagnoses)
    - most_common_fn_co_label: among false negatives, which OTHER true label was
      most frequently also present (hints at what's "distracting" the model)
    """
    rows = []
    for i, c in enumerate(classes):
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        fp_mask = (y_true[:, i] == 0) & (y_pred[:, i] == 1)

        co_label = "-"
        if fn_mask.sum() > 0:
            other_idx = [j for j in range(len(classes)) if j != i]
            co_counts = y_true[fn_mask][:, other_idx].sum(axis=0)
            if co_counts.sum() > 0:
                co_label = classes[other_idx[int(np.argmax(co_counts))]]

        rows.append({
            "class": c,
            "false_negatives": int(fn_mask.sum()),
            "false_positives": int(fp_mask.sum()),
            "most_common_fn_co_label": co_label,
        })
    return pd.DataFrame(rows)
