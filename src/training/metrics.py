"""
Per-class and macro-averaged metrics for multi-label classification.

Macro-F1 is the primary model-selection metric (not accuracy) because
accuracy is trivially high and misleading on imbalanced multi-label data
(e.g. always predicting Hypertension=0 already gives ~94% "accuracy" on that
class alone). Macro-F1 weights every class equally, so the rare Hypertension
class has to actually be learned to move the score.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5,
                     classes=CLASSES) -> dict:
    """
    Args:
        y_true: (N, num_classes) binary ground truth.
        y_pred_probs: (N, num_classes) predicted probabilities (post-sigmoid).
        threshold: probability cutoff for converting to binary predictions.

    Returns:
        dict with per-class precision/recall/f1/auc, plus macro averages.
    """
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None, zero_division=0
    )

    auc_per_class = []
    for i in range(len(classes)):
        # AUC undefined if a class has only one label value present in y_true
        if len(np.unique(y_true[:, i])) < 2:
            auc_per_class.append(float("nan"))
        else:
            auc_per_class.append(roc_auc_score(y_true[:, i], y_pred_probs[:, i]))

    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "auc": float(auc_per_class[i]),
            "support": int(support[i]),
        }

    results = {
        "per_class": per_class,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "macro_auc": float(np.nanmean(auc_per_class)),
    }
    return results


def print_metrics_report(results: dict):
    print(f"{'Class':<8}{'Precision':<11}{'Recall':<11}{'F1':<11}{'AUC':<11}{'Support'}")
    for c, m in results["per_class"].items():
        print(f"{c:<8}{m['precision']:<11.4f}{m['recall']:<11.4f}{m['f1']:<11.4f}"
              f"{m['auc']:<11.4f}{m['support']}")
    print("-" * 62)
    print(f"{'Macro':<8}{results['macro_precision']:<11.4f}{results['macro_recall']:<11.4f}"
          f"{results['macro_f1']:<11.4f}{results['macro_auc']:<11.4f}")
