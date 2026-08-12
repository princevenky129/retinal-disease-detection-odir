"""
Full test-set evaluation.

Reports per-class Precision/Recall/F1/AUC, with special attention called out
for Hypertension (minority class) and Other (heterogeneous class), plus an
optional "excluding Other" macro average since Other internally lumps ~12
different diseases together and is expected to underperform for reasons
unrelated to model quality.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics, print_metrics_report

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


@torch.no_grad()
def run_evaluation(model, test_dataset, cfg, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"],
    )

    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    per_class_thresholds = cfg["evaluation"].get("per_class_thresholds")
    if per_class_thresholds:
        threshold = np.array([per_class_thresholds[c] for c in CLASSES])
    else:
        threshold = cfg["evaluation"]["threshold"]
    results = compute_metrics(all_labels, all_probs, threshold=threshold)

    print("\n=== Full Test Set Results (all 8 classes) ===")
    print_metrics_report(results)

    print(f"\nHypertension (minority class) F1: {results['per_class']['H']['f1']:.4f} "
          f"| AUC: {results['per_class']['H']['auc']:.4f}")
    print(f"Other (heterogeneous class) F1: {results['per_class']['O']['f1']:.4f} "
          f"| AUC: {results['per_class']['O']['auc']:.4f}")

    if cfg["evaluation"].get("report_without_other", True):
        classes_no_other = [c for c in CLASSES if c != "O"]
        idx = [CLASSES.index(c) for c in classes_no_other]
        threshold_no_other = threshold[idx] if isinstance(threshold, np.ndarray) else threshold
        results_no_other = compute_metrics(
            all_labels[:, idx], all_probs[:, idx], threshold=threshold_no_other, classes=classes_no_other
        )
        print("\n=== Results Excluding 'Other' Class ===")
        print_metrics_report(results_no_other)

    return results, all_labels, all_probs
