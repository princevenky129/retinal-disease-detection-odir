"""
WeightedRandomSampler construction for ODIR-5K's class imbalance.

>>> THIS IS WHERE BALANCING HAPPENS (data-loading side). <<<
The complementary mechanism is Asymmetric Loss (src/losses/asymmetric_loss.py),
which handles balancing on the loss side. Together they form the two
independent strategies described in the project's imbalance-handling plan.

Because this is a MULTI-LABEL problem (each row can belong to several classes
at once), a single row doesn't have one "class" to weight by. We handle this
by giving each sample a weight equal to the weight of its RAREST positive
class (i.e. the sample is boosted according to the hardest-to-find label it
contains). This ensures samples containing Hypertension get drawn more often
even if they also happen to contain a common class like Diabetes.

IMPORTANT: after building the sampler, you MUST run the verification step in
notebooks/02_sampler_verification.ipynb -- pull several batches and confirm
rare classes appear more often than their raw frequency. This is a common
silent-failure point (e.g. forgetting `replacement=True`).
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def compute_class_weights(df, classes=CLASSES) -> dict:
    """Inverse-frequency weight per class: weight_c = total_samples / (num_classes * count_c)."""
    counts = df[classes].sum(axis=0)
    total = len(df)
    weights = {}
    for c in classes:
        count_c = max(counts[c], 1)  # avoid div-by-zero
        weights[c] = total / (len(classes) * count_c)
    return weights


def compute_sample_weights(df, classes=CLASSES) -> np.ndarray:
    """
    Per-sample weight = max class-weight among that sample's positive labels.
    Using max (not mean) ensures rare-class samples aren't diluted just because
    they co-occur with a common class.
    """
    class_weights = compute_class_weights(df, classes)
    class_weight_arr = np.array([class_weights[c] for c in classes])

    labels = df[classes].values.astype(np.float32)  # (N, 8)
    # For rows with no positive label (shouldn't happen in ODIR-5K, but guard anyway)
    has_positive = labels.sum(axis=1) > 0
    sample_weights = np.where(
        has_positive,
        (labels * class_weight_arr[None, :]).max(axis=1),
        class_weight_arr.mean(),  # fallback weight
    )
    return sample_weights


def build_weighted_sampler(df, classes=CLASSES) -> WeightedRandomSampler:
    sample_weights = compute_sample_weights(df, classes)
    sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,  # required for oversampling rare classes
    )
    return sampler


def verify_sampler_balance(dataset, sampler, classes=CLASSES, num_batches: int = 20, batch_size: int = 16):
    """
    Draw several batches through the sampler and report per-class occurrence
    counts, compared to the raw dataset's class frequencies. Used by
    notebooks/02_sampler_verification.ipynb. Rare classes (e.g. Hypertension)
    should appear noticeably more often per batch than their raw share.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    class_counts = np.zeros(len(classes))
    total_seen = 0

    for i, (_, labels) in enumerate(loader):
        class_counts += labels.numpy().sum(axis=0)
        total_seen += labels.shape[0]
        if i + 1 >= num_batches:
            break

    sampled_freq = class_counts / total_seen
    raw_freq = dataset.df[classes].mean(axis=0).values

    print(f"{'Class':<6}{'Raw freq':<12}{'Sampled freq':<14}{'Boost factor'}")
    for c, raw_f, samp_f in zip(classes, raw_freq, sampled_freq):
        boost = samp_f / raw_f if raw_f > 0 else float("nan")
        print(f"{c:<6}{raw_f:<12.4f}{samp_f:<14.4f}{boost:.2f}x")

    return raw_freq, sampled_freq
