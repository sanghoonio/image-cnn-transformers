"""Evaluation metrics for multi-label classification."""

import numpy as np
from sklearn.metrics import average_precision_score


def compute_mAP(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_positives: int = 0,
) -> dict:
    """Compute mean Average Precision and per-class AP.

    Args:
        y_true: Binary ground truth matrix (n_samples, n_classes)
        y_score: Predicted scores matrix (n_samples, n_classes)
        min_positives: Minimum positive examples required to include a class.
            Classes with fewer positives are excluded from mAP.

    Returns:
        Dict with "mAP", "per_class_ap" (list), "supported_classes" (list of indices),
        and "supported_mAP" (mAP over classes with >= min_positives examples).
    """
    n_classes = y_true.shape[1]
    per_class_ap = []
    supported_ap = []
    supported_classes = []

    for i in range(n_classes):
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0:
            per_class_ap.append(0.0)
            continue

        ap = float(average_precision_score(y_true[:, i], y_score[:, i]))
        per_class_ap.append(ap)

        if n_pos >= min_positives:
            supported_ap.append(ap)
            supported_classes.append(i)

    mAP = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    supported_mAP = float(np.mean(supported_ap)) if supported_ap else 0.0

    return {
        "mAP": mAP,
        "per_class_ap": per_class_ap,
        "supported_mAP": supported_mAP,
        "supported_classes": supported_classes,
    }
