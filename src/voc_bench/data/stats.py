"""Dataset verification and statistics."""

import json

import numpy as np

from .voc_dataset import VOCClassification, VOC_CLASSES


def compute_dataset_stats(root: str) -> dict:
    """Compute and return dataset statistics as a structured dict."""
    train_ds = VOCClassification(root, split="train")
    val_ds = VOCClassification(root, split="val")

    train_labels = train_ds.labels
    val_labels = val_ds.labels

    # Per-class counts
    train_class_counts = train_labels.sum(axis=0).astype(int).tolist()
    val_class_counts = val_labels.sum(axis=0).astype(int).tolist()

    # Label density
    train_labels_per_image = train_labels.sum(axis=1)
    val_labels_per_image = val_labels.sum(axis=1)

    stats = {
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "num_classes": len(VOC_CLASSES),
        "classes": VOC_CLASSES,
        "train_class_counts": dict(zip(VOC_CLASSES, train_class_counts)),
        "val_class_counts": dict(zip(VOC_CLASSES, val_class_counts)),
        "train_avg_labels_per_image": float(np.mean(train_labels_per_image)),
        "train_std_labels_per_image": float(np.std(train_labels_per_image)),
        "val_avg_labels_per_image": float(np.mean(val_labels_per_image)),
        "train_frac_3plus_labels": float(
            np.mean(train_labels_per_image >= 3)
        ),
    }
    return stats


def print_stats(stats: dict) -> None:
    """Print dataset statistics in a human-readable format."""
    print(f"Train images: {stats['train_images']}")
    print(f"Val images:   {stats['val_images']}")
    print(f"Classes:      {stats['num_classes']}")
    print()

    print("Per-class training counts:")
    for cls, count in sorted(
        stats["train_class_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {cls:15s}  {count:5d}")
    print()

    print(f"Avg labels/image (train): {stats['train_avg_labels_per_image']:.2f} "
          f"+/- {stats['train_std_labels_per_image']:.2f}")
    print(f"Avg labels/image (val):   {stats['val_avg_labels_per_image']:.2f}")
    print(f"Fraction with >=3 labels: {stats['train_frac_3plus_labels']:.3f}")
