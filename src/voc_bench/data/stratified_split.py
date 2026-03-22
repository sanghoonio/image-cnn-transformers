"""Iterative stratification for multi-label datasets."""

from pathlib import Path

import numpy as np

from .voc_dataset import VOCClassification


def iterative_stratification(
    labels: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """Select a stratified subset of a multi-label dataset.

    Uses scikit-multilearn's IterativeStratification to split the dataset
    while preserving approximate class frequencies across all labels.

    Args:
        labels: Binary label matrix of shape (n_samples, n_labels)
        fraction: Fraction of data to select (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Array of selected indices
    """
    if fraction >= 1.0:
        return np.arange(len(labels))

    from skmultilearn.model_selection import IterativeStratification

    n_samples = len(labels)

    # Shuffle data order with seed to produce different subsets per seed.
    # IterativeStratification is deterministic given input order, so
    # permuting the input achieves seed-dependent stratified subsets.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    shuffled_labels = labels[perm]

    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[1.0 - fraction, fraction],
    )

    # split() yields (train_indices, test_indices) for each fold.
    # With sample_distribution_per_fold=[1-fraction, fraction],
    # the first fold (train_indices) is the larger set.
    # We want the smaller subset, so take the train_indices from the
    # first split (which corresponds to the `fraction` proportion).
    for subset_indices, _ in stratifier.split(
        perm.reshape(-1, 1), shuffled_labels
    ):
        # Map back to original indices
        original_indices = perm[subset_indices]
        return np.sort(original_indices)


def create_subsets(
    data_root: str,
    output_dir: str = "data/subsets",
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Generate and save stratified subset indices for all fraction/seed combos.

    Args:
        data_root: Path to VOCdevkit
        output_dir: Directory to save .npy index files
        fractions: List of data fractions (default: [0.05, 0.10, 0.20, 0.50, 1.00])
        seeds: List of random seeds (default: [42, 123, 456])

    Returns:
        Dict mapping (fraction, seed) -> per-class counts in the subset
    """
    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.50, 1.00]
    if seeds is None:
        seeds = [42, 123, 456]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = VOCClassification(data_root, split="train")
    labels = dataset.labels
    n_total = len(labels)

    results = {}

    for fraction in fractions:
        for seed in seeds:
            indices = iterative_stratification(labels, fraction, seed)
            n_selected = len(indices)

            # Per-class counts in subset
            subset_labels = labels[indices]
            per_class_counts = subset_labels.sum(axis=0).astype(int)

            # Save indices
            fname = f"frac{fraction:.2f}_seed{seed}.npy"
            np.save(out_path / fname, indices)

            results[(fraction, seed)] = {
                "n_images": n_selected,
                "n_expected": int(round(n_total * fraction)),
                "per_class_counts": per_class_counts.tolist(),
                "min_class_count": int(per_class_counts.min()),
            }

    return results
