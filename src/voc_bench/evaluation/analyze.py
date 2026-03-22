"""Result aggregation and analysis."""

import json
from pathlib import Path

import pandas as pd


def aggregate_results(results_dir: str = "results") -> pd.DataFrame:
    """Collect all eval_results.json files into a single DataFrame.

    Args:
        results_dir: Root results directory

    Returns:
        DataFrame with one row per experiment run.
    """
    results_path = Path(results_dir)
    rows = []

    for eval_file in sorted(results_path.rglob("eval_results.json")):
        with open(eval_file) as f:
            data = json.load(f)

        # Flatten per_class_ap into separate columns
        per_class_ap = data.pop("per_class_ap", {})
        for cls_name, ap in per_class_ap.items():
            data[f"ap_{cls_name}"] = ap

        data["run_dir"] = str(eval_file.parent.relative_to(results_path))
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std mAP across seeds for each configuration.

    Groups by model_name, pretrained, fraction, augmentation.
    """
    if df.empty:
        return df

    group_cols = ["model_name", "pretrained", "fraction", "augmentation"]
    available_cols = [c for c in group_cols if c in df.columns]

    summary = df.groupby(available_cols).agg(
        mAP_mean=("mAP", "mean"),
        mAP_std=("mAP", "std"),
        n_seeds=("mAP", "count"),
    ).reset_index()

    summary["mAP_mean"] = summary["mAP_mean"].round(4)
    summary["mAP_std"] = summary["mAP_std"].round(4)

    return summary
