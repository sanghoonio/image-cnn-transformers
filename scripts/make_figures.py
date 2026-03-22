"""Generate M2 report figures from training results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("reports/figures")


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load metrics.jsonl into a DataFrame."""
    rows = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_eval(run_dir: Path) -> dict:
    """Load eval_results.json."""
    with open(run_dir / "eval_results.json") as f:
        return json.load(f)


def fig1_training_curves():
    """Training curves: val mAP and train loss per epoch for all 4 baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = {
        "resnet50_pretrained": "ResNet-50 (pretrained)",
        "resnet50_scratch": "ResNet-50 (scratch)",
        "vit_b16_pretrained": "ViT-B/16 (pretrained)",
        "vit_b16_scratch": "ViT-B/16 (scratch)",
    }
    colors = {
        "resnet50_pretrained": "#1f77b4",
        "resnet50_scratch": "#aec7e8",
        "vit_b16_pretrained": "#d62728",
        "vit_b16_scratch": "#ff9896",
    }

    for run_name, label in labels.items():
        run_dir = RESULTS_DIR / f"{run_name}_frac1.00_aug-standard_seed42"
        if not run_dir.exists():
            continue
        df = load_metrics(run_dir)
        color = colors[run_name]

        ax1.plot(df["epoch"], df["val_mAP"], marker="o", markersize=4,
                 label=label, color=color)
        ax2.plot(df["epoch"], df["train_loss"], marker="o", markersize=4,
                 label=label, color=color)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation mAP")
    ax1.set_title("Validation mAP vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "training_curves.pdf", bbox_inches="tight")
    print("Saved training_curves.png/pdf")
    plt.close(fig)


def fig2_mAP_comparison():
    """Bar chart: mAP for all 4 baselines."""
    configs = [
        ("resnet50_pretrained", "ResNet-50\npretrained"),
        ("resnet50_scratch", "ResNet-50\nscratch"),
        ("vit_b16_pretrained", "ViT-B/16\npretrained"),
        ("vit_b16_scratch", "ViT-B/16\nscratch"),
    ]
    colors = ["#1f77b4", "#aec7e8", "#d62728", "#ff9896"]

    mAPs = []
    names = []
    for run_name, label in configs:
        run_dir = RESULTS_DIR / f"{run_name}_frac1.00_aug-standard_seed42"
        if not run_dir.exists():
            continue
        eval_data = load_eval(run_dir)
        mAPs.append(eval_data["mAP"])
        names.append(label)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, mAPs, color=colors[:len(names)], edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, mAPs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("mAP")
    ax.set_title("Baseline mAP Comparison (100% data, seed=42)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mAP_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mAP_comparison.pdf", bbox_inches="tight")
    print("Saved mAP_comparison.png/pdf")
    plt.close(fig)


def fig3_per_class_ap():
    """Per-class AP heatmap for all 4 baselines."""
    configs = [
        ("resnet50_pretrained", "ResNet-50 (pretrained)"),
        ("resnet50_scratch", "ResNet-50 (scratch)"),
        ("vit_b16_pretrained", "ViT-B/16 (pretrained)"),
        ("vit_b16_scratch", "ViT-B/16 (scratch)"),
    ]

    all_ap = {}
    classes = None
    for run_name, label in configs:
        run_dir = RESULTS_DIR / f"{run_name}_frac1.00_aug-standard_seed42"
        if not run_dir.exists():
            continue
        eval_data = load_eval(run_dir)
        per_class = eval_data["per_class_ap"]
        if classes is None:
            classes = list(per_class.keys())
        all_ap[label] = [per_class[c] for c in classes]

    data = np.array(list(all_ap.values()))

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_ap)))
    ax.set_yticklabels(list(all_ap.keys()), fontsize=10)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="AP", shrink=0.8)
    ax.set_title("Per-Class Average Precision")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_ap.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "per_class_ap.pdf", bbox_inches="tight")
    print("Saved per_class_ap.png/pdf")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig1_training_curves()
    fig2_mAP_comparison()
    fig3_per_class_ap()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
