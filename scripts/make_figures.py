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


ARCH_LABELS = {
    "resnet50": "ResNet-50",
    "convnext_t": "ConvNeXt-T",
    "vit_b16": "ViT-B/16",
    "deit_s": "DeiT-S",
}
ARCH_COLORS = {
    "resnet50": "#1f77b4",
    "convnext_t": "#ff7f0e",
    "vit_b16": "#d62728",
    "deit_s": "#2ca02c",
}


def _load_summary() -> pd.DataFrame:
    """Load summary.csv and coerce the boolean column."""
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    if df["pretrained"].dtype == object:
        df["pretrained"] = df["pretrained"].map({"True": True, "False": False})
    return df


def fig_data_scaling():
    """Centerpiece: mAP vs training fraction, 8 lines (4 archs x 2 pretraining)."""
    df = _load_summary()
    df = df[df["augmentation"] == "standard"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name in ["resnet50", "convnext_t", "vit_b16", "deit_s"]:
        for pretrained in [True, False]:
            sub = df[
                (df["model_name"] == model_name) & (df["pretrained"] == pretrained)
            ].sort_values("fraction")
            if sub.empty:
                continue
            color = ARCH_COLORS[model_name]
            linestyle = "-" if pretrained else "--"
            label = f"{ARCH_LABELS[model_name]} ({'pretrained' if pretrained else 'scratch'})"
            ax.plot(
                sub["fraction"], sub["mAP_mean"],
                marker="o", linestyle=linestyle, color=color, label=label,
            )
            ax.fill_between(
                sub["fraction"],
                sub["mAP_mean"] - sub["mAP_std"],
                sub["mAP_mean"] + sub["mAP_std"],
                color=color, alpha=0.15,
            )

    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.10, 0.20, 0.50, 1.00])
    ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Validation mAP")
    ax.set_title("Data-Scaling: 4 Architectures x 2 Pretraining Conditions")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "data_scaling.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "data_scaling.pdf", bbox_inches="tight")
    print("Saved data_scaling.png/pdf")
    plt.close(fig)


def fig_augmentation_ablation():
    """Grouped bars: aug mode x architecture, mean +/- std across seeds."""
    df = _load_summary()
    df = df[
        (df["pretrained"] == True)
        & (df["fraction"] == 1.00)
        & (df["model_name"].isin(["resnet50", "vit_b16"]))
    ]

    aug_order = ["none", "standard", "strong"]
    arch_order = ["resnet50", "vit_b16"]

    x = np.arange(len(aug_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, arch in enumerate(arch_order):
        means, stds = [], []
        for aug in aug_order:
            row = df[(df["model_name"] == arch) & (df["augmentation"] == aug)]
            means.append(row["mAP_mean"].iloc[0])
            stds.append(row["mAP_std"].iloc[0])
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=4,
            color=ARCH_COLORS[arch], edgecolor="black", linewidth=0.5,
            label=f"{ARCH_LABELS[arch]} pretrained",
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in aug_order])
    ax.set_xlabel("Augmentation mode")
    ax.set_ylabel("Validation mAP")
    ax.set_title("Augmentation Ablation (full data, mean +/- std across 3 seeds)")
    ax.set_ylim(0.80, 0.96)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "augmentation_ablation.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "augmentation_ablation.pdf", bbox_inches="tight")
    print("Saved augmentation_ablation.png/pdf")
    plt.close(fig)


def fig_per_class_ap_full():
    """Heatmap: 8 (arch x pretrain) x 20 VOC classes at frac=1.00, mean across 3 seeds."""
    archs = [
        ("resnet50", True),
        ("convnext_t", True),
        ("vit_b16", True),
        ("deit_s", True),
        ("resnet50", False),
        ("convnext_t", False),
        ("vit_b16", False),
        ("deit_s", False),
    ]
    seeds = [42, 123, 456]

    all_ap = {}
    classes = None
    for model, pretrained in archs:
        pretrain_tag = "pretrained" if pretrained else "scratch"
        label = f"{ARCH_LABELS[model]} ({pretrain_tag})"
        per_seed = []
        for seed in seeds:
            run_dir = RESULTS_DIR / (
                f"{model}_{pretrain_tag}_frac1.00_aug-standard_seed{seed}"
            )
            if not run_dir.exists():
                continue
            eval_data = load_eval(run_dir)
            per_class = eval_data["per_class_ap"]
            if classes is None:
                classes = list(per_class.keys())
            per_seed.append([per_class[c] for c in classes])
        if per_seed:
            all_ap[label] = np.mean(per_seed, axis=0)

    data = np.array(list(all_ap.values()))

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_ap)))
    ax.set_yticklabels(list(all_ap.keys()), fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="AP", shrink=0.8)
    ax.set_title("Per-Class AP (full data, mean across 3 seeds)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_ap_full.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "per_class_ap_full.pdf", bbox_inches="tight")
    print("Saved per_class_ap_full.png/pdf")
    plt.close(fig)


def fig_training_curves_full():
    """Training curves for all 8 architectures at full data, seed=42.

    2x2 grid: rows = (pretrained, scratch); cols = (val mAP, train loss).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for row_idx, pretrained in enumerate([True, False]):
        pretrain_tag = "pretrained" if pretrained else "scratch"
        for arch in ["resnet50", "convnext_t", "vit_b16", "deit_s"]:
            run_dir = RESULTS_DIR / (
                f"{arch}_{pretrain_tag}_frac1.00_aug-standard_seed42"
            )
            if not run_dir.exists():
                continue
            df = load_metrics(run_dir)
            color = ARCH_COLORS[arch]
            label = ARCH_LABELS[arch]

            axes[row_idx, 0].plot(
                df["epoch"], df["val_mAP"], marker="o", markersize=3,
                color=color, label=label,
            )
            axes[row_idx, 1].plot(
                df["epoch"], df["train_loss"], marker="o", markersize=3,
                color=color, label=label,
            )

        axes[row_idx, 0].set_xlabel("Epoch")
        axes[row_idx, 0].set_ylabel("Validation mAP")
        axes[row_idx, 0].set_title(f"Validation mAP -- {pretrain_tag.capitalize()}")
        axes[row_idx, 0].set_ylim(0, 1.0)
        axes[row_idx, 0].legend(loc="lower right", fontsize=9)
        axes[row_idx, 0].grid(True, alpha=0.3)

        axes[row_idx, 1].set_xlabel("Epoch")
        axes[row_idx, 1].set_ylabel("Training Loss")
        axes[row_idx, 1].set_title(f"Training Loss -- {pretrain_tag.capitalize()}")
        axes[row_idx, 1].legend(loc="upper right", fontsize=9)
        axes[row_idx, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Training Dynamics: 4 Architectures x 2 Pretraining (full data, seed=42)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves_full.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "training_curves_full.pdf", bbox_inches="tight")
    print("Saved training_curves_full.png/pdf")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig1_training_curves()
    fig2_mAP_comparison()
    fig3_per_class_ap()
    fig_data_scaling()
    fig_augmentation_ablation()
    fig_per_class_ap_full()
    fig_training_curves_full()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
