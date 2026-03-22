"""Train a single model configuration."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from voc_bench.config import (
    ExperimentConfig,
    dump_config,
    load_config,
    resolve_output_dir,
    save_config,
)
from voc_bench.data.transforms import get_train_transform, get_val_transform
from voc_bench.data.voc_dataset import VOCClassification, VOC_CLASSES
from voc_bench.models.factory import build_model
from voc_bench.training.optimizer import build_optimizer
from voc_bench.training.trainer import set_seed, train


def main():
    parser = argparse.ArgumentParser(description="Train a model on PASCAL VOC 2012")
    parser.add_argument(
        "--config", type=str, nargs="+", required=True,
        help="Path(s) to YAML config file(s). Merged in order.",
    )
    # CLI overrides
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--dump-config", action="store_true",
        help="Print resolved config and exit without training",
    )
    args = parser.parse_args()

    # Build CLI overrides dict (only non-None values)
    overrides = {}
    for key in [
        "model_name", "fraction", "augmentation", "seed",
        "subset_seed", "epochs", "batch_size", "data_root", "output_dir",
    ]:
        cli_key = key.replace("_", "-")
        val = getattr(args, key.replace("-", "_"), None) if "-" in cli_key else getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if args.pretrained is not None:
        overrides["pretrained"] = args.pretrained == "true"

    cfg = load_config(*args.config, overrides=overrides)
    run_dir = resolve_output_dir(cfg)

    if args.dump_config:
        print(dump_config(cfg))
        return

    print(f"Run directory: {run_dir}")
    print(dump_config(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Load data
    train_transform = get_train_transform(cfg.augmentation)
    val_transform = get_val_transform()

    # Load subset indices if fraction < 1.0
    indices = None
    if cfg.fraction < 1.0:
        subset_path = Path("data/subsets") / f"frac{cfg.fraction:.2f}_seed{cfg.subset_seed}.npy"
        if not subset_path.exists():
            raise FileNotFoundError(
                f"Subset file not found: {subset_path}. "
                f"Run scripts/create_subsets.py first."
            )
        indices = np.load(subset_path)
        print(f"Using subset: {len(indices)} images from {subset_path}")

    train_ds = VOCClassification(
        cfg.data_root, split="train", transform=train_transform, indices=indices
    )
    val_ds = VOCClassification(
        cfg.data_root, split="val", transform=val_transform
    )

    # pin_memory only supported on CUDA, not MPS
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
    )

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # Build model and optimizer
    model = build_model(cfg.model_name, cfg.pretrained, cfg.num_classes)
    optimizer = build_optimizer(
        model, cfg.model_name, cfg.backbone_lr, cfg.head_lr, cfg.weight_decay
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model_name} ({'pretrained' if cfg.pretrained else 'scratch'})")
    print(f"Parameters: {param_count:,}")

    # Save resolved config
    save_config(cfg, run_dir / "config.yaml")

    # Train
    print(f"\nStarting training for {cfg.epochs} epochs...\n")
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=cfg.epochs,
        output_dir=run_dir,
        patience=cfg.patience,
        warmup_epochs=cfg.warmup_epochs,
    )

    # Save final eval results
    eval_results = {
        "model_name": cfg.model_name,
        "pretrained": cfg.pretrained,
        "fraction": cfg.fraction,
        "augmentation": cfg.augmentation,
        "seed": cfg.seed,
        "param_count": param_count,
        "per_class_ap": dict(zip(VOC_CLASSES, results["per_class_ap"])),
        **{k: v for k, v in results.items() if k != "per_class_ap"},
    }

    eval_path = run_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nFinal mAP: {results['best_mAP']:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
