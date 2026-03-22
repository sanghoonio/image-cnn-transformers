"""Training loop with early stopping, checkpointing, and JSONL logging."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from voc_bench.training.metrics import compute_mAP


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    output_dir: Path,
    patience: int = 5,
    warmup_epochs: int = 0,
    device: str | None = None,
) -> dict:
    """Run training with early stopping and logging.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Configured optimizer (with param groups)
        epochs: Maximum number of epochs
        output_dir: Directory to save metrics and checkpoints
        patience: Early stopping patience (epochs without improvement)
        warmup_epochs: Number of linear warmup epochs (0 = no warmup)
        device: Device to train on (auto-detected if None)

    Returns:
        Final evaluation results dict.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_model_path = output_dir / "best_model.pt"

    best_mAP = -1.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Get current LR (from first param group)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_mAP": round(val_metrics["mAP"], 4),
            "lr": current_lr,
        }

        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_mAP={val_metrics['mAP']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Early stopping
        if val_metrics["mAP"] > best_mAP:
            best_mAP = val_metrics["mAP"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    training_time = time.time() - start_time

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    final_metrics = evaluate(model, val_loader, criterion, device)

    return {
        "best_epoch": epoch - epochs_without_improvement,
        "best_mAP": round(best_mAP, 4),
        "training_time_sec": round(training_time, 1),
        **final_metrics,
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate model on a data loader.

    Returns:
        Dict with val_loss, mAP, per_class_ap, supported_mAP.
    """
    model.eval()
    all_labels = []
    all_scores = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    avg_loss = total_loss / max(n_batches, 1)

    map_results = compute_mAP(all_labels, all_scores, min_positives=5)

    return {
        "val_loss": avg_loss,
        **map_results,
    }
