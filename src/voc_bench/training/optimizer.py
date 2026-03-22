"""Optimizer construction with differential learning rates."""

import torch
from torch import nn

from voc_bench.models.factory import get_param_groups


def build_optimizer(
    model: nn.Module,
    model_name: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with differential learning rates.

    Args:
        model: The model
        model_name: Model name for param group splitting
        backbone_lr: Learning rate for backbone parameters
        head_lr: Learning rate for classification head
        weight_decay: Weight decay for AdamW
    """
    param_groups = get_param_groups(model, model_name, backbone_lr, head_lr)
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
