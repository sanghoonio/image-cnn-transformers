"""Model factory for all 4 architectures."""

from torch import nn
import timm
import torchvision.models as tv_models


def build_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = 20,
) -> nn.Module:
    """Build a model by name, replacing the classification head for multi-label.

    Args:
        name: One of "resnet50", "convnext_t", "vit_b16", "deit_s"
        pretrained: Whether to load ImageNet-pretrained weights
        num_classes: Number of output classes (20 for VOC)

    Returns:
        Model with replaced classification head outputting raw logits.
    """
    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "convnext_t":
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, num_classes
        )

    elif name == "vit_b16":
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif name == "deit_s":
        model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {name!r}. "
            f"Choose from: resnet50, convnext_t, vit_b16, deit_s"
        )

    return model


def get_param_groups(model: nn.Module, name: str, backbone_lr: float, head_lr: float):
    """Split model parameters into backbone and head groups for differential LR.

    Returns:
        List of param group dicts for torch.optim.
    """
    if name == "resnet50":
        head_params = set(model.fc.parameters())
    elif name == "convnext_t":
        head_params = set(model.classifier[2].parameters())
    elif name in ("vit_b16", "deit_s"):
        head_params = set(model.head.parameters())
    else:
        raise ValueError(f"Unknown model: {name!r}")

    backbone_params = [p for p in model.parameters() if p not in head_params]
    head_params_list = list(head_params)

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params_list, "lr": head_lr},
    ]
