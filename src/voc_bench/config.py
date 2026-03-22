"""Experiment configuration with YAML I/O via omegaconf."""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


@dataclass
class ExperimentConfig:
    # model
    model_name: str = "resnet50"  # resnet50 | convnext_t | vit_b16 | deit_s
    pretrained: bool = True
    num_classes: int = 20

    # data
    data_root: str = "data/VOCdevkit"
    fraction: float = 1.0  # 0.05 | 0.10 | 0.20 | 0.50 | 1.00
    augmentation: str = "standard"  # none | standard | strong
    subset_seed: int = 42
    num_workers: int = 4

    # training
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    batch_size: int = 64
    patience: int = 5
    seed: int = 42
    warmup_epochs: int = 0

    # output
    output_dir: str = "results"


def load_config(*paths: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    """Load and merge configs from YAML files, with optional CLI overrides.

    Merges in order: schema defaults -> file1 -> file2 -> ... -> overrides.
    """
    schema = OmegaConf.structured(ExperimentConfig)
    configs = [schema]
    for p in paths:
        configs.append(OmegaConf.load(p))
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    return OmegaConf.to_object(merged)


def save_config(cfg: ExperimentConfig | dict, path: str | Path) -> None:
    """Save config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, ExperimentConfig):
        conf = OmegaConf.structured(cfg)
    else:
        conf = OmegaConf.create(cfg)
    OmegaConf.save(conf, path)


def dump_config(cfg: ExperimentConfig | dict) -> str:
    """Return config as YAML string."""
    if isinstance(cfg, ExperimentConfig):
        conf = OmegaConf.structured(cfg)
    else:
        conf = OmegaConf.create(cfg)
    return OmegaConf.to_yaml(conf)


def resolve_output_dir(cfg: ExperimentConfig) -> Path:
    """Build the output directory path from config fields."""
    pretrain_tag = "pretrained" if cfg.pretrained else "scratch"
    frac_tag = f"frac{cfg.fraction:.2f}"
    aug_tag = f"aug-{cfg.augmentation}"
    seed_tag = f"seed{cfg.seed}"
    run_name = f"{cfg.model_name}_{pretrain_tag}_{frac_tag}_{aug_tag}_{seed_tag}"
    return Path(cfg.output_dir) / run_name
