# configs/

YAML configuration files for experiments. Configs merge in layers — base defaults, then model-specific overrides, then CLI flags.

## How merging works

```
base.yaml                      # shared defaults
  + resnet50_pretrained.yaml   # model overrides (name, LRs)
    + --fraction 0.05          # CLI overrides
```

The final resolved config is saved to `results/<run>/config.yaml` as the reproducibility record.

## Files

| File | Purpose |
|------|---------|
| `base.yaml` | Shared defaults: epochs=30, batch_size=64, augmentation=standard, patience=5, seed=42 |
| `resnet50_pretrained.yaml` | ResNet-50 fine-tuning: backbone_lr=1e-4, head_lr=1e-3 |
| `resnet50_scratch.yaml` | ResNet-50 from scratch: lr=1e-2 for both |
| `convnext_t_pretrained.yaml` | ConvNeXt-T fine-tuning: backbone_lr=1e-4, head_lr=1e-3 |
| `convnext_t_scratch.yaml` | ConvNeXt-T from scratch: lr=1e-2 for both |
| `vit_b16_pretrained.yaml` | ViT-B/16 fine-tuning: backbone_lr=1e-5, head_lr=1e-3 |
| `vit_b16_scratch.yaml` | ViT-B/16 from scratch: lr=1e-3 + 5 warmup epochs |
| `deit_s_pretrained.yaml` | DeiT-S fine-tuning: backbone_lr=1e-5, head_lr=1e-3 |
| `deit_s_scratch.yaml` | DeiT-S from scratch: lr=1e-3 + 5 warmup epochs |
| `sweep.yaml` | Defines the full experiment grid (models x fractions x seeds) |

## Config fields

```yaml
# model
model_name: resnet50        # resnet50 | convnext_t | vit_b16 | deit_s
pretrained: true
num_classes: 20

# data
data_root: data/VOCdevkit
fraction: 1.0               # 0.05 | 0.10 | 0.20 | 0.50 | 1.00
augmentation: standard       # none | standard | strong
subset_seed: 42

# training
backbone_lr: 1.0e-4         # lower for pretrained backbone
head_lr: 1.0e-3             # higher for new classification head
weight_decay: 0.01
epochs: 30
batch_size: 64
patience: 5                  # early stopping patience
seed: 42
warmup_epochs: 0             # linear LR warmup (used for ViTs from scratch)

# output
output_dir: results
```

## generated/

Created by `scripts/launch_sweep.py`. Contains one fully-resolved YAML per run plus a `sweep_configs.txt` manifest for the SLURM job array. This directory is not committed.
