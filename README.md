# CNNs vs. Vision Transformers on PASCAL VOC

DS 6050 Deep Learning project (UVA, Group 8). Comparing CNN and Vision Transformer data efficiency on PASCAL VOC 2012 multi-label classification across varying training set sizes (5% to 100%).

**Paper draft:** [Overleaf](https://www.overleaf.com/project/6993a42d2316a7126b7f7ed5)

## Models

We compare 4 architectures — 2 CNNs and 2 transformers — to isolate the effect of architectural inductive biases on data efficiency.

| Model | Family | Params | Key property |
|-------|--------|--------|-------------|
| **ResNet-50** | CNN | 25.6M | Classic CNN with built-in locality and translation equivariance via 3x3 convolutions + skip connections |
| **ConvNeXt-T** | CNN | 28.6M | Modernized CNN using transformer-era training recipes (7x7 kernels, LayerNorm, GELU). Tests whether the ViT advantage is from attention or training recipes |
| **ViT-B/16** | Transformer | 86M | Standard Vision Transformer. No spatial inductive biases — must learn everything from data. Needs large-scale pretraining |
| **DeiT-S** | Transformer | 22M | Data-efficient ViT trained with knowledge distillation from a CNN teacher. Tests whether distillation helps in low-data regimes |

All models output 20 raw logits (one per VOC class), trained with `BCEWithLogitsLoss` for multi-label classification.

## Pipeline

The experiment pipeline is config-driven: each run is fully specified by a YAML config file, and the config is saved alongside results as the reproducibility record.

### Step 1: Dataset verification

```bash
python scripts/verify_dataset.py --data-root data/VOCdevkit --download
```

Downloads PASCAL VOC 2012 and reports class distribution, label density (~1.5 labels/image), and dataset splits (5,717 train / 5,823 val).

### Step 2: Create stratified subsets

```bash
python scripts/create_subsets.py --data-root data/VOCdevkit
```

Generates iteratively stratified training subsets at 5%, 10%, 20%, 50%, and 100% of the data (3 seeds each). Uses multi-label stratification to preserve class frequencies across all 20 labels. Saves index files to `data/subsets/`.

### Step 3: Train a model

```bash
# Single run with config file
python scripts/train.py --config configs/base.yaml configs/resnet50_pretrained.yaml

# Override specific parameters via CLI
python scripts/train.py --config configs/base.yaml configs/vit_b16_pretrained.yaml \
    --fraction 0.05 --seed 123

# Preview resolved config without training
python scripts/train.py --config configs/base.yaml configs/resnet50_pretrained.yaml --dump-config
```

Trains one model configuration, logging per-epoch metrics as JSONL and saving the best checkpoint by validation mAP. Early stopping with patience=5.

### Step 4: Run full sweep (Rivanna)

```bash
# Generate per-run configs
python scripts/launch_sweep.py

# Submit as SLURM job array
sbatch --array=0-119%20 slurm/sweep_array.sh
```

Generates 120 configs (4 models x 2 pretrain x 5 fractions x 3 seeds) and submits them as a SLURM job array on Rivanna. The `%20` limits concurrent jobs.

### Step 5: Aggregate results

```bash
python scripts/aggregate_results.py --results-dir results
```

Collects all `eval_results.json` files into a summary CSV with mean +/- std mAP across seeds.

## Config System

Configs merge in order: `base.yaml` (defaults) -> model YAML (overrides) -> CLI flags (overrides).

```bash
# base.yaml provides shared defaults (epochs, batch_size, augmentation, etc.)
# Model YAMLs override model_name, pretrained, and learning rates
# CLI flags override anything

python scripts/train.py --config configs/base.yaml configs/vit_b16_pretrained.yaml --fraction 0.10
```

Available model configs in `configs/`:
- `resnet50_pretrained.yaml` / `resnet50_scratch.yaml`
- `convnext_t_pretrained.yaml` / `convnext_t_scratch.yaml`
- `vit_b16_pretrained.yaml` / `vit_b16_scratch.yaml`
- `deit_s_pretrained.yaml` / `deit_s_scratch.yaml`

## Results Format

Each training run saves to its own directory:

```
results/{model}_{pretrained|scratch}_frac{X.XX}_aug-{mode}_seed{N}/
  config.yaml        # exact resolved config (the reproducibility record)
  metrics.jsonl      # per-epoch: {"epoch":1, "train_loss":0.42, "val_mAP":0.72, ...}
  best_model.pt      # best model state_dict by val mAP
  eval_results.json  # final: mAP, per_class_ap, param_count, training_time, etc.
```

## Repository Structure

```
src/voc_bench/           # Installable Python package
  config.py              # Experiment config dataclass + YAML I/O
  data/                  # Dataset loading, transforms, stratification
  models/                # Model factory (ResNet-50, ConvNeXt-T, ViT-B/16, DeiT-S)
  training/              # Training loop, metrics, optimizer
  evaluation/            # Efficiency metrics, result aggregation
configs/                 # YAML config files (base + per-model)
scripts/                 # Standalone entry-point scripts
slurm/                   # SLURM job templates for Rivanna
literature/              # Papers, citations, literature review
plans/                   # Implementation plans and logs
reports/                 # LaTeX report source
```

## Setup

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment
uv sync
source .venv/bin/activate
uv pip install -e .

# Download dataset
python scripts/verify_dataset.py --data-root data/VOCdevkit --download

# Generate subsets
python scripts/create_subsets.py
```
