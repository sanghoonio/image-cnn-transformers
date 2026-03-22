---
date: 2026-03-22
status: in-progress
description: Milestone 2 implementation plan - experiment pipeline for CNN vs ViT comparison on PASCAL VOC
---

# Milestone 2 Implementation Plan

## Context

Instructor feedback on M1 was positive (strong experimental design, good lit review, multi-label framing called a "genuine contribution") but identified two gaps: (1) no concrete augmentation ablation plan despite the title, and (2) no evaluation beyond mAP. The team collaborates on code and runs experiments on UVA Rivanna (SLURM).

---

## Model Overview

We compare 4 architectures across two families (2 CNNs, 2 transformers):

### CNNs

**ResNet-50** -- the classic CNN baseline. Stacked 3x3 convolutions with skip connections. Built-in inductive biases: locality (kernel sees only neighbors), translation equivariance (same kernel everywhere), spatial hierarchy (pooling reduces resolution). 25.6M params.

**ConvNeXt-T** -- a "modernized" CNN that adopts transformer-era design choices (7x7 kernels, LayerNorm, GELU, inverted bottleneck) while keeping the convolutional core. Tests: *is the ViT advantage from attention itself, or just better training recipes?* 28.6M params.

### Transformers

**ViT-B/16** -- standard Vision Transformer. Splits image into 16x16 non-overlapping patches, embeds each as a token, processes through 12 transformer blocks with multi-head self-attention. No built-in spatial inductive biases -- must learn everything from data. Needs large-scale pretraining; struggles from scratch on small datasets. 86M params.

**DeiT-S** -- Data-efficient Image Transformer. Smaller ViT (22M params) trained with knowledge distillation from a CNN teacher (RegNet), injecting CNN-like spatial understanding indirectly. Tests: *does distillation help transformers in low-data regimes?*

### What each comparison reveals

| Comparison | Question |
|---|---|
| ResNet-50 vs ViT-B/16 | Classic CNN vs transformer: raw data efficiency gap |
| ResNet-50 vs ConvNeXt-T | Do modernized training recipes close the gap to ViTs? |
| ViT-B/16 vs DeiT-S | Does CNN distillation help transformers with less data? |
| Pretrained vs scratch | How much does pretraining compensate for lack of inductive bias? |
| 5%-100% data scaling | Where does each architecture's data-efficiency curve bend? |

All models get a final linear layer outputting 20 raw logits (one per VOC class), trained with `BCEWithLogitsLoss`.

---

## M2 Deliverables (from course slides + instructor feedback)

**Paper** (`G08_checkpoint.pdf`, up to 3 pages via Overleaf):
1. Abstract
2. Introduction -- problem context and motivation
3. Literature Survey -- expanded from M1
4. Method -- detailed architecture description (all 4 models, augmentation policy, training setup)
5. Preliminary Experiments -- initial baseline results with training curves
6. Next Steps -- planned improvements and full ablation plan
7. Member Contributions -- individual responsibilities

**Required elements:**
- At least one baseline model with training curves & metrics
- Error analysis (per-class AP breakdown)
- Ablation study plan (defined, not necessarily executed)
- Code repository URL

**Experiment results needed for M2:**
1. Dataset verification -- confirm counts, class distribution, label density
2. Stratified subsets defined -- iterative stratification, verify 5% subset
3. ResNet-50 pretrained baseline at 100% data (target mAP ~85-90%)
4. ViT-B/16 pretrained baseline at 100% data (target mAP ~85-92%)
5. ResNet-50 and ViT-B/16 from-scratch baselines at 100% data
6. Augmentation policy explicitly defined (exact transforms specified)
7. Training/validation curves for each baseline
8. Computational efficiency: param count, inference time

**= 4 configurations x 3 seeds = 12 training runs on Rivanna (GPU needed)**

**Due later (build infrastructure now, run after M2):**
- Full 120-run sweep across all 4 models x 2 pretrain x 5 fractions
- Augmentation ablation (none/standard/strong)
- Full per-class AP analysis, mAP-per-FLOP
- All report figures

The plan below builds the complete pipeline (so the full sweep is one command away) but only the 12 baseline runs need GPU time for M2.

---

## 1. Project Structure

```
src/
  voc_bench/
    __init__.py
    config.py              # dataclass config + YAML I/O (omegaconf)
    data/
      __init__.py
      voc_dataset.py       # VOC 2012 loading via ImageSets/Main txt files
      stratified_split.py  # iterative stratification for multi-label subsets
      transforms.py        # none / standard / strong augmentation pipelines
      stats.py             # dataset verification: class dist, label density
    models/
      __init__.py
      factory.py           # build_model() for all 4 architectures + param groups
    training/
      __init__.py
      trainer.py           # train loop, early stopping, checkpointing, seeding
      optimizer.py         # AdamW with differential LR (backbone vs head)
      metrics.py           # mAP, per-class AP, supported-class mAP
    evaluation/
      __init__.py
      efficiency.py        # param count, FLOPs (fvcore), GPU mem, timing
      analyze.py           # load results, produce summary tables
configs/
  base.yaml                # shared defaults
  resnet50_pretrained.yaml  # per-model overrides (8 total: 4 models x 2 pretrain)
  ...
  sweep.yaml               # grid definition: fractions, seeds, augmentation
scripts/
  verify_dataset.py        # download VOC, print stats, save class dist plot
  create_subsets.py         # generate and save stratified subset indices
  train.py                 # CLI entry point: load config, train, log, save
  launch_sweep.py          # generate config grid + SLURM job array
  aggregate_results.py     # collect all eval_results.json into summary CSV
slurm/
  train_single.sh          # single-run SLURM template
  sweep_array.sh           # job array template
results/                   # gitignored; structured output
notebooks/
  m2_figures.ipynb         # visualization for report
```

## 2. Config-Driven Design

Following [databio CLI principles](https://github.com/databio/lab.databio.org/discussions/71): the config file *is* the methods section. Every experiment is fully reproducible from its config.

**Core principles applied:**
- **Config + CLI override**: `train.py --config base.yaml --fraction 0.05 --seed 123` -- config defines defaults, flags override
- **Every step independent**: `verify_dataset.py`, `create_subsets.py`, `train.py`, `aggregate_results.py` are standalone scripts that read/write structured files
- **Deterministic by default**: fixed seed=42, deterministic cudnn, sorted outputs
- **Structured output everywhere**: JSONL for per-epoch metrics, JSON for final eval, no human-only print statements in scripts
- **Self-documenting**: `train.py --dump-config` prints the complete resolved config as YAML (what will actually run)

**Config structure** (single flat YAML per experiment):

```yaml
# model
model_name: resnet50        # resnet50 | convnext_t | vit_b16 | deit_s
pretrained: true
num_classes: 20

# data
data_root: /path/to/VOCdevkit
fraction: 1.0               # 0.05 | 0.10 | 0.20 | 0.50 | 1.00
augmentation: standard       # none | standard | strong
subset_seed: 42             # seed for stratified subset selection

# training
backbone_lr: 1.0e-4
head_lr: 1.0e-3
weight_decay: 0.01
epochs: 30
batch_size: 64
patience: 5
seed: 42
warmup_epochs: 0

# output
output_dir: results/
```

A `base.yaml` provides shared defaults. Per-model YAMLs override only the fields that differ (model_name, LRs, warmup). The sweep script merges base + model + CLI overrides and writes one resolved YAML per run -- that resolved YAML is saved alongside results as the reproducibility record.

**Implementation**: use `omegaconf` for YAML merge (`OmegaConf.merge(base, model, cli)`). Config is a Python dataclass for type safety.

## 3. Dependencies to Add

```
uv add timm omegaconf fvcore
```

For iterative stratification: use `scikit-multilearn` if compatible with Python 3.13; otherwise find an alternative maintained library.

## 4. Dataset (`data/voc_dataset.py`)

- Parse `ImageSets/Main/{class}_{train|val}.txt` files (20 classes x 2 splits)
- Each file has lines: `image_id  {1, -1, 0}` (positive, negative, difficult)
- Build a `(N, 20)` binary label matrix from these files
- `__getitem__` returns `(image_tensor, label_vector)`
- Verification script confirms: 5,717 train, 5,823 val, 20 classes, ~1.5 avg labels/image

## 5. Stratified Subsets (`data/stratified_split.py`)

- Iterative stratification (handles multi-label, unlike standard stratified sampling)
- Generate subsets at 5%, 10%, 20%, 50%, 100% of training data
- 3 seeds per fraction = 15 index files saved as `.npy`
- Verify: 5% subset (~286 images) has >=5 examples per class
- Save indices to `data/subsets/` and commit so all team members use identical splits

## 6. Models (`models/factory.py`)

Single `build_model(name, pretrained, num_classes=20)` function returning `(model, param_groups_fn)`:

| Model | Source | Head replacement |
|---|---|---|
| ResNet-50 | `torchvision.models.resnet50` | `model.fc = Linear(2048, 20)` |
| ConvNeXt-T | `torchvision.models.convnext_tiny` | `model.classifier[2] = Linear(768, 20)` |
| ViT-B/16 | `timm` `vit_base_patch16_224.augreg_in21k` | timm handles via `num_classes` arg |
| DeiT-S | `timm` `deit_small_patch16_224` | timm handles via `num_classes` arg |

4 models total (dropped ResNet-101 per instructor suggestion -- ConvNeXt-T is a more interesting CNN comparison point than just a deeper ResNet).

All output raw logits (no sigmoid) -- sigmoid is inside `BCEWithLogitsLoss`.

## 7. Training (`training/trainer.py`)

- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: AdamW with 2 param groups (differential LR)
  - Pretrained ResNets/ConvNeXt: backbone lr=1e-4, head lr=1e-3
  - Pretrained ViTs: backbone lr=1e-5, head lr=1e-3
  - From-scratch ResNets: lr=1e-2, from-scratch ViTs: lr=1e-3 + warmup
- **Scheduler**: CosineAnnealingLR
- **Early stopping**: patience=5 on val mAP
- **Epochs**: 30 max
- **Seeding**: `torch.manual_seed`, `cuda.manual_seed_all`, `np.random.seed`, `random.seed`, deterministic cudnn
- **Logging**: JSONL (one line per epoch, crash-resilient)
- **Checkpointing**: save best model `state_dict` by val mAP

## 8. Augmentation Pipelines (`data/transforms.py`)

| Mode | Transforms |
|---|---|
| `none` | Resize(256), CenterCrop(224), ToTensor, Normalize |
| `standard` | RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter(0.4,0.4,0.4,0.1), ToTensor, Normalize |
| `strong` | standard + RandAugment(num_ops=2, magnitude=9) before ToTensor |
| `val` (always) | Resize(256), CenterCrop(224), ToTensor, Normalize |

## 9. Evaluation

- **mAP**: `sklearn.metrics.average_precision_score(y_true, y_score, average=None)` -> per-class AP, then mean
- **Supported-class mAP**: exclude classes with <5 positives in the training subset
- **Per-class AP**: report all 20 classes for key configurations
- **Efficiency**: param count, FLOPs (fvcore), peak GPU memory, training wall time, avg inference time

## 10. Experiment Grid

### Core experiment: 4 models x 2 pretrain x 5 fractions x 3 seeds = **120 runs**
All use `standard` augmentation.

### Augmentation ablation: 2 best pretrained models x 3 aug levels x 5 fractions x 3 seeds = **30 runs**
(Some overlap with core experiment -- `standard` runs are shared.)

### Total: ~135 unique runs

## 11. SLURM Strategy

- `launch_sweep.py` generates a `configs/generated/` directory with one YAML per run and a `sweep_configs.txt` manifest
- Job array: `#SBATCH --array=0-N%20` (cap 20 concurrent jobs)
- GPU: configurable per model. V100 sufficient for ResNets/ConvNeXt/DeiT-S; A100 for ViT-B/16 (86M params). Default to standard GPU partition.
- Per job: 1 GPU, 8 CPUs, 32GB RAM, 2hr time limit
- Download VOC on login node first (compute nodes may lack internet)
- Batch size fallback: if ViT-B/16 OOMs at batch_size=64, reduce to 32
- **For M2**: only 12 runs needed (4 configs x 3 seeds). Can submit manually or via small job array.

## 12. Results Storage

All output is structured (JSON/JSONL) -- no human-only print formatting. Every run is self-contained and reproducible from its saved config.

```
results/{model}_{pretrained}_{fraction}_{aug}_seed{N}/
  config.yaml        # resolved config (the exact parameters used -- the reproducibility record)
  metrics.jsonl      # per-epoch: {"epoch":1, "train_loss":0.42, "val_loss":0.38, "val_mAP":0.72, ...}
  best_model.pt      # state_dict only
  eval_results.json  # final: mAP, per_class_ap dict, param_count, flops, gpu_mem_mb, timing
```

JSONL for metrics (crash-resilient append, works with `head`/`tail`/`wc -l`). JSON for final eval (one structured object). `aggregate_results.py` collects all `eval_results.json` -> single CSV/DataFrame.

## 13. Key Figures for Report

1. **Centerpiece**: mAP vs data fraction, lines per (model, pretrain), shaded std bands
2. Pretrained vs scratch gap (grouped bars)
3. Per-class AP heatmap (select configs at 5% and 100%)
4. Augmentation ablation (bar chart, top 2 models x 3 aug levels)
5. Efficiency scatter (mAP vs FLOPs, bubble size = param count)

## 14. Implementation Order

| Phase | What | Files | Where |
|---|---|---|---|
| **1. Foundation** | Package structure, config system, dataset loading, transforms, dataset verification | `config.py`, `voc_dataset.py`, `transforms.py`, `stats.py`, `verify_dataset.py` | Local (CPU) |
| **2. Subsets + Models** | Iterative stratification, subset generation, model factory, smoke test all 4 models | `stratified_split.py`, `create_subsets.py`, `factory.py` | Local (CPU) |
| **3. Training** | Training loop, metrics, optimizer, CLI entry point, local smoke test (2 epochs, 5% data, CPU) | `trainer.py`, `metrics.py`, `optimizer.py`, `train.py` | Local (CPU) |
| **4. SLURM + M2 baselines** | SLURM templates, config YAMLs, submit 12 M2 baseline runs | `slurm/`, `launch_sweep.py`, config YAMLs | Rivanna (GPU) |
| **5. Analysis** | Result aggregation, dataset stats report | `aggregate_results.py` | Local (CPU) |
| **6. README** | Document pipeline structure, usage examples, model descriptions, and step-by-step walkthrough | `README.md` | Local |

Phases 1-3 are all local development with no GPU. Phase 4 is the only part requiring Rivanna. The full sweep infrastructure (`launch_sweep.py`, job arrays for 120+ runs) is built in Phase 4 but only the 12 M2 baselines are submitted. Phase 6 writes a comprehensive README after everything works.

## 15. Risks

- **scikit-multilearn on 3.13**: May not install. Find alternative maintained library if needed.
- **ViT scratch on 5% data**: May produce near-random mAP (~5%). Expected -- report as key finding.
- **Queue limits**: Use `--array=%20` to cap concurrent jobs.
- **VOC download on compute nodes**: Download on login node first.

## 16. README (`README.md`)

Rewrite the project README at the end to serve as pipeline documentation. Contents:

1. **Project overview** -- what we're studying (CNN vs ViT data efficiency on multi-label classification) and why
2. **Models** -- description of each architecture (ResNet-50, ConvNeXt-T, ViT-B/16, DeiT-S), what makes them architecturally distinct, and what each comparison tests
3. **Pipeline steps** -- what each script does, in order:
   - `verify_dataset.py`: downloads VOC 2012, reports class distribution and label density
   - `create_subsets.py`: generates iteratively stratified training subsets (5-100%)
   - `train.py`: trains a single model configuration, logs metrics as JSONL
   - `launch_sweep.py`: generates configs and SLURM job array for the full experiment grid
   - `aggregate_results.py`: collects all results into a summary CSV
4. **Quick start** -- brief examples of how to run each step
5. **Config system** -- how configs work (base + model + CLI override), how to inspect with `--dump-config`
6. **Results format** -- description of output directory structure and file formats
7. **Setup** -- `uv sync`, data download, Rivanna environment

## Verification

1. `python scripts/verify_dataset.py` -- confirms dataset stats match expected values
2. `python scripts/create_subsets.py` -- generates subsets, prints per-class counts for 5% subset
3. Smoke test: `python scripts/train.py --config configs/resnet50_pretrained.yaml` with fraction=0.05, epochs=2 -- verify metrics.jsonl is written and mAP is computed
4. Single SLURM job on Rivanna to verify end-to-end
5. Full sweep submission
6. `python scripts/aggregate_results.py` -- produces summary CSV
7. Notebook generates all figures
