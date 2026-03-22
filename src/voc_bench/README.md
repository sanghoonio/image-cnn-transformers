# src/voc_bench/

Python library for the experiment pipeline. Imported by `scripts/` — not run directly.

## Modules

### config.py

Experiment configuration as a Python dataclass with YAML I/O via omegaconf.

- `ExperimentConfig` — dataclass with all experiment parameters
- `load_config(*paths, overrides=)` — merge multiple YAMLs + dict overrides
- `save_config(cfg, path)` — write resolved config to YAML
- `dump_config(cfg)` — return config as YAML string
- `resolve_output_dir(cfg)` — build output path from config fields

### data/

| Module | Purpose |
|--------|---------|
| `voc_dataset.py` | `VOCClassification` dataset class. Parses `ImageSets/Main/{class}_{split}.txt` to build a (N, 20) binary label matrix. Returns (image_tensor, label_vector). |
| `transforms.py` | `get_train_transform(mode)` and `get_val_transform()`. Three train modes: `none` (resize+crop), `standard` (crop+flip+jitter), `strong` (standard + RandAugment). |
| `stratified_split.py` | `iterative_stratification()` using scikit-multilearn. Produces seed-dependent stratified subsets that preserve class frequencies across all 20 labels. |
| `stats.py` | `compute_dataset_stats()` — class counts, label density, split sizes. |

### models/

| Module | Purpose |
|--------|---------|
| `factory.py` | `build_model(name, pretrained, num_classes)` — constructs ResNet-50, ConvNeXt-T, ViT-B/16, or DeiT-S with the classification head replaced for 20-class multi-label output. `get_param_groups()` splits parameters into backbone and head groups for differential learning rates. |

### training/

| Module | Purpose |
|--------|---------|
| `trainer.py` | `train()` — full training loop with BCEWithLogitsLoss, cosine LR schedule, optional warmup, early stopping on val mAP, JSONL per-epoch logging, and best-model checkpointing. `evaluate()` — runs inference and computes mAP. Auto-detects device (CUDA > MPS > CPU). |
| `metrics.py` | `compute_mAP()` — per-class AP via sklearn, mean AP, and supported-class mAP (excludes classes with < N positives). |
| `optimizer.py` | `build_optimizer()` — constructs AdamW with two param groups (backbone LR, head LR). |

### evaluation/

| Module | Purpose |
|--------|---------|
| `efficiency.py` | Parameter count, FLOPs (via fvcore), inference timing with GPU sync, peak memory. |
| `analyze.py` | `aggregate_results()` — walks results dir, loads all eval_results.json into a DataFrame. `summary_table()` — mean +/- std mAP grouped by model/pretrained/fraction. |
