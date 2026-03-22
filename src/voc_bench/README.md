# src/voc_bench/

Python library for the experiment pipeline. Imported by `scripts/` — not run directly.

Built entirely on PyTorch. Models are loaded from `torchvision` (ResNet-50, ConvNeXt-T) and `timm` (ViT-B/16, DeiT-S) with only the final classification head replaced for 20-class multi-label output — no custom architectures. Augmentation uses `torchvision.transforms`. Metrics use `sklearn`. No custom algorithm implementations.

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
| `trainer.py` | `train()` — full training loop. `evaluate()` — runs inference and computes mAP. Auto-detects device (CUDA > MPS > CPU). See details below. |
| `metrics.py` | `compute_mAP()` — per-class AP via sklearn on sigmoid-transformed scores vs binary labels, then averaged. Also computes `supported_mAP` which excludes classes with fewer than N positive examples (useful at small data fractions where rare classes have too few examples for reliable AP). |
| `optimizer.py` | `build_optimizer()` — constructs AdamW with two parameter groups: a lower LR for the pretrained backbone (preserves learned features) and a higher LR for the new classification head (must learn from scratch). When training from scratch, both groups use the same rate. |

#### trainer.py details

| Component | What it does |
|-----------|-------------|
| **BCEWithLogitsLoss** | Binary Cross-Entropy on raw model outputs (logits). Since each image can have multiple labels, we treat classification as 20 independent yes/no predictions. The "with logits" part means sigmoid is applied inside the loss for numerical stability, rather than applying sigmoid to the model output first. |
| **Cosine LR schedule** | Learning rate decays following a cosine curve toward zero over the training run. Aggressive updates early when far from optimal, gentler updates as the model converges. Standard schedule for both CNNs and ViTs. |
| **Warmup** | For ViTs from scratch, LR starts near zero and linearly ramps up over the first N epochs (default 5) before cosine decay begins. ViTs are sensitive to large early gradients because randomly initialized attention weights produce unstable updates. ResNets don't need this. |
| **Early stopping** | After each epoch, mAP is computed on validation. If no improvement for `patience` consecutive epochs (default 5), training stops. Prevents overfitting, especially on small data fractions. |
| **Best-model checkpointing** | Every time val mAP improves, model weights are saved to `best_model.pt`. At the end of training, the best checkpoint is reloaded for final evaluation. Reported results come from the best epoch, not the last. |
| **JSONL logging** | One JSON line appended to `metrics.jsonl` per epoch: `{"epoch": 1, "train_loss": 0.42, "val_mAP": 0.72, ...}`. Append-per-line means partial logs survive job crashes. These are the training curves for the report. |

### evaluation/

| Module | Purpose |
|--------|---------|
| `efficiency.py` | Parameter count, FLOPs (via fvcore), inference timing with GPU sync, peak memory. |
| `analyze.py` | `aggregate_results()` — walks results dir, loads all eval_results.json into a DataFrame. `summary_table()` — mean +/- std mAP grouped by model/pretrained/fraction. |
