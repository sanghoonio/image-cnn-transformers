# scripts/

Standalone CLI entry points. Each script imports from `src/voc_bench/` and is run directly with `python scripts/<name>.py`.

## Scripts

### verify_dataset.py

Downloads PASCAL VOC 2012 and prints dataset statistics (class distribution, label density).

```bash
python scripts/verify_dataset.py --data-root data/VOCdevkit --download
python scripts/verify_dataset.py --output data/dataset_stats.json
```

### create_subsets.py

Generates iteratively stratified training subsets at 5%, 10%, 20%, 50%, 100% with 3 seeds each. Saves index files to `data/subsets/`.

```bash
python scripts/create_subsets.py --data-root data/VOCdevkit
```

### train.py

Trains a single model configuration. This is the main workhorse.

```bash
# Basic usage — merge base + model config
python scripts/train.py --config configs/base.yaml configs/resnet50_pretrained.yaml

# Override parameters via CLI
python scripts/train.py --config configs/base.yaml configs/vit_b16_pretrained.yaml \
    --fraction 0.05 --seed 123

# Preview resolved config without training
python scripts/train.py --config configs/base.yaml configs/resnet50_pretrained.yaml --dump-config
```

Outputs to `results/<run_name>/`: config.yaml, metrics.jsonl, best_model.pt, eval_results.json.

### launch_sweep.py

Generates per-run config files for the full experiment grid and prints the SLURM submission command.

```bash
python scripts/launch_sweep.py                    # generate configs
python scripts/launch_sweep.py --dry-run           # preview without writing
```

### aggregate_results.py

After training runs complete, collects all `eval_results.json` files into a summary CSV.

```bash
python scripts/aggregate_results.py --results-dir results --output results/summary.csv
```
