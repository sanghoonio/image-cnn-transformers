# slurm/

SLURM job templates for running experiments on UVA Rivanna HPC.

## Files

### train_single.sh

Run a single training configuration.

```bash
sbatch slurm/train_single.sh configs/resnet50_pretrained.yaml
```

### sweep_array.sh

Run the full experiment grid as a SLURM job array. Reads config paths from `configs/generated/sweep_configs.txt` (created by `scripts/launch_sweep.py`).

```bash
# Generate configs first
python scripts/launch_sweep.py

# Submit with 20 concurrent job limit
sbatch --array=0-119%20 slurm/sweep_array.sh
```

## Resource defaults

- 1 GPU, 8 CPUs, 32GB RAM, 2hr time limit
- Logs to `logs/slurm/`

## Setup on Rivanna

```bash
# On login node (has internet):
git pull
uv sync
uv pip install -e .
python scripts/verify_dataset.py --data-root data/VOCdevkit --download

# Then submit jobs from project root
```
