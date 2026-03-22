#!/bin/bash
#SBATCH --job-name=voc-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%A_%a.out

# Usage: sbatch --array=0-119%20 slurm/sweep_array.sh
# The %20 limits concurrent jobs to 20.

set -euo pipefail

MANIFEST=configs/generated/sweep_configs.txt

# Get the config path for this array task
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$CONFIG" ]; then
    echo "Error: No config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Activate environment
source .venv/bin/activate

mkdir -p logs/slurm

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "---"

python scripts/train.py --config "$CONFIG"
