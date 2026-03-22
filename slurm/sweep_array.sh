#!/bin/bash
#SBATCH --job-name='voc-sweep'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH --mem='32GB'
#SBATCH --time=04:00:00
#SBATCH --account='shakeri_ds6050'
#SBATCH --output='/home/sp5fd/image-cnn-transformers/logs/slurm/%A_%a.out'
#SBATCH --error='/home/sp5fd/image-cnn-transformers/logs/slurm/%A_%a_error.txt'

# Usage: sbatch --array=0-119%20 slurm/sweep_array.sh

set -euo pipefail

cd /home/sp5fd/image-cnn-transformers

MANIFEST=configs/generated/sweep_configs.txt

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$CONFIG" ]; then
    echo "Error: No config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

source .venv/bin/activate

mkdir -p logs/slurm

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "---"

python scripts/train.py --config "$CONFIG"
