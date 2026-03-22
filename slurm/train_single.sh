#!/bin/bash
#SBATCH --job-name='voc-train'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH --mem='32GB'
#SBATCH --time=04:00:00
#SBATCH --account='shakeri_ds6050'
#SBATCH --output='/home/sp5fd/image-cnn-transformers/logs/slurm/%x-%j.out'
#SBATCH --error='/home/sp5fd/image-cnn-transformers/logs/slurm/%x-%j_error.txt'

# Usage: sbatch slurm/train_single.sh configs/resnet50_pretrained.yaml [extra args]

set -euo pipefail

cd /home/sp5fd/image-cnn-transformers

CONFIG=${1:?Usage: sbatch slurm/train_single.sh <config.yaml> [extra args]}
shift
EXTRA_ARGS="$@"

source .venv/bin/activate

mkdir -p logs/slurm

echo "Running config: $CONFIG $EXTRA_ARGS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "---"

python scripts/train.py --config configs/base.yaml "$CONFIG" $EXTRA_ARGS
