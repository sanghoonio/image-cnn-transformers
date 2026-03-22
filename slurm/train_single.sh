#!/bin/bash
#SBATCH --job-name=voc-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out

# Usage: sbatch slurm/train_single.sh configs/resnet50_pretrained.yaml

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm/train_single.sh <config.yaml>}

# Activate environment (adjust for your Rivanna setup)
source .venv/bin/activate

mkdir -p logs/slurm

echo "Running config: $CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "---"

python scripts/train.py --config configs/base.yaml "$CONFIG"
