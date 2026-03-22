#!/bin/bash
# Submit minimum M2 baseline runs on Rivanna.
# 2 CNNs + 2 ViTs, pretrained + scratch, single seed each.
#
# Usage: bash slurm/m2_baselines.sh

set -euo pipefail

mkdir -p /home/sp5fd/image-cnn-transformers/logs/slurm

for model in resnet50_pretrained resnet50_scratch vit_b16_pretrained vit_b16_scratch; do
  echo "Submitting ${model} seed=42"
  sbatch slurm/train_single.sh configs/${model}.yaml --seed 42
done

echo "Submitted 4 jobs."
