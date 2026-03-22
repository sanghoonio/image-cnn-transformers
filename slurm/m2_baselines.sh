#!/bin/bash
# Submit the 12 M2 baseline runs on Rivanna.
# ResNet-50 + ViT-B/16, pretrained + scratch, 3 seeds each.
#
# Usage: bash slurm/m2_baselines.sh

set -euo pipefail

mkdir -p /home/sp5fd/image-cnn-transformers/logs/slurm

for model in resnet50_pretrained resnet50_scratch vit_b16_pretrained vit_b16_scratch; do
  for seed in 42 123 456; do
    echo "Submitting ${model} seed=${seed}"
    sbatch slurm/train_single.sh configs/${model}.yaml --seed $seed
  done
done

echo "Submitted 12 jobs."
