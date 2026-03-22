#!/bin/bash
# Pre-download all pretrained model weights on the login node.
# Compute nodes have no internet, so weights must be cached first.
# Disables SSL verification because Rivanna's certificates are broken.
#
# Usage (on Rivanna login node):
#   bash scripts/download_weights.sh

set -euo pipefail

cd /home/sp5fd/image-cnn-transformers
source .venv/bin/activate

python -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision.models as tv
import timm

print('Downloading ResNet-50 (ImageNet-1K)...')
tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)

print('Downloading ConvNeXt-T (ImageNet-1K)...')
tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

print('Downloading ViT-B/16 (ImageNet-21K)...')
timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)

print('Downloading DeiT-S (ImageNet-1K)...')
timm.create_model('deit_small_patch16_224', pretrained=True)

print('All weights cached.')
"
