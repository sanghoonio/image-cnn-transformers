# M2 Baseline Results

PASCAL VOC 2012 multi-label classification, 100% training data, standard augmentation, seed=42.

## Summary

| Model | Pretraining | mAP | Best Epoch | Params |
|-------|-------------|-----|------------|--------|
| ResNet-50 | ImageNet | 0.8532 | 4/30 | 23.5M |
| ResNet-50 | Scratch | 0.3092 | 29/30 | 23.5M |
| ViT-B/16 | ImageNet-21K | 0.9274 | 5/30 | 85.8M |
| ViT-B/16 | Scratch | 0.1659 | 3/30 | 85.8M |

## Per-Class AP

| Class | ResNet-50 (pretrained) | ResNet-50 (scratch) | ViT-B/16 (pretrained) | ViT-B/16 (scratch) |
|-------|----------------------|-------------------|---------------------|------------------|
| aeroplane | 0.97 | 0.63 | 1.00 | 0.36 |
| bicycle | 0.85 | 0.19 | 0.94 | 0.11 |
| bird | 0.94 | 0.28 | 0.98 | 0.17 |
| boat | 0.87 | 0.29 | 0.95 | 0.19 |
| bottle | 0.70 | 0.16 | 0.80 | 0.10 |
| bus | 0.93 | 0.52 | 0.95 | 0.17 |
| car | 0.83 | 0.39 | 0.89 | 0.26 |
| cat | 0.97 | 0.37 | 0.99 | 0.21 |
| chair | 0.75 | 0.40 | 0.84 | 0.16 |
| cow | 0.84 | 0.13 | 0.99 | 0.06 |
| diningtable | 0.70 | 0.20 | 0.83 | 0.09 |
| dog | 0.94 | 0.33 | 0.99 | 0.19 |
| horse | 0.93 | 0.14 | 0.99 | 0.06 |
| motorbike | 0.89 | 0.31 | 0.96 | 0.12 |
| person | 0.95 | 0.70 | 0.97 | 0.53 |
| pottedplant | 0.63 | 0.12 | 0.74 | 0.07 |
| sheep | 0.91 | 0.19 | 0.99 | 0.14 |
| sofa | 0.66 | 0.20 | 0.84 | 0.09 |
| train | 0.96 | 0.31 | 1.00 | 0.13 |
| tvmonitor | 0.86 | 0.33 | 0.92 | 0.10 |

Source: `results/*/eval_results.json`, verified 2026-03-22.
