# Pre-training and Data Augmentation: What Closes the ViT-CNN Gap?

## Executive Summary

The Vision Transformer (ViT) demonstrated that a pure transformer, applied to sequences of image patches, could match or exceed CNNs on image classification --- but only when pre-trained on datasets of 14 million to 300 million images. With only ImageNet-1K for pre-training (1.28 million images), ViT underperformed established CNNs by a wide margin. Two strategies consistently close this gap: pre-training on large external datasets and aggressive data augmentation. This report focuses on the evidence for both interventions and what they mean for our project, training on PASCAL VOC with roughly 11,000 images.

## Introduction

Our DS 6050 project investigates how Vision Transformers compare to CNNs for image classification across varying training dataset sizes. We plan to run experiments on the PASCAL VOC dataset (~11K images, 20 classes), a setting where data is scarce relative to the benchmarks that typically favor transformers.

This question matters because transformers have rapidly overtaken CNNs as the dominant architecture in many vision tasks, yet their data requirements remain poorly understood in practice. If you're choosing an architecture for a new task with limited labeled data, you face a real tradeoff: transformers offer flexibility and strong scaling behavior, but CNNs may generalize better out of the box.

This report synthesizes findings from a subset of our literature review, focusing specifically on two interventions --- pre-training and data augmentation --- that shape the ViT-CNN comparison most directly.

### Key Concepts

- **Inductive bias**: Assumptions built into a model architecture that constrain the hypothesis space. For CNNs, the key inductive biases are *locality* (each neuron connects only to a small spatial neighborhood) and *translation equivariance* (a shifted input produces a correspondingly shifted output). ViTs lack both --- self-attention operates over all spatial positions with no built-in notion of proximity.

- **Self-attention**: The mechanism at the core of transformers. Given a sequence of patch embeddings, self-attention computes pairwise affinities between all positions and uses those to produce weighted combinations. This gives transformers their flexibility --- and their data hunger.

- **Training from scratch vs. pre-training + fine-tuning**: A critical distinction. "From scratch" means initializing weights randomly and training on the target dataset alone. Pre-training means first training on a large dataset (ImageNet-1K, ImageNet-21K, or JFT-300M), then fine-tuning on the target. Many reported results depend heavily on which regime is used.

- **ImageNet-1K / ImageNet-21K / JFT-300M**: The three dataset scales that recur throughout. ImageNet-1K contains ~1.28 million images across 1,000 classes. ImageNet-21K contains ~14 million images across 21,000 classes. JFT-300M is a Google-internal dataset of ~300 million images. These represent roughly 1x, 10x, and 250x scaling.

## Findings

### The Original Gap: ViT Needs Massive Data

The original Vision Transformer paper [@dosovitskiy2021vit] established the problem. Pre-trained on JFT-300M, ViT-Large/16 achieved 87.76% top-1 on ImageNet. With only ImageNet-1K pre-training, the same model managed only 76.53% --- roughly ResNet-50 territory with far more parameters. The paper varied pre-training scale directly: on ImageNet-1K, BiT ResNets outperformed ViTs, and larger ViT models degraded more than smaller ones (ViT-L underperformed ViT-B). At ImageNet-21K scale (14M images) ViT and CNN performance converged, and only at JFT-300M (300M images) did ViT pull ahead decisively.

The mechanistic explanation comes from Raghu et al. [@raghu2021dovit]: well-trained ViTs learn a mix of local and global attention in their lower layers --- a learned analog of CNN locality. When data is insufficient, **lower layers fail to learn local attention patterns**, and performance collapses. Larger models are more affected; ViT-B/32 tolerated ImageNet-only training, but ViT-L/16 and ViT-H/14 degraded sharply. CNNs never face this failure mode because locality is architectural, not learned.

### Pre-training vs. Training from Scratch

**Pre-training resolves the core failure mode** by supplying enough examples for the lower layers to converge on useful attention patterns before fine-tuning begins. The Mauricio et al. survey [@mauricio2023survey] of 17 studies found pre-training to be an important factor in several individual studies. One study explicitly found that pre-trained CNNs beat ViTs, but ViTs beat non-pre-trained CNNs --- suggesting that training regime can matter more than architecture choice.

CNNs also benefit from pre-training, but less dramatically. Big Transfer (BiT) [@kolesnikov2020bit] showed CNN accuracy scales with pre-training data (ResNet-152x4: 81.3% with ImageNet-1K, 85.4% with ImageNet-21K, 87.5% with JFT-300M), but the relative jump is smaller because CNNs already generalize well at moderate data scales thanks to their inductive biases. BiT also demonstrated strong few-shot transfer: 76.8% on ImageNet with only 10 examples per class.

DeiT [@touvron2021deit] showed the pre-training effect at smaller downstream scales. On CIFAR-10, DeiT-B trained from scratch reached 98.0%, while the same model with ImageNet pre-training and fine-tuning reached 99.1%. The gap is modest on CIFAR-10 but would likely widen on harder tasks with similar data volumes.

The Steiner et al. AugReg study [@steiner2022trainvit] added an important caveat: for small downstream datasets (Oxford Pets ~3K images, RESISC45 ~30K images), **transfer learning from a pre-trained model always outperformed training from scratch**, regardless of how good the augmentation/regularization recipe was. Pre-training is not optional when data is truly scarce.

CoAtNet [@dai2021coatnet] showed that architecture can partially substitute for pre-training scale. A well-designed hybrid (convolution in early stages, attention in later stages) pre-trained on ImageNet-21K matched ViT-Huge pre-trained on JFT-300M --- the same accuracy with **23x less pre-training data**. If you lack access to massive pre-training corpora, hybrid architectures stretch whatever pre-training data you have much further.

### Data Augmentation: Substituting for Data Scale

The AugReg study [@steiner2022trainvit] provides the most systematic investigation. Its central finding: **the right combination of augmentation and regularization can substitute for approximately 10x more training data**. ViT models trained on ImageNet-1K with tuned AugReg matched the same models trained on ImageNet-21K (10x more data) without AugReg. Augmentation helps more than regularization; when compute is limited, invest in augmentation first.

DeiT [@touvron2021deit] demonstrated this concretely: ViTs trained competitively on ImageNet-1K alone using RandAugment, Mixup, CutMix, random erasing, stochastic depth, and repeated augmentation over 300 epochs. DeiT-Base reached 81.8% on ImageNet-1K --- approaching EfficientNet-B4, though at lower throughput. Without these augmentations, vanilla ViT with ImageNet-1K pre-training alone managed only ~76.5%.

DeiT's distillation approach deserves separate mention. Using a CNN teacher to train a ViT student transferred inductive biases through the training signal. CNN teachers produced better ViT students than transformer teachers of comparable accuracy [@touvron2021deit] --- the CNN's implicit knowledge of locality was transmitted through distillation. Hard distillation significantly outperformed soft distillation (83.0% vs. 81.8% for DeiT-B at 224x224), with the best DeiT variant reaching 85.2% at 384x384 resolution.

**CNNs benefit from augmentation too, but the relative gain is smaller.** ConvNeXt [@liu2022convnext] showed that updating a ResNet-50's training recipe (AdamW, 300 epochs, modern augmentation) improved accuracy from 76.1% to 78.8% without any architectural change --- a meaningful 2.7-point gain. But CNNs start from a higher baseline without augmentation because their inductive biases provide built-in regularization. ViTs, lacking those biases, depend on augmentation more heavily to prevent overfitting on limited data.

The ConViT experiments [@dascoli2021convit] quantified the interaction between inductive bias and data scale. Training on subsampled ImageNet fractions, the gap between DeiT-S (no convolutional bias) and ConViT-S (soft convolutional bias) shrank from 37% relative improvement at 5% of ImageNet to just 2% at full ImageNet. The less data you have, the more augmentation and inductive bias matter. At 10% of ImageNet (~128K images), even freezing ConViT's convolutional layers entirely with convolutional initialization (fixed convolutional prior, only FFN layers learning) outperformed DeiT (54.3% vs. 47.8%). With random initialization instead, the frozen model scored only 44.8%.

## Evidence Gaps

Most comparisons use ImageNet-1K (1.28M images) as the "small" dataset. Evaluations below 100K images are rare, and below 10K almost nonexistent. None of the primary papers train on a dataset as small as PASCAL VOC (~11K images) in a standard classification setting. PASCAL VOC is also inherently multi-label, while the ViT-CNN literature focuses on single-label classification --- it is unclear whether data-efficiency relationships transfer to multi-label settings. Many improved ViT training recipes require longer schedules (300+ epochs), making it hard to fully disentangle architectural from compute effects.

## Conclusions

1. **Pre-training is the single most effective intervention for ViTs on limited data.** It resolves the fundamental failure mode --- inability to learn local attention patterns --- by providing enough data to initialize useful representations. For datasets below ~30K images, no augmentation recipe matches the benefit of pre-training.

2. **Aggressive augmentation is the second lever**, and the two compound. DeiT-style augmentation can substitute for roughly 10x more training data. This is essential whether or not you pre-train, but especially critical when training from scratch.

3. **CNNs are less sensitive to both interventions** because their inductive biases provide a strong prior. Pre-trained CNNs remain competitive baselines, and even from-scratch CNNs with modern training recipes match transformer accuracy.

4. **For our PASCAL VOC experiments (~11K images):** pure ViTs from scratch will almost certainly underperform CNNs. Pre-trained ViTs with heavy augmentation are the viable path to competitive transformer performance. Pre-trained CNNs (ResNet, EfficientNet) will serve as strong baselines.

## References

- dosovitskiy2021vit --- Dosovitskiy et al., ICLR (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
- raghu2021dovit --- Raghu et al., NeurIPS (2021). "Do Vision Transformers See Like Convolutional Neural Networks?"
- mauricio2023survey --- Mauricio et al., Applied Sciences (2023). "Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review."
- kolesnikov2020bit --- Kolesnikov et al., ECCV (2020). "Big Transfer (BiT): General Visual Representation Learning."
- touvron2021deit --- Touvron et al., ICML (2021). "Training Data-Efficient Image Transformers & Distillation Through Attention."
- steiner2022trainvit --- Steiner et al., TMLR (2022). "How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers."
- dai2021coatnet --- Dai et al., NeurIPS (2021). "CoAtNet: Marrying Convolution and Attention for All Data Sizes."
- liu2022convnext --- Liu et al., CVPR (2022). "A ConvNet for the 2020s."
- dascoli2021convit --- d'Ascoli et al., ICML (2021). "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases."
