# Vision Transformers vs. CNNs: How Dataset Size Shapes Image Classification Performance

## Executive Summary

The introduction of the Vision Transformer (ViT) in 2020 demonstrated that a pure transformer architecture, applied directly to sequences of image patches, could match or exceed state-of-the-art CNNs on image classification — but only when pre-trained on datasets of 14 million to 300 million images. Trained from scratch on ImageNet-1K (1.28 million images), ViT underperformed well-established CNNs by a wide margin. That gap prompted a lot of follow-up work asking *why* transformers struggle with limited data and *how* to fix it.

The answer is fairly consistent across papers: CNNs benefit from strong inductive biases — locality and translation equivariance — that are hardcoded into their architecture. Transformers lack these priors and must learn equivalent structure from data, which requires far more examples. When data is scarce, ViTs fail to learn local attention patterns in their early layers, a failure mode that has no CNN analog. But the gap is not inherent to the transformer paradigm. Data augmentation and regularization can substitute for roughly an order of magnitude more training data. Knowledge distillation from CNN teachers transfers inductive biases to transformer students. Hybrid architectures that combine convolutional and attention layers perform well across all data regimes. And purely convolutional networks, when equipped with modern training recipes and transformer-inspired design choices, can match transformer accuracy outright.

For our project, training on PASCAL VOC with roughly 11,000 images, the implications are concrete. Pure ViTs trained from scratch will almost certainly underperform CNNs in this regime. Pre-trained ViTs (with ImageNet or larger), aggressive augmentation, distillation-based training, or hybrid architectures represent the most viable paths to competitive transformer performance at this scale.

## Introduction

Our DS 6050 project investigates a central question in modern computer vision: **How do Vision Transformers compare to CNNs for image classification across varying training dataset sizes?** We plan to run experiments on the PASCAL VOC dataset (~11K images, 20 classes), a setting where data is scarce relative to the benchmarks that typically favor transformers.

This question matters because transformers have rapidly overtaken CNNs as the dominant architecture in many vision tasks, yet their data requirements remain poorly understood in practice. If you're choosing an architecture for a new task with limited labeled data, you face a real tradeoff: transformers offer flexibility and strong scaling behavior, but CNNs may generalize better out of the box. Understanding where each architecture works well, and why, directly affects experimental design decisions.

This report synthesizes findings from 16 papers spanning the period 2015–2023, covering the original ViT proposal, data-efficient training methods, mechanistic analyses of representation learning, hybrid architectures, modernized CNNs, and a systematic literature review. The goal is to give every group member a grounded understanding of what the literature says, what it does not say, and what it means for our experimental design.

### Key Concepts

A few concepts come up constantly across these papers:

- **Inductive bias**: Assumptions built into a model architecture that constrain the hypothesis space. For CNNs, the key inductive biases are *locality* (each neuron connects only to a small spatial neighborhood) and *translation equivariance* (a shifted input produces a correspondingly shifted output). ViTs lack both — self-attention operates over all spatial positions with no built-in notion of proximity.

- **Self-attention**: The mechanism at the core of transformers. Given a sequence of patch embeddings, self-attention computes pairwise affinities between all positions and uses those to produce weighted combinations. This is what gives transformers their flexibility — and their data hunger.

- **Training from scratch vs. pre-training + fine-tuning**: A critical distinction in this literature. "From scratch" means initializing weights randomly and training on the target dataset alone. Pre-training means first training on a large dataset (ImageNet-1K, ImageNet-21K, or JFT-300M), then fine-tuning on the target. Many reported results depend heavily on which regime is used.

- **ImageNet-1K / ImageNet-21K / JFT-300M**: The three dataset scales that recur throughout. ImageNet-1K contains ~1.28 million images across 1,000 classes. ImageNet-21K contains ~14 million images across 21,000 classes. JFT-300M is a Google-internal dataset of ~300 million images. These represent roughly 1x, 10x, and 250x scaling.

## Findings

### The Original Gap: ViT Needs Massive Data

The original Vision Transformer paper [@dosovitskiy2021vit] laid out the problem. Dosovitskiy et al. split images into fixed-size patches (typically 16x16 pixels), linearly embedded them, prepended a classification token, added positional embeddings, and fed the resulting sequence through a standard transformer encoder. The architecture is deliberately minimal — no convolutions, no hierarchical processing, no built-in locality.

The results cut both ways. Pre-trained on JFT-300M (300 million images), ViT-Large/16 achieved 87.76% top-1 accuracy on ImageNet, surpassing the best CNNs at the time. But trained from scratch on ImageNet-1K alone, the same ViT-Large/16 managed only around 76.5% — roughly the accuracy of a basic ResNet-50 with a fraction of the parameters [@dosovitskiy2021vit]. The paper states this directly: transformers "do not generalize well when trained on insufficient amounts of data."

The paper also varied pre-training dataset size directly. On small datasets like CIFAR-10 (50K images) and ImageNet-1K, a BiT ResNet outperformed ViT, with the gap widening as models grew larger. Only at the scale of ImageNet-21K (14M images) did ViT begin to match CNN performance, and only at JFT-300M did ViT pull ahead decisively. As the authors summarized: there exists a "performance crossover" between architectures, and it occurs at a dataset scale beyond what most practitioners have access to.

### Why ViTs Fail on Small Data: A Mechanistic View

Raghu et al. [@raghu2021dovit] opened the hood on ViT internals and compared them directly to CNN representations. Their analysis turned up several differences that help explain the data-efficiency gap:

First, ViTs have a much more **uniform representation structure** across layers. In a ResNet, lower layers capture local features (edges, textures) and higher layers capture global features (object parts, scenes), with clear transitions between stages. In a ViT, representations are more similar across the full depth of the network — the lower layers already contain a mix of local and global information.

Second, in the lower layers of a well-trained ViT (pre-trained on JFT-300M), some attention heads learn to attend locally (nearby patches) while others attend globally (distant patches). This mix of local and global attention is a learned analog of the hard-coded locality in CNNs. But here is the critical finding: **when trained with insufficient data, the lower attention layers do not learn to attend locally** [@raghu2021dovit]. The models that fail to learn local attention patterns are exactly the ones that perform poorly. Since CNNs have locality hardcoded through their architecture, they never face this failure mode.

Third, the effect is worse for larger models. ViT-B/32 (the smallest variant tested) still managed to learn local attention when trained on ImageNet alone and showed "similar performance in both settings" (JFT-pretrained vs. ImageNet-only). But ViT-L/16 and ViT-H/14 showed clear degradation, suggesting that larger models are more under-specified and require more data to converge to useful representations.

The dataset scale effects were not uniform across the network. Even with only 3% of the training data, lower-layer representations closely resembled those learned from the full dataset. It was the higher layers — responsible for more abstract, task-specific representations — that required larger amounts of data to converge [@raghu2021dovit].

### Closing the Gap Through Training Recipes

A recurring finding across these papers: much of the perceived architectural gap between ViTs and CNNs is actually a gap in training methodology.

**DeiT** [@touvron2021deit] demonstrated that ViTs can be trained competitively on ImageNet-1K alone, without any external data, given the right training recipe. The key ingredients were aggressive data augmentation (RandAugment, Mixup, CutMix, random erasing), careful regularization (stochastic depth, repeated augmentation), and a long training schedule (300 epochs). DeiT-Base achieved 81.8% top-1 on ImageNet-1K, and with a higher resolution fine-tuning step, DeiT-Base reached 83.1% — competitive with EfficientNet-B4 at comparable compute.

DeiT also introduced a **distillation token** that enables knowledge transfer from a CNN teacher to a ViT student. Interestingly, CNN teachers produced better ViT students than transformer teachers, even when the transformer teacher was more accurate [@touvron2021deit]. The authors attributed this to the transfer of inductive biases — the CNN teacher's implicit knowledge of locality and translation equivariance was transmitted through the distillation process. Hard distillation (using the teacher's argmax labels) outperformed soft distillation (using the teacher's probability distribution), with DeiT reaching 85.2% when combining distillation with fine-tuning at 384x384 resolution.

The smaller dataset experiments in DeiT are worth noting for our project. Training DeiT-Small from scratch on CIFAR-10 (50K images) yielded 97.5% accuracy, compared to 98.5% with ImageNet pre-training and 99.1% with pre-training followed by fine-tuning. The gap between scratch and pre-trained is meaningful but not catastrophic on CIFAR-10 — though the gap would likely widen on more complex tasks with similar data volumes.

**Steiner et al.** [@steiner2022trainvit] ran a large-scale study of training recipe effects, coining the term "AugReg" (augmentation + regularization). Their central finding: **the right combination of augmentation and regularization can substitute for approximately 10x more training data**. ViT models of various sizes trained on ImageNet-1K with carefully tuned AugReg matched the performance of the same models trained on ImageNet-21K (10x more data) without AugReg.

Some practical takeaways from this work. First, augmentation tends to help more than regularization; when compute is limited, invest in augmentation first [@steiner2022trainvit]. Second, more data leads to more generic models that transfer better, but AugReg partially closes this gap. Third, for small downstream datasets (e.g., Oxford Pets with ~3K images or RESISC45 with ~30K images), transfer learning from a pre-trained model always outperformed training from scratch, regardless of how good the AugReg recipe was. Fourth, when choosing model variants under a fixed compute budget, prefer larger patch sizes (e.g., /32 over /16) over thinner models — the reduced spatial resolution hurts less than the loss of model capacity.

### Hybrid Architectures: Combining the Best of Both

If ViTs lack inductive bias and CNNs lack flexible long-range attention, why not combine them? Several papers explored this, and the results are consistently encouraging.

**CoAtNet** [@dai2021coatnet] systematically studied how to stack convolution and attention layers for different data regimes. The paper articulated the core tradeoff clearly: "convolution generalizes better, attention has higher model capacity." Dai et al. evaluated every possible combination of convolution (C) and transformer (T) stages in a four-stage architecture and found:

- On ImageNet-1K (the low-data regime, relatively speaking), more convolution was better: C-C-C-C ≈ C-C-C-T ≥ C-C-T-T > C-T-T-T >> pure ViT. Fully convolution-based variants and those with convolution in the early stages had the smallest generalization gaps.
- On JFT (the high-data regime), more transformer stages won: C-C-T-T ≈ C-T-T-T > pure ViT > C-C-C-T > C-C-C-C. The flexible attention layers could leverage the additional data, while convolution layers saturated.

CoAtNet achieved 86.0% top-1 on ImageNet-1K alone (matching the best pure CNNs) and 88.56% with ImageNet-21K pre-training, matching ViT-Huge pre-trained on JFT-300M — with **23x less pre-training data** [@dai2021coatnet]. Put differently: a well-designed hybrid can match a pure transformer that saw 23 times more data during pre-training. The paper also explicitly attributed the generalization advantage of convolution to **translation equivariance**, noting that standard ViT's absolute positional embeddings lack this property, "which partially explains why ConvNets are usually better than Transformers when the dataset is not enormously large."

**ConViT** [@dascoli2021convit] took a different approach: instead of stacking separate convolution and attention layers, it introduced a soft mechanism that lets each attention head choose its own balance between local (convolutional) and global (transformer-like) attention. The gated positional self-attention (GPSA) mechanism initializes attention heads to behave like convolutions and then allows each head to "escape locality" through a learned gating parameter.

The sample efficiency experiments in ConViT are the cleanest quantification of how inductive bias scales with data. Training ConViT-S and DeiT-S on subsampled fractions of ImageNet:

| Data fraction | DeiT-S | ConViT-S | Relative improvement |
|:---:|:---:|:---:|:---:|
| 5% | 34.8% | 47.8% | +37% |
| 10% | 48.0% | 59.6% | +24% |
| 30% | 66.1% | 73.7% | +12% |
| 50% | 74.6% | 78.2% | +5% |
| 100% | 79.9% | 81.4% | +2% |

The pattern is clear: **the benefit of convolutional inductive bias is inversely proportional to dataset size** [@dascoli2021convit]. At 5% of ImageNet (about 64K images — comparable to PASCAL VOC scale), the soft convolutional bias provides a 37% relative improvement. At 100% of ImageNet, the benefit shrinks to 2%. d'Ascoli et al. also found that the benefit of inductive bias increases with model size — larger models are more under-specified and benefit more from the convolutional prior, particularly on small datasets.

One experiment from ConViT really drives this home: **freezing** the GPSA layers entirely (so they act as fixed random convolutions, with only the FFN layers learning) produced a model that still outperformed DeiT on 10% of ImageNet data (54.3% vs. 47.8%). Even a random convolutional prior was better than no prior at all in the low-data regime.

**DHVT** (Dynamic Hybrid Vision Transformer) [@lu2022bridging] specifically targeted the small-dataset regime. Lu et al. identified two structural problems that explain ViT's failure on small datasets: (1) inability to learn local spatial attention patterns (confirming Raghu et al.'s finding), and (2) failure to develop diverse channel representations, since scarce data prevents individual attention heads from learning sufficiently different features.

DHVT addressed both problems through a Dynamic Aggregation FFN (integrating depthwise convolution and squeeze-excitation into the feed-forward network) and a novel "head token" mechanism that forces different attention heads to interact and produce complementary representations. The CIFAR-100 results (50K training images, 100 classes) speak for themselves [@lu2022bridging]:

- Pure DeiT-S: 66.55% (worse than the smaller DeiT-T at 67.59% — scaling up *hurts*)
- Swin-T: 78.07%
- Strong CNNs (Res2NeXt-29, etc.): 82–83%
- DHVT-S: **85.68%** — surpassing all compared CNNs

Worth pausing on that last number: DHVT-S trained from scratch on CIFAR-100 outperformed a ResNet-50 that had been pre-trained on ImageNet-1K and then fine-tuned on CIFAR-100 (85.68% vs. 85.44%) [@lu2022bridging]. A well-designed hybrid transformer, trained entirely on 50K images, beat a CNN that had the benefit of 1.28 million images of pre-training.

### The Role of Convolutional Stems

**Xiao et al.** [@xiao2021earlyconv] zeroed in on a seemingly minor architectural detail that turns out to matter a lot: ViT's patchify stem. The standard ViT uses a single strided convolution (16x16 kernel, stride 16) to convert image patches into token embeddings — an aggressive downsampling that runs counter to decades of CNN design wisdom, where small strides and gradual spatial reduction are standard.

Replacing this patchify stem with a stack of five stride-2, 3x3 convolutions (dubbed ViT_C) improved nearly everything: 1-2% accuracy gains on ImageNet-1K, faster convergence, robustness to optimizer choice (the accuracy gap between AdamW and SGD shrank from ~10% to less than 0.2% for larger models), and better hyperparameter stability [@xiao2021earlyconv].

Here's the uncomfortable part: on ImageNet-1K, the baseline ViT with patchify stem (ViT_P) **underperformed both RegNetY and ResNet** across the entire model complexity spectrum [@xiao2021earlyconv]. ViT_P didn't outperform state-of-the-art CNNs even with ImageNet-21K pre-training. Only ViT_C managed to surpass CNNs, and only when given the ImageNet-21K pre-training advantage. The benefit of the convolutional stem was consistent across all data scales, suggesting it is not merely a small-data crutch but a structural improvement to ViT's optimization landscape.

### The CNN Strikes Back: ConvNeXt

If so many strategies for improving ViT involve adding CNN-like components, what happens if you go the other direction? Can CNNs absorb transformer-inspired design choices and close the gap from their side?

**Liu et al.** [@liu2022convnext] showed that they can, convincingly, with ConvNeXt. Starting from a standard ResNet-50 and incrementally applying design choices from transformers — stage compute ratios, patchify-like downsampling, depthwise convolutions, inverted bottleneck blocks, larger kernels (7x7), GELU activation, LayerNorm, fewer activation functions — they produced a pure CNN that matched or exceeded Swin Transformer performance at every scale.

The first step alone — simply updating the training recipe (AdamW optimizer, 300 epochs, modern augmentation) without changing the architecture — improved ResNet-50 from 76.1% to 78.8%, a 2.7-point gain [@liu2022convnext]. Again: training recipes account for a surprising fraction of the perceived CNN-vs-transformer gap.

ConvNeXt accuracy on ImageNet-1K trained from scratch [@liu2022convnext]:
- ConvNeXt-T (29M params): 82.1% vs. Swin-T: 81.3%
- ConvNeXt-S (50M params): 83.1% vs. Swin-S: 83.0%
- ConvNeXt-B (89M params): 83.8% vs. Swin-B: 83.5%

With ImageNet-22K pre-training:
- ConvNeXt-L (384x384): 87.5% vs. Swin-L (384x384): 87.3%
- ConvNeXt-XL (384x384): 87.8%

These results complicate the narrative that transformers are inherently superior at large scale. Liu et al. explicitly address the "widely held view" that transformers benefit more from large-scale pre-training, finding that "properly designed ConvNets are not inferior to vision Transformers when pre-trained with large datasets" [@liu2022convnext]. ConvNeXt also matched or beat Swin on COCO detection and ADE20K segmentation, tasks where transformers were thought to have a clearer advantage.

### What the Survey Literature Says

Mauricio et al. [@mauricio2023survey] conducted a systematic review of 17 studies comparing ViTs and CNNs across various image classification tasks (medical imaging, agriculture, traffic signs, safety-critical systems, etc.). The findings were more mixed than the benchmark-focused papers might suggest:

- In most reviewed studies, ViT-based models matched or outperformed CNNs, particularly for medical imaging and tasks with complex spatial relationships.
- However, findings on small-dataset performance were **conflicting**. Some studies found ViTs performed well on small datasets due to the self-attention mechanism extracting richer patch relationships. Others found ViTs failed to generalize when training data was limited, with large gaps between training and test accuracy.
- **Pre-training was identified as crucial** across studies. Without pre-training, ViTs generally underperformed CNNs; with pre-training, the advantage reversed.
- **Hybrid models (combining CNN and ViT components) often outperformed both standalone architectures** in the reviewed studies.

The survey's bottom line: "CNN's can generalize better with smaller datasets and get better accuracy than ViTs, but ViTs have the advantage of learning information better with fewer images" — a distinction between *generalization* (test-time performance on unseen data) and *representation learning* (extracting information from individual examples) [@mauricio2023survey]. That distinction is worth keeping in mind as we design our experiments.

### Additional Context from Foundational and Supporting Work

Several foundational CNN papers and supporting transformer variants fill in context that the primary papers assume.

**ResNet** [@he2016resnet] introduced skip connections and enabled training of networks exceeding 100 layers. It remains the dominant CNN baseline in virtually every transformer comparison. ResNet-50 achieves roughly 76–77% top-1 on ImageNet-1K with standard training — the number against which most ViT results are compared. ResNet was trained solely on ImageNet-1K, which shows CNNs can reach strong accuracy with moderate-sized datasets and no external pre-training.

**VGGNet** [@simonyan2015vggnet] demonstrated the "depth hypothesis" for CNNs and achieved strong results on PASCAL VOC, the dataset we plan to use. VGG's parameter count (138–144M) now looks enormous by modern standards — both CNNs and transformers achieve far higher accuracy with far fewer parameters — but its principle that deeper representations improve classification quality carries through to both architecture families.

**EfficientNet** [@tan2019efficientnet] was the state of the art in CNN design right before transformers arrived. Through compound scaling of depth, width, and resolution, EfficientNet-B7 achieved 84.3% top-1 on ImageNet-1K with 66M parameters — no pre-training on larger datasets required. This was the ceiling transformers had to beat, and DeiT explicitly benchmarked against it. EfficientNet also transferred well to smaller datasets (91.7% on CIFAR-100).

**Swin Transformer** [@liu2021swin] showed that transformers could borrow CNN design principles (hierarchical feature maps, local windowed attention with linear complexity) and compete on ImageNet-1K without needing JFT-scale data. Swin-T achieved 81.3% on ImageNet-1K, surpassing both DeiT-S (79.8%) and ResNet-50. With ImageNet-22K pre-training, Swin-L reached 87.3%. Swin was the first transformer to work well as a general-purpose vision backbone across classification, detection, and segmentation at practical data scales.

**T2T-ViT** [@yuan2021t2tvit] addressed vanilla ViT's training inefficiency by introducing a progressive tokenization module that models local structure before feeding patches into the transformer. T2T-ViT-14 achieved 81.5% on ImageNet-1K trained from scratch, outperforming DeiT-Small (79.8%) without requiring a CNN teacher for distillation, and surpassing ResNets of similar size by 1.4–2.7%. This was early evidence that ViT's data inefficiency is a design problem, not a fundamental limitation of attention.

**Big Transfer (BiT)** [@kolesnikov2020bit] provides the CNN-side perspective on scaling pre-training data. Using a ResNet-152x4 with Group Normalization and Weight Standardization, Kolesnikov et al. showed that CNN performance scales strongly with pre-training data: 81.3% with ImageNet-1K, 85.4% with ImageNet-21K, and 87.5% with JFT-300M. A finding that applies to both CNNs and transformers: **model capacity and dataset size must scale in tandem**. Larger models don't benefit from small datasets, and small models don't benefit from large datasets. BiT also showed strong few-shot transfer: with only 10 examples per class on ImageNet, BiT-L hit 76.8%, a reminder that CNNs with good pre-training can be quite data-efficient at downstream tasks.

## Evidence Gaps and Limitations

There are real gaps in what these papers cover:

**Limited evaluation on truly small datasets.** Most comparisons use ImageNet-1K (1.28M images) as the "small" dataset. Evaluations on datasets below 100K images are rare, and those below 10K images are almost nonexistent. CIFAR-10/100 (50K images) appears in several papers, and DHVT [@lu2022bridging] and ConViT [@dascoli2021convit] provide data at subsampled ImageNet scales. But none of the primary papers train on a dataset as small as PASCAL VOC (~11K images) in a standard classification setting. Our experimental results will be partially extrapolating from observed trends rather than directly matching the literature's experimental conditions.

**PASCAL VOC is absent from the ViT-vs-CNN comparison literature.** VGGNet [@simonyan2015vggnet] achieved strong results on PASCAL VOC for detection and segmentation, but the recent ViT-vs-CNN literature does not include PASCAL VOC classification comparisons. This is likely because PASCAL VOC is considered dated as a classification benchmark and does not have a standard train/test protocol for the multi-label classification task it was originally designed for. We should be mindful that our experimental setup may not perfectly align with any published comparison.

**Multi-label vs. single-label classification.** PASCAL VOC is inherently a multi-label dataset (images often contain multiple object classes). The vast majority of the ViT-vs-CNN comparison literature focuses on single-label classification (ImageNet, CIFAR). It is unclear whether the data-efficiency relationships observed for single-label tasks transfer directly to multi-label settings, where the supervision signal per image is richer but the task is also more complex.

**Pre-training dataset overlap.** Many of the reported results involve models pre-trained on ImageNet-21K or JFT-300M. When these models are evaluated on ImageNet-1K (a subset of ImageNet-21K), the evaluation and pre-training distributions overlap. The extent to which this inflates reported accuracy is rarely discussed. For our project, we would be fine-tuning ImageNet-pre-trained models on PASCAL VOC, which has minimal overlap with ImageNet's categories.

**Training compute as a confound.** Many of the improved ViT training recipes (DeiT, AugReg) require longer training schedules (300+ epochs) and heavier augmentation than standard CNN training. Whether the improvements stem from architectural changes or simply from spending more compute during training is not always clearly disentangled.

**No theoretical framework.** Raghu et al. and d'Ascoli et al. offer empirical and mechanistic explanations for the data-efficiency gap, but nobody has a theory that predicts, given a dataset size and task complexity, which architecture family will perform better. The heuristics work, but they're empirical.

## Conclusions

Across these 16 papers, several conclusions hold up consistently:

1. **Pure ViTs trained from scratch need substantially more data than CNNs to reach competitive performance.** At ImageNet-1K scale (~1.28M images), vanilla ViT lags CNNs by a significant margin. At smaller scales (50K images on CIFAR-100), the gap widens to 15–20 percentage points. The root cause is the absence of inductive biases — particularly locality — that must be learned from data in transformers but are hardcoded in CNNs.

2. **The gap is not inherent to the transformer paradigm.** It can be substantially closed through: (a) training recipes — augmentation, regularization, and distillation (DeiT, AugReg); (b) architectural modifications — convolutional stems (ViT_C), hybrid convolution-attention stacking (CoAtNet), soft inductive biases (ConViT), and specialized small-data designs (DHVT); (c) pre-training on larger datasets followed by fine-tuning.

3. **CNNs are not obsolete.** ConvNeXt demonstrates that a pure CNN, equipped with modern design choices and training recipes, matches or exceeds transformer accuracy at every data scale tested — including with large-scale pre-training. The performance gap between architectures has historically been confounded with differences in training methodology.

4. **Hybrid architectures are the most robust choice across data regimes.** CoAtNet matched ViT-Huge/JFT-300M accuracy using only ImageNet-21K pre-training (23x less data). Hybrids get CNN generalization at small scale and transformer capacity at large scale.

5. **For our PASCAL VOC experiments (~11K images), we should expect:**
   - Pure ViTs trained from scratch to perform poorly relative to CNNs.
   - Transfer learning (pre-training on ImageNet, fine-tuning on VOC) to be essential for competitive ViT performance.
   - Heavy data augmentation (following DeiT/AugReg recipes) to partially compensate for limited data.
   - Pre-trained CNNs (ResNet, EfficientNet) to serve as strong baselines.
   - Hybrid architectures or ViTs with convolutional stems to offer the best balance if we want to compare transformer capabilities at this scale.

## References

- dosovitskiy2021vit — Dosovitskiy et al., ICLR (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
- touvron2021deit — Touvron et al., ICML (2021). "Training Data-Efficient Image Transformers & Distillation Through Attention."
- steiner2022trainvit — Steiner et al., TMLR (2022). "How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers."
- liu2022convnext — Liu et al., CVPR (2022). "A ConvNet for the 2020s."
- dascoli2021convit — d'Ascoli et al., ICML (2021). "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases."
- lu2022bridging — Lu et al., NeurIPS (2022). "Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets."
- raghu2021dovit — Raghu et al., NeurIPS (2021). "Do Vision Transformers See Like Convolutional Neural Networks?"
- dai2021coatnet — Dai et al., NeurIPS (2021). "CoAtNet: Marrying Convolution and Attention for All Data Sizes."
- mauricio2023survey — Mauricio et al., Applied Sciences (2023). "Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review."
- xiao2021earlyconv — Xiao et al., NeurIPS (2021). "Early Convolutions Help Transformers See Better."
- he2016resnet — He et al., CVPR (2016). "Deep Residual Learning for Image Recognition."
- simonyan2015vggnet — Simonyan & Zisserman, ICLR (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition."
- tan2019efficientnet — Tan & Le, ICML (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
- liu2021swin — Liu et al., ICCV (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."
- yuan2021t2tvit — Yuan et al., ICCV (2021). "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet."
- kolesnikov2020bit — Kolesnikov et al., ECCV (2020). "Big Transfer (BiT): General Visual Representation Learning."
