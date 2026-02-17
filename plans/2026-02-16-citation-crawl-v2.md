---
date: 2026-02-16
status: complete
description: First-hop citation crawl on 6 anchor papers (ViT, DeiT, HowToTrainViT, ConvNeXt, ConViT, Swin) via Semantic Scholar API
---

# First-Hop Citation Crawl Results (v2)

**Date:** 2026-02-16
**Source:** Semantic Scholar Graph API v1
**Method:** For each anchor, fetched up to 100 references and 100 citations. Filtered by relevance scoring (title/abstract keyword matching for: ViT-vs-CNN comparison, data efficiency, inductive bias, image classification, training dataset size effects, augmentation, transfer learning). Ranked by combined relevance score and citation count. Cross-anchor overlap computed across all 6 networks.

## Caveats

- Semantic Scholar limits to 100 results per endpoint. For highly-cited papers like ViT (57K+) and Swin (29K+), the 100 citations returned are a **tiny sample** biased toward recent papers.
- The references lists are complete or near-complete (all anchors have <100 references).
- Cross-anchor overlap is computed across the filtered results only.

## Summary Statistics

| # | Anchor | ArXiv ID | Raw Relevant | Top Shown |
|---|--------|----------|-------------|-----------|
| 1 | ViT (Dosovitskiy 2020) | 2010.11929 | 44 | 20 |
| 2 | DeiT (Touvron 2020) | 2012.12877 | 51 | 20 |
| 3 | How to Train Your ViT (Steiner 2021) | 2106.10270 | 59 | 20 |
| 4 | ConvNeXt (Liu 2022) | 2201.03545 | 43 | 20 |
| 5 | ConViT (d'Ascoli 2021) | 2103.10697 | 51 | 20 |
| 6 | Swin Transformer (Liu 2021) | 2103.14030 | 46 | 20 |

**Total unique relevant papers:** 232
**High-relevance cross-anchor papers (score >= 15, 2+ networks):** 25

---

## Anchor 1: ViT -- An Image is Worth 16x16 Words (Dosovitskiy 2020)
ArXiv:2010.11929 | 44 relevant papers, showing top 20

### Top 20 Papers

1. **Big Transfer (BiT): General Visual Representation Learning** [REF]
   - Kolesnikov (2019) | Cites: **1,317** | ArXiv:1912.11370 | DOI:10.1007/978-3-030-58558-7_29
   - Relevance: data efficiency; small data regime; transfer/pretraining
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT, Swin, ViT**

2. **Data-Efficient Image Recognition with Contrastive Predictive Coding** [REF]
   - Henaff (2019) | Cites: **1,530** | ArXiv:1905.09272
   - Relevance: data efficiency; transfer/pretraining

3. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762
   - Relevance: foundational transformer architecture; self-attention mechanism
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

4. **Batch Normalization: Accelerating Deep Network Training** [REF]
   - Ioffe (2015) | Cites: **45,980** | ArXiv:1502.03167
   - Relevance: training methodology

5. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark; dataset scaling
   - **CROSS-ANCHOR: ConvNeXt, DeiT, HowToTrainViT, Swin, ViT**

6. **A Simple Framework for Contrastive Learning of Visual Representations** [REF]
   - Chen (2020) | Cites: **22,848** | ArXiv:2002.05709
   - Relevance: augmentation strategies; pretraining

7. **On Robustness and Transferability of Convolutional Neural Networks** [REF]
   - Djolonga (2020) | Cites: **168** | ArXiv:2007.08558 | DOI:10.1109/CVPR46437.2021.01619
   - Relevance: CNN transferability; scaling effects

8. **Fixing the train-test resolution discrepancy** [REF]
   - Touvron (2019) | Cites: **468** | ArXiv:1906.06423
   - Relevance: augmentation; resolution effects on transfer
   - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

9. **Exploring the Limits of Weakly Supervised Pretraining** [REF]
   - Mahajan (2018) | Cites: **1,440** | ArXiv:1805.00932 | DOI:10.1007/978-3-030-01216-8_12
   - Relevance: pretraining data scale effects

10. **Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation** [REF]
    - Wang (2020) | Cites: **789** | ArXiv:2003.07853 | DOI:10.1007/978-3-030-58548-8_7
    - Relevance: axial attention in vision; augmentation; scaling

11. **Self-Supervised Learning of Video-Induced Visual Invariances** [REF]
    - Tschannen (2019) | Cites: **65** | ArXiv:1912.02783 | DOI:10.1109/cvpr42600.2020.01382
    - Relevance: pretraining strategies

12. **Stand-Alone Self-Attention in Vision Models** [REF]
    - Ramachandran (2019) | Cites: **1,333** | ArXiv:1906.05909
    - Relevance: self-attention replacing convolution; hybrid approaches
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

13. **Attention Augmented Convolutional Networks** [REF]
    - Bello (2019) | Cites: **1,132** | ArXiv:1904.09925 | DOI:10.1109/ICCV.2019.00338
    - Relevance: combining CNN + attention; augmenting convolutions
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

14. **Revisiting Unreasonable Effectiveness of Data in Deep Learning Era** [REF]
    - Sun (2017) | Cites: **2,637** | ArXiv:1707.02968 | DOI:10.1109/ICCV.2017.97
    - Relevance: dataset size effects on model performance; scaling
    - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

15. **Weight Standardization** [REF]
    - Qiao (2019) | Cites: **147** | ArXiv:1903.10520
    - Relevance: training normalization technique

16. **VisualBERT: A Simple and Performant Baseline for Vision and Language** [REF]
    - Li (2019) | Cites: **2,224** | ArXiv:1908.03557
    - Relevance: transformer pretraining for vision
    - **CROSS-ANCHOR: DeiT, ViT**

17. **Exploring Self-Attention for Image Recognition** [REF]
    - Zhao (2020) | Cites: **901** | ArXiv:2004.13621 | DOI:10.1109/CVPR42600.2020.01009
    - Relevance: self-attention in vision models
    - **CROSS-ANCHOR: Swin, ViT**

18. **Scaling Autoregressive Video Models** [REF]
    - Weissenborn (2019) | Cites: **238** | ArXiv:1906.02634
    - Relevance: scaling; attention mechanism

19. **Fixing the train-test resolution discrepancy: FixEfficientNet** [REF]
    - Touvron (2020) | Cites: **112** | ArXiv:2003.08237
    - Relevance: resolution and transfer learning
    - **CROSS-ANCHOR: DeiT, ViT**

20. **Leveraging distillation token and weaker teacher model to improve DeiT transfer learning capability** [CIT]
    - Reswara (2026) | Cites: **0** | DOI:10.11591/ijict.v15i1.pp198-206
    - Relevance: data efficiency; DeiT distillation; transfer learning
    - **CROSS-ANCHOR: DeiT, Swin, ViT**

---

## Anchor 2: DeiT -- Training Data-Efficient Image Transformers (Touvron 2020)
ArXiv:2012.12877 | 51 relevant papers, showing top 20

### Top 20 Papers

1. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762
   - Relevance: foundational transformer; attention mechanism
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

2. **Transferring Inductive Biases through Knowledge Distillation** [REF]
   - Abnar (2020) | Cites: **69** | ArXiv:2006.00555
   - Relevance: **inductive bias transfer; CNN-to-transformer distillation**
   - **CROSS-ANCHOR: ConViT, DeiT**

3. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark
   - **CROSS-ANCHOR: ConvNeXt, DeiT, HowToTrainViT, Swin, ViT**

4. **Fixing the train-test resolution discrepancy** [REF]
   - Touvron (2019) | Cites: **468** | ArXiv:1906.06423
   - Relevance: augmentation; transfer learning effects
   - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

5. **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** [REF]
   - Goyal (2017) | Cites: **3,975** | ArXiv:1706.02677
   - Relevance: ImageNet training; scaling

6. **Random Erasing Data Augmentation** [REF]
   - Zhong (2017) | Cites: **4,009** | ArXiv:1708.04896 | DOI:10.1609/AAAI.V34I07.7000
   - Relevance: data augmentation
   - **CROSS-ANCHOR: ConvNeXt, DeiT, Swin**

7. **Stand-Alone Self-Attention in Vision Models** [REF]
   - Ramachandran (2019) | Cites: **1,333** | ArXiv:1906.05909
   - Relevance: self-attention vs convolution
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

8. **Attention Augmented Convolutional Networks** [REF]
   - Bello (2019) | Cites: **1,132** | ArXiv:1904.09925 | DOI:10.1109/ICCV.2019.00338
   - Relevance: CNN + attention hybrid
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

9. **Revisiting Unreasonable Effectiveness of Data in Deep Learning Era** [REF]
   - Sun (2017) | Cites: **2,637** | ArXiv:1707.02968 | DOI:10.1109/ICCV.2017.97
   - Relevance: dataset size effects; scaling
   - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

10. **Reducing Transformer Depth on Demand with Structured Dropout** [REF]
    - Fan (2019) | Cites: **667** | ArXiv:1909.11556
    - Relevance: efficient transformer training

11. **VisualBERT: A Simple and Performant Baseline for Vision and Language** [REF]
    - Li (2019) | Cites: **2,224** | ArXiv:1908.03557
    - Relevance: transformer pretraining; attention mechanism
    - **CROSS-ANCHOR: DeiT, ViT**

12. **Augment Your Batch: Improving Generalization Through Instance Repetition** [REF]
    - Hoffer (2020) | Cites: **250** | DOI:10.1109/cvpr42600.2020.00815
    - Relevance: augmentation; transfer learning
    - **CROSS-ANCHOR: DeiT, Swin**

13. **Bag of Tricks for Image Classification with Convolutional Neural Networks** [REF]
    - He (2018) | Cites: **1,545** | ArXiv:1812.01187 | DOI:10.1109/CVPR.2019.00065
    - Relevance: image classification; augmentation; training tricks

14. **Semi-Supervised Masked Autoencoders: Unlocking Vision Transformer Potential with Limited Data** [CIT]
    - Faysal (2026) | Cites: **0** | ArXiv:2601.20072
    - Relevance: **data efficiency; small data regime; ViT with limited labels**
    - **CROSS-ANCHOR: DeiT, HowToTrainViT**

15. **Fixing the train-test resolution discrepancy: FixEfficientNet** [REF]
    - Touvron (2020) | Cites: **112** | ArXiv:2003.08237
    - Relevance: resolution transfer
    - **CROSS-ANCHOR: DeiT, ViT**

16. **Global Self-Attention Networks for Image Recognition** [REF]
    - Shen (2020) | Cites: **32** | ArXiv:2010.03019
    - Relevance: global attention for vision

17. **Leveraging distillation token and weaker teacher model to improve DeiT transfer learning capability** [CIT]
    - Reswara (2026) | Cites: **0** | DOI:10.11591/ijict.v15i1.pp198-206
    - Relevance: data efficiency; distillation; transfer learning
    - **CROSS-ANCHOR: DeiT, Swin, ViT**

18. **Training-Efficient Text-to-Music Generation with State-Space Modeling** [CIT]
    - Lee (2026) | Cites: **0** | ArXiv:2601.14786
    - Relevance: data efficiency; CNN-vs-transformer architectural comparison

19. **Vanilla Group Equivariant Vision Transformer: Simple and Effective** [CIT]
    - Fu (2026) | Cites: **0** | ArXiv:2602.08047
    - Relevance: **inductive bias; data efficiency; attention design**

20. **Krause Synchronization Transformers** [CIT]
    - Liu (2026) | Cites: **0** | ArXiv:2602.11534
    - Relevance: inductive bias; attention mechanism
    - **CROSS-ANCHOR: DeiT, Swin, ViT**

---

## Anchor 3: How to Train Your ViT (Steiner 2021)
ArXiv:2106.10270 | 59 relevant papers, showing top 20

### Top 20 Papers

1. **Big Transfer (BiT): General Visual Representation Learning** [REF]
   - Kolesnikov (2019) | Cites: **1,317** | ArXiv:1912.11370 | DOI:10.1007/978-3-030-58558-7_29
   - Relevance: **data efficiency; small data regime; transfer learning at scale**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT, Swin, ViT**

2. **CoAtNet: Marrying Convolution and Attention for All Data Sizes** [REF]
   - Dai (2021) | Cites: **1,494** | ArXiv:2106.04803
   - Relevance: **inductive bias; hybrid CNN-attention; data size effects**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

3. **Emerging Properties in Self-Supervised Vision Transformers** [REF]
   - Caron (2021) | Cites: **8,167** | ArXiv:2104.14294 | DOI:10.1109/ICCV48922.2021.00951
   - Relevance: CNN-vs-transformer representation comparison; vision transformer

4. **SiT: Self-supervised vIsion Transformer** [REF]
   - Ahmed (2021) | Cites: **157** | ArXiv:2104.03602
   - Relevance: **small data regime; self-supervised pretraining; scaling**

5. **EfficientNetV2: Smaller Models and Faster Training** [REF]
   - Tan (2021) | Cites: **3,860** | ArXiv:2104.00298
   - Relevance: augmentation; efficient training; pretraining
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

6. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark
   - **CROSS-ANCHOR: ConvNeXt, DeiT, HowToTrainViT, Swin, ViT**

7. **CvT: Introducing Convolutions to Vision Transformers** [REF]
   - Wu (2021) | Cites: **2,305** | ArXiv:2103.15808 | DOI:10.1109/ICCV48922.2021.00009
   - Relevance: **CNN-transformer hybrid; vision transformer**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

8. **Fixing the train-test resolution discrepancy** [REF]
   - Touvron (2019) | Cites: **468** | ArXiv:1906.06423
   - Relevance: augmentation; transfer learning
   - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

9. **Bottleneck Transformers for Visual Recognition** [REF]
   - Srinivas (2021) | Cites: **1,135** | ArXiv:2101.11605 | DOI:10.1109/CVPR46437.2021.01625
   - Relevance: attention mechanism; hybrid architecture
   - **CROSS-ANCHOR: ConViT, ConvNeXt, HowToTrainViT, Swin**

10. **Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training** [REF]
    - Changpinyo (2021) | Cites: **1,390** | ArXiv:2102.08981 | DOI:10.1109/CVPR46437.2021.00356
    - Relevance: pretraining at scale

11. **LiT: Zero-Shot Transfer with Locked-image text Tuning** [REF]
    - Zhai (2021) | Cites: **676** | ArXiv:2111.07991 | DOI:10.1109/CVPR52688.2022.01759
    - Relevance: transfer learning

12. **Revisiting Unreasonable Effectiveness of Data in Deep Learning Era** [REF]
    - Sun (2017) | Cites: **2,637** | ArXiv:1707.02968 | DOI:10.1109/ICCV.2017.97
    - Relevance: **data scale effects on performance; scaling laws**
    - **CROSS-ANCHOR: DeiT, HowToTrainViT, ViT**

13. **Pyramid Vision Transformer** [REF]
    - Wang (2021) | Cites: **4,664** | ArXiv:2102.12122 | DOI:10.1109/ICCV48922.2021.00061
    - Relevance: hierarchical vision transformer
    - **CROSS-ANCHOR: HowToTrainViT, Swin**

14. **Do Better ImageNet Models Transfer Better?** [REF]
    - Kornblith (2018) | Cites: **1,469** | ArXiv:1805.08974 | DOI:10.1109/CVPR.2019.00277
    - Relevance: **image classification; transfer learning effectiveness**

15. **ImageNet-21K Pretraining for the Masses** [REF]
    - Ridnik (2021) | Cites: **869** | ArXiv:2104.10972
    - Relevance: ImageNet pretraining; transfer learning

16. **Remote Sensing Image Scene Classification: Benchmark and State of the Art** [REF]
    - Cheng (2017) | Cites: **2,619** | ArXiv:1703.00121 | DOI:10.1109/JPROC.2017.2675998
    - Relevance: classification benchmarks; scaling

17. **A Comparative Study of Vision Transformers and CNNs for Few-Shot Rigid Transformation** [CIT]
    - Kaya (2025) | Cites: **0** | ArXiv:2510.04794
    - Relevance: **inductive bias; CNN-vs-transformer comparison; small data regime**

18. **Segmenter: Transformer for Semantic Segmentation** [REF]
    - Strudel (2021) | Cites: **1,815** | ArXiv:2105.05633 | DOI:10.1109/ICCV48922.2021.00717
    - Relevance: transfer learning from ViT

19. **Semi-Supervised Masked Autoencoders: Unlocking Vision Transformer Potential with Limited Data** [CIT]
    - Faysal (2026) | Cites: **0** | ArXiv:2601.20072
    - Relevance: **data efficiency; small data regime; ViT with limited labels**
    - **CROSS-ANCHOR: DeiT, HowToTrainViT**

20. **BlindTuner: Privacy-Preserving Fine-Tuning of Transformers Based on Homomorphic Encryption** [CIT]
    - Panzade (2025) | Cites: **2** | DOI:10.1109/JIOT.2025.3552321
    - Relevance: data efficiency; fine-tuning strategies

---

## Anchor 4: ConvNeXt -- A ConvNet for the 2020s (Liu 2022)
ArXiv:2201.03545 | 43 relevant papers, showing top 20

### Top 20 Papers

1. **Big Transfer (BiT): General Visual Representation Learning** [REF]
   - Kolesnikov (2019) | Cites: **1,317** | ArXiv:1912.11370 | DOI:10.1007/978-3-030-58558-7_29
   - Relevance: **data efficiency; transfer learning at scale; small data**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT, Swin, ViT**

2. **CoAtNet: Marrying Convolution and Attention for All Data Sizes** [REF]
   - Dai (2021) | Cites: **1,494** | ArXiv:2106.04803
   - Relevance: **inductive bias; hybrid CNN-attention for all data sizes**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

3. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762
   - Relevance: foundational transformer
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

4. **Global Filter Networks for Image Classification** [REF]
   - Rao (2021) | Cites: **618** | ArXiv:2107.00645
   - Relevance: **inductive bias; image classification; attention alternatives**

5. **Fast R-CNN** [REF]
   - Girshick (2015) | Cites: **27,703** | ArXiv:1504.08083
   - Relevance: CNN-transformer comparison context; pretraining

6. **EfficientNetV2: Smaller Models and Faster Training** [REF]
   - Tan (2021) | Cites: **3,860** | ArXiv:2104.00298
   - Relevance: augmentation; efficient training
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

7. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark
   - **CROSS-ANCHOR: ConvNeXt, DeiT, HowToTrainViT, Swin, ViT**

8. **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation** [REF]
   - Girshick (2013) | Cites: **28,399** | ArXiv:1311.2524 | DOI:10.1109/CVPR.2014.81
   - Relevance: CNN feature learning; transfer

9. **Towards Robust Vision Transformer** [REF]
   - Mao (2021) | Cites: **234** | ArXiv:2105.07926 | DOI:10.1109/CVPR52688.2022.01173
   - Relevance: ViT augmentation; robustness; attention

10. **CvT: Introducing Convolutions to Vision Transformers** [REF]
    - Wu (2021) | Cites: **2,305** | ArXiv:2103.15808 | DOI:10.1109/ICCV48922.2021.00009
    - Relevance: **CNN-transformer hybrid**
    - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT**

11. **Two-Stream Convolutional Networks for Action Recognition in Videos** [REF]
    - Simonyan (2014) | Cites: **8,022** | ArXiv:1406.2199
    - Relevance: convolutional architecture; training data requirements

12. **Early Convolutions Help Transformers See Better** [REF]
    - Xiao (2021) | Cites: **892** | ArXiv:2106.14881
    - Relevance: **CNN-transformer hybrid; convolutional inductive bias helps ViTs**

13. **Semantic Understanding of Scenes Through the ADE20K Dataset** [REF]
    - Zhou (2016) | Cites: **2,203** | ArXiv:1608.05442 | DOI:10.1007/s11263-018-1140-0
    - Relevance: evaluation benchmark
    - **CROSS-ANCHOR: ConvNeXt, Swin**

14. **Co-Scale Conv-Attentional Image Transformers** [REF]
    - Xu (2021) | Cites: **437** | ArXiv:2104.06399 | DOI:10.1109/ICCV48922.2021.00983
    - Relevance: **CNN-attention hybrid**

15. **Natural Adversarial Examples** [REF]
    - Hendrycks (2019) | Cites: **1,783** | ArXiv:1907.07174 | DOI:10.1109/CVPR46437.2021.01501
    - Relevance: augmentation; robustness evaluation

16. **Multiscale Vision Transformers** [REF]
    - Fan (2021) | Cites: **1,532** | ArXiv:2104.11227 | DOI:10.1109/ICCV48922.2021.00675
    - Relevance: transfer/pretraining; scaling; vision transformer

17. **Bottleneck Transformers for Visual Recognition** [REF]
    - Srinivas (2021) | Cites: **1,135** | ArXiv:2101.11605 | DOI:10.1109/CVPR46437.2021.01625
    - Relevance: hybrid attention mechanism
    - **CROSS-ANCHOR: ConViT, ConvNeXt, HowToTrainViT, Swin**

18. **Faster R-CNN** [REF]
    - Ren (2015) | Cites: **70,251** | ArXiv:1506.01497 | DOI:10.1109/TPAMI.2016.2577031
    - Relevance: object detection baseline

19. **Random Erasing Data Augmentation** [REF]
    - Zhong (2017) | Cites: **4,009** | ArXiv:1708.04896 | DOI:10.1609/AAAI.V34I07.7000
    - Relevance: augmentation
    - **CROSS-ANCHOR: ConvNeXt, DeiT, Swin**

20. **Stand-Alone Self-Attention in Vision Models** [REF]
    - Ramachandran (2019) | Cites: **1,333** | ArXiv:1906.05909
    - Relevance: **self-attention replacing convolution**
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

---

## Anchor 5: ConViT -- Improving Vision Transformers with Soft Convolutional Inductive Biases (d'Ascoli 2021)
ArXiv:2103.10697 | 51 relevant papers, showing top 20

### Top 20 Papers

1. **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** [REF]
   - Yuan (2021) | Cites: **2,378** | ArXiv:2101.11986 | DOI:10.1109/ICCV48922.2021.00060
   - Relevance: **data efficiency; training from scratch; small data regime**
   - **CROSS-ANCHOR: ConViT, Swin**

2. **ResMLP: Feedforward Networks for Image Classification With Data-Efficient Training** [REF]
   - Touvron (2021) | Cites: **823** | ArXiv:2105.03404 | DOI:10.1109/TPAMI.2022.3206148
   - Relevance: **data-efficient training; image classification**
   - **CROSS-ANCHOR: ConViT, Swin**

3. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762
   - Relevance: foundational transformer; attention mechanism
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

4. **Transferring Inductive Biases through Knowledge Distillation** [REF]
   - Abnar (2020) | Cites: **69** | ArXiv:2006.00555
   - Relevance: **inductive bias; CNN-to-transformer knowledge transfer**
   - **CROSS-ANCHOR: ConViT, DeiT**

5. **Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth** [REF]
   - Dong (2021) | Cites: **493** | ArXiv:2103.03404
   - Relevance: **inductive bias; limitations of pure attention**

6. **Bottleneck Transformers for Visual Recognition** [REF]
   - Srinivas (2021) | Cites: **1,135** | ArXiv:2101.11605 | DOI:10.1109/CVPR46437.2021.01625
   - Relevance: hybrid attention mechanism
   - **CROSS-ANCHOR: ConViT, ConvNeXt, HowToTrainViT, Swin**

7. **Towards Learning Convolutions from Scratch** [REF]
   - Neyshabur (2020) | Cites: **74** | ArXiv:2007.13657
   - Relevance: **inductive bias; learning convolutional structure**

8. **Stand-Alone Self-Attention in Vision Models** [REF]
   - Ramachandran (2019) | Cites: **1,333** | ArXiv:1906.05909
   - Relevance: self-attention vs convolution
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

9. **Attention Augmented Convolutional Networks** [REF]
   - Bello (2019) | Cites: **1,132** | ArXiv:1904.09925 | DOI:10.1109/ICCV.2019.00338
   - Relevance: CNN + attention augmentation
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

10. **Revisiting Spatial Invariance with Low-Rank Local Connectivity** [REF]
    - Elsayed (2020) | Cites: **51** | ArXiv:2002.02959
    - Relevance: **inductive bias; spatial locality**

11. **Finding the Needle in the Haystack with Convolutions: on the benefits of architectural bias** [REF]
    - d'Ascoli (2019) | Cites: **37** | ArXiv:1906.06766
    - Relevance: **CNN-vs-transformer; benefits of convolutional bias**

12. **IBiT: Utilizing Inductive Biases to Create a More Data Efficient Attention Mechanism** [CIT]
    - Giri (2025) | Cites: **0** | ArXiv:2509.22719
    - Relevance: **inductive bias; data efficiency; small data regime**

13. **Benchmarking Vision Transformers and CNNs for Ulos Batak Pattern Recognition** [CIT]
    - Lamudur (2025) | Cites: **0** | DOI:10.1109/ICITISEE68184.2025.11355022
    - Relevance: **inductive bias; small data regime; ViT vs CNN benchmarking**

14. **MedNeXt: accurate medical image classification and segmentation** [CIT]
    - Xue (2026) | Cites: **0** | DOI:10.1371/journal.pone.0340108
    - Relevance: **CNN-vs-transformer; small data; image classification**

15. **ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases** [REF]
    - Unknown (2021) | Cites: **1**
    - Relevance: inductive bias; vision transformer design

16. **Distilling structural knowledge from CNNs to vision transformers for data-efficient visual recognition** [CIT]
    - Chen (2026) | Cites: **0** | DOI:10.1016/j.neunet.2026.108601
    - Relevance: **data efficiency; CNN-to-ViT distillation**

17. **A Comparative Analysis of CNNs, Vision Transformers and Hybrid Architectures for Lung CT Classification** [CIT]
    - Gulcan (2025) | Cites: **0** | DOI:10.1109/UBMK67458.2025.11206987
    - Relevance: **CNN-vs-transformer comparison; image classification; augmentation**

18. **A Study on the Limitations of CNNs and ConViT in Still Image Recognition** [CIT]
    - Qiu (2025) | Cites: **0** | DOI:10.1109/ICBEBH66536.2025.11276334
    - Relevance: **CNN-vs-transformer comparison; ConViT limitations**

19. **HMCFormer: hierarchical multi-scale convolutional transformer** [CIT]
    - Feng (2025) | Cites: **0** | DOI:10.7717/peerj-cs.3088
    - Relevance: hybrid CNN-transformer architecture

20. **Decoding vision transformer variations for image classification** [CIT]
    - Montrezol (2026) | Cites: **0** | DOI:10.1016/j.mlwa.2026.100844
    - Relevance: image classification; vision transformer variants
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT**

---

## Anchor 6: Swin Transformer -- Hierarchical Vision Transformer using Shifted Windows (Liu 2021)
ArXiv:2103.14030 | 46 relevant papers, showing top 20

### Top 20 Papers

1. **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** [REF]
   - Yuan (2021) | Cites: **2,378** | ArXiv:2101.11986 | DOI:10.1109/ICCV48922.2021.00060
   - Relevance: **data efficiency; training from scratch; small data**
   - **CROSS-ANCHOR: ConViT, Swin**

2. **Big Transfer (BiT): General Visual Representation Learning** [REF]
   - Kolesnikov (2019) | Cites: **1,317** | ArXiv:1912.11370 | DOI:10.1007/978-3-030-58558-7_29
   - Relevance: **data efficiency; small data regime; transfer learning**
   - **CROSS-ANCHOR: ConvNeXt, HowToTrainViT, Swin, ViT**

3. **ResMLP: Feedforward Networks for Image Classification With Data-Efficient Training** [REF]
   - Touvron (2021) | Cites: **823** | ArXiv:2105.03404 | DOI:10.1109/TPAMI.2022.3206148
   - Relevance: **data-efficient training; image classification**
   - **CROSS-ANCHOR: ConViT, Swin**

4. **YOLOv4: Optimal Speed and Accuracy of Object Detection** [REF]
   - Bochkovskiy (2020) | Cites: **14,739** | ArXiv:2004.10934
   - Relevance: augmentation; training strategy; scaling

5. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762
   - Relevance: foundational transformer
   - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

6. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark
   - **CROSS-ANCHOR: ConvNeXt, DeiT, HowToTrainViT, Swin, ViT**

7. **Toward Transformer-Based Object Detection** [REF]
   - Beal (2020) | Cites: **244** | ArXiv:2012.09958
   - Relevance: CNN-vs-transformer comparison; transfer learning

8. **Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation** [REF]
   - Ghiasi (2020) | Cites: **1,179** | ArXiv:2012.07177 | DOI:10.1109/CVPR46437.2021.00294
   - Relevance: **data efficiency; augmentation**

9. **Long Range Arena: A Benchmark for Efficient Transformers** [REF]
   - Tay (2020) | Cites: **844** | ArXiv:2011.04006
   - Relevance: **CNN-vs-transformer benchmarking; attention efficiency**

10. **Semantic Understanding of Scenes Through the ADE20K Dataset** [REF]
    - Zhou (2016) | Cites: **2,203** | ArXiv:1608.05442 | DOI:10.1007/s11263-018-1140-0
    - Relevance: evaluation benchmark
    - **CROSS-ANCHOR: ConvNeXt, Swin**

11. **Bottleneck Transformers for Visual Recognition** [REF]
    - Srinivas (2021) | Cites: **1,135** | ArXiv:2101.11605 | DOI:10.1109/CVPR46437.2021.01625
    - Relevance: hybrid attention mechanism
    - **CROSS-ANCHOR: ConViT, ConvNeXt, HowToTrainViT, Swin**

12. **Random Erasing Data Augmentation** [REF]
    - Zhong (2017) | Cites: **4,009** | ArXiv:1708.04896 | DOI:10.1609/AAAI.V34I07.7000
    - Relevance: data augmentation
    - **CROSS-ANCHOR: ConvNeXt, DeiT, Swin**

13. **Stand-Alone Self-Attention in Vision Models** [REF]
    - Ramachandran (2019) | Cites: **1,333** | ArXiv:1906.05909
    - Relevance: self-attention vs convolution
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

14. **Attention Augmented Convolutional Networks** [REF]
    - Bello (2019) | Cites: **1,132** | ArXiv:1904.09925 | DOI:10.1109/ICCV.2019.00338
    - Relevance: CNN + attention augmentation
    - **CROSS-ANCHOR: ConViT, ConvNeXt, DeiT, Swin, ViT**

15. **Pyramid Vision Transformer** [REF]
    - Wang (2021) | Cites: **4,664** | ArXiv:2102.12122 | DOI:10.1109/ICCV48922.2021.00061
    - Relevance: hierarchical vision transformer
    - **CROSS-ANCHOR: HowToTrainViT, Swin**

16. **Learning Transferable Visual Models From Natural Language Supervision** [REF]
    - Radford (2021) | Cites: **42,989** | ArXiv:2103.00020
    - Relevance: transfer learning; pretraining

17. **Deformable ConvNets V2: More Deformable, Better Results** [REF]
    - Zhu (2018) | Cites: **2,436** | ArXiv:1811.11168 | DOI:10.1109/CVPR.2019.00953
    - Relevance: deformable convolution

18. **Hybrid Task Cascade for Instance Segmentation** [REF]
    - Chen (2019) | Cites: **1,461** | ArXiv:1901.07518 | DOI:10.1109/CVPR.2019.00511
    - Relevance: hybrid architecture

19. **An Analysis of Scale Invariance in Object Detection - SNIP** [REF]
    - Singh (2017) | Cites: **794** | ArXiv:1711.08189 | DOI:10.1109/CVPR.2018.00377
    - Relevance: transfer learning; scale effects

20. **Augment Your Batch: Improving Generalization Through Instance Repetition** [REF]
    - Hoffer (2020) | Cites: **250** | DOI:10.1109/cvpr42600.2020.00815
    - Relevance: augmentation; transfer learning
    - **CROSS-ANCHOR: DeiT, Swin**

---

## Cross-Anchor Papers

Papers appearing in the filtered results of 2+ anchors, with relevance score >= 15. These are the most structurally central and topically relevant works in this citation network.

### Appears in 5 anchor networks

- **Attention is All you Need** -- Vaswani (2017) -- 165,637 cites | ArXiv:1706.03762
  - ViT(ref), DeiT(ref), ConvNeXt(ref), ConViT(ref), Swin(ref)
  - Foundational transformer architecture; attention mechanism

- **ImageNet: A large-scale hierarchical image database** -- Deng (2009) -- 70,819 cites | DOI:10.1109/CVPR.2009.5206848
  - ViT(ref), DeiT(ref), HowToTrainViT(ref), ConvNeXt(ref), Swin(ref)
  - Core image classification benchmark

- **Stand-Alone Self-Attention in Vision Models** -- Ramachandran (2019) -- 1,333 cites | ArXiv:1906.05909
  - ViT(ref), DeiT(ref), ConvNeXt(ref), ConViT(ref), Swin(ref)
  - Self-attention replacing convolution; augmentation; hybrid approaches

- **Attention Augmented Convolutional Networks** -- Bello (2019) -- 1,132 cites | ArXiv:1904.09925
  - ViT(ref), DeiT(ref), ConvNeXt(ref), ConViT(ref), Swin(ref)
  - CNN + attention hybrid; augmenting convolutions with attention

### Appears in 4 anchor networks

- **Big Transfer (BiT): General Visual Representation Learning** -- Kolesnikov (2019) -- 1,317 cites | ArXiv:1912.11370
  - ViT(ref), HowToTrainViT(ref), ConvNeXt(ref), Swin(ref)
  - **Data efficiency; small data regime; transfer learning at scale**

- **Bottleneck Transformers for Visual Recognition** -- Srinivas (2021) -- 1,135 cites | ArXiv:2101.11605
  - HowToTrainViT(ref), ConvNeXt(ref), ConViT(ref), Swin(ref)
  - Hybrid attention mechanism

### Appears in 3 anchor networks

- **Fixing the train-test resolution discrepancy** -- Touvron (2019) -- 468 cites | ArXiv:1906.06423
  - ViT(ref), DeiT(ref), HowToTrainViT(ref)
  - Augmentation; resolution effects on transfer

- **Revisiting Unreasonable Effectiveness of Data** -- Sun (2017) -- 2,637 cites | ArXiv:1707.02968
  - ViT(ref), DeiT(ref), HowToTrainViT(ref)
  - **Dataset size effects on performance; data scaling laws**

- **Random Erasing Data Augmentation** -- Zhong (2017) -- 4,009 cites | ArXiv:1708.04896
  - DeiT(ref), ConvNeXt(ref), Swin(ref)
  - Augmentation strategy

- **Leveraging distillation token and weaker teacher model to improve DeiT transfer learning** -- Reswara (2026) -- 0 cites
  - ViT(cit), DeiT(cit), Swin(cit)
  - Data efficiency; distillation; transfer learning

- **Krause Synchronization Transformers** -- Liu (2026) -- 0 cites | ArXiv:2602.11534
  - ViT(cit), DeiT(cit), Swin(cit)
  - Inductive bias; attention mechanism

- **Decoding vision transformer variations for image classification** -- Montrezol (2026) -- 0 cites
  - DeiT(cit), ConvNeXt(cit), ConViT(cit)
  - Image classification; vision transformer comparison guide

### Appears in 2 anchor networks (high relevance only)

- **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** -- Yuan (2021) -- 2,378 cites | ArXiv:2101.11986
  - ConViT(ref), Swin(ref)
  - **Data efficiency; training from scratch; small data regime**

- **CoAtNet: Marrying Convolution and Attention for All Data Sizes** -- Dai (2021) -- 1,494 cites | ArXiv:2106.04803
  - HowToTrainViT(ref), ConvNeXt(ref)
  - **Inductive bias; hybrid CNN-attention; data size effects**

- **ResMLP: Feedforward Networks for Image Classification With Data-Efficient Training** -- Touvron (2021) -- 823 cites | ArXiv:2105.03404
  - ConViT(ref), Swin(ref)
  - **Data efficiency; image classification**

- **Transferring Inductive Biases through Knowledge Distillation** -- Abnar (2020) -- 69 cites | ArXiv:2006.00555
  - DeiT(ref), ConViT(ref)
  - **Inductive bias; CNN-to-transformer knowledge transfer**

- **EfficientNetV2: Smaller Models and Faster Training** -- Tan (2021) -- 3,860 cites | ArXiv:2104.00298
  - HowToTrainViT(ref), ConvNeXt(ref)
  - Augmentation; efficient training

- **CvT: Introducing Convolutions to Vision Transformers** -- Wu (2021) -- 2,305 cites | ArXiv:2103.15808
  - HowToTrainViT(ref), ConvNeXt(ref)
  - **CNN-transformer hybrid; vision transformer design**

- **Semi-Supervised Masked Autoencoders: Unlocking Vision Transformer Potential with Limited Data** -- Faysal (2026) -- 0 cites | ArXiv:2601.20072
  - DeiT(cit), HowToTrainViT(cit)
  - **Data efficiency; small data regime; ViT with limited labels**

- **ViT-5: Vision Transformers for The Mid-2020s** -- (2026) -- 0 cites | ArXiv:2602.08071
  - DeiT(cit), ConvNeXt(cit)
  - Vision transformer architecture evolution

---

## Key Papers for the Research Question (ViT vs CNN, Data Efficiency, Inductive Bias)

These papers from the crawl are most directly relevant to comparing transformers vs CNNs with respect to data requirements and inductive biases:

### Must-Read (highest topical relevance)

| Paper | Year | Cites | ArXiv | Key Topic |
|-------|------|-------|-------|-----------|
| Transferring Inductive Biases through Knowledge Distillation | 2020 | 69 | 2006.00555 | Inductive bias transfer from CNNs to transformers |
| CoAtNet: Marrying Convolution and Attention for All Data Sizes | 2021 | 1,494 | 2106.04803 | Hybrid arch; data size effects on CNN vs transformer |
| Big Transfer (BiT) | 2019 | 1,317 | 1912.11370 | Transfer learning; data efficiency at scale |
| Tokens-to-Token ViT: Training from Scratch on ImageNet | 2021 | 2,378 | 2101.11986 | Training ViTs without large pretraining data |
| Attention is Not All You Need: Pure Attention Loses Rank | 2021 | 493 | 2103.03404 | Limitations of pure attention (inductive bias argument) |
| Finding the Needle in the Haystack with Convolutions | 2019 | 37 | 1906.06766 | Benefits of convolutional architectural bias |
| Towards Learning Convolutions from Scratch | 2020 | 74 | 2007.13657 | Can networks learn convolutional inductive bias? |
| Stand-Alone Self-Attention in Vision Models | 2019 | 1,333 | 1906.05909 | Self-attention as replacement for convolution |
| Early Convolutions Help Transformers See Better | 2021 | 892 | 2106.14881 | Convolutional stem improves ViT data efficiency |
| Data-Efficient Image Recognition with Contrastive Predictive Coding | 2019 | 1,530 | 1905.09272 | Data-efficient recognition approaches |
| Revisiting Unreasonable Effectiveness of Data | 2017 | 2,637 | 1707.02968 | Dataset size impact on deep learning |
| Do Better ImageNet Models Transfer Better? | 2018 | 1,469 | 1805.08974 | Transfer learning effectiveness |
| ResMLP: Data-Efficient Training | 2021 | 823 | 2105.03404 | Data-efficient alternative architectures |
| CvT: Introducing Convolutions to Vision Transformers | 2021 | 2,305 | 2103.15808 | CNN-transformer hybrid |
| IBiT: Utilizing Inductive Biases for Data Efficient Attention | 2025 | 0 | 2509.22719 | Inductive bias for data-efficient attention |
| Semi-Supervised Masked Autoencoders: ViT with Limited Data | 2026 | 0 | 2601.20072 | ViT data efficiency with limited labels |
| A Comparative Study of ViTs and CNNs for Few-Shot | 2025 | 0 | 2510.04794 | Direct ViT-vs-CNN comparison in few-shot |

### Supporting Context

| Paper | Year | Cites | ArXiv | Key Topic |
|-------|------|-------|-------|-----------|
| Attention Augmented Convolutional Networks | 2019 | 1,132 | 1904.09925 | Augmenting CNNs with attention |
| Emerging Properties in Self-Supervised Vision Transformers | 2021 | 8,167 | 2104.14294 | ViT vs CNN representation differences |
| Global Filter Networks for Image Classification | 2021 | 618 | 2107.00645 | Inductive bias; attention alternatives |
| On Robustness and Transferability of CNNs | 2020 | 168 | 2007.08558 | CNN transfer learning; scaling |
| Bottleneck Transformers for Visual Recognition | 2021 | 1,135 | 2101.11605 | Hybrid architecture |
| EfficientNetV2 | 2021 | 3,860 | 2104.00298 | Efficient CNN training |
| Long Range Arena | 2020 | 844 | 2011.04006 | Benchmarking transformers vs other architectures |
| Bag of Tricks for Image Classification with CNNs | 2018 | 1,545 | 1812.01187 | CNN training recipes |
| ImageNet-21K Pretraining for the Masses | 2021 | 869 | 2104.10972 | Pretraining scale effects |
| Pyramid Vision Transformer | 2021 | 4,664 | 2102.12122 | Hierarchical ViT design |
