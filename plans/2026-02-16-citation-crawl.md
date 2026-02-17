---
date: 2026-02-16
status: complete
description: First-hop citation crawl on 5 anchor papers via Semantic Scholar API
---

# First-Hop Citation Crawl Results

**Date:** 2026-02-16  
**Source:** Semantic Scholar Graph API v1  
**Method:** For each anchor, fetched up to 100 references and 100 citations. Filtered by title/abstract keyword relevance. Ranked by citation count.

## Caveats

- Semantic Scholar limits to 100 results per endpoint. For highly-cited papers (ResNet: 220K+, VGGNet: 109K+, EfficientNet: 21K+), the 100 citations returned are a **tiny sample** of all citing works and are biased toward recent/popular papers.
- The references lists are complete or near-complete for all anchors (all have <100 references).
- Cross-anchor overlap is computed only across the filtered top-N lists shown here, not across all results.

## Summary Statistics

| # | Anchor | Raw Refs | Raw Cits | Refs Filtered | Cits Filtered |
|---|--------|----------|----------|--------------|--------------|
| 1 | Bridging the Gap (Lu 2022) | 89 | 86 | 79 | 72 |
| 2 | Do ViTs See Like CNNs (Raghu 2021) | 61 | 100 | 37 | 73 |
| 3 | ResNet (He 2016) | 54 | 100 | 24 | 40 |
| 4 | EfficientNet (Tan 2019) | 54 | 100 | 36 | 63 |
| 5 | VGGNet (Simonyan 2015) | 43 | 100 | 31 | 57 |

---
## Anchor 1: Bridging the Gap (Lu 2022)
ArXiv:2210.05958 | Refs: 89 raw / 79 filtered | Cits: 86 raw / 72 filtered

### References (papers this anchor cites)

1. **Deep Residual Learning for Image Recognition** [REF]
   - He (2015) | Cites: **220,082** | ArXiv:1512.03385 | DOI:10.1109/cvpr.2016.90
   - Relevance: visual recognition task **[MULTI-ANCHOR: 1, 4]**

2. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762 | no DOI
   - Relevance: vision transformer architecture **[MULTI-ANCHOR: 1, 2]**

3. **ImageNet classification with deep convolutional neural networks** [REF]
   - Krizhevsky (2012) | Cites: **126,741** | no ArXiv | DOI:10.1145/3065386
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

4. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** [REF]
   - Devlin (2019) | Cites: **109,971** | ArXiv:1810.04805 | DOI:10.18653/v1/N19-1423
   - Relevance: vision transformer architecture; transfer learning / pretraining **[MULTI-ANCHOR: 1, 2]**

5. **U-Net: Convolutional Networks for Biomedical Image Segmentation** [REF]
   - Ronneberger (2015) | Cites: **90,883** | ArXiv:1505.04597 | DOI:10.1007/978-3-319-24574-4_28
   - Relevance: CNN architecture; visual recognition task

6. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** [REF]
   - Ren (2015) | Cites: **70,251** | ArXiv:1506.01497 | DOI:10.1109/TPAMI.2016.2577031
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 1, 3]**

7. **Going deeper with convolutions** [REF]
   - Szegedy (2014) | Cites: **46,466** | ArXiv:1409.4842 | DOI:10.1109/CVPR.2015.7298594
   - Relevance: general computer vision **[MULTI-ANCHOR: 1, 3, 4, 5]**

8. **ImageNet Large Scale Visual Recognition Challenge** [REF]
   - Russakovsky (2014) | Cites: **41,804** | ArXiv:1409.0575 | DOI:10.1007/s11263-015-0816-y
   - Relevance: image classification benchmark; visual recognition task **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

9. **Densely Connected Convolutional Networks** [REF]
   - Huang (2016) | Cites: **41,635** | ArXiv:1608.06993 | DOI:10.1109/CVPR.2017.243
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

10. **Mask R-CNN** [REF]
   - He (2017) | Cites: **30,840** | ArXiv:1703.06870 | no DOI
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

11. **Decoupled Weight Decay Regularization** [REF]
   - Loshchilov (2017) | Cites: **30,577** | no ArXiv | no DOI
   - Relevance: training regularization

12. **Rethinking the Inception Architecture for Computer Vision** [REF]
   - Szegedy (2015) | Cites: **30,135** | ArXiv:1512.00567 | DOI:10.1109/CVPR.2016.308
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

13. **Focal Loss for Dense Object Detection** [REF]
   - Lin (2017) | Cites: **29,841** | no ArXiv | DOI:10.1109/ICCV.2017.324
   - Relevance: visual recognition task

14. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** [REF]
   - Liu (2021) | Cites: **29,360** | ArXiv:2103.14030 | DOI:10.1109/ICCV48922.2021.00986
   - Relevance: vision transformer architecture

15. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** [REF]
   - Howard (2017) | Cites: **23,922** | ArXiv:1704.04861 | no DOI
   - Relevance: CNN architecture; deep learning foundations **[MULTI-ANCHOR: 1, 4]**

### Citations (papers that cite this anchor)

1. **Pruning Self-Attentions Into Convolutional Layers in Single Path** [CIT]
   - He (2021) | Cites: **60** | ArXiv:2111.11802 | DOI:10.1109/TPAMI.2024.3355890
   - Relevance: vision transformer architecture; CNN architecture; inductive bias analysis

2. **Depth-Wise Convolutions in Vision Transformers for Efficient Training on Small Datasets** [CIT]
   - Zhang (2024) | Cites: **41** | ArXiv:2407.19394 | DOI:10.1016/j.neucom.2024.128998
   - Relevance: vision transformer architecture; data efficiency; model scaling

3. **Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection** [CIT]
   - Truong (2024) | Cites: **40** | ArXiv:2406.17376 | DOI:10.21437/Interspeech.2024-659
   - Relevance: vision transformer architecture; architecture comparison

4. **IndoHerb: Indonesia medicinal plants recognition using transfer learning and deep learning** [CIT]
   - Musyaffa (2023) | Cites: **18** | ArXiv:2308.01604 | DOI:10.1016/j.heliyon.2024.e40606
   - Relevance: transfer learning / pretraining; visual recognition task; deep learning foundations

5. **Structural Attention: Rethinking Transformer for Unpaired Medical Image Synthesis** [CIT]
   - Phan (2024) | Cites: **17** | ArXiv:2406.18967 | DOI:10.48550/arXiv.2406.18967
   - Relevance: vision transformer architecture; inductive bias analysis

6. **Equipping computational pathology systems with artifact processing pipelines: a showcase for computation and performance trade-offs** [CIT]
   - Kanwal (2024) | Cites: **15** | ArXiv:2403.07743 | DOI:10.1186/s12911-024-02676-z
   - Relevance: general computer vision

7. **A Hyperparameter-Free Attention Module Based on Feature Map Mathematical Calculation for Remote-Sensing Image Scene Classification** [CIT]
   - Wan (2024) | Cites: **15** | no ArXiv | DOI:10.1109/TGRS.2023.3335627
   - Relevance: vision transformer architecture; image classification benchmark

8. **Cardiac signals classification via optional multimodal multiscale receptive fields CNN-enhanced Transformer** [CIT]
   - Zhang (2024) | Cites: **12** | no ArXiv | DOI:10.1016/j.knosys.2024.112175
   - Relevance: vision transformer architecture; CNN architecture; image classification benchmark

9. **US-Net: U-shaped network with Convolutional Attention Mechanism for ultrasound medical images** [CIT]
   - Xie (2024) | Cites: **10** | no ArXiv | DOI:10.1016/j.cag.2024.104054
   - Relevance: vision transformer architecture; CNN architecture

10. **Improving Generalized Zero-Shot Learning SSVEP Classification Performance From Data-Efficient Perspective** [CIT]
   - Wang (2023) | Cites: **10** | no ArXiv | DOI:10.1109/TNSRE.2023.3324148
   - Relevance: image classification benchmark

---
## Anchor 2: Do ViTs See Like CNNs (Raghu 2021)
ArXiv:2108.08810 | Refs: 61 raw / 37 filtered | Cits: 100 raw / 73 filtered

### References (papers this anchor cites)

1. **Attention is All you Need** [REF]
   - Vaswani (2017) | Cites: **165,637** | ArXiv:1706.03762 | no DOI
   - Relevance: vision transformer architecture **[MULTI-ANCHOR: 1, 2]**

2. **ImageNet classification with deep convolutional neural networks** [REF]
   - Krizhevsky (2012) | Cites: **126,741** | no ArXiv | DOI:10.1145/3065386
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

3. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** [REF]
   - Devlin (2019) | Cites: **109,971** | ArXiv:1810.04805 | DOI:10.18653/v1/N19-1423
   - Relevance: vision transformer architecture; transfer learning / pretraining **[MULTI-ANCHOR: 1, 2]**

4. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | no ArXiv | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark **[MULTI-ANCHOR: 2, 5]**

5. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** [REF]
   - Dosovitskiy (2020) | Cites: **57,043** | ArXiv:2010.11929 | no DOI
   - Relevance: vision transformer architecture; visual recognition task

6. **ImageNet Large Scale Visual Recognition Challenge** [REF]
   - Russakovsky (2014) | Cites: **41,804** | ArXiv:1409.0575 | DOI:10.1007/s11263-015-0816-y
   - Relevance: image classification benchmark; visual recognition task **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

7. **End-to-End Object Detection with Transformers** [REF]
   - Carion (2020) | Cites: **16,889** | ArXiv:2005.12872 | DOI:10.1007/978-3-030-58452-8_13
   - Relevance: vision transformer architecture; visual recognition task **[MULTI-ANCHOR: 1, 2]**

8. **Emerging Properties in Self-Supervised Vision Transformers** [REF]
   - Caron (2021) | Cites: **8,167** | ArXiv:2104.14294 | DOI:10.1109/ICCV48922.2021.00951
   - Relevance: vision transformer architecture; architecture comparison

9. **MLP-Mixer: An all-MLP Architecture for Vision** [REF]
   - Tolstikhin (2021) | Cites: **3,374** | ArXiv:2105.01601 | no DOI
   - Relevance: general computer vision

10. **Revisiting Unreasonable Effectiveness of Data in Deep Learning Era** [REF]
   - Sun (2017) | Cites: **2,637** | ArXiv:1707.02968 | DOI:10.1109/ICCV.2017.97
   - Relevance: deep learning foundations

11. **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** [REF]
   - Yuan (2021) | Cites: **2,378** | ArXiv:2101.11986 | DOI:10.1109/ICCV48922.2021.00060
   - Relevance: vision transformer architecture; data efficiency; image classification benchmark **[MULTI-ANCHOR: 1, 2]**

12. **An Empirical Study of Training Self-Supervised Vision Transformers** [REF]
   - Chen (2021) | Cites: **2,231** | ArXiv:2104.02057 | DOI:10.1109/ICCV48922.2021.00950
   - Relevance: vision transformer architecture **[MULTI-ANCHOR: 1, 2]**

13. **Understanding the Effective Receptive Field in Deep Convolutional Neural Networks** [REF]
   - Luo (2016) | Cites: **2,084** | ArXiv:1701.04128 | no DOI
   - Relevance: CNN architecture; deep learning foundations

14. **Image Transformer** [REF]
   - Parmar (2018) | Cites: **1,852** | ArXiv:1802.05751 | no DOI
   - Relevance: vision transformer architecture

15. **Generative Pretraining From Pixels** [REF]
   - Chen (2020) | Cites: **1,728** | no ArXiv | no DOI
   - Relevance: transfer learning / pretraining **[MULTI-ANCHOR: 1, 2]**

### Citations (papers that cite this anchor)

1. **Enhancing pine wilt disease detection with synthetic data and external attention-based transformers** [CIT]
   - Amin (2025) | Cites: **32** | no ArXiv | DOI:10.1016/j.engappai.2025.111655
   - Relevance: vision transformer architecture

2. **Ensembling Pruned Attention Heads For Uncertainty-Aware Efficient Transformers** [CIT]
   - Gabetni (2025) | Cites: **3** | ArXiv:2510.18358 | DOI:10.48550/arXiv.2510.18358
   - Relevance: vision transformer architecture

3. **Benchmark of plankton images classification: emphasizing features extraction over classifier complexity** [CIT]
   - Panaïotis (2026) | Cites: **2** | no ArXiv | DOI:10.5194/essd-18-945-2026
   - Relevance: image classification benchmark

4. **Investigating the Utility of Explainable Artificial Intelligence for Neuroimaging‐Based Dementia Diagnosis and Prognosis** [CIT]
   - Martin (2026) | Cites: **1** | no ArXiv | DOI:10.1002/hbm.70456
   - Relevance: general computer vision

5. **A non-invasive MRI-based multimodal fusion deep learning model (MF-DLM) for predicting overall survival in bladder cancer: a multicentre retrospective study** [CIT]
   - Cai (2025) | Cites: **1** | no ArXiv | DOI:10.1016/j.eclinm.2025.103640
   - Relevance: deep learning foundations

6. **BENCHMARKING VISION TRANSFORMER KLASIFIKASI VISUAL MASAKAN PADANG DENGAN ROBUSTNESS MELALUI AUGMENTASI DATA** [CIT]
   - Pradhana (2025) | Cites: **1** | no ArXiv | DOI:10.33005/sitasi.v5i1.2527
   - Relevance: vision transformer architecture; image classification benchmark

7. **Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era** [CIT]
   - Lu (2025) | Cites: **1** | ArXiv:2511.06024 | DOI:10.48550/arXiv.2511.06024
   - Relevance: vision transformer architecture; visual recognition task; deep learning foundations

8. **Do Students Debias Like Teachers? On the Distillability of Bias Mitigation Methods** [CIT]
   - Cheng (2025) | Cites: **1** | ArXiv:2510.26038 | DOI:10.48550/arXiv.2510.26038
   - Relevance: transfer learning / pretraining

9. **Adaptive recruitment of cortex-wide recurrence for visual object recognition** [CIT]
   - Oyarzo (2025) | Cites: **1** | no ArXiv | DOI:10.1101/2025.10.17.682937
   - Relevance: visual recognition task

10. **PAGE-4D: Disentangled Pose and Geometry Estimation for VGGT-4D Perception** [CIT]
   - Zhou (2025) | Cites: **1** | ArXiv:2510.17568 | no DOI
   - Relevance: CNN architecture

---
## Anchor 3: ResNet (He 2016)
ArXiv:1512.03385 | Refs: 54 raw / 24 filtered | Cits: 100 raw / 40 filtered

### References (papers this anchor cites)

1. **ImageNet classification with deep convolutional neural networks** [REF]
   - Krizhevsky (2012) | Cites: **126,741** | no ArXiv | DOI:10.1145/3065386
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

2. **Very Deep Convolutional Networks for Large-Scale Image Recognition** [REF]
   - Simonyan (2014) | Cites: **109,093** | ArXiv:1409.1556 | no DOI
   - Relevance: CNN architecture; visual recognition task

3. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** [REF]
   - Ren (2015) | Cites: **70,251** | ArXiv:1506.01497 | DOI:10.1109/TPAMI.2016.2577031
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 1, 3]**

4. **Going deeper with convolutions** [REF]
   - Szegedy (2014) | Cites: **46,466** | ArXiv:1409.4842 | DOI:10.1109/CVPR.2015.7298594
   - Relevance: general computer vision **[MULTI-ANCHOR: 1, 3, 4, 5]**

5. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** [REF]
   - Ioffe (2015) | Cites: **45,980** | ArXiv:1502.03167 | no DOI
   - Relevance: general computer vision **[MULTI-ANCHOR: 3, 4]**

6. **ImageNet Large Scale Visual Recognition Challenge** [REF]
   - Russakovsky (2014) | Cites: **41,804** | ArXiv:1409.0575 | DOI:10.1007/s11263-015-0816-y
   - Relevance: image classification benchmark; visual recognition task **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

7. **Fully convolutional networks for semantic segmentation** [REF]
   - Shelhamer (2014) | Cites: **41,027** | ArXiv:1411.4038 | DOI:10.1109/CVPR.2015.7298965
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

8. **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation** [REF]
   - Girshick (2013) | Cites: **28,399** | ArXiv:1311.2524 | DOI:10.1109/CVPR.2014.81
   - Relevance: visual recognition task **[MULTI-ANCHOR: 3, 5]**

9. **Fast R-CNN** [REF]
   - Girshick (2015) | Cites: **27,703** | ArXiv:1504.08083 | no DOI
   - Relevance: CNN architecture; architecture comparison

10. **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification** [REF]
   - He (2015) | Cites: **20,050** | ArXiv:1502.01852 | DOI:10.1109/ICCV.2015.123
   - Relevance: image classification benchmark

11. **Neural Networks for Pattern Recognition** [REF]
   - Wulf (1993) | Cites: **16,936** | no ArXiv | DOI:10.1016/S0065-2458(08)60404-0
   - Relevance: visual recognition task; deep learning foundations

12. **International Journal of Computer Vision manuscript No. (will be inserted by the editor) The PASCAL Visual Object Classes (VOC) Challenge** [REF]
   - Unknown (n.d.) | Cites: **15,516** | no ArXiv | no DOI
   - Relevance: general computer vision **[MULTI-ANCHOR: 3, 5]**

13. **Caffe: Convolutional Architecture for Fast Feature Embedding** [REF]
   - Jia (2014) | Cites: **14,821** | ArXiv:1408.5093 | DOI:10.1145/2647868.2654889
   - Relevance: CNN architecture

14. **Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition** [REF]
   - He (2014) | Cites: **12,314** | ArXiv:1406.4729 | DOI:10.1007/978-3-319-10578-9_23
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

15. **Backpropagation Applied to Handwritten Zip Code Recognition** [REF]
   - LeCun (1989) | Cites: **11,855** | no ArXiv | DOI:10.1162/neco.1989.1.4.541
   - Relevance: visual recognition task **[MULTI-ANCHOR: 3, 5]**

### Citations (papers that cite this anchor)

1. **DiffusionEngine: Diffusion model is scalable data engine for object detection** [CIT]
   - Zhang (2026) | Cites: **2** | no ArXiv | DOI:10.1016/j.patcog.2025.112141
   - Relevance: visual recognition task

2. **DCM-Net: A novel dual-branch CNN-Mamba cross-layer feature fusion network for medical image segmentation** [CIT]
   - Liu (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.bspc.2025.109267
   - Relevance: CNN architecture; visual recognition task

3. **Deep learning-based remote sensing image super-resolution: Recent advances and challenges** [CIT]
   - Yang (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.neucom.2026.132939
   - Relevance: deep learning foundations

4. **Blockchain-empowered cluster distillation federated learning for heterogeneous smart grids** [CIT]
   - Zhou (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.inffus.2025.103913
   - Relevance: transfer learning / pretraining

5. **A novel obstructive sleep apnea detection model based on multi-scale convolutional neural networks** [CIT]
   - Liang (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.eswa.2025.130312
   - Relevance: CNN architecture; deep learning foundations

6. **BFCNet: A bidirectional feature coupling network for cross-modal emotion recognition based on brain functional connectivity** [CIT]
   - Fu (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109492
   - Relevance: vision transformer architecture; visual recognition task

7. **Deep learning-based myocardial infarction detection and localization via multi-lead ECG and constant-Q nonstationary gabor transform** [CIT]
   - Eltrass (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109541
   - Relevance: deep learning foundations

8. **MDA-UNet: A multi-dimensional attention model to enhance fuzzy boundary segmentation in MRI of liver metastases** [CIT]
   - Ou (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109517
   - Relevance: vision transformer architecture; visual recognition task

9. **Pixel-level polyp segmentation network based on parallel feature enhancement and attention mechanism** [CIT]
   - Xu (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109548
   - Relevance: vision transformer architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

10. **Automated Segmentation of Pterygium Lesions Using Multiscale Deep Learning Networks.** [CIT]
   - Zulkifley (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.exer.2026.110928
   - Relevance: visual recognition task; deep learning foundations

---
## Anchor 4: EfficientNet (Tan 2019)
ArXiv:1905.11946 | Refs: 54 raw / 36 filtered | Cits: 100 raw / 63 filtered

### References (papers this anchor cites)

1. **Deep Residual Learning for Image Recognition** [REF]
   - He (2015) | Cites: **220,082** | ArXiv:1512.03385 | DOI:10.1109/cvpr.2016.90
   - Relevance: visual recognition task **[MULTI-ANCHOR: 1, 4]**

2. **ImageNet classification with deep convolutional neural networks** [REF]
   - Krizhevsky (2012) | Cites: **126,741** | no ArXiv | DOI:10.1145/3065386
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

3. **Going deeper with convolutions** [REF]
   - Szegedy (2014) | Cites: **46,466** | ArXiv:1409.4842 | DOI:10.1109/CVPR.2015.7298594
   - Relevance: general computer vision **[MULTI-ANCHOR: 1, 3, 4, 5]**

4. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** [REF]
   - Ioffe (2015) | Cites: **45,980** | ArXiv:1502.03167 | no DOI
   - Relevance: general computer vision **[MULTI-ANCHOR: 3, 4]**

5. **Dropout: a simple way to prevent neural networks from overfitting** [REF]
   - Srivastava (2014) | Cites: **42,400** | no ArXiv | DOI:10.5555/2627435.2670313
   - Relevance: deep learning foundations

6. **ImageNet Large Scale Visual Recognition Challenge** [REF]
   - Russakovsky (2014) | Cites: **41,804** | ArXiv:1409.0575 | DOI:10.1007/s11263-015-0816-y
   - Relevance: image classification benchmark; visual recognition task **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

7. **Densely Connected Convolutional Networks** [REF]
   - Huang (2016) | Cites: **41,635** | ArXiv:1608.06993 | DOI:10.1109/CVPR.2017.243
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

8. **Mask R-CNN** [REF]
   - He (2017) | Cites: **30,840** | ArXiv:1703.06870 | no DOI
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

9. **Rethinking the Inception Architecture for Computer Vision** [REF]
   - Szegedy (2015) | Cites: **30,135** | ArXiv:1512.00567 | DOI:10.1109/CVPR.2016.308
   - Relevance: CNN architecture **[MULTI-ANCHOR: 1, 4]**

10. **Feature Pyramid Networks for Object Detection** [REF]
   - Lin (2016) | Cites: **25,697** | ArXiv:1612.03144 | DOI:10.1109/CVPR.2017.106
   - Relevance: visual recognition task

11. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** [REF]
   - Howard (2017) | Cites: **23,922** | ArXiv:1704.04861 | no DOI
   - Relevance: CNN architecture; deep learning foundations **[MULTI-ANCHOR: 1, 4]**

12. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** [REF]
   - Sandler (2018) | Cites: **23,146** | ArXiv:1801.04381 | DOI:10.1109/CVPR.2018.00474
   - Relevance: CNN architecture

13. **Xception: Deep Learning with Depthwise Separable Convolutions** [REF]
   - Chollet (2016) | Cites: **16,932** | ArXiv:1610.02357 | DOI:10.1109/CVPR.2017.195
   - Relevance: model scaling; deep learning foundations

14. **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning** [REF]
   - Szegedy (2016) | Cites: **15,190** | ArXiv:1602.07261 | DOI:10.1609/aaai.v31i1.11231
   - Relevance: CNN architecture

15. **Aggregated Residual Transformations for Deep Neural Networks** [REF]
   - Xie (2016) | Cites: **11,376** | ArXiv:1611.05431 | DOI:10.1109/CVPR.2017.634
   - Relevance: deep learning foundations **[MULTI-ANCHOR: 1, 4]**

### Citations (papers that cite this anchor)

1. **U-MobileViT: A Lightweight Vision Transformer-based Backbone for Panoptic Driving Segmentation** [CIT]
   - Nguyen (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.image.2025.117461
   - Relevance: vision transformer architecture; visual recognition task

2. **Bayesian optimization-enhanced vision transformer for damage detection in steel truss structures using continuous wavelet transform analysis** [CIT]
   - Hi̇çyılmaz (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.istruc.2026.111059
   - Relevance: vision transformer architecture

3. **Harnessing the power of Squidle+ to develop flexible machine learning models.** [CIT]
   - Günzel (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.marenvres.2026.107888
   - Relevance: general computer vision

4. **PatchFlow: Leveraging a Flow-Based Model with Patch Features** [CIT]
   - Zhang (2026) | Cites: **1** | ArXiv:2602.05238 | no DOI
   - Relevance: general computer vision

5. **MSANet: Electromagnetic ultrasonic signal recognition and grading of submarine pipeline defects based on a multi-sensory attention network** [CIT]
   - Zhang (2026) | Cites: **1** | no ArXiv | DOI:10.1016/j.ijpvp.2025.105692
   - Relevance: vision transformer architecture; visual recognition task

6. **Attention-based deep learning model for clinical assessment of focal liver lesions using ultrasound imaging** [CIT]
   - Lingamaiah (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109563
   - Relevance: vision transformer architecture; deep learning foundations

7. **DB-HDFFN: Dual Branch Hierarchical Dynamic Feature Fusion Network for medical image classification** [CIT]
   - Wen (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109538
   - Relevance: image classification benchmark **[MULTI-ANCHOR: 4, 5]**

8. **Multi-perspective domain-invariant network with energy density-based data augmentation for domain generalization fault diagnosis** [CIT]
   - Hong (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.eswa.2026.131583
   - Relevance: training regularization

9. **Glottal localization and segmentation using two-stage convolutional neural network during tracheal intubation** [CIT]
   - Wang (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2025.109214
   - Relevance: CNN architecture; visual recognition task; deep learning foundations

10. **A wheat seedling detection model based on efficient feature extraction and coordinate attention mechanism** [CIT]
   - Wang (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.eja.2026.127993
   - Relevance: vision transformer architecture

---
## Anchor 5: VGGNet (Simonyan 2015)
ArXiv:1409.1556 | Refs: 43 raw / 31 filtered | Cits: 100 raw / 57 filtered

### References (papers this anchor cites)

1. **ImageNet classification with deep convolutional neural networks** [REF]
   - Krizhevsky (2012) | Cites: **126,741** | no ArXiv | DOI:10.1145/3065386
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

2. **ImageNet: A large-scale hierarchical image database** [REF]
   - Deng (2009) | Cites: **70,819** | no ArXiv | DOI:10.1109/CVPR.2009.5206848
   - Relevance: image classification benchmark **[MULTI-ANCHOR: 2, 5]**

3. **Going deeper with convolutions** [REF]
   - Szegedy (2014) | Cites: **46,466** | ArXiv:1409.4842 | DOI:10.1109/CVPR.2015.7298594
   - Relevance: general computer vision **[MULTI-ANCHOR: 1, 3, 4, 5]**

4. **ImageNet Large Scale Visual Recognition Challenge** [REF]
   - Russakovsky (2014) | Cites: **41,804** | ArXiv:1409.0575 | DOI:10.1007/s11263-015-0816-y
   - Relevance: image classification benchmark; visual recognition task **[MULTI-ANCHOR: 1, 2, 3, 4, 5]**

5. **Fully convolutional networks for semantic segmentation** [REF]
   - Shelhamer (2014) | Cites: **41,027** | ArXiv:1411.4038 | DOI:10.1109/CVPR.2015.7298965
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

6. **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation** [REF]
   - Girshick (2013) | Cites: **28,399** | ArXiv:1311.2524 | DOI:10.1109/CVPR.2014.81
   - Relevance: visual recognition task **[MULTI-ANCHOR: 3, 5]**

7. **Visualizing and Understanding Convolutional Networks** [REF]
   - Zeiler (2013) | Cites: **16,751** | ArXiv:1311.2901 | DOI:10.1007/978-3-319-10590-1_53
   - Relevance: CNN architecture; architecture comparison

8. **International Journal of Computer Vision manuscript No. (will be inserted by the editor) The PASCAL Visual Object Classes (VOC) Challenge** [REF]
   - Unknown (n.d.) | Cites: **15,516** | no ArXiv | no DOI
   - Relevance: general computer vision **[MULTI-ANCHOR: 3, 5]**

9. **Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition** [REF]
   - He (2014) | Cites: **12,314** | ArXiv:1406.4729 | DOI:10.1007/978-3-319-10578-9_23
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

10. **Backpropagation Applied to Handwritten Zip Code Recognition** [REF]
   - LeCun (1989) | Cites: **11,855** | no ArXiv | DOI:10.1162/neco.1989.1.4.541
   - Relevance: visual recognition task **[MULTI-ANCHOR: 3, 5]**

11. **Two-Stream Convolutional Networks for Action Recognition in Videos** [REF]
   - Simonyan (2014) | Cites: **8,022** | ArXiv:1406.2199 | no DOI
   - Relevance: CNN architecture; visual recognition task

12. **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps** [REF]
   - Simonyan (2013) | Cites: **8,017** | ArXiv:1312.6034 | no DOI
   - Relevance: CNN architecture; image classification benchmark

13. **OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks** [REF]
   - Sermanet (2013) | Cites: **5,105** | ArXiv:1312.6229 | no DOI
   - Relevance: CNN architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

14. **CNN Features Off-the-Shelf: An Astounding Baseline for Recognition** [REF]
   - Razavian (2014) | Cites: **5,061** | ArXiv:1403.6382 | DOI:10.1109/CVPRW.2014.131
   - Relevance: CNN architecture; visual recognition task

15. **DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition** [REF]
   - Donahue (2013) | Cites: **5,061** | ArXiv:1310.1531 | no DOI
   - Relevance: CNN architecture; visual recognition task

### Citations (papers that cite this anchor)

1. **HeShare: Energy-Aware and Efficient Multi-Task GPU Sharing in Heterogeneous GPU-Based Computing Systems** [CIT]
   - Jiang (2026) | Cites: **1** | no ArXiv | DOI:10.1109/TC.2025.3628924
   - Relevance: general computer vision

2. **A Multimodal Multi-Drone Cooperation System for Real-Time Human Searching** [CIT]
   - Peng (2026) | Cites: **1** | no ArXiv | DOI:10.1109/TMC.2025.3619530
   - Relevance: general computer vision

3. **Federated CNN-Transformer: Enabling Distributed Sensing-Assisted Beam Prediction in ISAC Systems for IoT Applications** [CIT]
   - Zhao (2026) | Cites: **1** | no ArXiv | DOI:10.1109/JIOT.2025.3593588
   - Relevance: vision transformer architecture; CNN architecture

4. **Pixel-level polyp segmentation network based on parallel feature enhancement and attention mechanism** [CIT]
   - Xu (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109548
   - Relevance: vision transformer architecture; visual recognition task **[MULTI-ANCHOR: 3, 5]**

5. **DB-HDFFN: Dual Branch Hierarchical Dynamic Feature Fusion Network for medical image classification** [CIT]
   - Wen (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.bspc.2026.109538
   - Relevance: image classification benchmark **[MULTI-ANCHOR: 4, 5]**

6. **RA-GCN: Residual attention based graph convolutional network for multi-label pattern image retrieval** [CIT]
   - Li (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.patcog.2025.112647
   - Relevance: vision transformer architecture; CNN architecture **[MULTI-ANCHOR: 3, 5]**

7. **Unleash and integrate the power of pre-trained ViTs via feature fusion for open-vocabulary object detection** [CIT]
   - Gao (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.displa.2025.103321
   - Relevance: vision transformer architecture; transfer learning / pretraining; visual recognition task **[MULTI-ANCHOR: 3, 5]**

8. **Lightweight deep learning approach for retinal OCT image classification: A CNN with hybrid pooling and optimized learning** [CIT]
   - Dave (2026) | Cites: **0** | no ArXiv | DOI:10.11591/ijict.v15i1.pp414-427
   - Relevance: CNN architecture; image classification benchmark; deep learning foundations

9. **Efficient dual-branch high-resolution transformer design for crack segmentation** [CIT]
   - Li (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.asoc.2025.114513
   - Relevance: vision transformer architecture; visual recognition task

10. **A transformer based multi-task deep learning model for urban livability evaluation by fusing remote sensing and textual geospatial data** [CIT]
   - Zhou (2026) | Cites: **0** | no ArXiv | DOI:10.1016/j.rse.2026.115232
   - Relevance: vision transformer architecture; deep learning foundations

---
## Cross-Anchor Papers

Papers appearing in the filtered results of 2+ anchors. These are the most structurally central works in this citation network.

### Appears in 5 anchor networks

- **ImageNet classification with deep convolutional neural networks** -- Krizhevsky (2012) -- 126,741 cites
  - A1(ref), A2(ref), A3(ref), A4(ref), A5(ref)
- **ImageNet Large Scale Visual Recognition Challenge** -- Russakovsky (2014) -- 41,804 cites | ArXiv:1409.0575
  - A1(ref), A2(ref), A3(ref), A4(ref), A5(ref)

### Appears in 4 anchor networks

- **Going deeper with convolutions** -- Szegedy (2014) -- 46,466 cites | ArXiv:1409.4842
  - A1(ref), A3(ref), A4(ref), A5(ref)

### Appears in 2 anchor networks

- **Deep Residual Learning for Image Recognition** -- He (2015) -- 220,082 cites | ArXiv:1512.03385
  - A1(ref), A4(ref)
- **Attention is All you Need** -- Vaswani (2017) -- 165,637 cites | ArXiv:1706.03762
  - A1(ref), A2(ref)
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** -- Devlin (2019) -- 109,971 cites | ArXiv:1810.04805
  - A1(ref), A2(ref)
- **ImageNet: A large-scale hierarchical image database** -- Deng (2009) -- 70,819 cites
  - A2(ref), A5(ref)
- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** -- Ren (2015) -- 70,251 cites | ArXiv:1506.01497
  - A1(ref), A3(ref)
- **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** -- Ioffe (2015) -- 45,980 cites | ArXiv:1502.03167
  - A3(ref), A4(ref)
- **Densely Connected Convolutional Networks** -- Huang (2016) -- 41,635 cites | ArXiv:1608.06993
  - A1(ref), A4(ref)
- **Fully convolutional networks for semantic segmentation** -- Shelhamer (2014) -- 41,027 cites | ArXiv:1411.4038
  - A3(ref), A5(ref)
- **Mask R-CNN** -- He (2017) -- 30,840 cites | ArXiv:1703.06870
  - A1(ref), A4(ref)
- **Rethinking the Inception Architecture for Computer Vision** -- Szegedy (2015) -- 30,135 cites | ArXiv:1512.00567
  - A1(ref), A4(ref)
- **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation** -- Girshick (2013) -- 28,399 cites | ArXiv:1311.2524
  - A3(ref), A5(ref)
- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** -- Howard (2017) -- 23,922 cites | ArXiv:1704.04861
  - A1(ref), A4(ref)
- **End-to-End Object Detection with Transformers** -- Carion (2020) -- 16,889 cites | ArXiv:2005.12872
  - A1(ref), A2(ref)
- **International Journal of Computer Vision manuscript No. (will be inserted by the editor) The PASCAL Visual Object Classes (VOC) Challenge** -- Unknown (n.d.) -- 15,516 cites
  - A3(ref), A5(ref)
- **Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition** -- He (2014) -- 12,314 cites | ArXiv:1406.4729
  - A3(ref), A5(ref)
- **Backpropagation Applied to Handwritten Zip Code Recognition** -- LeCun (1989) -- 11,855 cites
  - A3(ref), A5(ref)
- **Aggregated Residual Transformations for Deep Neural Networks** -- Xie (2016) -- 11,376 cites | ArXiv:1611.05431
  - A1(ref), A4(ref)
- **OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks** -- Sermanet (2013) -- 5,105 cites | ArXiv:1312.6229
  - A3(ref), A5(ref)
- **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** -- Yuan (2021) -- 2,378 cites | ArXiv:2101.11986
  - A1(ref), A2(ref)
- **An Empirical Study of Training Self-Supervised Vision Transformers** -- Chen (2021) -- 2,231 cites | ArXiv:2104.02057
  - A1(ref), A2(ref)
- **Generative Pretraining From Pixels** -- Chen (2020) -- 1,728 cites
  - A1(ref), A2(ref)
- **Pixel-level polyp segmentation network based on parallel feature enhancement and attention mechanism** -- Xu (2026) -- 0 cites
  - A3(cit), A5(cit)
- **RA-GCN: Residual attention based graph convolutional network for multi-label pattern image retrieval** -- Li (2026) -- 0 cites
  - A3(cit), A5(cit)
- **Unleash and integrate the power of pre-trained ViTs via feature fusion for open-vocabulary object detection** -- Gao (2026) -- 0 cites
  - A3(cit), A5(cit)
- **DB-HDFFN: Dual Branch Hierarchical Dynamic Feature Fusion Network for medical image classification** -- Wen (2026) -- 0 cites
  - A4(cit), A5(cit)
