# Conceptual Guide: CNN and Vision Transformer Architectures

This companion report to the literature review provides visual, diagrammatic explanations of the core architectures compared in the ViT-vs-CNN literature. The goal is to build intuition for *how* each architecture processes an image, *where* their inductive biases come from, and *why* these structural differences drive the data-efficiency gap discussed in the main report.

---

## 1. The CNN Pipeline

A convolutional neural network processes an image through a hierarchy of stages. Each stage applies local filters (convolutions) followed by spatial downsampling (pooling or strided convolutions), progressively building from low-level features (edges, textures) to high-level features (object parts, scenes).

```mermaid
flowchart TD
    A["Input Image (224 x 224 x 3)"]

    subgraph Stage1["Stage 1 — Edges (112 x 112)"]
        B["Conv 3x3 + ReLU → Conv 3x3 + ReLU → MaxPool"]
    end

    subgraph Stage2["Stage 2 — Textures (56 x 56)"]
        C["Conv 3x3 + ReLU → Conv 3x3 + ReLU → MaxPool"]
    end

    subgraph Stage3["Stage 3 — Parts (28 x 28)"]
        D["Conv 3x3 + ReLU → Conv 3x3 + ReLU → MaxPool"]
    end

    subgraph Stage4["Stage 4 — Semantics (14 x 14)"]
        E["Conv 3x3 + ReLU → Conv 3x3 + ReLU → Global Avg Pool"]
    end

    F["FC → Softmax → Class Prediction"]

    A --> Stage1 --> Stage2 --> Stage3 --> Stage4 --> F
```

**Key properties:**
- **Locality**: Each convolution filter sees only a small spatial region (e.g. 3x3 pixels). The network's "view" of the image grows gradually through stacked layers — this is the *receptive field*.
- **Translation equivariance**: The same filter is applied at every spatial position. A cat in the top-left produces the same feature activations as a cat in the bottom-right, just shifted in the feature map.
- **Hierarchical processing**: Early layers detect edges and textures; middle layers combine these into parts; later layers recognize objects. This is not learned — it is forced by the architecture.

### 1.1 The Convolution Operation

The fundamental building block. A small filter slides across the input, computing a dot product at each position. The output is a feature map where each value represents the presence of a local pattern.

```mermaid
flowchart TD
    subgraph input["Input Feature Map"]
        I1["5 x 5 spatial grid\n(one channel shown)"]
    end

    subgraph kernel["Learned Filter"]
        K1["3 x 3 kernel\nweights shared\nacross all positions"]
    end

    subgraph output["Output Feature Map"]
        O1["3 x 3 spatial grid\neach value = dot product\nof filter with local patch"]
    end

    I1 -- "slide filter\nacross spatial\npositions" --> O1
    K1 -- "same weights\neverywhere\n(weight sharing)" --> O1
```

Weight sharing is the source of both **parameter efficiency** (one 3x3 filter has just 9 weights regardless of image size) and **translation equivariance** (shifting the input shifts the output identically).

### 1.2 ResNet: Skip Connections

ResNet (He et al., 2016) solved the degradation problem in deep networks by introducing skip connections that let gradients flow directly through the network. Instead of learning a mapping H(x), each block learns a *residual* F(x) = H(x) - x, so the output is F(x) + x.

```mermaid
flowchart LR
    subgraph ResBlock["Residual Block"]
        direction TB
        X["Input x"] --> CONV1["Conv 3x3 + BN + ReLU"]
        CONV1 --> CONV2["Conv 3x3 + BN"]
        X -- "skip connection\n(identity)" --> ADD["Add: F(x) + x"]
        CONV2 --> ADD
        ADD --> RELU["ReLU"]
    end

    IN["Input"] --> ResBlock --> OUT["Output"]
```

This seemingly simple change is what enabled networks deeper than ~20 layers to train effectively. A ResNet-50 stacks 16 such blocks across 4 stages, with downsampling between stages. The skip connection ensures that even if a block learns nothing useful (F(x) ≈ 0), the signal passes through unchanged — the network can only help, never hurt, by adding depth.

---

## 2. The Vision Transformer (ViT) Pipeline

A Vision Transformer treats an image as a sequence of patches, embeds each patch into a vector, and processes the resulting sequence through a standard transformer encoder. There are no convolutions, no pooling, and no hierarchical stages — just self-attention operating on a flat sequence.

```mermaid
flowchart LR
    subgraph Input
        A["Image\n(224 x 224 x 3)"]
    end

    subgraph Patchify["Patch Embedding"]
        B["Split into\n14 x 14 = 196\npatches of 16x16"]
        C["Linear projection\nof each patch\n→ D-dim vector"]
    end

    subgraph Tokens["Token Sequence"]
        D["[CLS] token\n+ 196 patch tokens\n+ position embeddings"]
    end

    subgraph Encoder["Transformer Encoder (L layers)"]
        E["Multi-Head\nSelf-Attention"]
        F["Feed-Forward\nNetwork (MLP)"]
        G["Layer Norm\n+ Residual"]
    end

    subgraph Head["Classification Head"]
        H["MLP on [CLS]\n→ Softmax"]
    end

    A --> B --> C --> D --> E --> F --> G
    G -. "repeat L times" .-> E
    G --> H
```

**Key properties:**
- **No locality bias**: Self-attention computes relationships between *all* patch pairs. A patch in the corner can directly attend to a patch at the center. There is no constraint that forces local processing first.
- **No translation equivariance**: Position is encoded through learned embeddings (absolute coordinates), not through shared weights. Moving an object in the image changes its position embeddings.
- **Flat processing**: All layers operate on the same spatial resolution (196 tokens). There is no progressive downsampling or feature hierarchy — the model must learn any hierarchical structure from data.

### 2.1 Patch Embedding in Detail

This is how ViT converts a 2D image into a 1D sequence. The image is divided into a grid of non-overlapping patches, and each patch is linearly projected into the model's embedding dimension.

```mermaid
flowchart TD
    subgraph image["Input Image (224 x 224)"]
        IMG["Divided into\n14 rows x 14 cols\n= 196 patches\nEach patch: 16 x 16 x 3\n= 768 values"]
    end

    subgraph embed["Embedding"]
        FLAT["Flatten each patch\n→ 768-dim vector"]
        PROJ["Linear projection\n768 → D dimensions"]
        POS["Add learned\nposition embedding\n(1 per patch)"]
        CLS["Prepend [CLS]\ntoken"]
    end

    subgraph seq["Resulting Sequence"]
        SEQ["197 tokens\neach D-dimensional\n[CLS], p₁, p₂, ..., p₁₉₆"]
    end

    IMG --> FLAT --> PROJ --> POS --> CLS --> SEQ
```

The patchify stem uses a single 16x16 convolution with stride 16 — an aggressive downsampling that discards all sub-patch spatial structure in one step. This is the opposite of CNN design wisdom, where gradual downsampling with small strides is standard. As Xiao et al. (2021) showed, replacing this with a stack of 3x3 convolutions improves optimization stability and accuracy.

### 2.2 Self-Attention: The Core Mechanism

Self-attention is what allows every token to "look at" every other token. For each token, it computes query (Q), key (K), and value (V) vectors, then uses dot-product attention to produce a weighted combination of all values.

```mermaid
flowchart TD
    subgraph sa["Self-Attention for one token"]
        INPUT["Input token xᵢ"]
        Q["Query: qᵢ = Wq · xᵢ"]
        K["Keys from ALL tokens:\nK = Wk · [x₁...x₁₉₇]"]
        V["Values from ALL tokens:\nV = Wv · [x₁...x₁₉₇]"]

        DOT["Attention scores:\nαᵢⱼ = softmax(qᵢ · kⱼ / √d)\nfor all j = 1..197"]

        OUT["Output:\nyᵢ = Σⱼ αᵢⱼ · vⱼ\n(weighted sum of all values)"]
    end

    INPUT --> Q
    INPUT --> K
    INPUT --> V
    Q --> DOT
    K --> DOT
    DOT --> OUT
    V --> OUT
```

**Multi-head attention** runs H independent attention operations in parallel (each with its own Q, K, V projections), then concatenates and projects the results. This allows different heads to attend to different relationship types — some might learn local attention (nearby patches), others global attention (distant patches).

**Cost**: Self-attention has O(N²) complexity where N is the sequence length. For 196 patches this is manageable; for high-resolution images with thousands of patches, this becomes prohibitive — motivating architectures like Swin Transformer.

### 2.3 A Single Transformer Encoder Block

Each layer of the transformer encoder applies multi-head self-attention followed by a feed-forward network, with layer normalization and residual connections around each.

```mermaid
flowchart TD
    X["Input tokens\n(197 x D)"] --> LN1["Layer Norm"]
    LN1 --> MHSA["Multi-Head\nSelf-Attention"]
    MHSA --> ADD1["Add (residual)"]
    X --> ADD1

    ADD1 --> LN2["Layer Norm"]
    LN2 --> FFN["Feed-Forward Network\nLinear → GELU → Linear\n(D → 4D → D)"]
    FFN --> ADD2["Add (residual)"]
    ADD1 --> ADD2

    ADD2 --> OUT["Output tokens\n(197 x D)"]
```

The feed-forward network operates independently on each token (no cross-token interaction). All cross-token communication happens exclusively through self-attention.

---

## 3. CNN vs. ViT: Structural Comparison

The following diagram contrasts how each architecture processes the same input image, highlighting the fundamental structural differences.

```mermaid
flowchart TD
    IMG["Input Image\n224 x 224 x 3"]

    subgraph CNN["CNN Path"]
        direction TB
        C1["Stage 1: 112x112\nLocal edges\n(receptive field: 3x3)"]
        C2["Stage 2: 56x56\nTextures, corners\n(RF: ~10x10)"]
        C3["Stage 3: 28x28\nObject parts\n(RF: ~40x40)"]
        C4["Stage 4: 14x14\nSemantic concepts\n(RF: ~100x100)"]
        C5["Global Pool → FC"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph ViT["ViT Path"]
        direction TB
        V1["Patch Embed:\n196 tokens (14x14 grid)\nEach token = 16x16 patch"]
        V2["Transformer Layer 1\nAll 196 tokens attend\nto all 196 tokens"]
        V3["Transformer Layer 2\n(same resolution,\nsame token count)"]
        V4["...\nTransformer Layer L"]
        V5["[CLS] token → FC"]
        V1 --> V2 --> V3 --> V4 --> V5
    end

    IMG --> C1
    IMG --> V1
```

| Property | CNN | ViT |
|---|---|---|
| Spatial processing | Hierarchical (progressive downsampling) | Flat (fixed resolution throughout) |
| Receptive field | Grows gradually across layers | Global from the first layer |
| Locality | Hardcoded (small filters) | Must be learned from data |
| Translation equivariance | Built in (weight sharing) | Absent (learned position embeddings) |
| Parameter sharing | Across spatial positions | Across sequence positions (same attention weights) |
| Complexity in spatial dim | O(N) per layer | O(N²) per layer |
| Inductive bias | Strong (good for small data) | Weak (needs large data or augmentation) |

---

## 4. Swin Transformer: Bridging the Gap

Swin Transformer (Liu et al., 2021) re-introduces two CNN design principles into the transformer framework: **hierarchical feature maps** and **local attention windows**. This makes the architecture more efficient and gives it structural properties closer to a CNN while retaining the benefits of self-attention.

```mermaid
flowchart TD
    IMG["Image\n224 x 224 x 3"]

    subgraph S1["Stage 1 — Patch Partition + Linear Embed"]
        P1["4x4 patches → 56x56 tokens\nC = 96"]
    end

    subgraph S2["Stage 2 — Patch Merging + Swin Blocks"]
        P2["Merge 2x2 → 28x28 tokens\nC = 192"]
        W2["Window Attention\n(7x7 windows)"]
        SW2["Shifted Window\nAttention"]
    end

    subgraph S3["Stage 3 — Patch Merging + Swin Blocks"]
        P3["Merge 2x2 → 14x14 tokens\nC = 384"]
        W3["Window + Shifted\nWindow Attention"]
    end

    subgraph S4["Stage 4 — Patch Merging + Swin Blocks"]
        P4["Merge 2x2 → 7x7 tokens\nC = 768"]
        W4["Window + Shifted\nWindow Attention"]
    end

    HEAD["Global Pool → FC"]

    IMG --> P1 --> P2 --> W2 --> SW2 --> P3 --> W3 --> P4 --> W4 --> HEAD
```

### 4.1 Windowed and Shifted Window Attention

Instead of computing attention among all tokens (O(N²)), Swin restricts attention to local windows. To allow cross-window information flow, consecutive transformer blocks alternate between regular and shifted window partitions.

```mermaid
flowchart LR
    subgraph regular["Layer L: Regular Windows"]
        RW["Feature map partitioned\ninto non-overlapping\n7x7 windows\n\nAttention computed\nINSIDE each window\n\nNo cross-window\ncommunication"]
    end

    subgraph shifted["Layer L+1: Shifted Windows"]
        SW["Windows shifted by\n(3, 3) pixels\n\nNew windows span\nboundaries of old windows\n\nEnables cross-window\ninformation flow"]
    end

    regular --> shifted
```

This alternation gives Swin a growing effective receptive field across layers (similar to stacked CNN layers) while keeping the computational cost linear in image size rather than quadratic.

---

## 5. Hybrid Architectures: CoAtNet

CoAtNet (Dai et al., 2021) directly stacks convolutional stages and transformer stages in a single architecture. The principle: use convolution early (where locality bias helps most and spatial resolution is highest) and attention later (where global reasoning is needed and the token count is smaller).

```mermaid
flowchart LR
    IMG["Image\n224 x 224"]

    subgraph ConvStages["Convolutional Stages"]
        S0["S0: Conv Stem\n112 x 112"]
        S1["S1: MBConv Blocks\n56 x 56\n(depthwise separable\nconvolutions)"]
    end

    subgraph TransStages["Transformer Stages"]
        S2["S2: Relative\nSelf-Attention\n28 x 28"]
        S3["S3: Relative\nSelf-Attention\n14 x 14"]
    end

    HEAD["Global Pool\n→ FC"]

    IMG --> S0 --> S1 --> S2 --> S3 --> HEAD
```

CoAtNet's C-C-T-T configuration works well across data regimes:
- **Small data** (ImageNet-1K): The convolutional stages provide inductive bias where it matters most — early spatial processing — while the transformer stages add capacity.
- **Large data** (JFT-300M): The transformer stages benefit from the additional data in ways that convolution stages cannot, since attention can learn arbitrarily complex spatial relationships.

The result: CoAtNet matched ViT-Huge pre-trained on JFT-300M (300M images) using only ImageNet-21K pre-training (14M images) — 23x less data.

---

## 6. Why ViTs Fail on Small Data: Visual Explanation

This diagram illustrates the core mechanistic finding from Raghu et al. (2021): well-trained ViTs learn a mixture of local and global attention in their lower layers, but data-starved ViTs fail to develop local attention patterns.

```mermaid
flowchart TD
    subgraph trained["ViT — Sufficient Training Data"]
        direction TB
        T_LOW["Lower Layers"]
        T_LOCAL["Some heads attend\nLOCALLY\n(nearby patches)"]
        T_GLOBAL["Other heads attend\nGLOBALLY\n(distant patches)"]
        T_MIX["Mixed local+global\nrepresentation\n(analogous to CNN\nearly layers)"]
        T_HIGH["Higher Layers\n→ Abstract features"]

        T_LOW --> T_LOCAL
        T_LOW --> T_GLOBAL
        T_LOCAL --> T_MIX
        T_GLOBAL --> T_MIX
        T_MIX --> T_HIGH
    end

    subgraph starved["ViT — Insufficient Training Data"]
        direction TB
        S_LOW["Lower Layers"]
        S_NOPE["Heads DO NOT learn\nlocal attention patterns"]
        S_DIFFUSE["Diffuse, unfocused\nattention across\nall positions"]
        S_FAIL["Higher Layers\n→ Poor representations\n→ Low accuracy"]

        S_LOW --> S_NOPE
        S_LOW --> S_DIFFUSE
        S_NOPE --> S_FAIL
        S_DIFFUSE --> S_FAIL
    end

    subgraph cnn["CNN — Any Data Size"]
        direction TB
        CNN_LOW["Lower Layers"]
        CNN_LOCAL["Local features\n(HARDCODED by\n3x3 convolutions)"]
        CNN_HIGH["Higher Layers\n→ Semantic features"]

        CNN_LOW --> CNN_LOCAL --> CNN_HIGH
    end
```

The CNN never faces this failure mode because locality is an architectural constraint, not a learned behavior. This is the fundamental source of the CNN's data-efficiency advantage.

---

## 7. The Spectrum of Inductive Bias

Across the architectures in the literature, there is a spectrum from strong inductive bias (CNNs) to weak inductive bias (vanilla ViT), with various hybrid and modified designs in between.

```mermaid
flowchart LR
    subgraph spectrum["Inductive Bias Spectrum"]
        direction LR
        STRONG["STRONG BIAS\n\nConvNeXt\nResNet\nEfficientNet\nVGG"]
        MODERATE["MODERATE BIAS\n\nCoAtNet (C-C-T-T)\nConViT (soft conv)\nSwin (windowed)\nDHVT (conv FFN)\nViT + Conv Stem"]
        WEAK["WEAK BIAS\n\nDeiT (augmentation)\nT2T-ViT (tokenizer)\nVanilla ViT"]

        STRONG --- MODERATE --- WEAK
    end

    subgraph data["Data Required"]
        direction LR
        LESS["Fewer images\nneeded"] --- MORE["More images\nneeded"]
    end

    subgraph flex["Model Flexibility"]
        direction LR
        CONSTRAINED["More constrained\nrepresentations"] --- FLEXIBLE["More flexible\nrepresentations"]
    end
```

The tradeoff: stronger bias means less data is needed to learn good representations, but the model is more constrained in what it can represent. Weaker bias means the model can potentially learn richer representations, but only if given enough data to discover the right structure.

For a small dataset like PASCAL VOC (~11K images), architectures toward the left of this spectrum (strong bias) will generalize better. As dataset size increases into the millions, architectures toward the right (weak bias) can leverage their flexibility to surpass the constrained models.

---

## 8. Summary: Architecture Selection for Our Project

```mermaid
flowchart TD
    Q{"Dataset size?"}
    Q -->|"< 50K images\n(PASCAL VOC)"| SMALL
    Q -->|"50K - 1M images\n(CIFAR, subsampled IN)"| MEDIUM
    Q -->|"> 1M images\n(ImageNet-1K+)"| LARGE

    subgraph SMALL["Small Data Regime"]
        S1["Best: Pre-trained CNN\n(ResNet, EfficientNet)"]
        S2["Good: Pre-trained hybrid\n(CoAtNet, Swin)"]
        S3["Viable: Pre-trained ViT\nwith heavy augmentation"]
        S4["Poor: ViT from scratch"]
    end

    subgraph MEDIUM["Medium Data Regime"]
        M1["Strong: CNN with\nmodern training"]
        M2["Strong: Hybrid\narchitectures"]
        M3["Competitive: ViT with\nDeiT/AugReg recipes"]
    end

    subgraph LARGE["Large Data Regime"]
        L1["Strong: All architectures\ncompetitive"]
        L2["ViTs begin to\npull ahead at scale"]
        L3["Hybrids most\nrobust overall"]
    end
```

For our PASCAL VOC experiments, we sit firmly in the small-data regime. The literature points to pre-trained CNNs as the strongest baselines, with pre-trained hybrids and augmented ViTs as viable comparison points — and pure ViTs from scratch as the expected low performer that demonstrates the data-efficiency gap.
