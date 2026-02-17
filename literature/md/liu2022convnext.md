A ConvNet for the 2020s

Zhuang Liu1,2* Hanzi Mao1 Chao-Yuan Wu1 Christoph Feichtenhofer1 Trevor Darrell2 Saining Xie1†

1Facebook AI Research (FAIR)

2UC Berkeley

Code: https://github.com/facebookresearch/ConvNeXt

2
2
0
2

r
a

M
2

]

V
C
.
s
c
[

2
v
5
4
5
3
0
.
1
0
2
2
:
v
i
X
r
a

Abstract

The “Roaring 20s” of visual recognition began with the
introduction of Vision Transformers (ViTs), which quickly
superseded ConvNets as the state-of-the-art image classiﬁca-
tion model. A vanilla ViT, on the other hand, faces difﬁculties
when applied to general computer vision tasks such as object
detection and semantic segmentation. It is the hierarchical
Transformers (e.g., Swin Transformers) that reintroduced sev-
eral ConvNet priors, making Transformers practically viable
as a generic vision backbone and demonstrating remarkable
performance on a wide variety of vision tasks. However,
the effectiveness of such hybrid approaches is still largely
credited to the intrinsic superiority of Transformers, rather
than the inherent inductive biases of convolutions. In this
work, we reexamine the design spaces and test the limits of
what a pure ConvNet can achieve. We gradually “modernize”
a standard ResNet toward the design of a vision Transformer,
and discover several key components that contribute to the
performance difference along the way. The outcome of this
exploration is a family of pure ConvNet models dubbed Con-
vNeXt. Constructed entirely from standard ConvNet modules,
ConvNeXts compete favorably with Transformers in terms of
accuracy and scalability, achieving 87.8% ImageNet top-1
accuracy and outperforming Swin Transformers on COCO
detection and ADE20K segmentation, while maintaining the
simplicity and efﬁciency of standard ConvNets.

1. Introduction

Looking back at the 2010s, the decade was marked by
the monumental progress and impact of deep learning. The
primary driver was the renaissance of neural networks, partic-
ularly convolutional neural networks (ConvNets). Through
the decade, the ﬁeld of visual recognition successfully
shifted from engineering features to designing (ConvNet)
architectures. Although the invention of back-propagation-
trained ConvNets dates all the way back to the 1980s [42],
it was not until late 2012 that we saw its true potential for

*Work done during an internship at Facebook AI Research.
†Corresponding author.

Figure 1. ImageNet-1K classiﬁcation results for • ConvNets and
◦ vision Transformers. Each bubble’s area is proportional to FLOPs
of a variant in a model family. ImageNet-1K/22K models here
take 2242/3842 images respectively. ResNet and ViT results were
obtained with improved training procedures over the original papers.
We demonstrate that a standard ConvNet model can achieve the
same level of scalability as hierarchical vision Transformers while
being much simpler in design.

visual feature learning. The introduction of AlexNet [40]
precipitated the “ImageNet moment” [59], ushering in a new
era of computer vision. The ﬁeld has since evolved at a
rapid speed. Representative ConvNets like VGGNet [64],
Inceptions [68], ResNe(X)t [28, 87], DenseNet [36], Mo-
bileNet [34], EfﬁcientNet [71] and RegNet [54] focused on
different aspects of accuracy, efﬁciency and scalability, and
popularized many useful design principles.

The full dominance of ConvNets in computer vision was
not a coincidence: in many application scenarios, a “sliding
window” strategy is intrinsic to visual processing, particu-
larly when working with high-resolution images. ConvNets
have several built-in inductive biases that make them well-
suited to a wide variety of computer vision applications. The
most important one is translation equivariance, which is a de-
sirable property for tasks like objection detection. ConvNets
are also inherently efﬁcient due to the fact that when used in
a sliding-window manner, the computations are shared [62].
For many decades, this has been the default use of ConvNets,
generally on limited object categories such as digits [43],
faces [58, 76] and pedestrians [19, 63]. Entering the 2010s,

4816256GFLOPsDiameter

the region-based detectors [23, 24, 27, 57] further elevated
ConvNets to the position of being the fundamental building
block in a visual recognition system.

Around the same time, the odyssey of neural network
design for natural language processing (NLP) took a very
different path, as the Transformers replaced recurrent neural
networks to become the dominant backbone architecture.
Despite the disparity in the task of interest between language
and vision domains, the two streams surprisingly converged
in the year 2020, as the introduction of Vision Transformers
(ViT) completely altered the landscape of network architec-
ture design. Except for the initial “patchify” layer, which
splits an image into a sequence of patches, ViT introduces no
image-speciﬁc inductive bias and makes minimal changes
to the original NLP Transformers. One primary focus of
ViT is on the scaling behavior: with the help of larger model
and dataset sizes, Transformers can outperform standard
ResNets by a signiﬁcant margin. Those results on image
classiﬁcation tasks are inspiring, but computer vision is not
limited to image classiﬁcation. As discussed previously,
solutions to numerous computer vision tasks in the past
decade depended signiﬁcantly on a sliding-window, fully-
convolutional paradigm. Without the ConvNet inductive
biases, a vanilla ViT model faces many challenges in being
adopted as a generic vision backbone. The biggest chal-
lenge is ViT’s global attention design, which has a quadratic
complexity with respect to the input size. This might be
acceptable for ImageNet classiﬁcation, but quickly becomes
intractable with higher-resolution inputs.

Hierarchical Transformers employ a hybrid approach to
bridge this gap. For example, the “sliding window” strategy
(e.g. attention within local windows) was reintroduced to
Transformers, allowing them to behave more similarly to
ConvNets. Swin Transformer [45] is a milestone work in this
direction, demonstrating for the ﬁrst time that Transformers
can be adopted as a generic vision backbone and achieve
state-of-the-art performance across a range of computer vi-
sion tasks beyond image classiﬁcation. Swin Transformer’s
success and rapid adoption also revealed one thing:
the
essence of convolution is not becoming irrelevant; rather, it
remains much desired and has never faded.

Under this perspective, many of the advancements of
Transformers for computer vision have been aimed at bring-
ing back convolutions. These attempts, however, come
at a cost: a naive implementation of sliding window self-
attention can be expensive [55]; with advanced approaches
such as cyclic shifting [45], the speed can be optimized but
the system becomes more sophisticated in design. On the
other hand, it is almost ironic that a ConvNet already satisﬁes
many of those desired properties, albeit in a straightforward,
no-frills way. The only reason ConvNets appear to be losing
steam is that (hierarchical) Transformers surpass them in
many vision tasks, and the performance difference is usually

attributed to the superior scaling behavior of Transformers,
with multi-head self-attention being the key component.

Unlike ConvNets, which have progressively improved
over the last decade, the adoption of Vision Transformers
was a step change. In recent literature, system-level com-
parisons (e.g. a Swin Transformer vs. a ResNet) are usually
adopted when comparing the two. ConvNets and hierar-
chical vision Transformers become different and similar at
the same time: they are both equipped with similar induc-
tive biases, but differ signiﬁcantly in the training procedure
and macro/micro-level architecture design. In this work,
we investigate the architectural distinctions between Con-
vNets and Transformers and try to identify the confounding
variables when comparing the network performance. Our
research is intended to bridge the gap between the pre-ViT
and post-ViT eras for ConvNets, as well as to test the limits
of what a pure ConvNet can achieve.

To do this, we start with a standard ResNet (e.g. ResNet-
50) trained with an improved procedure. We gradually “mod-
ernize” the architecture to the construction of a hierarchical
vision Transformer (e.g. Swin-T). Our exploration is directed
by a key question: How do design decisions in Transformers
impact ConvNets’ performance? We discover several key
components that contribute to the performance difference
along the way. As a result, we propose a family of pure
ConvNets dubbed ConvNeXt. We evaluate ConvNeXts on a
variety of vision tasks such as ImageNet classiﬁcation [17],
object detection/segmentation on COCO [44], and semantic
segmentation on ADE20K [92]. Surprisingly, ConvNeXts,
constructed entirely from standard ConvNet modules, com-
pete favorably with Transformers in terms of accuracy, scal-
ability and robustness across all major benchmarks. Con-
vNeXt maintains the efﬁciency of standard ConvNets, and
the fully-convolutional nature for both training and testing
makes it extremely simple to implement.

We hope the new observations and discussions can chal-
lenge some common beliefs and encourage people to rethink
the importance of convolutions in computer vision.

2. Modernizing a ConvNet: a Roadmap

In this section, we provide a trajectory going from a
ResNet to a ConvNet that bears a resemblance to Transform-
ers. We consider two model sizes in terms of FLOPs, one is
the ResNet-50 / Swin-T regime with FLOPs around 4.5×109
and the other being ResNet-200 / Swin-B regime which has
FLOPs around 15.0 × 109. For simplicity, we will present
the results with the ResNet-50 / Swin-T complexity models.
The conclusions for higher capacity models are consistent
and results can be found in Appendix C.

At a high level, our explorations are directed to inves-
tigate and follow different levels of designs from a Swin
Transformer while maintaining the network’s simplicity as
a standard ConvNet. The roadmap of our exploration is as

only did vision Transformers bring a new set of modules
and architectural design decisions, but they also introduced
different training techniques (e.g. AdamW optimizer) to vi-
sion. This pertains mostly to the optimization strategy and
associated hyper-parameter settings. Thus, the ﬁrst step
of our exploration is to train a baseline model with the vi-
sion Transformer training procedure, in this case, ResNet-
50/200. Recent studies [7, 81] demonstrate that a set of
modern training techniques can signiﬁcantly enhance the
performance of a simple ResNet-50 model. In our study,
we use a training recipe that is close to DeiT’s [73] and
Swin Transformer’s [45]. The training is extended to 300
epochs from the original 90 epochs for ResNets. We use the
AdamW optimizer [46], data augmentation techniques such
as Mixup [90], Cutmix [89], RandAugment [14], Random
Erasing [91], and regularization schemes including Stochas-
tic Depth [36] and Label Smoothing [69]. The complete set
of hyper-parameters we use can be found in Appendix A.1.
By itself, this enhanced training recipe increased the perfor-
mance of the ResNet-50 model from 76.1% [1] to 78.8%
(+2.7%), implying that a signiﬁcant portion of the perfor-
mance difference between traditional ConvNets and vision
Transformers may be due to the training techniques. We will
use this ﬁxed training recipe with the same hyperparameters
throughout the “modernization” process. Each reported ac-
curacy on the ResNet-50 regime is an average obtained from
training with three different random seeds.

2.2. Macro Design

We now analyze Swin Transformers’ macro network de-
sign. Swin Transformers follow ConvNets [28, 65] to use a
multi-stage design, where each stage has a different feature
map resolution. There are two interesting design considera-
tions: the stage compute ratio, and the “stem cell” structure.

Changing stage compute ratio. The original design of the
computation distribution across stages in ResNet was largely
empirical. The heavy “res4” stage was meant to be compat-
ible with downstream tasks like object detection, where a
detector head operates on the 14×14 feature plane. Swin-T,
on the other hand, followed the same principle but with a
slightly different stage compute ratio of 1:1:3:1. For larger
Swin Transformers, the ratio is 1:1:9:1. Following the de-
sign, we adjust the number of blocks in each stage from
(3, 4, 6, 3) in ResNet-50 to (3, 3, 9, 3), which also aligns
the FLOPs with Swin-T. This improves the model accuracy
from 78.8% to 79.4%. Notably, researchers have thoroughly
investigated the distribution of computation [53, 54], and a
more optimal design is likely to exist.

From now on, we will use this stage compute ratio.

Changing stem to “Patchify”. Typically, the stem cell de-
sign is concerned with how the input images will be pro-
cessed at the network’s beginning. Due to the redundancy

Figure 2. We modernize a standard ConvNet (ResNet) towards
the design of a hierarchical vision Transformer (Swin), without
introducing any attention-based modules. The foreground bars are
model accuracies in the ResNet-50/Swin-T FLOP regime; results
for the ResNet-200/Swin-B regime are shown with the gray bars. A
hatched bar means the modiﬁcation is not adopted. Detailed results
for both regimes are in the appendix. Many Transformer archi-
tectural choices can be incorporated in a ConvNet, and they lead
to increasingly better performance. In the end, our pure ConvNet
model, named ConvNeXt, can outperform the Swin Transformer.

follows. Our starting point is a ResNet-50 model. We ﬁrst
train it with similar training techniques used to train vision
Transformers and obtain much improved results compared to
the original ResNet-50. This will be our baseline. We then
study a series of design decisions which we summarized
as 1) macro design, 2) ResNeXt, 3) inverted bottleneck, 4)
large kernel size, and 5) various layer-wise micro designs. In
Figure 2, we show the procedure and the results we are able
to achieve with each step of the “network modernization”.
Since network complexity is closely correlated with the ﬁ-
nal performance, the FLOPs are roughly controlled over the
course of the exploration, though at intermediate steps the
FLOPs might be higher or lower than the reference models.
All models are trained and evaluated on ImageNet-1K.

2.1. Training Techniques

Apart from the design of the network architecture, the
training procedure also affects the ultimate performance. Not

ResNet-50/200Swin-T/Bstageratio“patchify”stemdepthconvwidth↑invertingdimsmove↑d. convkernelsz.→5kernelsz.→7kernelsz.→9kernelsz.→11ReLU➝GELUfeweractivationsfewernormsBN➝LNsep.d.s.convConvNeXt-T/BMacroDesignResNeXtInvertedBottleneckLargeKernelMicroDesign…78.979.579.578.580.480.778.979.880.380.680.680.680.881.581.281.681.781.881.3ImageNet Top1 Acc (%)GFLOPs4.14.54.42.45.34.64.14.14.24.24.34.24.24.24.24.54.578.879.479.578.380.580.679.980.480.680.680.580.681.381.481.582.081.3inherent in natural images, a common stem cell will aggres-
sively downsample the input images to an appropriate feature
map size in both standard ConvNets and vision Transformers.
The stem cell in standard ResNet contains a 7×7 convolution
layer with stride 2, followed by a max pool, which results
in a 4× downsampling of the input images. In vision Trans-
formers, a more aggressive “patchify” strategy is used as
the stem cell, which corresponds to a large kernel size (e.g.
kernel size = 14 or 16) and non-overlapping convolution.
Swin Transformer uses a similar “patchify” layer, but with
a smaller patch size of 4 to accommodate the architecture’s
multi-stage design. We replace the ResNet-style stem cell
with a patchify layer implemented using a 4×4, stride 4 con-
volutional layer. The accuracy has changed from 79.4% to
79.5%. This suggests that the stem cell in a ResNet may be
substituted with a simpler “patchify” layer à la ViT which
will result in similar performance.

We will use the “patchify stem” (4×4 non-overlapping

convolution) in the network.

2.3. ResNeXt-ify

In this part, we attempt to adopt the idea of ResNeXt [87],
which has a better FLOPs/accuracy trade-off than a vanilla
ResNet. The core component is grouped convolution, where
the convolutional ﬁlters are separated into different groups.
At a high level, ResNeXt’s guiding principle is to “use more
groups, expand width”. More precisely, ResNeXt employs
grouped convolution for the 3×3 conv layer in a bottleneck
block. As this signiﬁcantly reduces the FLOPs, the network
width is expanded to compensate for the capacity loss.

In our case we use depthwise convolution, a special case
of grouped convolution where the number of groups equals
the number of channels. Depthwise conv has been popular-
ized by MobileNet [34] and Xception [11]. We note that
depthwise convolution is similar to the weighted sum op-
eration in self-attention, which operates on a per-channel
basis, i.e., only mixing information in the spatial dimension.
The combination of depthwise conv and 1 × 1 convs leads
to a separation of spatial and channel mixing, a property
shared by vision Transformers, where each operation either
mixes information across spatial or channel dimension, but
not both. The use of depthwise convolution effectively re-
duces the network FLOPs and, as expected, the accuracy.
Following the strategy proposed in ResNeXt, we increase the
network width to the same number of channels as Swin-T’s
(from 64 to 96). This brings the network performance to
80.5% with increased FLOPs (5.3G).

We will now employ the ResNeXt design.

2.4. Inverted Bottleneck

One important design in every Transformer block is that it
creates an inverted bottleneck, i.e., the hidden dimension of
the MLP block is four times wider than the input dimension

Figure 3. Block modiﬁcations and resulted speciﬁcations. (a) is
a ResNeXt block; in (b) we create an inverted bottleneck block and
in (c) the position of the spatial depthwise conv layer is moved up.

(see Figure 4). Interestingly, this Transformer design is con-
nected to the inverted bottleneck design with an expansion
ratio of 4 used in ConvNets. The idea was popularized by
MobileNetV2 [61], and has subsequently gained traction in
several advanced ConvNet architectures [70, 71].

Here we explore the inverted bottleneck design. Figure 3
(a) to (b) illustrate the conﬁgurations. Despite the increased
FLOPs for the depthwise convolution layer, this change
reduces the whole network FLOPs to 4.6G, due to the signif-
icant FLOPs reduction in the downsampling residual blocks’
shortcut 1×1 conv layer. Interestingly, this results in slightly
improved performance (80.5% to 80.6%). In the ResNet-200
/ Swin-B regime, this step brings even more gain (81.9% to
82.6%) also with reduced FLOPs.

We will now use inverted bottlenecks.

2.5. Large Kernel Sizes

In this part of the exploration, we focus on the behav-
ior of large convolutional kernels. One of the most distin-
guishing aspects of vision Transformers is their non-local
self-attention, which enables each layer to have a global
receptive ﬁeld. While large kernel sizes have been used in
the past with ConvNets [40, 68], the gold standard (popular-
ized by VGGNet [65]) is to stack small kernel-sized (3×3)
conv layers, which have efﬁcient hardware implementations
on modern GPUs [41]. Although Swin Transformers rein-
troduced the local window to the self-attention block, the
window size is at least 7×7, signiﬁcantly larger than the
ResNe(X)t kernel size of 3×3. Here we revisit the use of
large kernel-sized convolutions for ConvNets.

Moving up depthwise conv layer. To explore large kernels,
one prerequisite is to move up the position of the depthwise
conv layer (Figure 3 (b) to (c)). That is a design decision
also evident in Transformers: the MSA block is placed prior
to the MLP layers. As we have an inverted bottleneck block,
this is a natural design choice — the complex/inefﬁcient
modules (MSA, large-kernel conv) will have fewer channels,
while the efﬁcient, dense 1×1 layers will do the heavy lifting.
This intermediate step reduces the FLOPs to 4.1G, resulting
in a temporary performance degradation to 79.9%.

Increasing the kernel size. With all of these preparations,
the beneﬁt of adopting larger kernel-sized convolutions is sig-

(a)d3×3,96➝961×1,384➝961×1,96➝384d3×3,384➝3841×1,96➝3841×1,384➝961×1,96➝384d3×3,96➝961×1,384➝96(b)(c)niﬁcant. We experimented with several kernel sizes, includ-
ing 3, 5, 7, 9, and 11. The network’s performance increases
from 79.9% (3×3) to 80.6% (7×7), while the network’s
FLOPs stay roughly the same. Additionally, we observe that
the beneﬁt of larger kernel sizes reaches a saturation point at
7×7. We veriﬁed this behavior in the large capacity model
too: a ResNet-200 regime model does not exhibit further
gain when we increase the kernel size beyond 7×7.
We will use 7×7 depthwise conv in each block.
At this point, we have concluded our examination of
network architectures on a macro scale. Intriguingly, a sig-
niﬁcant portion of the design choices taken in a vision Trans-
former may be mapped to ConvNet instantiations.

2.6. Micro Design

In this section, we investigate several other architectural
differences at a micro scale — most of the explorations here
are done at the layer level, focusing on speciﬁc choices of
activation functions and normalization layers.

Replacing ReLU with GELU One discrepancy between
NLP and vision architectures is the speciﬁcs of which ac-
tivation functions to use. Numerous activation functions
have been developed over time, but the Rectiﬁed Linear Unit
(ReLU) [49] is still extensively used in ConvNets due to its
simplicity and efﬁciency. ReLU is also used as an activation
function in the original Transformer paper [77]. The Gaus-
sian Error Linear Unit, or GELU [32], which can be thought
of as a smoother variant of ReLU, is utilized in the most
advanced Transformers, including Google’s BERT [18] and
OpenAI’s GPT-2 [52], and, most recently, ViTs. We ﬁnd
that ReLU can be substituted with GELU in our ConvNet
too, although the accuracy stays unchanged (80.6%).

Fewer activation functions. One minor distinction be-
tween a Transformer and a ResNet block is that Transform-
ers have fewer activation functions. Consider a Transformer
block with key/query/value linear embedding layers, the pro-
jection layer, and two linear layers in an MLP block. There
is only one activation function present in the MLP block. In
comparison, it is common practice to append an activation
function to each convolutional layer, including the 1 × 1
convs. Here we examine how performance changes when
we stick to the same strategy. As depicted in Figure 4, we
eliminate all GELU layers from the residual block except
for one between two 1 × 1 layers, replicating the style of a
Transformer block. This process improves the result by 0.7%
to 81.3%, practically matching the performance of Swin-T.
We will now use a single GELU activation in each block.

Fewer normalization layers. Transformer blocks usually
have fewer normalization layers as well. Here we remove
two BatchNorm (BN) layers, leaving only one BN layer
before the conv 1 × 1 layers. This further boosts the perfor-
mance to 81.4%, already surpassing Swin-T’s result. Note

Figure 4. Block designs for a ResNet, a Swin Transformer, and a
ConvNeXt. Swin Transformer’s block is more sophisticated due to
the presence of multiple specialized modules and two residual con-
nections. For simplicity, we note the linear layers in Transformer
MLP blocks also as “1×1 convs” since they are equivalent.

that we have even fewer normalization layers per block than
Transformers, as empirically we ﬁnd that adding one ad-
ditional BN layer at the beginning of the block does not
improve the performance.

Substituting BN with LN. BatchNorm [38] is an essen-
tial component in ConvNets as it improves the convergence
and reduces overﬁtting. However, BN also has many in-
tricacies that can have a detrimental effect on the model’s
performance [84]. There have been numerous attempts at
developing alternative normalization [60, 75, 83] techniques,
but BN has remained the preferred option in most vision
tasks. On the other hand, the simpler Layer Normaliza-
tion [5] (LN) has been used in Transformers, resulting in
good performance across different application scenarios.

Directly substituting LN for BN in the original ResNet
will result in suboptimal performance [83]. With all the mod-
iﬁcations in network architecture and training techniques,
here we revisit the impact of using LN in place of BN. We
observe that our ConvNet model does not have any difﬁcul-
ties training with LN; in fact, the performance is slightly
better, obtaining an accuracy of 81.5%.

From now on, we will use one LayerNorm as our choice

of normalization in each residual block.

Separate downsampling layers. In ResNet, the spatial
downsampling is achieved by the residual block at the start of

BN, ReLUBN, ReLUBNReLU256-d1×1, 643×3, 641×1, 256d7×7, 961×1, 3841×1, 9696-dLNGELU96-d1×1, 96×3MSA, w7×7, H=31×1, 961×1, 3841×1, 96GELULN96-d+ rel. pos.win. shiftLNResNet BlockConvNeXt BlockSwin Transformer Blockeach stage, using 3×3 conv with stride 2 (and 1×1 conv with
stride 2 at the shortcut connection). In Swin Transformers, a
separate downsampling layer is added between stages. We
explore a similar strategy in which we use 2×2 conv layers
with stride 2 for spatial downsampling. This modiﬁcation
surprisingly leads to diverged training. Further investigation
shows that, adding normalization layers wherever spatial
resolution is changed can help stablize training. These in-
clude several LN layers also used in Swin Transformers: one
before each downsampling layer, one after the stem, and one
after the ﬁnal global average pooling. We can improve the
accuracy to 82.0%, signiﬁcantly exceeding Swin-T’s 81.3%.
We will use separate downsampling layers. This brings

us to our ﬁnal model, which we have dubbed ConvNeXt.

A comparison of ResNet, Swin, and ConvNeXt block struc-
tures can be found in Figure 4. A comparison of ResNet-50,
Swin-T and ConvNeXt-T’s detailed architecture speciﬁca-
tions can be found in Table 9.

Closing remarks. We have ﬁnished our ﬁrst “playthrough”
and discovered ConvNeXt, a pure ConvNet, that can outper-
form the Swin Transformer for ImageNet-1K classiﬁcation
in this compute regime. It is worth noting that all design
choices discussed so far are adapted from vision Transform-
ers. In addition, these designs are not novel even in the
ConvNet literature — they have all been researched sepa-
rately, but not collectively, over the last decade. Our Con-
vNeXt model has approximately the same FLOPs, #params.,
throughput, and memory use as the Swin Transformer, but
does not require specialized modules such as shifted window
attention or relative position biases.

These ﬁndings are encouraging but not yet completely
convincing — our exploration thus far has been limited to
a small scale, but vision Transformers’ scaling behavior is
what truly distinguishes them. Additionally, the question of
whether a ConvNet can compete with Swin Transformers
on downstream tasks such as object detection and semantic
segmentation is a central concern for computer vision practi-
tioners. In the next section, we will scale up our ConvNeXt
models both in terms of data and model size, and evaluate
them on a diverse set of visual recognition tasks.

3. Empirical Evaluations on ImageNet

We construct different ConvNeXt variants, ConvNeXt-
T/S/B/L, to be of similar complexities to Swin-T/S/B/L [45].
ConvNeXt-T/B is the end product of the “modernizing” pro-
cedure on ResNet-50/200 regime, respectively. In addition,
we build a larger ConvNeXt-XL to further test the scalabil-
ity of ConvNeXt. The variants only differ in the number
of channels C, and the number of blocks B in each stage.
Following both ResNets and Swin Transformers, the number
of channels doubles at each new stage. We summarize the
conﬁgurations below:

• ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
• ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
• ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
• ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
• ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

3.1. Settings

The ImageNet-1K dataset consists of 1000 object classes
with 1.2M training images. We report ImageNet-1K top-1
accuracy on the validation set. We also conduct pre-training
on ImageNet-22K, a larger dataset of 21841 classes (a super-
set of the 1000 ImageNet-1K classes) with ∼14M images
for pre-training, and then ﬁne-tune the pre-trained model on
ImageNet-1K for evaluation. We summarize our training
setups below. More details can be found in Appendix A.

Training on ImageNet-1K. We train ConvNeXts for 300
epochs using AdamW [46] with a learning rate of 4e-3.
There is a 20-epoch linear warmup and a cosine decaying
schedule afterward. We use a batch size of 4096 and a
weight decay of 0.05. For data augmentations, we adopt
common schemes including Mixup [90], Cutmix [89], Ran-
dAugment [14], and Random Erasing [91]. We regularize
the networks with Stochastic Depth [37] and Label Smooth-
ing [69]. Layer Scale [74] of initial value 1e-6 is applied.
We use Exponential Moving Average (EMA) [51] as we ﬁnd
it alleviates larger models’ overﬁtting.

Pre-training on ImageNet-22K. We pre-train ConvNeXts
on ImageNet-22K for 90 epochs with a warmup of 5 epochs.
We do not use EMA. Other settings follow ImageNet-1K.

Fine-tuning on ImageNet-1K. We ﬁne-tune ImageNet-
22K pre-trained models on ImageNet-1K for 30 epochs. We
use AdamW, a learning rate of 5e-5, cosine learning rate
schedule, layer-wise learning rate decay [6, 12], no warmup,
a batch size of 512, and weight decay of 1e-8. The default
pre-training, ﬁne-tuning, and testing resolution is 2242. Ad-
ditionally, we ﬁne-tune at a larger resolution of 3842, for
both ImageNet-22K and ImageNet-1K pre-trained models.
Compared with ViTs/Swin Transformers, ConvNeXts are
simpler to ﬁne-tune at different resolutions, as the network
is fully-convolutional and there is no need to adjust the input
patch size or interpolate absolute/relative position biases.

3.2. Results

ImageNet-1K. Table 1 (upper) shows the result compari-
son with two recent Transformer variants, DeiT [73] and
Swin Transformers [45], as well as two ConvNets from
architecture search - RegNets [54], EfﬁcientNets [71] and
EfﬁcientNetsV2 [72]. ConvNeXt competes favorably with
two strong ConvNet baselines (RegNet [54] and Efﬁcient-
Net [71]) in terms of the accuracy-computation trade-off, as
well as the inference throughputs. ConvNeXt also outper-
forms Swin Transformer of similar complexities across the

model

image
size

#param. FLOPs

throughput
(image / s)

IN-1K
top-1 acc.

ImageNet-1K trained models
• RegNetY-16G [54] 2242
84M 16.0G
6002
• EffNet-B7 [71]
66M 37.0G
4802 120M 53.0G
• EffNetV2-L [72]
2242
◦ DeiT-S [73]
22M 4.6G
2242
◦ DeiT-B [73]
87M 17.6G
2242
◦ Swin-T
28M 4.5G
2242
• ConvNeXt-T
29M 4.5G
2242
◦ Swin-S
50M 8.7G
2242
• ConvNeXt-S
50M 8.7G
2242
◦ Swin-B
88M 15.4G
2242
• ConvNeXt-B
89M 15.4G
3842
◦ Swin-B
88M 47.1G
3842
• ConvNeXt-B
89M 45.0G
2242 198M 34.4G
• ConvNeXt-L
3842 198M 101.0G
• ConvNeXt-L
ImageNet-22K pre-trained models
3842 388M 204.6G
• R-101x3 [39]
4802 937M 840.5G
• R-152x4 [39]
4802 120M 53.0G
• EffNetV2-L [72]
• EffNetV2-XL [72] 4802 208M 94.0G
◦ ViT-B/16 ((cid:84)) [67] 3842
87M 55.5G
◦ ViT-L/16 ((cid:84)) [67] 3842 305M 191.1G
2242
• ConvNeXt-T
29M 4.5G
3842
• ConvNeXt-T
29M 13.1G
2242
• ConvNeXt-S
50M 8.7G
3842
• ConvNeXt-S
50M 25.5G
2242
◦ Swin-B
88M 15.4G
2242
• ConvNeXt-B
89M 15.4G
3842
◦ Swin-B
88M 47.0G
3842
• ConvNeXt-B
89M 45.1G
2242 197M 34.5G
◦ Swin-L
2242 198M 34.4G
• ConvNeXt-L
3842 197M 103.9G
◦ Swin-L
3842 198M 101.0G
• ConvNeXt-L
2242 350M 60.9G
• ConvNeXt-XL
3842 350M 179.0G
• ConvNeXt-XL

334.7
55.1
83.7
978.5
302.1
757.9
774.7
436.7
447.1
286.6
292.1
85.1
95.7
146.8
50.4

-
-
83.7
56.5
93.1
28.5
774.7
282.8
447.1
163.5
286.6
292.1
85.1
95.7
145.0
146.8
46.0
50.4
89.3
30.2

82.9
84.3
85.7
79.8
81.8
81.3
82.1
83.0
83.1
83.5
83.8
84.5
85.1
84.3
85.5

84.4
85.4
86.8
87.3
85.4
86.8
82.9
84.1
84.6
85.8
85.2
85.8
86.4
86.8
86.3
86.6
87.3
87.5
87.0
87.8

Table 1. Classiﬁcation accuracy on ImageNet-1K. Similar to
Transformers, ConvNeXt also shows promising scaling behavior
with higher-capacity models and a larger (pre-training) dataset. In-
ference throughput is measured on a V100 GPU, following [45]. On
an A100 GPU, ConvNeXt can have a much higher throughput than
Swin Transformer. See Appendix E. ((cid:84))ViT results with 90-epoch
AugReg [67] training, provided through personal communication
with the authors.

board, sometimes with a substantial margin (e.g. 0.8% for
ConvNeXt-T). Without specialized modules such as shifted
windows or relative position bias, ConvNeXts also enjoy
improved throughput compared to Swin Transformers.

A highlight from the results is ConvNeXt-B at 3842: it
outperforms Swin-B by 0.6% (85.1% vs. 84.5%), but with
12.5% higher inference throughput (95.7 vs. 85.1 image/s).
We note that the FLOPs/throughput advantage of ConvNeXt-

B over Swin-B becomes larger when the resolution increases
from 2242 to 3842. Additionally, we observe an improved
result of 85.5% when further scaling to ConvNeXt-L.

ImageNet-22K. We present results with models ﬁne-tuned
from ImageNet-22K pre-training at Table 1 (lower). These
experiments are important since a widely held view is that
vision Transformers have fewer inductive biases thus can per-
form better than ConvNets when pre-trained on a larger scale.
Our results demonstrate that properly designed ConvNets
are not inferior to vision Transformers when pre-trained
with large dataset — ConvNeXts still perform on par or
better than similarly-sized Swin Transformers, with slightly
higher throughput. Additionally, our ConvNeXt-XL model
achieves an accuracy of 87.8% — a decent improvement
over ConvNeXt-L at 3842, demonstrating that ConvNeXts
are scalable architectures.

On ImageNet-1K, EfﬁcientNetV2-L, a searched architec-
ture equipped with advanced modules (such as Squeeze-and-
Excitation [35]) and progressive training procedure achieves
top performance. However, with ImageNet-22K pre-training,
ConvNeXt is able to outperform EfﬁcientNetV2, further
demonstrating the importance of large-scale training.

In Appendix B, we discuss robustness and out-of-domain

generalization results for ConvNeXt.

3.3. Isotropic ConvNeXt vs. ViT

In this ablation, we examine if our ConvNeXt block de-
sign is generalizable to ViT-style [20] isotropic architec-
tures which have no downsampling layers and keep the
same feature resolutions (e.g. 14×14) at all depths. We
construct isotropic ConvNeXt-S/B/L using the same feature
dimensions as ViT-S/B/L (384/768/1024). Depths are set
at 18/18/36 to match the number of parameters and FLOPs.
The block structure remains the same (Fig. 4). We use the
supervised training results from DeiT [73] for ViT-S/B and
MAE [26] for ViT-L, as they employ improved training
procedures over the original ViTs [20]. ConvNeXt models
are trained with the same settings as before, but with longer
warmup epochs. Results for ImageNet-1K at 2242 resolution
are in Table 2. We observe ConvNeXt can perform generally
on par with ViT, showing that our ConvNeXt block design
is competitive when used in non-hierarchical models.

model

#param. FLOPs

◦ ViT-S
22M 4.6G
• ConvNeXt-S (iso.)
22M 4.3G
◦ ViT-B
87M 17.6G
• ConvNeXt-B (iso.)
87M 16.9G
◦ ViT-L
304M 61.6G
• ConvNeXt-L (iso.) 306M 59.7G

throughput
(image / s)
978.5
1038.7
302.1
320.1
93.1
94.4

training
mem. (GB)
4.9
4.2
9.1
7.7
22.5
20.4

IN-1K
acc.
79.8
79.7
81.8
82.0
82.6
82.6

Table 2. Comparing isotropic ConvNeXt and ViT. Training
memory is measured on V100 GPUs with 32 per-GPU batch size.

backbone

FLOPs FPS APbox APbox

50 APbox

75 APmask APmask

50 APmask
75

Mask-RCNN 3× schedule

◦ Swin-T
• ConvNeXt-T

267G 23.1 46.0
262G 25.6 46.2

68.1
67.9

50.3
50.8

Cascade Mask-RCNN 3× schedule

• ResNet-50
739G 16.2 46.3
• X101-32
819G 13.8 48.1
• X101-64
972G 12.6 48.3
◦ Swin-T
745G 12.2 50.4
• ConvNeXt-T
741G 13.5 50.4
◦ Swin-S
838G 11.4 51.9
• ConvNeXt-S
827G 12.0 51.9
◦ Swin-B
982G 10.7 51.9
• ConvNeXt-B
964G 11.4 52.7
◦ Swin-B‡
982G 10.7 53.0
• ConvNeXt-B‡
964G 11.5 54.0
◦ Swin-L‡
1382G 9.2 53.9
• ConvNeXt-L‡
1354G 10.0 54.8
• ConvNeXt-XL‡ 1898G 8.6 55.2

64.3
66.5
66.4
69.2
69.1
70.7
70.8
70.5
71.3
71.8
73.1
72.4
73.8
74.2

50.5
52.4
52.3
54.7
54.8
56.3
56.5
56.4
57.2
57.5
58.8
58.8
59.8
59.9

41.6
41.7

40.1
41.6
41.7
43.7
43.7
45.0
45.0
45.0
45.6
45.8
46.9
46.7
47.6
47.7

65.1
65.0

61.7
63.9
64.0
66.6
66.5
68.2
68.4
68.1
68.9
69.4
70.6
70.1
71.3
71.6

44.9
44.9

43.4
45.2
45.1
47.3
47.3
48.8
49.1
48.9
49.5
49.7
51.3
50.8
51.7
52.2

Table 3. COCO object detection and segmentation results using
Mask-RCNN and Cascade Mask-RCNN. ‡ indicates that the model
is pre-trained on ImageNet-22K. ImageNet-1K pre-trained Swin
results are from their Github repository [3]. AP numbers of the
ResNet-50 and X101 models are from [45]. We measure FPS on
an A100 GPU. FLOPs are calculated with image size (1280, 800).

4. Empirical Evaluation on Downstream Tasks

Object detection and segmentation on COCO. We ﬁne-
tune Mask R-CNN [27] and Cascade Mask R-CNN [9] on
the COCO dataset with ConvNeXt backbones. Following
Swin Transformer [45], we use multi-scale training, AdamW
optimizer, and a 3× schedule. Further details and hyper-
parameter settings can be found in Appendix A.3.

Table 3 shows object detection and instance segmentation
results comparing Swin Transformer, ConvNeXt, and tradi-
tional ConvNet such as ResNeXt. Across different model
complexities, ConvNeXt achieves on-par or better perfor-
mance than Swin Transformer. When scaled up to bigger
models (ConvNeXt-B/L/XL) pre-trained on ImageNet-22K,
in many cases ConvNeXt is signiﬁcantly better (e.g. +1.0 AP)
than Swin Transformers in terms of box and mask AP.

Semantic segmentation on ADE20K. We also evaluate
ConvNeXt backbones on the ADE20K semantic segmen-
tation task with UperNet [85]. All model variants are trained
for 160K iterations with a batch size of 16. Other experimen-
tal settings follow [6] (see Appendix A.3 for more details).
In Table 4, we report validation mIoU with multi-scale test-
ing. ConvNeXt models can achieve competitive performance
across different model capacities, further validating the ef-
fectiveness of our architecture design.

backbone

input crop. mIoU #param. FLOPs

◦ Swin-T
• ConvNeXt-T
◦ Swin-S
• ConvNeXt-S
◦ Swin-B
• ConvNeXt-B

◦ Swin-B‡
• ConvNeXt-B‡
◦ Swin-L‡
• ConvNeXt-L‡
• ConvNeXt-XL‡

ImageNet-1K pre-trained
45.8
46.7
49.5
49.6
49.7
49.9

5122
5122
5122
5122
5122
5122
ImageNet-22K pre-trained
6402
6402
6402
6402
6402

51.7
53.1
53.5
53.7
54.0

945G
60M
60M
939G
81M 1038G
82M 1027G
121M 1188G
122M 1170G

121M 1841G
122M 1828G
234M 2468G
235M 2458G
391M 3335G

Table 4. ADE20K validation results using UperNet [85]. ‡ in-
dicates IN-22K pre-training. Swins’ results are from its GitHub
repository [2]. Following Swin, we report mIoU results with multi-
scale testing. FLOPs are based on input sizes of (2048, 512) and
(2560, 640) for IN-1K and IN-22K pre-trained models, respectively.

It is natural to ask whether the design of
convolutions.
ConvNeXt will render it practically inefﬁcient. As demon-
strated throughout the paper, the inference throughputs of
ConvNeXts are comparable to or exceed that of Swin Trans-
formers. This is true for both classiﬁcation and other tasks
requiring higher-resolution inputs (see Table 1,3 for com-
parisons of throughput/FPS). Furthermore, we notice that
training ConvNeXts requires less memory than training Swin
Transformers. For example, training Cascade Mask-RCNN
using ConvNeXt-B backbone consumes 17.4GB of peak
memory with a per-GPU batch size of 2, while the reference
number for Swin-B is 18.5GB. In comparison to vanilla ViT,
both ConvNeXt and Swin Transformer exhibit a more favor-
able accuracy-FLOPs trade-off due to the local computations.
It is worth noting that this improved efﬁciency is a result of
the ConvNet inductive bias, and is not directly related to the
self-attention mechanism in vision Transformers.

5. Related Work

Hybrid models. In both the pre- and post-ViT eras, the
hybrid model combining convolutions and self-attentions
has been actively studied. Prior to ViT, the focus was
on augmenting a ConvNet with self-attention/non-local
modules [8, 55, 66, 79] to capture long-range dependen-
cies. The original ViT [20] ﬁrst studied a hybrid conﬁg-
uration, and a large body of follow-up works focused on
reintroducing convolutional priors to ViT, either in an ex-
plicit [15, 16, 21, 82, 86, 88] or implicit [45] fashion.

Remarks on model efﬁciency. Under similar FLOPs, mod-
els with depthwise convolutions are known to be slower
and consume more memory than ConvNets with only dense

Recent convolution-based approaches. Han et al. [25]
show that local Transformer attention is equivalent to in-
homogeneous dynamic depthwise conv. The MSA block in

Swin is then replaced with a dynamic or regular depthwise
convolution, achieving comparable performance to Swin.
A concurrent work ConvMixer [4] demonstrates that, in
small-scale settings, depthwise convolution can be used as a
promising mixing strategy. ConvMixer uses a smaller patch
size to achieve the best results, making the throughput much
lower than other baselines. GFNet [56] adopts Fast Fourier
Transform (FFT) for token mixing. FFT is also a form of con-
volution, but with a global kernel size and circular padding.
Unlike many recent Transformer or ConvNet designs, one
primary goal of our study is to provide an in-depth look at
the process of modernizing a standard ResNet and achieving
state-of-the-art performance.

6. Conclusions

In the 2020s, vision Transformers, particularly hierar-
chical ones such as Swin Transformers, began to overtake
ConvNets as the favored choice for generic vision backbones.
The widely held belief is that vision Transformers are more
accurate, efﬁcient, and scalable than ConvNets. We propose
ConvNeXts, a pure ConvNet model that can compete favor-
ably with state-of-the-art hierarchical vision Transformers
across multiple computer vision benchmarks, while retaining
the simplicity and efﬁciency of standard ConvNets. In some
ways, our observations are surprising while our ConvNeXt
model itself is not completely new — many design choices
have all been examined separately over the last decade, but
not collectively. We hope that the new results reported in this
study will challenge several widely held views and prompt
people to rethink the importance of convolution in computer
vision.

Acknowledgments. We thank Kaiming He, Eric Mintun,
Xingyi Zhou, Ross Girshick, and Yann LeCun for valuable
discussions and feedback.

Appendix

In this Appendix, we provide further experimental details
(§A), robustness evaluation results (§B), more modernization
experiment results (§C), and a detailed network speciﬁcation
(§D). We further benchmark model throughput on A100
GPUs (§E). Finally, we discuss the limitations (§F) and
societal impact (§G) of our work.

A. Experimental Settings

A.1. ImageNet (Pre-)training

We provide ConvNeXts’ ImageNet-1K training and
ImageNet-22K pre-training settings in Table 5. The settings
are used for our main results in Table 1 (Section 3.2). All
ConvNeXt variants use the same setting, except the stochas-
tic depth rate is customized for model variants.

For experiments in “modernizing a ConvNet” (Section 2),
we also use Table 5’s setting for ImageNet-1K, except EMA
is disabled, as we ﬁnd using EMA severely hurts models
with BatchNorm layers.

For isotropic ConvNeXts (Section 3.3), the setting for
ImageNet-1K in Table A is also adopted, but warmup is ex-
tended to 50 epochs, and layer scale is disabled for isotropic
ConvNeXt-S/B. The stochastic depth rates are 0.1/0.2/0.5
for isotropic ConvNeXt-S/B/L.

(pre-)training conﬁg

weight init
optimizer
base learning rate
weight decay
optimizer momentum
batch size
training epochs
learning rate schedule
warmup epochs
warmup schedule
layer-wise lr decay [6, 12]
randaugment [14]
mixup [90]
cutmix [89]
random erasing [91]
label smoothing [69]
stochastic depth [37]
layer scale [74]
head init scale [74]
gradient clip
exp. mov. avg. (EMA) [51]

ConvNeXt-T/S/B/L ConvNeXt-T/S/B/L/XL

ImageNet-1K
2242
trunc. normal (0.2)
AdamW
4e-3
0.05
β1, β2=0.9, 0.999
4096
300
cosine decay
20
linear
None
(9, 0.5)
0.8
1.0
0.25
0.1
0.1/0.4/0.5/0.5
1e-6
None
None
0.9999

ImageNet-22K
2242
trunc. normal (0.2)
AdamW
4e-3
0.05
β1, β2=0.9, 0.999
4096
90
cosine decay
5
linear
None
(9, 0.5)
0.8
1.0
0.25
0.1
0.0/0.0/0.1/0.1/0.2
1e-6
None
None
None

Table 5. ImageNet-1K/22K (pre-)training settings. Multiple
stochastic depth rates (e.g., 0.1/0.4/0.5/0.5) are for each model
(e.g., ConvNeXt-T/S/B/L) respectively.

A.2. ImageNet Fine-tuning

We list the settings for ﬁne-tuning on ImageNet-1K in
Table 6. The ﬁne-tuning starts from the ﬁnal model weights
obtained in pre-training, without using the EMA weights,
even if in pre-training EMA is used and EMA accuracy is
reported. This is because we do not observe improvement if
we ﬁne-tune with the EMA weights (consistent with observa-
tions in [73]). The only exception is ConvNeXt-L pre-trained
on ImageNet-1K, where the model accuracy is signiﬁcantly
lower than the EMA accuracy due to overﬁtting, and we
select its best EMA model during pre-training as the starting
point for ﬁne-tuning.

In ﬁne-tuning, we use layer-wise learning rate decay [6,
12] with every 3 consecutive blocks forming a group. When
the model is ﬁne-tuned at 3842 resolution, we use a crop ratio
of 1.0 (i.e., no cropping) during testing following [2, 74, 80],
instead of 0.875 at 2242.

pre-training conﬁg

ﬁne-tuning conﬁg

optimizer
base learning rate
weight decay
optimizer momentum
batch size
training epochs
learning rate schedule
layer-wise lr decay
warmup epochs
warmup schedule
randaugment
mixup
cutmix
random erasing
label smoothing
stochastic depth
layer scale
head init scale
gradient clip
exp. mov. avg. (EMA)

ConvNeXt-B/L
ImageNet-1K
2242
ImageNet-1K
3842
AdamW
5e-5
1e-8
β1, β2=0.9, 0.999
512
30
cosine decay
0.7
None
N/A
(9, 0.5)
None
None
0.25
0.1
0.8/0.95
pre-trained
0.001
None
None

ConvNeXt-T/S/B/L/XL
ImageNet-22K
2242
ImageNet-1K
2242 and 3842
AdamW
5e-5
1e-8
β1, β2=0.9, 0.999
512
30
cosine decay
0.8
None
N/A
(9, 0.5)
None
None
0.25
0.1
0.0/0.1/0.2/0.3/0.4
pre-trained
0.001
None
None(T-L)/0.9999(XL)

Table 6. ImageNet-1K ﬁne-tuning settings. Multiple values (e.g.,
0.8/0.95) are for each model (e.g., ConvNeXt-B/L) respectively.

A.3. Downstream Tasks

For ADE20K and COCO experiments, we follow the
training settings used in BEiT [6] and Swin [45]. We also
use MMDetection [10] and MMSegmentation [13] toolboxes.
We use the ﬁnal model weights (instead of EMA weights)
from ImageNet pre-training as network initializations.

We conduct a lightweight sweep for COCO experiments
including learning rate {1e-4, 2e-4}, layer-wise learning rate
decay [6] {0.7, 0.8, 0.9, 0.95}, and stochastic depth rate
{0.3, 0.4, 0.5, 0.6, 0.7, 0.8}. We ﬁne-tune the ImageNet-22K
pre-trained Swin-B/L on COCO using the same sweep. We
use the ofﬁcial code and pre-trained model weights [3].

The hyperparameters we sweep for ADE20K experiments
include learning rate {8e-5, 1e-4}, layer-wise learning rate
decay {0.8, 0.9}, and stochastic depth rate {0.3, 0.4, 0.5}.
We report validation mIoU results using multi-scale testing.
Additional single-scale testing results are in Table 7.

backbone

input crop. mIoU

ImageNet-1K pre-trained

• ConvNeXt-T
• ConvNeXt-S
• ConvNeXt-B

5122
5122
5122

ImageNet-22K pre-trained

• ConvNeXt-B‡
• ConvNeXt-L‡
• ConvNeXt-XL‡

6402
6402
6402

46.0
48.7
49.1

52.6
53.2
53.6

Table 7. ADE20K validation results with single-scale testing.

B. Robustness Evaluation

Additional robustness evaluation results for ConvNeXt
models are presented in Table 8. We directly test our
ImageNet-1K trained/ﬁne-tuned classiﬁcation models on sev-
eral robustness benchmark datasets such as ImageNet-A [33],
ImageNet-R [30], ImageNet-Sketch [78] and ImageNet-
C/¯C [31, 48] datasets. We report mean corruption error
(mCE) for ImageNet-C, corruption error for ImageNet-¯C,
and top-1 Accuracy for all other datasets.

ConvNeXt (in particular the large-scale model variants)
exhibits promising robustness behaviors, outperforming
state-of-the-art robust transformer models [47] on several
benchmarks. With extra ImageNet-22K data, ConvNeXt-
XL demonstrates strong domain generalization capabilities
(e.g. achieving 69.3%/68.2%/55.0% accuracy on ImageNet-
A/R/Sketch benchmarks, respectively). We note that these ro-
bustness evaluation results were acquired without using any
specialized modules or additional ﬁne-tuning procedures.

Data/Size FLOPs / Params Clean C (↓) ¯C (↓) A

R

SK

4.1 / 25.6

76.1

76.7

57.7

0.0 36.1 24.1

Model

ResNet-50

Swin-T [45]
RVT-S* [47]
ConvNeXt-T
Swin-B [45]
RVT-B* [47]
ConvNeXt-B

1K/2242

1K/2242
1K/2242
1K/2242
1K/2242
1K/2242
1K/2242

4.5 / 28.3
4.7 / 23.3
4.5 / 28.6
15.4 / 87.8
17.7 / 91.8
15.4 / 88.6

22K/3842
ConvNeXt-B
22K/3842
ConvNeXt-L
ConvNeXt-XL 22K/3842

45.1 / 88.6
101.0 / 197.8
179.0 / 350.2

81.2
81.9
82.1
83.4
82.6
83.8

86.8
87.5
87.8

62.0
49.4
53.2
54.4
46.8
46.8

43.1
40.2
38.8

-

21.6 41.3 29.1
37.5 25.7 47.7 34.7
40.0 24.2 47.2 33.8
35.8 46.6 32.4
30.8 28.5 48.7 36.0
34.4 36.7 51.3 38.2

-

30.7 62.3 64.9 51.6
29.9 65.5 66.7 52.8
27.1 69.3 68.2 55.0

Table 8. Robustness evaluation of ConvNeXt. We do not make
use of any specialized modules or additional ﬁne-tuning procedures.

C. Modernizing ResNets: detailed results

Here we provide detailed tabulated results for the mod-
ernization experiments, at both ResNet-50 / Swin-T and
ResNet-200 / Swin-B regimes. The ImageNet-1K top-1 ac-
curacies and FLOPs for each step are shown in Table 10
and 11. ResNet-50 regime experiments are run with 3 ran-
dom seeds.

For ResNet-200, the initial number of blocks at each stage
is (3, 24, 36, 3). We change it to Swin-B’s (3, 3, 27, 3) at
the step of changing stage ratio. This drastically reduces the
FLOPs, so at the same time, we also increase the width from
64 to 84 to keep the FLOPs at a similar level. After the step
of adopting depthwise convolutions, we further increase the
width to 128 (same as Swin-B’s) as a separate step.

The observations on the ResNet-200 regime are mostly
consistent with those on ResNet-50 as described in the main
paper. One interesting difference is that inverting dimensions
brings a larger improvement at ResNet-200 regime than at
ResNet-50 regime (+0.79% vs. +0.14%). The performance

output size

stem

56×56

• ResNet-50
7×7, 64, stride 2
3×3 max pool, stride 2

res2

56×56

res3

28×28

res4

14×14

res5

7×7











 × 3

1×1, 64
3×3, 64
1×1, 256



 × 4

1×1, 128
3×3, 128
1×1, 512





1×1, 256
3×3, 256
1×1, 1024





1×1, 512
3×3, 512
1×1, 2048



 × 6



 × 3

• ConvNeXt-T

◦ Swin-T

4×4, 96, stride 4

4×4, 96, stride 4







d7×7, 96
1×1, 384
1×1, 96

 × 3





d7×7, 192
1×1, 768
1×1, 192





d7×7, 384
1×1, 1536
1×1, 384





d7×7, 768
1×1, 3072
1×1, 768



 × 3



 × 9



 × 3

























(cid:21)

(cid:21)

1×1, 96×3
MSA, w7×7, H=3, rel. pos.
1×1, 96
(cid:20) 1×1, 384
1×1, 96
1×1, 192×3
MSA, w7×7, H=6, rel. pos.
1×1, 192
(cid:20) 1×1, 768
1×1, 192
1×1, 384×3
MSA, w7×7, H=12, rel. pos.
1×1, 384
(cid:20) 1×1, 1536
1×1, 384
1×1, 768×3
MSA, w7×7, H=24, rel. pos.
1×1, 768
(cid:20) 1×1, 3072
1×1, 768

(cid:21)

(cid:21)









× 2

× 2

× 6

× 2

FLOPs
# params.

4.1 × 109
25.6 × 106

4.5 × 109
28.6 × 106

4.5 × 109
28.3 × 106

Table 9. Detailed architecture speciﬁcations for ResNet-50, ConvNeXt-T and Swin-T.

model
ResNet-50 (PyTorch [1])
ResNet-50 (enhanced recipe)
stage ratio
“patchify” stem
depthwise conv
increase width
inverting dimensions
move up depthwise conv
kernel size → 5
kernel size → 7
kernel size → 9
kernel size → 11
ReLU → GELU
fewer activations
fewer norms
BN → LN
separate d.s. conv (ConvNeXt-T)
Swin-T [45]

IN-1K acc.
76.13
78.82 ± 0.07
79.36 ± 0.07
79.51 ± 0.18
78.28 ± 0.08
80.50 ± 0.02
80.64 ± 0.03
79.92 ± 0.08
80.35 ± 0.08
80.57 ± 0.14
80.57 ± 0.06
80.47 ± 0.11
80.62 ± 0.14
81.27 ± 0.06
81.41 ± 0.09
81.47 ± 0.09
81.97 ± 0.06
81.30

GFLOPs
4.09
4.09
4.53
4.42
2.35
5.27
4.64
4.07
4.10
4.15
4.21
4.29
4.15
4.15
4.15
4.46
4.49
4.50

Table 10. Detailed results for modernizing a ResNet-50. Mean
and standard deviation are obtained by training the network with
three different random seeds.

model
ResNet-200 [29]
ResNet-200 (enhanced recipe)
stage ratio and increase width
“patchify” stem
depthwise conv
increase width
inverting dimensions
move up depthwise conv
kernel size → 5
kernel size → 7
kernel size → 9
kernel size → 11
ReLU → GELU
fewer activations
fewer norms
BN → LN
separate d.s. conv (ConvNeXt-B)
Swin-B [45]

IN-1K acc.
78.20
81.14
81.33
81.59
80.54
81.85
82.64
82.04
82.32
82.30
82.27
82.18
82.19
82.71
83.17
83.35
83.60
83.50

GFLOPs
15.01
15.01
14.52
14.38
7.23
16.76
15.68
14.63
14.70
14.81
14.95
15.13
14.81
14.81
14.81
14.81
15.35
15.43

Table 11. Detailed results for modernizing a ResNet-200.

D. Detailed Architectures

gained by increasing kernel size also seems to saturate at
kernel size 5 instead of 7. Using fewer normalization layers
also has a bigger gain compared with the ResNet-50 regime
(+0.46% vs. +0.14%).

We present a detailed architecture comparison between
ResNet-50, ConvNeXt-T and Swin-T in Table 9. For differ-
ently sized ConvNeXts, only the number of blocks and the
number of channels at each stage differ from ConvNeXt-T

(see Section 3 for details). ConvNeXts enjoy the simplic-
ity of standard ConvNets, but compete favorably with Swin
Transformers in visual recognition.

E. Benchmarking on A100 GPUs

Following Swin Transformer [45], the ImageNet models’
inference throughputs in Table 1 are benchmarked using a
V100 GPU, where ConvNeXt is slightly faster in inference
than Swin Transformer with a similar number of parameters.
We now benchmark them on the more advanced A100 GPUs,
which support the TensorFloat32 (TF32) tensor cores. We
employ PyTorch [50] version 1.10 to use the latest “Channel
Last” memory layout [22] for further speedup.

We present the results in Table 12. Swin Transformers and
ConvNeXts both achieve faster inference throughput than
V100 GPUs, but ConvNeXts’ advantage is now signiﬁcantly
greater, sometimes up to 49% faster. This preliminary study
shows promising signals that ConvNeXt, employed with
standard ConvNet modules and simple in design, could be
practically more efﬁcient models on modern hardwares.

857.3

model

FLOPs

throughput
(image / s)
1325.6

image
size
2242
2242
2242
2242
2242
2242
3842
3842
2242
2242
3842 103.9G
3842 101.0G 211.4 (+34%)

4.5G
4.5G 1943.5 (+47%)
8.7G
8.7G 1275.3 (+49%)
15.4G
15.4G 969.0 (+46%)
47.1G
45.0G 336.6 (+39%)
34.5G
34.4G 611.5 (+40%)

◦ Swin-T
• ConvNeXt-T
◦ Swin-S
• ConvNeXt-S
◦ Swin-B
• ConvNeXt-B
◦ Swin-B
• ConvNeXt-B
◦ Swin-L
• ConvNeXt-L
◦ Swin-L
• ConvNeXt-L
• ConvNeXt-XL 2242
60.9G
• ConvNeXt-XL 3842 179.0G

424.4
147.4

157.9

662.8

435.9

242.5

IN-1K / 22K
trained, 1K acc.
81.3 / –
82.1 / –
83.0 / –
83.1 / –
83.5 / 85.2
83.8 / 85.8
84.5 / 86.4
85.1 / 86.8
– / 86.3
84.3 / 86.6
– / 87.3
85.5 / 87.5
– / 87.0
– / 87.8

Table 12. Inference throughput comparisons on an A100 GPU.
Using TF32 data format and “channel last” memory layout, Con-
vNeXt enjoys up to ∼49% higher throughput compared with a
Swin Transformer with similar FLOPs.

F. Limitations

We demonstrate ConvNeXt, a pure ConvNet model, can
perform as good as a hierarchical vision Transformer on
image classiﬁcation, object detection, instance and semantic
segmentation tasks. While our goal is to offer a broad range
of evaluation tasks, we recognize computer vision applica-
tions are even more diverse. ConvNeXt may be more suited
for certain tasks, while Transformers may be more ﬂexible
for others. A case in point is multi-modal learning, in which
a cross-attention module may be preferable for modeling

feature interactions across many modalities. Additionally,
Transformers may be more ﬂexible when used for tasks re-
quiring discretized, sparse, or structured outputs. We believe
the architecture choice should meet the needs of the task at
hand while striving for simplicity.

G. Societal Impact

In the 2020s, research on visual representation learn-
ing began to place enormous demands on computing re-
sources. While larger models and datasets improve per-
formance across the board, they also introduce a slew of
challenges. ViT, Swin, and ConvNeXt all perform best with
their huge model variants. Investigating those model designs
inevitably results in an increase in carbon emissions. One
important direction, and a motivation for our paper, is to
strive for simplicity — with more sophisticated modules,
the network’s design space expands enormously, obscuring
critical components that contribute to the performance dif-
ference. Additionally, large models and datasets present
issues in terms of model robustness and fairness. Further
investigation on the robustness behavior of ConvNeXt vs.
Transformer will be an interesting research direction. In
terms of data, our ﬁndings indicate that ConvNeXt models
beneﬁt from pre-training on large-scale datasets. While our
method makes use of the publicly available ImageNet-22K
dataset, individuals may wish to acquire their own data for
pre-training. A more circumspect and responsible approach
to data selection is required to avoid potential concerns with
data biases.

References

[1] PyTorch Vision Models. https://pytorch.org/
vision/stable/models.html. Accessed: 2021-10-
01.

[2] GitHub repository: Swin transformer. https://github.

com/microsoft/Swin-Transformer, 2021.

[3] GitHub repository: Swin transformer for object detection.
https://github.com/SwinTransformer/Swin-
Transformer-Object-Detection, 2021.

[4] Anonymous. Patches are all you need? Openreview, 2021.
[5] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.

Layer normalization. arXiv:1607.06450, 2016.

[6] Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-
training of image transformers. arXiv:2106.08254, 2021.
[7] Irwan Bello, William Fedus, Xianzhi Du, Ekin Dogus Cubuk,
Aravind Srinivas, Tsung-Yi Lin, Jonathon Shlens, and Barret
Zoph. Revisiting resnets: Improved training and scaling
strategies. NeurIPS, 2021.

[8] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens,
and Quoc V Le. Attention augmented convolutional networks.
In ICCV, 2019.

[9] Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: Delv-
ing into high quality object detection. In CVPR, 2018.

[10] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu
Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu,
Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tian-
heng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue
Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang,
Chen Change Loy, and Dahua Lin. MMDetection: Open
mmlab detection toolbox and benchmark. arXiv:1906.07155,
2019.

[11] François Chollet. Xception: Deep learning with depthwise

separable convolutions. In CVPR, 2017.

[12] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christo-
pher D Manning. ELECTRA: Pre-training text encoders as
discriminators rather than generators. In ICLR, 2020.
[13] MMSegmentation contributors. MMSegmentation: Openmm-
lab semantic segmentation toolbox and benchmark. https:
/ / github . com / open - mmlab / mmsegmentation,
2020.

[14] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V
Le. Randaugment: Practical automated data augmentation
with a reduced search space. In CVPR Workshops, 2020.
[15] Zihang Dai, Hanxiao Liu, Quoc V Le, and Mingxing Tan.
Coatnet: Marrying convolution and attention for all data sizes.
NeurIPS, 2021.

[16] Stéphane d’Ascoli, Hugo Touvron, Matthew Leavitt, Ari Mor-
cos, Giulio Biroli, and Levent Sagun. ConViT: Improving
vision transformers with soft convolutional inductive biases.
ICML, 2021.

[17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li
Fei-Fei. ImageNet: A large-scale hierarchical image database.
In CVPR, 2009.

[18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: Pre-training of deep bidirectional trans-
formers for language understanding. In NAACL, 2019.
[19] Piotr Dollár, Serge Belongie, and Pietro Perona. The fastest

pedestrian detector in the west. In BMVC, 2010.

[20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at
scale. In ICLR, 2021.

[21] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li,
Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer.
Multiscale vision transformers. ICCV, 2021.

[22] Vitaly Fedyunin. Tutorial: Channel last memory format
in PyTorch. https://pytorch.org/tutorials/
intermediate/memory_format_tutorial.html,
2021. Accessed: 2021-10-01.

[23] Ross Girshick. Fast R-CNN. In ICCV, 2015.
[24] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra
Malik. Rich feature hierarchies for accurate object detection
and semantic segmentation. In CVPR, 2014.

[25] Qi Han, Zejia Fan, Qi Dai, Lei Sun, Ming-Ming Cheng, Ji-
aying Liu, and Jingdong Wang. Demystifying local vision
transformer: Sparse connectivity, weight sharing, and dy-
namic weight. arXiv:2106.04263, 2021.

[26] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Dollár, and Ross Girshick. Masked autoencoders are scalable
vision learners. arXiv:2111.06377, 2021.

[27] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Gir-

shick. Mask R-CNN. In ICCV, 2017.

[28] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In CVPR, 2016.
[29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Identity mappings in deep residual networks. In ECCV, 2016.
[30] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kada-
vath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu,
Samyak Parajuli, Mike Guo, et al. The many faces of robust-
ness: A critical analysis of out-of-distribution generalization.
In ICCV, 2021.

[31] Dan Hendrycks and Thomas Dietterich. Benchmarking neural
network robustness to common corruptions and perturbations.
In ICLR, 2018.

[32] Dan Hendrycks and Kevin Gimpel. Gaussian error linear

units (gelus). arXiv:1606.08415, 2016.

[33] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt,
and Dawn Song. Natural adversarial examples. In CVPR,
2021.

[34] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry
Kalenichenko, Weijun Wang, Tobias Weyand, Marco An-
dreetto, and Hartwig Adam. MobileNets: Efﬁcient con-
volutional neural networks for mobile vision applications.
arXiv:1704.04861, 2017.

[35] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation

networks. In CVPR, 2018.

[36] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kil-
ian Q Weinberger. Densely connected convolutional networks.
In CVPR, 2017.

[37] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q
Weinberger. Deep networks with stochastic depth. In ECCV,
2016.

[38] Sergey Ioffe. Batch renormalization: Towards reducing mini-
batch dependence in batch-normalized models. In NeurIPS,
2017.

[39] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan
Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.
Big Transfer (BiT): General visual representation learning. In
ECCV, 2020.

[40] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet
classiﬁcation with deep convolutional neural networks. In
NeurIPS, 2012.

[41] Andrew Lavin and Scott Gray. Fast algorithms for convolu-

tional neural networks. In CVPR, 2016.

[42] Yann LeCun, Bernhard Boser, John S Denker, Donnie Hen-
derson, Richard E Howard, Wayne Hubbard, and Lawrence D
Jackel. Backpropagation applied to handwritten zip code
recognition. Neural computation, 1989.

[43] Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner,
et al. Gradient-based learning applied to document recogni-
tion. Proceedings of the IEEE, 1998.

[44] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence
Zitnick. Microsoft COCO: Common objects in context. In
ECCV. 2014.

[45] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng
Zhang, Stephen Lin, and Baining Guo. Swin transformer:
Hierarchical vision transformer using shifted windows. 2021.
[46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay

regularization. In ICLR, 2019.

[47] Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie
Duan, Shaokai Ye, Yuan He, and Hui Xue. Towards robust
vision transformer. arXiv preprint arXiv:2105.07926, 2021.
[48] Eric Mintun, Alexander Kirillov, and Saining Xie. On in-
teraction between augmentations and corruptions in natural
corruption robustness. NeurIPS, 2021.

[49] Vinod Nair and Geoffrey E Hinton. Rectiﬁed linear units
improve restricted boltzmann machines. In ICML, 2010.
[50] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An
imperative style, high-performance deep learning library. In
NeurIPS, 2019.

[51] Boris T Polyak and Anatoli B Juditsky. Acceleration of
stochastic approximation by averaging. SIAM Journal on
Control and Optimization, 1992.

[52] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario
Amodei, and Ilya Sutskever. Language models are unsuper-
vised multitask learners. 2019.

[53] Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen
Lo, and Piotr Dollár. On network design spaces for visual
recognition. In ICCV, 2019.

[54] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaim-
ing He, and Piotr Dollár. Designing network design spaces.
In CVPR, 2020.

[55] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan
Bello, Anselm Levskaya, and Jonathon Shlens. Stand-alone
self-attention in vision models. NeurIPS, 2019.

[56] Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and
Jie Zhou. Global ﬁlter networks for image classiﬁcation.
NeurIPS, 2021.

[57] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
Faster R-CNN: Towards real-time object detection with region
proposal networks. In NeurIPS, 2015.

[58] Henry A Rowley, Shumeet Baluja, and Takeo Kanade. Neural

network-based face detection. TPAMI, 1998.

[59] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-
jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li
Fei-Fei. ImageNet Large Scale Visual Recognition Challenge.
IJCV, 2015.

[60] Tim Salimans and Diederik P Kingma. Weight normalization:
A simple reparameterization to accelerate training of deep
neural networks. In NeurIPS, 2016.

[61] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zh-
moginov, and Liang-Chieh Chen. Mobilenetv2: Inverted
residuals and linear bottlenecks. In CVPR, 2018.

[62] Pierre Sermanet, David Eigen, Xiang Zhang, Michael Math-
ieu, Rob Fergus, and Yann LeCun. Overfeat: Integrated
recognition, localization and detection using convolutional
networks. In ICLR, 2014.

[63] Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and
Yann LeCun. Pedestrian detection with unsupervised multi-
stage feature learning. In CVPR, 2013.

[64] Karen Simonyan and Andrew Zisserman. Two-stream convo-
lutional networks for action recognition in videos. In NeurIPS,
2014.

[65] Karen Simonyan and Andrew Zisserman. Very deep convolu-
tional networks for large-scale image recognition. In ICLR,
2015.

[66] Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon
Shlens, Pieter Abbeel, and Ashish Vaswani. Bottleneck trans-
formers for visual recognition. In CVPR, 2021.

[67] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross
Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train
your vit? data, augmentation, and regularization in vision
transformers. arXiv preprint arXiv:2106.10270, 2021.
[68] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet,
Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent
Vanhoucke, and Andrew Rabinovich. Going deeper with
convolutions. In CVPR, 2015.

[69] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe,
Jonathon Shlens, and Zbigniew Wojna. Rethinking the incep-
tion architecture for computer vision. In CVPR, 2016.
[70] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan,
Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet:
Platform-aware neural architecture search for mobile.
In
CVPR, 2019.

[71] Mingxing Tan and Quoc Le. Efﬁcientnet: Rethinking model

scaling for convolutional neural networks. In ICML, 2019.

[72] Mingxing Tan and Quoc Le. Efﬁcientnetv2: Smaller models

and faster training. In ICML, 2021.

[73] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco
Massa, Alexandre Sablayrolles, and Hervé Jégou. Training
data-efﬁcient image transformers & distillation through atten-
tion. arXiv:2012.12877, 2020.

[74] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles,
Gabriel Synnaeve, and Hervé Jégou. Going deeper with
image transformers. ICCV, 2021.

[75] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. In-
stance normalization: The missing ingredient for fast styliza-
tion. arXiv:1607.08022, 2016.

[76] Régis Vaillant, Christophe Monrocq, and Yann Le Cun. Orig-
inal approach for the localisation of objects in images. Vision,
Image and Signal Processing, 1994.

[77] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS, 2017.
[78] Haohan Wang, Songwei Ge, Eric P Xing, and Zachary C
Lipton. Learning robust global representations by penalizing
local predictive power. NeurIPS, 2019.

[79] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming

He. Non-local neural networks. In CVPR, 2018.

[80] Ross Wightman. GitHub repository: Pytorch image mod-
els. https://github.com/rwightman/pytorch-
image-models, 2019.

[81] Ross Wightman, Hugo Touvron, and Hervé Jégou. Resnet
strikes back: An improved training procedure in timm.
arXiv:2110.00476, 2021.

[82] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang
Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions
to vision transformers. ICCV, 2021.

[83] Yuxin Wu and Kaiming He. Group normalization. In ECCV,

2018.

[84] Yuxin Wu and Justin Johnson. Rethinking "batch" in batch-

norm. arXiv:2105.07576, 2021.

[85] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and
Jian Sun. Uniﬁed perceptual parsing for scene understanding.
In ECCV, 2018.

[86] Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr
Dollár, and Ross Girshick. Early convolutions help transform-
ers see better. In NeurIPS, 2021.

[87] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and
Kaiming He. Aggregated residual transformations for deep
neural networks. In CVPR, 2017.

[88] Weijian Xu, Yifan Xu, Tyler Chang, and Zhuowen Tu. Co-
scale conv-attentional image transformers. ICCV, 2021.
[89] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk
Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regu-
larization strategy to train strong classiﬁers with localizable
features. In ICCV, 2019.

[90] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David
Lopez-Paz. mixup: Beyond empirical risk minimization. In
ICLR, 2018.

[91] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and
Yi Yang. Random erasing data augmentation. In AAAI, 2020.
[92] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler,
Adela Barriuso, and Antonio Torralba. Semantic understand-
ing of scenes through the ADE20K dataset. IJCV, 2019.


