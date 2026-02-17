2
2
0
2
c
e
D
0
3

]

V
C
.
s
c
[

2
v
8
5
9
5
0
.
0
1
2
2
:
v
i
X
r
a

Bridging the Gap Between Vision Transformers and
Convolutional Neural Networks on Small Datasets

Zhiying Lu, Hongtao Xie∗, Chuanbin Liu∗, Yongdong Zhang
University of Science and Technology of China, Hefei, China
arieseirack@mail.ustc.edu.cn, {htxie,liucb92,zhyd73}@ustc.edu.cn

Abstract

There still remains an extreme performance gap between Vision Transformers
(ViTs) and Convolutional Neural Networks (CNNs) when training from scratch on
small datasets, which is concluded to the lack of inductive bias. In this paper, we
further consider this problem and point out two weaknesses of ViTs in inductive
biases, that is, the spatial relevance and diverse channel representation. First,
on spatial aspect, objects are locally compact and relevant, thus ﬁne-grained feature
needs to be extracted from a token and its neighbors. While the lack of data hinders
ViTs to attend the spatial relevance. Second, on channel aspect, representation
exhibits diversity on different channels. But the scarce data can not enable ViTs
to learn strong enough representation for accurate recognition. To this end, we
propose Dynamic Hybrid Vision Transformer (DHVT) as the solution to enhance
the two inductive biases. On spatial aspect, we adopt a hybrid structure, in which
convolution is integrated into patch embedding and multi-layer perceptron module,
forcing the model to capture the token features as well as their neighboring features.
On channel aspect, we introduce a dynamic feature aggregation module in MLP
and a brand new "head token" design in multi-head self-attention module to help
re-calibrate channel representation and make different channel group representation
interacts with each other. The fusion of weak channel representation forms a strong
enough representation for classiﬁcation. With this design, we successfully eliminate
the performance gap between CNNs and ViTs, and our DHVT achieves a series
of state-of-the-art performance with a lightweight model, 85.68% on CIFAR-100
with 22.8M parameters, 82.3% on ImageNet-1K with 24.0M parameters. Code is
available at https://github.com/ArieSeirack/DHVT.

1

Introduction

Convolutional Neural Networks (CNNs) have dominated in Computer Vision (CV) ﬁeld as the
backbone for various tasks like classiﬁcation [1, 2, 3, 4, 5, 6, 7], object detection [8, 9, 10] and
segmentation [11, 12, 13]. These years have witnessed the rapid growth of another promising
alternative architecture paradigm, Vision Transformers (ViTs). They have already exhibited great
performance in common tasks, such as classiﬁcation [14, 15, 16, 17, 18, 19], object detection
[20, 21, 22] and segmentation [23, 24].

ViT [14] is the pioneering model that brings Transformer architecture [25] from Natural Language
Processing (NLP) into CV. It has a higher performance upper bound than standard CNNs, while it
is at the cost of expensive computation and extremely huge amount of training data. The vanilla
ViT needs to be ﬁrstly pre-trained on the huge dataset JFT-300M [14] and then ﬁne-tuned on the
common dataset ImageNet-1K [26]. Under this experimental setting, it shows higher performance
than standard CNNs. However, when training from scratch on ImageNet-1K only, the accuracy is

∗Corresponding author

Preprint. Under review.

much lower. From the practical perspective, most of the datasets are even smaller than ImageNet-1K,
and not all the researchers can hold the burden of pre-training their own model on large datasets and
then ﬁne-tuning on the target small datasets. Thus, an effective architecture for training ViTs from
scratch on small datasets is demanded.

Recent works [27, 28, 29] explore the reasons for the difference in data efﬁciency between ViT
and CNNs, and draw a conclusion to the lack of inductive bias. In [27], it points out that with
not enough data, ViT does not learn to attend locally in earlier layers. And in [28], it says that
the stronger the inductive biases, the stronger the representations. Large datasets tend to help ViT
learn strong representations. Locality constraints improve the performance of ViT. Meanwhile, in
recent work [29], it demonstrates that convolutional constraints can enable strongly sample-efﬁcient
training in the small-data regime. The insufﬁcient training data makes ViT hard to derive the
inductive bias of attending locality, thus many recent works strive to introduce local inductive bias
by integrating convolution into ViTs [18, 15, 30, 31, 32] and modify it to hierarchical structure
[33, 34, 16, 17, 35], making ViTs more like traditional CNNs. This style of hybrid structure shows
comparable performance with strong CNNs when training from scratch on medium dataset ImageNet-
1K only. But the performance gap on much smaller datasets still remains.

Here, we consider that the scarce training data weakens the inductive biases in ViTs. Two kinds of
inductive bias need to be enhanced and better exploited to improve the data efﬁciency, that is, the
spatial relevance and diverse channel representation. On spatial aspect, tokens are relevant and
objects are locally compact. The important ﬁne-grained low-level feature needs to be extracted from
the token and its neighbors at the earlier layers. Rethinking the feature extraction framework in ViTs,
the module for feature representation is the multi-layer perceptron (MLP) and its receptive ﬁeld can
be seen as only itself. So ViTs depend on the multi-head self-attention (MHSA) module to model and
capture the relation between tokens. As is pointed out in work [27], with less training data, lower
attention layers do not learn to attend locally. In other words, they do not focus on neighboring tokens
and aggregate local information in the early stage. As is known, capturing local features in lower
layers facilitates the whole representation pipeline. The deep layers sequentially process the low-level
texture feature into high-level semantic features for ﬁnal recognition. Thus ViTs have an extreme
performance gap compared with CNNs when training from scratch on small datasets. On channel
aspect, feature representation exhibits diversity in different channels. And ViT has its own inductive
bias that different channel group encodes different feature representation of the object, and the whole
token vector forms the representation of the object. As is pointed out in work [28], large datasets
tend to help ViT learn strong representation. The insufﬁcient data can not enable ViTs to learn strong
enough representation, thus the whole representation is poor for accurate classiﬁcation.

In this paper, we solve the performance gap of training from scratch on small datasets between CNNs
and ViTs and provide a hybrid architecture called Dynamic Hybrid Vision Transformer (DHVT)
as a substitute. We ﬁrst introduce a hybrid model to address the issue on spatial aspect. The
proposed hybrid model integrates a sequence of convolution layers in the patch embedding stage to
eliminate the non-overlapping problem, preserving ﬁne-grained low-level features, and it involves
depth-wise convolution [36] in MLP for local feature extraction. In addition, we design two modules
for making feature representation stronger to solve the problem on channel view. To be speciﬁc, in
MLP, depth-wise convolution is adopted for the patch tokens, and the class token is identically passed
through without any computation. We then leverage the output patch tokens to produce channel
weight like Squeeze-Excitation (SE) [4] for the class token. This operation helps re-calibrate each
channel for the class token to reinforce its feature representation. Moreover, in order to enhance
interaction among different semantic representations of different channel groups and owing to the
variable length of the token sequence in vision transformer structure, we devise a brand new token
mechanism called "head token". The number of head tokens is the same as the number of attention
heads in MHSA. Head tokens are generated by segmenting and projecting input tokens along the
channel. The head tokens will be concatenated with all other tokens to pass through the MHSA. Each
channel group in the corresponding attention head in the MHSA now is able to interact with others.
Though maybe the representation in each channel and channel group is poor for classiﬁcation on
account of insufﬁcient training data, the head tokens help re-calibrate each learned feature pattern
and enable a stronger integral representation of the object, which is beneﬁcial to ﬁnal recognition.

We conduct experiments of training from scratch on various small datasets, the common dataset
CIFAR-100, and small domain datasets Clipart, Painting and Sketch from DomainNet [37] to examine
the performance of our model. On CIFAR-100, our proposed models show a signiﬁcant performance

2

margin with strong CNNs like ResNeXt, DenseNet and Res2Net. The Tiny model achieves 83.54%
with only 5.8M parameters, and our Small model reaches the state-of-the-art 85.68% accuracy with
only 22.8M parameters, outperforming a series of strong CNNs. Therefore, we eliminate the gap
between CNNs and ViTs, providing an alternative architecture that can train from scratch on small
datasets. We also evaluate the performance of DHVT when training from scratch on ImageNet-1K.
Our proposed DHVT-S achieves competitive 82.3% accuracy with only 24.0M parameters, which is
the state-of-the-art non-hierarchical vision transformer structure as far as we know, demonstrating the
effectiveness of our model on larger datasets. In summary, our main contributions are:

1. We conclude that the data efﬁciency on small datasets can be addressed by strengthening two
inductive biases in ViTs, which are spatial relevance and diverse channel representation.

2. On spatial aspect, we adopt a hybrid model integrated with convolution, preserving ﬁne-grained
low-level features at the earlier stage and forcing the model to extract tokens feature and corresponding
neighbor feature.

3. On channel aspect, we leverage the output patch tokens to re-calibrate class token channel-wise,
producing better feature representation. We further introduce "head token", a novel design that
helps fuse diverse feature representation encoded in different channel groups into a stronger integral
representation.

2 Related Work

Vision Transformers. Convolutional Neural Networks [38, 39, 1, 40, 2, 41] dominated the computer
vision ﬁelds in the past decade, with its intrinsic inductive biases designed for image recognition.
The past two years witnessed the rise of Vision Transformer models in various vision tasks [42, 43,
23, 20, 44, 45]. Although there exist previous works introducing attention mechanism into CNNs
[4, 46, 47], the pioneering full transformer architecture in computer vision are iGPT [48] and ViT
[14]. ViT is widely adopted as the architecture paradigm for vision tasks especially image recognition.
It processes the image as a token sequence and exploits relations among tokens. It uses "class
token" like BERT [49] to exchange information at every layer and for ﬁnal classiﬁcation. It performs
well when pre-trained on huge datasets. But when training from scratch on ImageNet-1K only, it
underperforms ResNets, demonstrating a data-hungry problem.

Data-efﬁcient ViTs. Many of the subsequent modiﬁcations on ViT strive for a more data-efﬁcient
architecture that can perform well without pre-training on larger datasets. The methods can be divided
into different groups. [42, 50] use knowledge distillation strategy and stronger data-augmentation
methods to enable training from scratch. [51] points out that using convolution in the patch embedding
stage greatly beneﬁts ViTs training. [52, 15, 18, 53, 54] leverage convolution for patch embedding
to eliminate the discontinuity brought by non-overlapping patch embedding in vanilla ViT, and
such design becomes a paradigm in subsequent works. To further introduce inductive bias into
ViT, [15, 30, 34, 55, 56] integrate depth-wise convolution into feed forward network, resulting in
a hybrid architecture combining the self-attention and convolution. To make ViTs more similar
to standard CNNs, [16, 54, 17, 34, 33, 35, 57, 32] re-design the spatial and channel dimension of
vanilla ViT, producing a series of hierarchical style vision transformer. [31, 58, 59] design another
parallel convolution branch and enable the interaction with the self-attention branch, making the two
branch complements each other. The above architectures introduce strong inductive bias and become
data-efﬁcient when training from scratch on ImageNet-1K. In addition, works like [60, 61, 62]
investigate channel-wise representation through conducting self-attention channel-wise, while we
enhance channel representation by dynamically aggregating patch token features to enhance class
token channel-wise and compatibly involve channel group-wise head tokens into vanilla self-attention.
Finally, works like [63, 64, 65], suggesting that the number of tokens can be variable.

ViTs for small datasets. There exists several works on solving the training from scratch problem
on small datasets. Though the above modiﬁed vision transformers perform well when trained on
ImageNet-1K, they fail to compete with standard CNNs when training on much smaller datasets like
CIFAR-100. Work [66] introduces a self-supervised style training strategy and a loss function to help
train ViTs on small datasets. CCT [67] adopts a convolutional tokenization module and replaces the
class token with the ﬁnal sequence pooling operation. SL-ViT [68] adopts shifted patch tokenization
module and modiﬁes self-attention to make it focus more locally. Though the previous works reduce
the performance gap between standard CNNs ResNets[1], they fail to be sub-optimal when compared

3

with strong CNNs. Our proposed method leverages local constraints and enhances representation
interaction, successfully bridging the performance gap on small datasets.

Figure 1: Overview of the proposed Dynamic Hybrid Vision Transformer (DHVT). DHVT follows a
non-hierarchical structure, where each encoder layer contains two pre-norm and shortcut, a Head-
Interacted Multi-Head Self-Attention (HI-MHSA) and a Dynamic Aggregation Feed Forward (DAFF).

3 Methods

3.1 Overview of DHVT

As shown in Fig. 1, the framework of our proposed DHVT is similar to vanilla ViT. We choose a
non-hierarchical structure, where every encoder block shares the same parameter setting, processing
the same shape of features. Under this structure, we can deal with variable length of token sequence.
We keep the design of using class token to interact with all the patch tokens and for ﬁnal prediction.
In the patch embedding module, the input image will be split into patches ﬁrst. Given the input
image with resolution H × W and the target patch size P , the resulting length of the patch token
sequence will be N = HW/P 2. Our modiﬁed patch embedding is called Sequential Overlapping
Patch Embedding (SOPE), which contains several successive convolution layers of 3 × 3 convolution
with stride s = 2, Batch Normalization and GELU [69] activation. The relation between the number
of convolution layers and the patch size is P = 2k. SOPE is able to eliminate the discontinuity
brought by the vanilla patch embedding module, preserving important low-level features. It is able
to provide position information to some extent. We also adopt two afﬁne transformations before
and after the series of convolution layers. This operation rescales and shifts the input feature, and it
acts like normalization, making the training performance more stable on small datasets. The whole
process of SOPE can be formulated as follows.

Af f (x) = Diag(α)x + β

Gi(x) = GELU (BN (Conv(x))), i = 1, . . . , k

SOP E(x) = Reshape(Af f (Gk(. . . (G2(G1(Af f (x)))))))

(1)

(2)

(3)

In Eq.1, α and β are learnable parameters, and initialized as 1 and 0 respectively. After the sequence
of convolution layers, the feature maps are then reshaped as patch tokens and concatenated with a
class token. Then the sequence of tokens will be fed into encoder layers. After SOPE, the token
sequence will pass through layers of encoder, where each encoder contains Layer Normalization [70],
multi-head self-attention and feed forward network. Here we modiﬁed the MHSA as Head-Interacted
Multi-Head Self-Attention (HI-MHSA) and feed forward network as Dynamic Aggregation Feed
Forward (DAFF). We will introduce them in the following sections. After the ﬁnal encoder layer, the
output class token will be fed into the linear head for ﬁnal prediction.

4

HI-MHSALayerNormDAFFLayerNormSequential Overlapping Patch EmbeddingEncoder LayerEncoder LayerLFC HeadClassification Resultcls tokenpatch token3×3 Conv, s=2BatchNormGELUkReshapeAffineAffine3.2 Dynamic Aggregation Feed Forward

Figure 2: The structure of Dynamic Aggregation Feed Forward (DAFF).

The vanilla feed forward network (FFN) in ViT is formed by two fully-connected layers and GELU
activation. All the tokens, either patch tokens or class token, will be processed by FFN. Here we
integrate depth-wise convolution [36] (DWCONV) in FFN and resulting in a hybrid model. Such
hybrid model is similar to standard CNNs because it can be seen as using convolution to do feature
representation. With the inductive bias brought by depth-wise convolution, the model is forced to
capture neighboring features, solving the problem on spatial view. It greatly reduces the performance
gap when training from scratch on small datasets, and converges faster than standard CNNs. However,
such a structure still performs worse than stronger CNNs. More investigations are required to solve
the problem on channel aspect.

We propose two methods that make the whole model more dynamic and learn stronger feature
representation under insufﬁcient data. The ﬁrst proposed module is Dynamic Aggregation Feed
Forward (DAFF). We aggregate the feature of patch tokens into the class token in a channel attention
way, similar to the Squeeze-Excitation operation in SENet [4], as is shown in Fig. 2. Class token
is split before the projection layers. Then the patch tokens will go through a depth-wise integrated
multi-layer perceptron with a shortcut inside. The output patch tokens will then be averaged into a
weight vector W. After the squeeze-excitation operation, the output weight vector will be multiplied
with class token channel-wise. Then the re-calibrated class token will be concatenated with output
patch tokens to restore the token sequence. We use Xc, Xp to denote class token and patch tokens
respectively. The process can be formulated as:

W = Linear(GELU (Linear((Average(Xp))))

Xc = Xc (cid:12) W

(4)

(5)

3.3 Head Token

The second design to enhance feature representation is "head token", which is a brand new mechanism
as far as we know. There are two reasons why we introduce head token here. First, in the original
MHSA module, each attention head has not interacted with others, which means each head only
focuses on itself to calculate attention. Second, channel groups in different heads are responsible
for different feature representations, which is the inductive bias of ViTs. And as we pointed out
above, the lack of training data can not enable models to learn strong representation. Under this
circumstance, the representation in each channel group is too weak for recognition. After introducing
head tokens into attention calculation, the channel group in each head are able to interact with those
in other heads, and different representation can be fused into an integral representation of the object.
Representation learned by insufﬁcient data may be poor in each channel, but their combination will
produce a strong enough representation. The structure of vision transformer also guarantees this
mechanism because the length of input tokens is variable, except for the hierarchical structure vision
transformer with window attention such as[17, 35].

The process of generating head tokens is shown in Fig. 3 (a). We denote the number of patch tokens
as N , so the length of the input sequence is N + 1. According to the pre-deﬁned number of heads h,
each D-dimensional token, including class token, will be reshaped into h parts. Each part contains d
channels, where D = d × h. We average all the separated tokens in their own parts. Thus we get

5

FCFCGELUDWCONVAvgPoolFCFCGELUsccls tokenpatch tokenssplitcconcatFigure 3: Pipeline of Head-Interacted Multi-Head Self-Attention (HI-MHSA).

totally h tokens and each one is d-dimensional. All such intermediate tokens will be projected into
D-dimension again, resulting in h head tokens in total. The head tokens will be added with head
embedding, which provides positional information for head tokens. Head embedding is a group of
learnable parameters, just like positional embedding. Finally, they are concatenated with patch tokens
and class token, forming the token sequence for standard MHSA, as Eq. 7, in which XH denotes
head tokens. We do not change the attention calculation in MHSA. Head tokens will also be linearly
projected into query, key and value, and they will interact with all other tokens. After MHSA, the
head tokens will be averaged and added to class token, just as Fig. 3 (b) shows. Head tokens can be
derived as Eq. 6 shows. We use Ehead to denote head embedding.

XH = GELU (Linear((Average(Reshape(X))))) + Ehead

X = [Xc; Xp; XH ] = [Xc; X1

p, . . . , XN

p ; X1

H , . . . , Xh
H ]

(6)

(7)

4 Experiments

All the experiments presented in our paper are based on image classiﬁcation. We do not conduct
experiments on downstream tasks. We ﬁrst introduce the training datasets and experimental settings
in Section 4.1. The performance comparisons are shown in Section 4.2. We also show the result of
the ablation study in Section 4.3. And ﬁnally, we present an example of visualization in 4.4.

4.1 Datasets and Experimental Settings

Datasets. Our main focus is training from scratch on small datasets. There are two factors to consider
whether a dataset is small: the total number of training data in the dataset and the average number
of training data for each class. Some datasets are small on the ﬁrst factor, but large on the second.
The example is CIFAR-10 [71], with 50000 training data in total for 10 classes, has an average of
5000 instances in each class. Considering this, we do not choose CIFAR-10 as our target dataset here.
We choose 5 different datasets here. The main performance comparisons are on CIFAR-100 [71].
And we choose three datasets from DomainNet [37], a benchmark commonly for domain adaptation
tasks. They have a large domain-shift from common medium dataset ImageNet-1K [26], making the
ﬁne-tuning experiments non-trivial, as pointed in [66]. Finally, we also choose ImageNet-1K to test
the performance of our proposed model. The details of the datasets are shown in Table 1.

6

N+1DddddN+1hdddd1hdhDhD = d×hReshapeN+1+hDAverageReshapeinputtokenscheadtokensMulti-Head Self-AttentionNDDhoutputtokenss(a) Generate Head Tokens (b) Multi-Head Self-Attentioncls tokenpatch tokenssplitcconcathead tokenAverage1DcN+1Dhead embeddingFCGELUTable 1: The details of training datasets. We report the train and test size of each dataset, including
the number of classes. We also show the average images per class in the training set.

Dataset

Train size Test size Classes Average images per class

CIFAR-100 [71]
ClipArt [37]
Sketch [37]
Painting [37]
ImageNet-1K [26]

50000
33525
48212
50416
1281167

10000
14604
20916
21850
100000

100
345
345
345
1000

500
97
140
146
1281

Model Variants. We propose two architecture variants. Detailed information on model variants can
be seen in Supplementary Materials.

• DHVT-T: 12 encoder layers, embedding dimension of 192, MLP ratios of 4, attention heads

of 4 on CIFAR-100 and DomainNet, and 3 on ImageNet-1K.

• DHVT-S: 12 encoder layers, embedding dimension of 384, MLP ratios of 4, attention heads

of 8 on CIFAR-100, 6 on DomainNet and ImageNet-1K.

Implementation Details. When training our DHVT, we keep the image size in CIFAR-100 as its
original resolution 32 × 32, and the patch size is set to 4 or 2. For ImageNet-1K, ClipArt, Painting and
Sketch, we adopt resolution 224 × 224, and the patch size comes to 16. All the data augmentations
are the same as those in DeiT [42]. We do not tune data-augmentation hyperparameters for better
performance. On all of the datasets, we train our network from random initialization with the AdamW
[72] optimizer with a cosine decay learning-rate scheduler. We set the batch size of 512 and 256 for
DHVT-T and DHVT-S when training on CIFAR-100, with an initial learning rate of 0.001, and a
weight decay of 0.05, a warm-up epoch of 5. When on ClipArt, Sketch and Painting, we use a batch
size of 256 and 128 respectively for DHVT-T and DHVT-S, with an initial learning rate of 0.001, a
warm-up epoch of 20 and a weight decay of 0.05. For ImageNet-1K, we use the batch size of 512 for
both models with and initial learning rate of 0.0005 and weight decay of 0.05, a warm-up epoch of
10. The training epochs on all datasets are 300, except that WRN28-10 [73] is trained for 200 epochs
on CIFAR. All of the training devices are Nvidia 3090 GPUs. We use Pytorch tools and our code is
modiﬁed from timm1.

4.2 Performance Comparisons

Method

ResNet-50
DHVT-T
DHVT-S

Table 2: Results on DomainNet
Painting

#Params ClipArt

24.2M
6.1M
23.8M

71.90
71.73
73.89

64.36
63.34
66.08

Sketch

67.45
66.60
68.72

Table 3: Results on ImageNet-1K

Method

#Params

ImageNet-1K

DHVT-T
DHVT-S

6.2M
24.0M

76.5
82.3

Results on DomainNet. We also conduct experiments on other small datasets. Here we choose three
datasets from DomainNet as our target. We use the implementation of ResNet-50 in Pytorch ofﬁcial
code for performance comparison. All of the data-augmentations, such as Mixup [74] and CutMix
[75] and AutoAugment [76], are also adopted for training ResNet-50 from scratch on these datasets.
All of the results reported are the best out of four runs. As is shown in Table 2, our model shows
better results than standard ResNet-50, demonstrating its performance across different small datasets.
The whole comparison of all of the DomainNet datasets with more baseline models is shown in
Supplementary Materials.

Results on ImageNet-1K To test the train-from-scratch performance of our model on the common
medium-size dataset ImageNet-1K, we also conduct experiments on it. We follow the same ex-
perimental settings as in DeiT [42]. The results are shown in Table 3. Surprisingly, our DHVT-T
reaches 76.47 accuracy and our DHVT-S reaches 82.3 accuracy. As far as we know, this is the
best performance under such a non-hierarchical vision transformer structure with class token. And

1https://github.com/rwightman/pytorch-image-models

7

our model outperforms many of the state-of-the-art methods with comparable parameters. This
experiment shows that our model not only behaves well on small datasets but also exhibits powerful
performance on larger datasets. We will show the performance comparison with other methods that
train from scratch on ImageNet-1K in the Supplementary Materials.

Table 4: Performance comparison of different methods on the CIFAR-100 dataset. All models are
trained from random initialization. "(cid:63)" denotes that we re-implement the method under the same
training scheme. The other results are cited from the corresponding works.

Type

Method

Patch Size

#Params GFLOPs Acc (%)

CNN

ViT

Hybrid

WRN28-10 [73]
SENet-29 [4]
ResNeXt-29, 8×64d [40]
SKNet-29 [41]
DenseNet-BC (k = 40) [2]
Res2NeXt-29, 6c×24w×6s-SE [77]

DeiT-T [42](cid:63)
DeiT-S [42](cid:63)
DeiT-T [42](cid:63)
DeiT-S [42](cid:63)
PVT-T [35]
PVT-S [35]
Swin-T [35]
NesT-T [35]
NesT-S [35]
NesT-B [35]

CCT-7/3×1 [67]
CvT-13 [18](cid:63)
CvT-13 [18](cid:63)
DHVT-T (Ours)
DHVT-S (Ours)
DHVT-T (Ours)
DHVT-S (Ours)

1
1
1
1
1
1

4
4
2
2
1
1
1
1
1
1

4
4
2
4
4
2
2

36.5M
35.0M
34.4M
27.7M
25.6M
36.9M

5.4M
21.4M
5.4M
21.4M
15.8M
27.0M
27.5M
6.2M
23.4M
90.1M

3.7M
19.6M
19.6M
6.0M
23.4M
5.8M
22.8M

5.2
5.4
5.4
4.2
9.3
5.9

0.4
1.4
1.4
5.5
0.6
1.2
1.4
1.7
6.6
26.5

1.0
1.1
4.5
0.4
1.5
1.4
5.6

80.75
82.22
82.23
82.67
82.82
83.44

67.59
66.55
65.86
63.77
69.62
69.79
78.07
78.69
81.70
82.56

80.92
79.24
81.81
80.93
82.91
83.54
85.68

Results on CIFAR-100. We mainly compare the performance of our proposed model on CIFAR-100.
Patch size set to 1 means taking raw pixel input. For comparison with other methods, we directly cite
the results reported in the corresponding paper. The results of our model are the best out of ﬁve runs
with different random seeds. As is shown in Table 4, CNN models occasionally have more parameters
and conduct fewer computations, while ViT models have much fewer parameters and conduct much
higher computations. Our model DHVT-T reaches 83.54 with 5.8M parameters. And DHVT-S
reaches 85.68 with only 22.8M parameters. With much fewer parameters, our model achieves much
higher performance against other ViT-based models and strong CNNs ResNeXt, SENet, SKNet,
DenseNet and Res2Net. And compared with other ViT and Hybrid models, we exhibit a signiﬁcant
performance improvement under reasonable parameters and computational burdens. We not only
bridge the performance gap between CNNs and ViTs but also push the state-of-the-art result to a
higher level. Moreover, scaling up and smaller patch size beneﬁt our method. Both DeiT and PVT
fail to achieve higher performance when scaling up. And when the patch size gets smaller, the
performance of DeiT even drops. These results are reasonable because insufﬁcient data is hard to
train a large model from scratch. And smaller patch size further intensiﬁes the non-overlapping
problem of vanilla ViT and thus decreases the performance. More experiment results such as training
from scratch on 224×224 resolution can be seen in Supplementary Materials.

4.3 Ablation Studies

All the results in the ablation study are the average over four runs with different random seeds.
The model for the ablation study is DHVT-T, with a patch size of 4 and training from scratch on
CIFAR-100 with the same data augmentation as in Section 4.2. Here DHVT-T is trained with a
learning rate of 0.001, a warm-up epoch of 10 and batch size of 512, and a total epoch of 300. The

8

baseline is DeiT-T with 4 heads and the patch size is set to 4. The results are shown in the following
tables.

The importance of positional information. We have a baseline performance of 67.59 from DeiT-T
with 4 heads, training from scratch with 300 epochs. When removing absolute positional embedding,
the performance drops drastically to 58.72, demonstrating the importance of position information
in vision transformers. SOPE is able to provide positional information to some extent because
such absolute positional information can be derived from zero padding. As is shown in Table 5,
when adopting SOPE and removing absolute position embedding, the performance does not drop so
drastically. But only depending on SOPE to provide position information is not enough.

Table 5: Ablation study on SOPE and DAFF
Abs. PE SOPE DAFF

Acc (%)

(cid:33)
(cid:37)

(cid:33)
(cid:37)

(cid:33)
(cid:37)

(cid:33)
(cid:37)

(cid:37)
(cid:37)

(cid:33)
(cid:33)

(cid:37)
(cid:37)

(cid:33)
(cid:33)

(cid:37)
(cid:37)

(cid:37)
(cid:37)

67.59 (+0.00)
58.72 (-8.87)

73.68 (+6.09)
69.65 (+2.06)

(cid:33) 79.47 (+11.88)
(cid:33) 79.75 (+12.16)

(cid:33) 80.17 (+12.58)
(cid:33) 80.35 (+12.76)

Table 6: Ablation study on head token

Abs. PE

SOPE
&
DAFF

(cid:33)

(cid:33)

(cid:37)

(cid:37)

(cid:37)

(cid:33)

Head Token Acc (%)

(cid:37)

(cid:33)

(cid:33)

67.59
(+0.00)

69.10
(+1.51)

80.85
(+13.26)

The role of DAFF. When adopting DAFF, the performance gain increases greatly to 79.47, because
DAFF solves the problem on both spatial and channel aspects, introducing strong local constraints
and re-calibrating channel feature representation. It is sensible to see that removing absolute position
embedding can increase performance. The positional information has been encoded into tokens
through the depth-wise convolution in DAFF, and the absolute position embedding will break
translation invariance. When both SOPE and DAFF are adopted, the positional information will
be encoded comprehensively, and SOPE will also help address the non-overlapping problem here,
preserving ﬁne-grained low-level features in the early stage.

The role of head tokens. From Table 6, we can also see the stable performance gain brought by head
tokens across different model structures. When introducing head tokens into DeiT-T, the performance
gets a +1.51 gain, demonstrating its effectiveness. As we said before, head tokens guarantee the
interaction among different channel groups, better fusing the diverse representation. The resulting
integral representation is now strong enough for classiﬁcation. When adopting all three modiﬁcations,
we get a +13.26 accuracy gain, successfully bridging the performance gap with CNNs.

4.4 Visualization

We visualize the attention maps of head tokens to patch tokens in Fig. 4. Each row represents one
image. The results are samples in the second encoder layer. We can see that different head token
activates on different patch tokens, exhibiting their diverse representations. On such low layers,
low-level ﬁne-grained features are able to be captured in our model. More visualization results are
shown in the Supplementary Materials.

5 Limitation

Though we achieve a much higher performance than existing methods, such performance gain comes
at the expense of computation. The performance when patch size set to 2 boosts higher than using
patch size of 4. But the computation expense rises quadratically. In practical usage, we suggest
choose a good patch size for better trade-off between performance and computation.

9

Figure 4: Visualization of the attention map of head tokens to patch tokens on low layer

6 Conclusion

In this paper, we present an alternative vision transformer architecture DHVT, which can train
from scratch on small datasets and reach state-of-the-art performance on a series of datasets. The
weak inductive biases of spatial relevance and diverse channel representation brought by insufﬁcient
training data are strengthened in our model. The highlighted head token design is able to transfer to
variants of ViT model to enable better feature representation.

7 Acknowledgements

This work is supported by the National Nature Science Foundation of China (62121002, 62022076,
62232006, U1936210, 62272436), the Youth Innovation Promotion Association Chinese Academy of
Sciences (Y2021122), Anhui Provincial Natural Science Foundation (2208085QF190), the China
Postdoctoral Science Foundation 2021M703081.

References

[1] He, K., X. Zhang, S. Ren, et al. Deep residual learning for image recognition. In Proceedings
of the IEEE conference on computer vision and pattern recognition, pages 770–778. 2016.

[2] Huang, G., Z. Liu, L. Van Der Maaten, et al. Densely connected convolutional networks.
In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
4700–4708. 2017.

[3] Liu, C., H. Xie, Z. Zha, et al. Bidirectional attention-recognition model for ﬁne-grained object

classiﬁcation. IEEE Transactions on Multimedia, 22(7):1785–1795, 2019.

[4] Hu, J., L. Shen, G. Sun. Squeeze-and-excitation networks.

In Proceedings of the IEEE

conference on computer vision and pattern recognition, pages 7132–7141. 2018.

[5] Min, S., H. Yao, H. Xie, et al. Domain-oriented semantic embedding for zero-shot learning.

IEEE Transactions on Multimedia, 23:3919–3930, 2020.

[6] Fu, J., H. Zheng, T. Mei. Look closer to see better: Recurrent attention convolutional neural
network for ﬁne-grained image recognition. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 4438–4446. 2017.

[7] Min, S., H. Yao, H. Xie, et al. Multi-objective matrix normalization for ﬁne-grained visual

recognition. IEEE Transactions on Image Processing, 29:4996–5009, 2020.

10

head token 1head token 2head token 3head token 4[8] Ren, S., K. He, R. Girshick, et al. Faster r-cnn: Towards real-time object detection with region

proposal networks. Advances in neural information processing systems, 28, 2015.

[9] Lin, T.-Y., P. Goyal, R. Girshick, et al. Focal loss for dense object detection. In Proceedings of

the IEEE international conference on computer vision, pages 2980–2988. 2017.

[10] Tian, Z., C. Shen, H. Chen, et al. Fcos: Fully convolutional one-stage object detection. In
Proceedings of the IEEE/CVF international conference on computer vision, pages 9627–9636.
2019.

[11] He, K., G. Gkioxari, P. Dollár, et al. Mask r-cnn. In Proceedings of the IEEE international

conference on computer vision, pages 2961–2969. 2017.

[12] Chen, L.-C., G. Papandreou, I. Kokkinos, et al. Deeplab: Semantic image segmentation with
deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on
pattern analysis and machine intelligence, 40(4):834–848, 2017.

[13] Ronneberger, O., P. Fischer, T. Brox. U-net: Convolutional networks for biomedical image
segmentation. In International Conference on Medical image computing and computer-assisted
intervention, pages 234–241. Springer, 2015.

[14] Dosovitskiy, A., L. Beyer, A. Kolesnikov, et al. An image is worth 16x16 words: Transformers

for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[15] Yuan, K., S. Guo, Z. Liu, et al. Incorporating convolution designs into visual transformers. In
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages
579–588. 2021.

[16] Wang, W., E. Xie, X. Li, et al. Pyramid vision transformer: A versatile backbone for dense
prediction without convolutions. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 568–578. 2021.

[17] Liu, Z., Y. Lin, Y. Cao, et al. Swin transformer: Hierarchical vision transformer using shifted
windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
10012–10022. 2021.

[18] Wu, H., B. Xiao, N. Codella, et al. Cvt: Introducing convolutions to vision transformers. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22–31.
2021.

[19] He, J., J.-N. Chen, S. Liu, et al. Transfg: A transformer architecture for ﬁne-grained recognition.
In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, vol. 36, pages 852–860. 2022.

[20] Carion, N., F. Massa, G. Synnaeve, et al. End-to-end object detection with transformers. In

European conference on computer vision, pages 213–229. Springer, 2020.

[21] Dai, Z., B. Cai, Y. Lin, et al. Up-detr: Unsupervised pre-training for object detection with
transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1601–1610. 2021.

[22] Zhu, X., W. Su, L. Lu, et al. Deformable detr: Deformable transformers for end-to-end object

detection. arXiv preprint arXiv:2010.04159, 2020.

[23] Strudel, R., R. Garcia, I. Laptev, et al. Segmenter: Transformer for semantic segmentation. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7262–7272.
2021.

[24] Guo, R., D. Niu, L. Qu, et al. Sotr: Segmenting objects with transformers. In Proceedings of
the IEEE/CVF International Conference on Computer Vision, pages 7157–7166. 2021.

[25] Vaswani, A., N. Shazeer, N. Parmar, et al. Attention is all you need. Advances in neural

information processing systems, 30, 2017.

[26] Russakovsky, O., J. Deng, H. Su, et al. Imagenet large scale visual recognition challenge.

International journal of computer vision, 115(3):211–252, 2015.

11

[27] Raghu, M., T. Unterthiner, S. Kornblith, et al. Do vision transformers see like convolutional

neural networks? Advances in Neural Information Processing Systems, 34, 2021.

[28] Park, N., S. Kim. How do vision transformers work? arXiv preprint arXiv:2202.06709, 2022.

[29] d’Ascoli, S., H. Touvron, M. L. Leavitt, et al. Convit: Improving vision transformers with
soft convolutional inductive biases. In International Conference on Machine Learning, pages
2286–2296. PMLR, 2021.

[30] Li, Y., K. Zhang, J. Cao, et al. Localvit: Bringing locality to vision transformers. arXiv preprint

arXiv:2104.05707, 2021.

[31] Peng, Z., W. Huang, S. Gu, et al. Conformer: Local features coupling global representations for
visual recognition. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 367–376. 2021.

[32] Chen, Z., L. Xie, J. Niu, et al. Visformer: The vision-friendly transformer. CoRR,

abs/2104.12533, 2021.

[33] Zhao, Y., G. Wang, C. Tang, et al. A battle of network structures: An empirical study of cnn,

transformer, and mlp. arXiv preprint arXiv:2108.13002, 2021.

[34] Heo, B., S. Yun, D. Han, et al. Rethinking spatial dimensions of vision transformers. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11936–
11945. 2021.

[35] Zhang, Z., H. Zhang, L. Zhao, et al. Nested hierarchical transformer: Towards accurate, data-
efﬁcient and interpretable visual understanding. In AAAI Conference on Artiﬁcial Intelligence
(AAAI), 2022. 2022.

[36] Howard, A. G., M. Zhu, B. Chen, et al. Mobilenets: Efﬁcient convolutional neural networks for

mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

[37] Peng, X., Q. Bai, X. Xia, et al. Moment matching for multi-source domain adaptation. In
Proceedings of the IEEE/CVF international conference on computer vision, pages 1406–1415.
2019.

[38] Krizhevsky, A., I. Sutskever, G. E. Hinton. Imagenet classiﬁcation with deep convolutional

neural networks. Advances in neural information processing systems, 25, 2012.

[39] Szegedy, C., W. Liu, Y. Jia, et al. Going deeper with convolutions. In Proceedings of the IEEE

conference on computer vision and pattern recognition, pages 1–9. 2015.

[40] Xie, S., R. Girshick, P. Dollár, et al. Aggregated residual transformations for deep neural
networks. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 1492–1500. 2017.

[41] Li, X., W. Wang, X. Hu, et al. Selective kernel networks. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pages 510–519. 2019.

[42] Touvron, H., M. Cord, M. Douze, et al. Training data-efﬁcient image transformers & distillation
through attention. In International Conference on Machine Learning, pages 10347–10357.
PMLR, 2021.

[43] Jiang, Y., S. Chang, Z. Wang. Transgan: Two pure transformers can make one strong gan, and

that can scale up. Advances in Neural Information Processing Systems, 34, 2021.

[44] Arnab, A., M. Dehghani, G. Heigold, et al. Vivit: A video vision transformer. In Proceedings
of the IEEE/CVF International Conference on Computer Vision, pages 6836–6846. 2021.

[45] Chen, X., S. Xie, K. He. An empirical study of training self-supervised vision transformers. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9640–9649.
2021.

12

[46] Wang, X., R. Girshick, A. Gupta, et al. Non-local neural networks. In Proceedings of the IEEE

conference on computer vision and pattern recognition, pages 7794–7803. 2018.

[47] Hu, H., Z. Zhang, Z. Xie, et al. Local relation networks for image recognition. In Proceedings
of the IEEE/CVF International Conference on Computer Vision, pages 3464–3473. 2019.

[48] Chen, M., A. Radford, R. Child, et al. Generative pretraining from pixels. In International

Conference on Machine Learning, pages 1691–1703. PMLR, 2020.

[49] Devlin, J., M.-W. Chang, K. Lee, et al. Bert: Pre-training of deep bidirectional transformers for

language understanding. arXiv preprint arXiv:1810.04805, 2018.

[50] Jiang, Z.-H., Q. Hou, L. Yuan, et al. All tokens matter: Token labeling for training better vision

transformers. Advances in Neural Information Processing Systems, 34, 2021.

[51] Xiao, T., M. Singh, E. Mintun, et al. Early convolutions help transformers see better. Advances

in Neural Information Processing Systems, 34:30392–30400, 2021.

[52] Yuan, L., Y. Chen, T. Wang, et al. Tokens-to-token vit: Training vision transformers from
scratch on imagenet. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 558–567. 2021.

[53] Chu, X., Z. Tian, B. Zhang, et al. Conditional positional encodings for vision transformers.

arXiv preprint arXiv:2102.10882, 2021.

[54] Wang, W., E. Xie, X. Li, et al. Pvt v2: Improved baselines with pyramid vision transformer.

Computational Visual Media, pages 1–10, 2022.

[55] Ren, S., D. Zhou, S. He, et al. Shunted self-attention via multi-scale token aggregation, 2021.

[56] Guo, J., K. Han, H. Wu, et al. Cmt: Convolutional neural networks meet vision transformers. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
12175–12185. 2022.

[57] Mao, X., G. Qi, Y. Chen, et al. Towards robust vision transformer.

arXiv preprint

arXiv:2105.07926, 2021.

[58] Chen, Q., Q. Wu, J. Wang, et al. Mixformer: Mixing features across windows and dimensions.

arXiv preprint arXiv:2204.02557, 2022.

[59] Xu, Y., Q. Zhang, J. Zhang, et al. Vitae: Vision transformer advanced by exploring intrinsic

inductive bias. Advances in Neural Information Processing Systems, 34, 2021.

[60] Ali, A., H. Touvron, M. Caron, et al. Xcit: Cross-covariance image transformers. Advances in

neural information processing systems, 34, 2021.

[61] Ding, M., B. Xiao, N. Codella, et al. Davit: Dual attention vision transformer. arXiv preprint

arXiv:2204.03645, 2022.

[62] Xu, W., Y. Xu, T. Chang, et al. Co-scale conv-attentional image transformers. In Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9981–9990. 2021.

[63] Ryoo, M., A. Piergiovanni, A. Arnab, et al. Tokenlearner: Adaptive space-time tokenization for

videos. Advances in Neural Information Processing Systems, 34, 2021.

[64] Fang, J., L. Xie, X. Wang, et al. Msg-transformer: Exchanging local spatial information by

manipulating messenger tokens. In CVPR. 2022.

[65] Liang, Y., C. Ge, Z. Tong, et al. Not all patches are what you need: Expediting vision trans-
formers via token reorganizations. In International Conference on Learning Representations.
2022.

[66] Liu, Y., E. Sangineto, W. Bi, et al. Efﬁcient training of visual transformers with small datasets.

Advances in Neural Information Processing Systems, 34, 2021.

13

[67] Hassani, A., S. Walton, N. Shah, et al. Escaping the big data paradigm with compact transform-

ers. arXiv preprint arXiv:2104.05704, 2021.

[68] Lee, S. H., S. Lee, B. C. Song. Vision transformer for small-size datasets, 2021.

[69] Hendrycks, D., K. Gimpel. Gaussian error linear units (gelus), 2016.

[70] Ba, J. L., J. R. Kiros, G. E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016.

[71] Krizhevsky, A., G. Hinton. Learning multiple layers of features from tiny images. Master’s

thesis, Department of Computer Science, University of Toronto, 2009.

[72] Loshchilov,

I., F. Hutter.

Decoupled weight decay regularization.

arXiv preprint

arXiv:1711.05101, 2017.

[73] Zagoruyko, S., N. Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146,

2016.

[74] Zhang, H., M. Cisse, Y. N. Dauphin, et al. mixup: Beyond empirical risk minimization. In

International Conference on Learning Representations. 2018.

[75] Yun, S., D. Han, S. J. Oh, et al. Cutmix: Regularization strategy to train strong classiﬁers with
localizable features. In Proceedings of the IEEE/CVF international conference on computer
vision, pages 6023–6032. 2019.

[76] Cubuk, E. D., B. Zoph, D. Mane, et al. Autoaugment: Learning augmentation strategies from
data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 113–123. 2019.

[77] Gao, S.-H., M.-M. Cheng, K. Zhao, et al. Res2net: A new multi-scale backbone architecture.
IEEE transactions on pattern analysis and machine intelligence, 43(2):652–662, 2019.

[78] Radosavovic, I., R. P. Kosaraju, R. Girshick, et al. Designing network design spaces.

In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
10428–10436. 2020.

[79] Liu, Z., H. Mao, C.-Y. Wu, et al. A convnet for the 2020s. arXiv preprint arXiv:2201.03545,

2022.

[80] Chen, C.-F. R., Q. Fan, R. Panda. Crossvit: Cross-attention multi-scale vision transformer for
image classiﬁcation. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 357–366. 2021.

[81] Han, K., A. Xiao, E. Wu, et al. Transformer in transformer. Advances in Neural Information

Processing Systems, 34, 2021.

[82] Chu, X., Z. Tian, Y. Wang, et al. Twins: Revisiting the design of spatial attention in vision

transformers. Advances in Neural Information Processing Systems, 34, 2021.

[83] Touvron, H., M. Cord, A. Sablayrolles, et al. Going deeper with image transformers.

In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 32–42.
2021.

[84] Zhang, P., X. Dai, J. Yang, et al. Multi-scale vision longformer: A new vision transformer for
high-resolution image encoding. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 2998–3008. 2021.

[85] Li, J., Y. Yan, S. Liao, et al. Local-to-global self-attention in vision transformers. arXiv preprint

arXiv:2107.04735, 2021.

[86] Yang, J., C. Li, P. Zhang, et al. Focal self-attention for local-global interactions in vision

transformers. arXiv preprint arXiv:2107.00641, 2021.

14

[87] Szegedy, C., V. Vanhoucke, S. Ioffe, et al. Rethinking the inception architecture for computer
vision. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 2818–2826. 2016.

[88] Zhong, Z., L. Zheng, G. Kang, et al. Random erasing data augmentation. In Proceedings of the

AAAI conference on artiﬁcial intelligence, vol. 34, pages 13001–13008. 2020.

15

A Pseudo Code of Dynamic Hybrid Vision Transformer

A.1 Pseudo Code of the Dynamic Aggregation Feed Forward (DAFF)

class DAFF ( nn . Module ) :

def __init__ ( self , in_dim , hid_dim , out_dim , kernel_size = 3 ) :

self . conv1 = nn . Conv2d ( in_dim , hid_dim , kernel_size =1 ,

stride =1 , padding = 0 )

self . conv2 = nn . Conv2d (

hid_dim , hid_dim , kernel_size =3 , stride =1 ,
padding = ( kernel_size - 1 ) // 2 , groups = hid_dim )

self . conv3 = nn . Conv2d ( hid_dim , out_dim , kernel_size =1 ,

stride =1 , padding = 0 )

self . act = nn . GELU ()
self . squeeze = nn . Adap tiveAvgP ool2d (( 1 , 1 ) )
self . compress = nn . Linear ( in_dim , in_dim // 4 )
self . excitation = nn . Linear ( in_dim // 4 , in_dim )
self . bn1 = nn . BatchNorm2d ( hid_dim )
self . bn2 = nn . BatchNorm2d ( hid_dim )
self . bn3 = nn . BatchNorm2d ( out_dim )

def forward ( self , x ) :

B , N , C = x . size ()
cls_token , tokens = torch . split (x , [1 , N - 1 ] , dim = 1 )
x = tokens . reshape (B , int ( math . sqrt ( N - 1 ) ) ,

int ( math . sqrt ( N - 1 ) ) , C ) . permute (0 , 3 , 1 , 2 )

x = self . act ( self . bn1 ( self . conv1 ( x ) ) )
x = x + self . act ( self . bn2 ( self . conv2 ( x ) ) )
x = self . bn3 ( self . conv3 ( x ) )

weight = self . squeeze ( x ) . flatten ( 1 ) . reshape (B , 1 , C )
weight = self . excitation ( self . act ( self . compress ( weight ) ) )
cls_token = cls_token * weight
tokens = x . flatten ( 2 ) . permute (0 , 2 , 1 )
out = torch . cat (( cls_token , tokens ) , dim = 1 )
return out

2

2Code is modiﬁed from https://github.com/coeusguo/ceit

16

A.2 Pseudo Code of the Sequential Overlapping Patch Embedding (SOPE)

def conv3x3 ( in_dim , out_dim ) :

return torch . nn . Sequential (

nn . Conv2d ( in_dim , out_dim , kernel_size =3 , stride =2 , padding = 1 ) ,
nn . BatchNorm2d ( out_dim )

)

class Affine ( nn . Module ) :

def __init__ ( self , dim ) :

self . alpha = nn . Parameter ( torch . ones ( [1 , dim , 1 , 1 ] ) )
self . beta = nn . Parameter ( torch . zeros ( [1 , dim , 1 , 1 ] ) )

def forward ( self , x ) :

x = x * self . alpha + self . beta
return x

class SOPE ( nn . Module ) :

def __init__ ( self , patch_size , embed_dim ) :

self . pre_affine = Affine ( 3 )
self . post_affine = Affine ( embed_dim )
if patch_size [ 0 ] = = 16 :

self . proj = torch . nn . Sequential (
conv3x3 (3 , embed_dim // 8 , 2 ) ,
nn . GELU () ,
conv3x3 ( embed_dim // 8 , embed_dim // 4 , 2 ) ,
nn . GELU () ,
conv3x3 ( embed_dim // 4 , embed_dim // 2 , 2 ) ,
nn . GELU () ,
conv3x3 ( embed_dim // 2 , embed_dim , 2 ) ,
)

elif patch_size [ 0 ] = = 4 :

self . proj = torch . nn . Sequential (
conv3x3 (3 , embed_dim // 2 , 2 ) ,
nn . GELU () ,
conv3x3 ( embed_dim // 2 , embed_dim , 2 ) ,
)

elif patch_size [ 0 ] = = 2 :

self . proj = torch . nn . Sequential (
conv3x3 (3 , embed_dim , 2 ) ,
nn . GELU () ,
)

def forward ( self , x ) :

B , C , H , W = x . shape
x = self . pre_affine ( x )
x = self . proj ( x )
x = self . post_affine ( x )
Hp , Wp = x . shape [ 2 ] , x . shape [ 3 ]
x = x . flatten ( 2 ) . transpose (1 , 2 )
return x

3

3Code is modiﬁed from https://github.com/facebookresearch/xcit

17

A.3 Pseudo Code of the Head-Interacted Multi-Head Self-Attention (HI-MHSA)

class Attention ( nn . Module ) :

def __init__ ( self , dim , num_heads = 8 ) :

super () . __init__ ()
self . num_heads = num_heads
head_dim = dim // num_heads
self . scale = head_dim ** - 0 . 5
self . qkv = nn . Linear ( dim , dim * 3 , bias = True )
self . proj = nn . Linear ( dim , dim )
self . act = nn . GELU ()
self . ht_proj = nn . Linear ( head_dim , dim , bias = True )
self . ht_norm = nn . LayerNorm ( head_dim )
self . pos_embed = nn . Parameter (

torch . zeros (1 , self . num_heads , dim ) )

def forward ( self , x ) :
B , N , C = x . shape

# head token
head_pos = self . pos_embed . expand ( x . shape [ 0 ] , -1 , - 1 )
ht = x . reshape (B , -1 , self . num_heads , C // self . num_heads ) .

permute (0 , 2 , 1 , 3 )

ht = ht . mean ( dim = 2 )
ht = self . ht_proj ( ht )

. reshape (B , -1 , self . num_heads , C // self . num_heads )

ht = self . act ( self . ht_norm ( ht ) ) . flatten ( 2 )
ht = ht + head_pos
x = torch . cat ( [x , ht ] , dim = 1 )

# common MHSA
qkv = self . qkv ( x ) . reshape (B , N + self . num_heads , 3 ,

self . num_heads , C // self . num_heads )
. permute (2 , 0 , 3 , 1 , 4 )

q , k , v = qkv [ 0 ] , qkv [ 1 ] , qkv [ 2 ]
attn = ( q @ k . transpose ( -2 , - 1 ) ) * self . scale
attn = attn . softmax ( dim = - 1 )
attn = self . attn_drop ( attn )
x = ( attn @ v ) . transpose (1 , 2 ) . reshape (B , N + self . num_heads , C )
x = self . proj ( x )

# split , average and add
cls , patch , ht = torch . split (x , [1 , N -1 , self . num_heads ] , dim = 1 )
cls = cls + torch . mean ( ht , dim =1 , keepdim = True )
x = torch . cat ( [ cls , patch ] , dim = 1 )

return x

18

B CIFAR-100 Dataset

B.1 Fine-tuning on CIFAR-100

We analyze ﬁne-tuning results in this section. All the models are pre-trained on ImageNet-1K [26]
only and then ﬁne-tuned on CIFAR-100 [71] datasets. Results are shown in Table 7. We cite the
reported results from corresponding papers. When ﬁne-tuning our DHVT, we use AdamW optimizer
with cosine learning rate scheduler and 2 warm-up epochs, a batch size of 256, an initial learning rate
of 0.0005, weight decay of 1e-8, and ﬁne-tuning epochs of 100. We ﬁne-tune our model on the image
size of 224×224 and we use a patch size of 16, head numbers of 3 and 6 for DHVT-T and DHVT-S
respectively, the same as the pre-trained model on ImageNet-1K.

Table 7: Pretrained on ImageNet-1K and then ﬁne-tuned on the CIFAR-100 (top-1 accuracy, 100
ﬁne-tuning epochs). "FT Epochs" denotes ﬁne-tuning epochs, and "Img Size" denotes the input
image size in ﬁne-tuning.

Method

GFLOPs

ImageNet-1K FT Epochs

Img Size CIFAR-100 Acc (%)

ResNet-50 [66]
ViT-B/16 [14]
ViT-L/16 [14]
T2T-ViT-14 [66]
Swin-T [66]
DeiT-B [42]
DeiT-B ↑384 [42]
CeiT-T [15]
CeiT-T ↑384 [15]
CeiT-S [15]
CeiT-S ↑384 [15]

DHVT-T (Ours)
DHVT-S (Ours)

3.8
18.7
65.8
5.2
4.5
17.3
52.8
1.2
3.6
4.5
12.9

1.2
4.7

-
77.9
76.5
81.5
81.3
81.8
83.1
76.4
78.8
82.0
83.3

76.5
82.3

100
10000
10000
100
100
7200
7200
100
100
100
100

100
100

224
384
384
224
224
224
384
224
384
224
384

224
224

85.44
87.13
86.35
87.33
88.22
90.8
90.8
88.4
88.0
90.8
90.8

86.73
88.87

From Table 7, we can see that our model has competitive transferring performance with Swin
Transformer, T2T-ViT. We fail in competing with DeiT [42] maybe because we only ﬁne-tuned
our model for 100 epochs. The longer epochs experiments are left for the future. And we also fail
to compete with CeiT [15] under comparable computational complexity. We consider that maybe
our model introduced too many inductive biases so that the ﬁne-tuning performance is constrained
also. However, the target of our method is mainly on train-from-scratch on small datasets. Thus the
ﬁne-tuning results are not so important in our consideration. And we can also see that we achieve
85.68 accuracy when training from scratch on CIFAR-100 only, as we reported in the main part
of this paper. Such a result even outperforms ResNet-50 pre-trained on ImageNet-1k, which only
reaches 85.44 accuracy when ﬁne-tuning. DHVT can beat the pre-trained and ﬁne-tuned ResNet-50
without any pre-training, suggesting the signiﬁcant performance of our model.

B.2 More Ablation Studies

In this part, we present more ablation studies on the minor operation or module variant in our
method. The experimental setup is the same as the main paper. We present ablation study results on
DAFF, SOPE, the number of attention heads, and the choice between Batch Normalization and Layer
Normalization.

B.2.1 Minor Operations in DAFF

We ﬁrst present the ablation study on the minor operation in DAFF to show why we ﬁnally adopt
such a structure. The inﬂuences of minor operation on the Feed Forward Network (FFN) in the
original ViT are shown in Table 8. (1) "Split CLS" means that the class token is split away from
other patch tokens and it will pass through FFN without any computation. And we can see a +1.17
improvement in accuracy. It may imply that the class token contains the global information of the
image and that the feature carried by the class token is quite different from other patch tokens. Thus

19

if both class token and patch tokens are projected by the same FFN, it would be hard to train the
model. (2) "Agg on CLS" means using Squeeze-Excitation [4] operation to dynamically aggregate
information from patch tokens and re-calibrate class token channel-wise. And we can see a further
improvement in the accuracy. This is a reasonable way to re-calibrate the class token because it
carries global information. We then use the global average pooling to gather information from patch
tokens and reﬁne the feature to enhance class token. (3) "AvgPool" means using Average Pooling
instead of Depth-wise Convolution (DWCONV)[36] to process the patch tokens. And we can see a
non-trivial improvement. Average Pooling can be seen as a deteriorated version of DWCONV. The
improvement brought by AvgPool implies that attending to and aggregating neighboring features is
indeed helpful and necessary.

Table 8: Ablation study on the Minor Operation on FFN of ViT

Variants

Acc (%)

ViT
ViT + Split CLS
ViT + Split CLS + Agg on CLS
ViT + Split CLS + AvgPool

67.59 (+0.00)
68.76 (+1.17)
69.34 (+1.75)
70.48 (+2.89)

We then conduct an ablation study on the inﬂuence brought by minor operations upon the full version
of DHVT. The results are shown in Table 9. The baseline DHVT-T with a patch size of 4 achieves
80.85 accuracy on CIFAR-100. (1) When the dynamic aggregation operation re-calibrates all the
tokens, rather than only the class token, the performance will drop drastically. (2) When removing
the shortcut alongside the DWCONV, the performance also drops. The shortcut retains the original
feature, and thus the feature holds global and local information simultaneously.

Table 9: Ablation study on the Minor Operation in DAFF

Variants

Acc (%)

DHVT
DHVT (Agg on all token)
DHVT (w/o shortcut)

80.85 (+0.00)
78.34 (-2.51)
80.14 (-0.71)

B.2.2 Afﬁne Operation in SOPE

We introduce two afﬁne transformations before and after the sequential convolution layers. The
pre-afﬁne is done by transforming the original input and the post-afﬁne is to adjust the feature after
the convolution sequence. This method renders stable training results on the CIFAR-100 dataset.
If we remove such an operation, the average performance will drop from 80.90 to 80.72. On the
Vanilla ViT with SOPE, the performance is 73.68, and if removing the two afﬁne transformations, the
accuracy will drop to 73.42.

B.2.3 The Number of Attention Head

It is well-known that DeiT-Tiny and DeiT-Small have 3 and 6 heads respectively. However, this
parameter is chosen by the experiment on ImageNet. Similarly, we set the number of heads as 4
and 8 for our DHVT-T and DHVT-S based on the experiment on CIFAR-100. The results can be
seen in Table 10. To be compatible with scalability, we adopt 4 heads in DHVT-T and 8 heads in
DHVT-S. We hypothesize this is because each attribute of the object in CIFAR does not need too
many channels for representation. So we keep the number of channels in each head less than usual.

B.2.4 The choice of Batch Normalization and Layer Normalization

Here in our work, BatchNorm is adopted at two positions: SOPE and DAFF. We use (A-B) to
denote normalization choice, where A is the normalization operation in SOPE and B is in DAFF.
The experiment is conducted on DHVT-T. We evaluate the inﬂuence from a batch size of 128, 256
and 512. From Table 11, we can see that using BN is indeed sensitive to the batch size, while its
performance is always superior to LN. From our point of view, the vanilla ViT adopts LN before
Multi-Head Self-Attention (MHSA) and Multi-layer Perceptron (MLP), aiming at regularizing each
token on channel dimension. This is important because MHSA uses dot-product operation, and LN

20

Table 10: The inﬂuence of number of attention head on CIFAR-100 dataset.

Method

#Head Acc (%)

ViT-T

DHVT-T

DHVT-S

3
4

3
4

4
6
8

66.50
67.59

80.92
80.98

82.27
82.59
82.82

helps control the value of the query and key, avoiding extreme values. In our work, LN is also adopted
at the same place, before MHSA and MLP. Further, we use convolution operation, and its output
should be regularized in terms of spatial dimension. When we replace BN with LN, the feature will
be reshaped into a sequence style, which may ignore the spatial relations. So it is more suitable to
use BN along with convolution for higher recognition accuracy.

Table 11: The inﬂuence of the choice between Batch Normalization (BN) and
Layer Normalization (LN) on CIFAR-100. "Var" and "BS" denote the variant
and the batch size respectively.

Acc (%) BS

Var

BN-BN

BN-LN

LN-BN

LN-LN

128

256

512

79.69

80.31

80.98

78.69

79.55

79.25

79.24

80.02

80.26

78.46

79.09

78.90

B.3 Training Efﬁciency on CIFAR-100

Figure 5: (a) Training Efﬁciency of applying head token only. (b) Training efﬁciency of applying
SOPE and DAFF and with or without absolute positional embedding.

In this section, we show the training efﬁciency of our method when applying each module. All the
experimental settings are the same as in the ablation study section. The baseline module is DeiT-Tiny
with 4 heads. Here we denote "without absolute positional embedding" as "woabs" and "wabs"
denotes the opposite. From Fig. 5 (a), we can see that introducing head token into the baseline can
facilitate the overall training process and reaches higher accuracy, proving the effectiveness of our
novel head token design. And from Fig. 5 (b), we can see that both SOPE and DAFF can improve the
whole training, and their combination has a positive inﬂuence on the model.

21

(a)(b)Figure 6: Training efﬁciency of head token when both SOPE and DAFF are applied and absolute
positional embedding is removed.

In addition, in Fig. 6, we use "DHVT full" to represent the full version of our model, including SOPE,
DAFF and head tokens, while removing absolute positional embedding. In such a circumstance, head
token is still able to give rise to the performance during most of the training epochs, and the ﬁnal
result is also a little higher than the one without head token.

B.4 Visualization on CIFAR-100

To further understand the feature interaction style in our proposed model, we provide more visualiza-
tion results in this section. First, we visualize the averaged attention map of all the tokens, including
class token, patch tokens and head tokens on the 2nd, 5th, 8th and 11th encoder layers. We provide
three example input images in total. Second, we visualize the attention of head tokens to patch tokens
in their corresponding head on the 2nd, 5th, 8th and 11th encoder layers. The model we visualize
here is DHVT-T training from scratch on the CIFAR-100 dataset, which contains 4 attention heads in
each layer and thus the corresponding number of head tokens is 4. And patch size is set to 4 here.

Figure 7: Visualization on CIFAR-100. (a) Averaged attention maps. (b) Attention of head tokens to
patch tokens in the corresponding heads.

22

(a) attention maps(b) attention of head tokens to patch tokensLayer 2Layer 5Layer 8Layer 11head token 1head token 2head token 3head token 4Figure 8: Visualization on CIFAR-100. (a) Averaged attention maps. (b) Attention of head tokens to
patch tokens in the corresponding heads.

Figure 9: Visualization on CIFAR-100. (a) Averaged attention maps. (b) Attention of head tokens to
patch tokens in the corresponding heads.

23

(a) attention maps(b) attention of head tokens to patch tokensLayer 2Layer 5Layer 8Layer 11head token 1head token 2head token 3head token 4(a) attention maps(b) attention of head tokens to patch tokensLayer 2Layer 5Layer 8Layer 11head token 1head token 2head token 3head token 4Note that head tokens are concatenated behind patch tokens, so the right-hand side of attention maps
represents the attention from all the tokens to head tokens. From the above results in Figure 7,8,9, we
can summarize two attributes that head tokens brought to the model. First, in the lower layers, such
as the 2nd layer, the model tends to attend to neighboring features and interacts with head tokens.
Going deeper, such as in the 5th layer, attention is scattered around all tokens and head tokens do not
receive much attention here. In higher layers, like in the 8th layer, attention focus on some of the
patch tokens and now head tokens receive more attention than in the 5th layer. Finally, in the layers
near the output layer, such as the 11th layer, patch tokens do not focus too much on head tokens, and
all the tokens converge their attention to the most prominent patch tokens.

Second, each head token represents a different representation as we visualized above. When head
tokens participate in attention calculation, they help the interaction of different representations, fusing
poor representation encoded in different channel groups into a strong integral representation. The
results are similar on ImageNet-1K and we provide a discussion later.

C DomainNet Dataset

C.1 Examples of DomainNet

In this part, we visualize some example images in the DomainNet [37] datasets as in Fig. 10. These
datasets have a domain shift from traditional natural image datasets like ImageNet-1K and CIFAR.
Also because of the scarce training data, models are hard to train from scratch on such datasets.
However, our proposed DHVT can address the issue with satisfactory results in both train-from-
scratch and pretrain-ﬁnetune scenario. Under a comparable amount of computational complexity, our
models exhibit non-trivial performance gain compared with baseline models on all of the six datasets.

Table 12: The statistics of training datasets. We report the train and test size of each DomainNet
dataset, including the number of classes. We also show the average images per class in the training
set.

Dataset

Train size Test size Classes Average images per class

ClipArt [37]
Sketch [37]
Painting [37]
Infograph [37]
Real [37]
Quickdraw [37]

33525
48212
50416
36023
120906
120750

14604
20916
21850
15582
52041
51750

345
345
345
345
345
345

97
140
146
104
350
350

Figure 10: Visualization of examples in each DomainNet dataset.

24

angelClipartbirthday cakeclarinetsnorkelceiling fansraccoonPaintingSketchbeachInfographrabbittoasterduckRealbowtiepeasQuickdrawC.2 Comprehensive Train-from-scratch Results on DomainNet

In this section, we present the whole performance comparison of train-from-scratch over all the
DomainNet datasets between our model and baseline models. The training scheme is shown in the
main paper in Section 4.1. From Table 13, our method demonstrates a consistent performance gap
over baseline models.

Table 13: Comprehensive Results on DomainNet. All the models are trained from scratch for 300
epochs under the same training schedule. The training resolution is 224×224. "C", "P", "S", "I",
"R", "Q" denotes the accuracy of ClipArt, Painting, Sketch, Infograph, Real and Quickdraw datasets
respectively. And we adopt ResNeXt-50, 32x4d variant.

Method

#Params GFLOPs

C

P

S

I

R

Q

ResNet-50 [1]
ResNeXt-50 [40]
CvT-13 [18]

DHVT-T
DHVT-S

24.2M
23.7M
19.7M

6.1M
23.8M

3.8
4.3
4.5

1.2
4.7

71.90
72.93
69.77

71.73
73.89

64.36
64.37
61.57

63.34
66.08

67.45
68.52
66.16

66.60
68.72

32.40
34.85
30.07

32.60
35.11

81.51
82.15
81.48

81.31
83.64

74.19
73.73
72.49

74.41
74.38

C.3 Fine-tuning on DomainNet

We analyze the ﬁne-tuning results on DomainNet datasets in this section. All the models are pre-
trained on ImageNet-1K only and then ﬁne-tuned on Clipart, Painting, and Sketch. Results are shown
in Table 14. We cite the reported results from corresponding papers. Note that the ﬁne-tuning epochs
in baseline models are 100, the same as we use. When ﬁne-tuning our DHVT, we use AdamW
optimizer with cosine learning rate scheduler and 2 warm-up epochs, a batch size of 256, an initial
learning rate of 0.0005, weight decay of 1e-8, and ﬁne-tuning epochs of 100. We ﬁne-tune our model
on the image size of 224×224 and we use a patch size of 16, head numbers of 3 and 6 for DHVT-T
and DHVT-S respectively, the same as the pre-trained model on ImageNet-1K.

Table 14: Pretrained on ImageNet-1K and then ﬁne-tuned on the DomainNet (top-1 accuracy (%),
100 ﬁne-tuning epochs). The ImageNet-1K column shows the accuracy of pretrained model on
ImageNet-1K.

Method

GFLOPs

ImageNet-1K Clipart

Painting

Sketch

ResNet-50 [66]
T2T-ViT-14 [66]
Swin-T [66]

DHVT-T (Ours)
DHVT-S (Ours)

3.8
5.2
4.5

1.2
4.7

-
81.5
81.3

76.5
82.3

75.22
74.59
73.51

77.88
80.06

66.58
72.29
72.99

72.05
74.18

67.77
72.18
72.37

70.79
73.32

From Table 14, we can see that our models show better performance than baseline methods ResNet-50,
Swin Transformer, and T2T-ViT. Especially on Clipart, our DHVT-S reaches more than 80 accuracy,
showing a signiﬁcantly better performance than baseline methods. Our tiny model achieves compara-
ble and even better accuracy than T2T-ViT and Swin Transformer with much lower computational
complexity on Clipart and Painting. From the main part of this paper, the performance of training
from scratch of DHVT-S is 68.72, as shown in the main part of this paper, which outperforms the
ﬁne-tuning result of ResNet-50, exhibiting the train-from-scratch capacity of our method.

D ImageNet-1K Dataset

D.1 Comparison on ImageNet-1K

We conduct experiments on ImageNet-1K dataset to test the performance of our proposed DHVT
on the common medium dataset. From the above Table 15, we can see that with fewer parameters
and comparable computational complexity, our DHVT achieves state-of-the-art results compared to

25

Table 15: Performance comparison of different methods on ImageNet-1K. All models are trained
from random initialization.

Method

#Params

Image Size GFLOPs Top-1 Acc (%)

RegNetY-800MF [78]
RegNetY-4.0GF [78]
ConvNeXt-T [79]

6.3M
20.6M
29M

T2T-ViT-7 [52]
DeiT-T [42]
PiT-Ti [34]
ConViT-Ti [29]
CrossViT-Ti [80]
TNT-T [81]
LocalViT-T [30]
ViTAE-T [59]
CeiT-T [15]
DHVT-T (Ours)

DeiT-S [42]
PVT-S [16]
PiT-S [34]
CrossViT-S [80]
PVT-Medium [16]
Conformer-Ti [31]
Swin-T [17]
ConViT-S [29]
TNT-S [81]
T2T-ViT-14 [52]
NesT-T [35]
CvT-13 [18]
Twins-SVT-S [82]
CaiT-XS24 [83]
CoaT-Lite Small [62]
CeiT-S[15]
ViL-S[84]
PVTv2-B2[54]
ViTAE-S[59]
LG-T[85]
Focal-T[86]
DHVT-S (Ours)

4.3M
5.7M
4.9M
5.7M
6.9M
6.2M
5.9M
4.8M
6.4M
6.2M

22.1M
24.5M
23.5M
26.7M
44.2M
23.5M
29.0M
27.8M
23.8M
21.5M
17.0M
20.0M
24.0M
26.6M
20.0M
24.2M
24.6M
25.4M
23.6M
32.6M
29.1M
24.1M

224
224
224

224
224
224
224
224
224
224
224
224
224

224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224
224

0.8
4.0
4.5

1.2
1.1
0.7
1.4
1.6
1.4
1.3
1.5
1.2
1.2

4.3
3.8
2.9
5.6
6.7
5.2
4.5
5.4
5.2
5.2
5.8
4.5
2.8
5.4
4.0
4.5
4.9
4.0
5.6
4.8
4.9
4.7

76.3
79.4
82.1

71.7
72.2
72.9
73.1
73.4
73.6
74.8
75.3
76.4
76.5

79.8
79.8
80.9
81.0
81.2
81.3
81.3
81.3
81.3
81.5
81.5
81.6
81.7
81.8
81.9
82.0
82.0
82.0
82.0
82.1
82.2
82.3

recent CNNs and ViTs. Both of our models are trained from scratch on ImageNet-1K datasets, with
an image size of 224×224, patch size of 16, optimizer of AdamW, and base learning rate of 0.0005
following cosine learning rate decay, weight decay of 0.05, a warm-up epoch of 10, batch size of 512.
All the data-augmentations and regularizations methods follow Deit [42], including random cropping,
random ﬂipping, label-smoothing [87], Mixup [74], CutMix [75] and random erasing [88].

Our DHVT-T reaches 76.5 accuracy with only 6.2M parameters, while DHVT-S achieves 82.3
accuracy with only 24.1M parameters. Our model not only outperforms the best non-hierarchical
vision transformer CeiT [15] but also shows competitive performance to most of the hierarchical
vision transformers like Swin Transformer [17] and hybrid architecture like ViTAE-S [59]. We also
show better performance than recent strong CNNs RegNet [78]and ConvNeXt [79]. We achieve such
results with much fewer parameters than existing methods, while our computational complexity is
also higher than theirs. This is a kind of mixed blessing. On one hand, our method can be seen as
using fewer parameters to conduct comprehensive and sufﬁcient computation. On the other hand,
such an amount of computation is a huge burden for both training and testing. We hope to reduce the
computational burden in future research while maintaining the same performance.

26

D.2 Visualization on ImageNet-1K

Figure 11: Averaged attention maps from DHVT-S training from scratch on ImageNet-1K.

We further visualize the attention maps of our proposed model training from scratch on ImageNet-1K
only. Here the model is DHVT-S with 6 attention heads and a patch size of 16. Note that head tokens
are concatenated behind patch tokens, so the right-hand side of attention maps represents the attention
from all the tokens to head tokens. With the introduction of head tokens, we can understand the
feature extraction and representation mechanism in our model.

In the input encoder layer, i.e. the 1st layer, all the tokens focus on themselves and the head tokens.
And in the early stage, i.e. from the 2nd to 6th layers, all the tokens focus more on themselves and do
not attend too much to head tokens. Further, in the middle stage, i.e. from the 7th to 9th layers, head
tokens draw more attention from other tokens. Finally in the late stage, i.e. in the 10th, 11th, and
12th layers, attention is more on prominent patch tokens.

From such attention style, we can conclude the feature extraction and representation mechanism as
the Early stage focuses on local and neighboring features, extracting low-level ﬁne-grained features.
Then feature representations interact and fuse to generate a strong enough representation in the middle
stage. The representation in each token is enhanced by such interaction. And in the late stage, the
model focus on the most prominent patch tokens to extract information for ﬁnal classiﬁcation.

In future research, it may be possible to only apply head token design in the middle stage of vision
transformers to save computation costs. We hope this visualization of the mechanism will inspire
more wonderful architectures in the future.

Table 16: Results on 224×224 resolution. All the models are trained from scratch for 100 epochs
under the same training schedule.

Method

#Params GFLOPs CIFAR-100 Clipart

Painting

Sketch

ResNet-50+Ldrloc [66]
SwinT+Ldrloc [66]
CvT-13+Ldrloc [66]
T2T-ViT+Ldrloc [66]

DHVT-T
DHVT-S

21.2M
24.1M
19.6M
21.2M

6.0M
23.7M

3.8
4.3
4.5
4.8

1.2
4.7

72.94
66.23
74.51
68.03

74.78
78.64

63.93
47.47
60.64
52.36

58.94
64.75

53.52
41.86
55.26
42.78

52.64
56.42

59.62
38.55
57.56
51.95

56.66
61.35

E Results on larger resolution

In order to make a comparison with paper [66], we conduct experiments under the same training
scheme. The models are trained from scratch on CIFAR-100, Clipart, Painting and Sketch for 100
epochs and the training resolution is 224×224. The patch size for SwinT, CvT, T2T-ViT and our

27

Layer 1Layer 2Layer 3Layer 4Layer 5Layer 6Layer 7Layer 8Layer 9Layer 10Layer 11Layer 12DHVT is set to 16. And the following Table 16 demonstrates the performance superiority of our
method.

F Model Variants

We present the variants and architecture parameters of our proposed model in this section. Note that
all the models remove the absolute positional embedding. For the CIFAR-100 dataset, the image size
is 32×32, and for DomainNet and ImageNet it is 224×224. In Table 17, "MLP" represents MLP
projection ratio and "S&E" is the reduction ratio in the squeeze-excitation operation.

Method

Dataset

Patch

DAFF
MLP

S&E

#heads

depth Dim #Params GFLOPs

Table 17: Model variants of DHVT

DHVT-T
DHVT-T
DHVT-S
DHVT-S

DHVT-T
DHVT-S

CIFAR
CIFAR
CIFAR
CIFAR

Domain
Domain

DHVT-T ImageNet
ImageNet
DHVT-S

4
2
4
2

16
16

16
16

4
4
4
4

4
4

4
4

4
4
4
4

4
4

4
4

4
4
8
8

4
6

3
6

12
12
12
12

12
12

12
12

192
192
384
384

192
384

192
384

6.0M
5.8M
23.4M
22.8M

6.1M
23.8M

6.2M
24.1M

0.4
1.4
1.5
5.6

1.2
4.7

1.2
4.7

28


