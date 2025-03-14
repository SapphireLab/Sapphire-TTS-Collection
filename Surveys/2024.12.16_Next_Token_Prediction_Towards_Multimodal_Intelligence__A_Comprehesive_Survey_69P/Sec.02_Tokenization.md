# 2·MultiModal Tokenization: 多模态分词


<table><tr><td width="50%">

Tokenization is the first and a fundamental step for multimodal sequential modeling under the next token prediction framework.
It decomposes information from various sources, such as images, videos, and audio clips, into a sequence of minimal, manageable units known as tokens for the NTP model to learn.
Table.02 provides an overview of the tokenizers used across various modalities in recent research.

Despite being derived from various modalities, these tokenization methods can all be categorized into two prototypes: **discrete tokenization**  and **continuous tokenization**.
In this section, we will initially introduce the general definition and basics techniques of training multimodal tokenizers ([Section.2.1](#Section.2.1)), then the fundamentals and applications of discrete tokens ([Section.2.2](#Section.2.2), [Section.2.3](#Section.2.3)) and continuous tokens ([Section.2.4](#Section.2.4), [Section.2.5](#Section.2.5)) in NTP framework.

</td><td>

在 **Next Token Prediction (NTP)** 框架下, Tokenization 是多模态序列建模的第一且基础的步骤.
它将不同来源的信息 (如图像, 视频, 音频片段) 分解为最小的可管理的单元即 Token 组成的序列, 以供 NTP 模型学习.
表格 02 总结了近期研究中不同模态所使用的 Tokenizers.

尽管来源于各种模态, 这些 Tokenization 方法可以分为两种原型: **离散 Tokenization** 和**连续 Tokenization**.

在本节中:
- [2.1 节](#Section.2.1)首先介绍一般定义和训练多模态 Tokenizers 的基础技术;
- [2.2 节](#Section.2.2)介绍离散 Tokens 的基本原理;
- [2.3 节](#Section.2.3)介绍离散 Tokens 的应用;
- [2.4 节](#Section.2.4)介绍连续 Tokens 的基本原理;
- [2.5 节](#Section.2.5)介绍连续 Tokens 的应用.

</td></tr></table>

## 2.1·Tokenization of Different Modalities: 不同模态的分词

<a id="Section.2.1"></a>

<table><tr><td width="50%">

We first define the tokenization process as a function $f$ that maps a sample $x$ from the raw multimodal space $X$ to a representation $z$ in the tokenizer's output representation space $Z_f$.

$$
f(x) = z,\tag{1}
$$

where $x\in X$ and $z\in Z_f$.

</td><td>

我们首先定义 Tokenization 过程为一个函数 $f$, 它将来自原始多模态空间 $X$ 的样本 $x$ 映射到 Tokenizer 的输出表示空间 $Z_f$ 中的表示 $z$.

$$
f(x) = z,\ x\in X,\ z\in Z_f.\tag{1}
$$

</td></tr></table>

### 2.1.1·Tokenizer Type: 分词器类型


<table><tr><td width="50%">

As illustrated in Fig.04, tokenizers for multimodal information can be categorized into two types: discrete and continuous.
This classification is based on how tokens are derived from the original data.
Both tokenization methods encode the original information into a latent representation space, but they differ in their approach.

Discrete tokenization performs quantization on the latent space, utilizing a fixed-size, discrete space similar to the vocabulary of language models.
In contrast, continuous tokenization does not involve quantization, resulting in a much larger representation space.

**Discrete**

In Equation.1, a discrete token implies that the representation space $Z_f$ comprises a finite number of discrete symbols.
The output space is called the codebook $C = \{c_1, c_2, \cdots, c_N\}$, where $c_i \in \mathbb{R}^0$, and each representation $z$ is composed of codes from this codebook, i.e., $z = \{z_1, z_2, \cdots, z_n\}$ with $z_i \in C$.
Language tokens are inherently discrete because they originate from a finite vocabulary.
Each word or subword unit is mapped to a unique token from this predefined set.
In contrast, modalities such as audio and images exist in continuous, high-dimensional spaces.
To process these modalities within the same framework (i.e., NTP) as for discrete language tokens, they need to be transformed into a discrete representation.

Quantization is a process that maps values from a continuous space to a discrete space, typically resulting in a much smaller representation space.
It is a default operation when a discrete representation is desired for tokenizing multimodal information.
Quantization is often combined with auto-encoder techniques to reduce the size of the latent space.
Typical examples include VQ-series tokenizers such as [VQ-VAE](../../Modules/VQ/2017.11.02_VQ-VAE.md) and [VQGAN](../../Modules/VQ/2020.12.17_VQGAN.md), which inherently feature discrete representations.
Details of the quantization process are introduced in [Section.2.2](#Section.2.2).

**Continuous**

In contrast to discrete tokenization, continuous tokenization represents data using a continuous space where tokens are derived directly from the data's inherent properties without enforcing quantization into a predefined codebook.
In this approach, the representation space $Z_f$ is not limited to a finite set of predetermined codes; rather, it preserves the continuous nature of the data.
Each token $z$ is sampled from a continuous distribution, allowing for a more nuanced and flexible representation that can capture the subtleties of the input data.
Continuous tokenization is particularly advantageous for modalities that naturally exist in a continuous form and require a rich representational capacity to capture their complex patterns.
For instance, in audio and visual data, continuous representations can effectively retain fine-grained temporal and spatial information that might be lost during discrete tokenization.

</td><td>

如图 04 所示, 多模态信息的 Tokenizer 可以分为两种类型: 离散和连续.
这一分类是基于 Token 如何从原始数据中产生.
两种 Tokenization 方法都将原始信息编码到潜在表示空间中, 但它们的方式不同.

离散 Tokenization 在潜在空间上进行量化, 利用固定大小的离散空间 (类似语言模型的词表).
连续 Tokenization 则与之相反, 不涉及到量化, 使得表示空间更大.

**离散 (Discrete)**

在 Tokenization 函数 $f$ 中, 一个离散 Token 意味着表示空间 $Z_f$ 由有限数量的离散符号组成.
输出空间称为**码本 Codebook** $C=\{c_1,c_2,\cdots,c_N\}$, 其中 $c_i\in \mathbb{R}^0$.
每个表示 $z$ 由该码本中的编码组成, 即 $z=\{z_1,z_2,\cdots,z_n\}, z_i\in C$.

语言 Token 天然地离散, 因为它们来自有限的词表.
每个单词或子词单元被映射到这一预定义集中的唯一 Token.
与此不同, 例如音频和图像等模态则存在于连续高维空间中.
为了在相同框架 (即 NTP) 中如离散语言 Token 一样处理这些模态, 它们需要被转换为离散表示.

量化 (Quantization) 是指将连续空间中的值映射到离散空间的过程, 通常会使得表示空间更小.
当需要对多模态信息进行 Tokenization 以获得离散表示时, 量化是默认操作.
量化通常和自编码器技术结合使用, 以减小潜在空间的大小.

经典示例包括 VQ 系列 Tokenizers (如 [VQ-VAE](../../Modules/VQ/2017.11.02_VQ-VAE.md), [VQGAN](../../Modules/VQ/2020.12.17_VQGAN.md)) 固有地具有离散表示.
量化过程的细节在 [2.2 节](#Section.2.2)中介绍.

**连续 (Continuous)**

和离散 Tokenization 不同, 连续 Tokenization 表示数据使用连续空间, 而 Token 则直接从数据本身的性质中产生而不强行量化到预定义的码本.
在这一方法中, 表示空间 $Z_f$ 不受限于预定义编码的有限集合, 而是保留了数据的连续本质.
每个 Token $z$ 都从连续分布中采样, 允许更具细微差别和灵活的表示, 能够捕捉输入数据的微妙之处.
连续 Tokenization 对于自然以连续形式存在并且需要丰富的表示能力来捕捉其复杂模式的模态特别有利.
例如, 在音频和视觉数据中, 连续表示可以有效地保留细粒度的时空信息, 而在离散 Tokenization 中则可能被削弱.

</td></tr></table>

### 2.1.2·Features of Tokenizers: 分词器特征


<table><tr><td width="50%">

Before diving into different tokenization techniques, we summarize the basic two features (Representation and Reconstruction) that an ideal multimodal tokenizer should possess to achieve better understanding and generation capabilities in the NTP framework.

**Representation Ability**

Effective representation encodes semantically relevant information into the latent space $Z$ while removing redundant information.
This is crucial for various downstream tasks that learn a conditional probability $P(Y|X)$ over the label space $Y$, conditioned on the multimodal input space $X$, by replacing it with $P(Y|Z)$.
Prominent tokenizers known for better representation include language-guided contrastive learning methods such as [CLIP](../../Models/_Basis/2021.02.26_CLIP.md) and fully self-supervised methods like [DINO](../../Models/_Basis/DINO.md).

**Reconstruction Ability**

For generating multimodal information, it is expected that the tokenization function $f$ is invertible or nearly invertible, meaning there is a detokenization function $g$ that can recover the original input from the representation space, satisfying $g(f(x)) = x$ or $g(f(x)) \approx x$.
Notable works that excel in reconstruction include Auto-Encoder (AE) series models such as [Variational Auto-Encoder (VAE)](../../Models/_Basis/VAE.md) and [VQ-VAE](../../Modules/VQ/2017.11.02_VQ-VAE.md).

It is important to note that these abilities are not mutually exclusive; their balance depends on the training techniques used.

</td><td>

在深入讨论不同 Tokenization 技术之前, 我们总结了基础的两个特征 (表示和重构), 一个理想的多模态 Tokenizer 应该具有在 NTP 框架中获得更好的理解和生成能力.

**表示能力 (Representation Ability)**

有效的表示将语义相关的信息编码到潜在空间 $Z$ 中并移除冗余信息.
这对于各种下游任务来说很重要, 它们通过将潜在空间 $Y$ 上以多模态输入空间 $X$ 为条件的条件概率 $P(Y|X)$ 替换为 $P(Y|Z)$ 来学习.

以更好的表示能力著称的卓越 Tokenizer 包括:
- 语言引导的对比学习方法 (如 [CLIP](../../Models/_Basis/2021.02.26_CLIP.md));
- 完全自监督方法 (如 [DINO](../../Models/_Basis/DINO.md)).

**重构能力 (Reconstruction Ability)**

为了生成多模态信息, 我们希望 Tokenization 函数 $f$ 是可逆或几乎可逆的, 意味着有一个 Detokenization 函数 $g$ 可以从表示空间中恢复原始输入, 即 $g(f(x)) = x$ 或 $g(f(x)) \approx x$.

在重构方面值得注意的工作包括自编码器系列 (如[变分自编码器 (VAE)](../../Models/_Basis/VAE.md) 和 [VQ-VAE](../../Modules/VQ/2017.11.02_VQ-VAE.md)).

重要的是需要注意这些能力并不互斥, 它们的平衡取决于所使用的训练技术.

</td></tr></table>

### 2.1.3·Training Methods for Tokenizers: 训练方法


<table><tr><td width="50%">

The training methodologies for tokenizers can be categorized into four groups, based on their respective training objectives: Auto-Encoding, Denoising Auto-Encoding, Supervised Training, and Contrastive Learning, as depicted in Figure.05.
Herein, we provide a summary of the core concepts associated with various tokenizers.

**Auto-Encoding**

Auto-Encoder (AE) is a type of artificial neural network designed to learn efficient data representations.
It consists of two main components: an encoder, which maps input data to a latent space with reduced dimensions, and a decoder, which reconstructs the input data from this latent representation.
The training goal for an Auto-Encoder is to minimize the reconstruction error, ensuring the decoded output closely resembles the original input.
Variants like [Variational Auto-Encoders (VAEs)](../../Models/_Basis/VAE.md) use probabilistic approaches to generate more robust and informative embeddings.
In multimodal generation models, tokenizers trained with auto-encoder methodologies are used to restore the multimodal input from the latent representation.
A special case is [diffusion models [URL]](https://sander.ai/2022/01/31/diffusion.html), which can also be viewed as an Auto-Encoder, enabling generation in a non-autoregressive manner ([MAR](../../Models/CV/2024.06.17_MAR.md)).
Discrete tokens are typically generated by quantizing ([Discrete VAE](../../Models/_Basis/2016.09.07_Discrete_VAE.md)) the continuous data representation within the latent space of auto-encoders.

**Denoising Auto-Encoding**

A Denoising Auto-Encoder (DAE) builds on the basic auto-encoder concept by introducing noise into the input data and training the model to reconstruct the original, noise-free version.
This approach encourages the model to learn robust features capable of handling data corruption, thereby improving its generalization capabilities.
In transformer-based models, a common technique known as Masked Language Modeling ([BERT](../../Models/TextLM/2018.10.11_BERT.md)) involves masking parts of the input tokens and training the model to predict them, which can be viewed as a special type of denoising auto-encoder.
This method has become mainstream across various modalities, popularized in language by [BERT](../../Models/TextLM/2018.10.11_BERT.md), in vision by [BEiT](../../Models/CV/2021.06.15_BEiT.md) and [MAE](../../Models/CV/2021.11.11_MAE.md), and in audio by [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md).

**Supervised Pretraining**

Some tokenizers are pretrained on specific tasks using supervised learning, aiming to acquire task-specific representations through labeled datasets.
These models are initially trained on large-scale datasets to capture specific features of the input data.
In the vision modality, supervised tasks include semantic segmentation, object detection, and depth estimation.
Models trained for these tasks, such as SAM~\citep{sam,samclip}, ViTDet~\citep{vitdet}, and MiDaS~\citep{midas}, are later used in LMMs as tokenizers, like in DeepSeek-VL~\citep{DeepSeek-VL} and Cambrain-1~\citep{Cambrian-1}, to extract diverse visual features from input data.
In the audio modality, Whisper~\citep{radford2023robust} is trained with 680,000 hours of labeled audio data in a weakly supervised manner.
Thanks to its robust and powerful speech feature extraction capabilities, Whisper is widely used in Speech LLMs ([SALMONN](../../Models/SpokenDialogue/2023.10.20_SALMONN.md); [Qwen-Audio](../../Models/SpeechLM/ST2T/2023.11.14_Qwen-Audio.md) [WavLLM](../../Models/SpeechLM/2024.03.31_WavLLM.md)) for extracting speech embeddings.

**Contrastive Learning**

Contrastive Learning is a self-supervised learning method that focuses on learning representations by distinguishing between positive and negative pairs.
The core idea is to bring similar (positive) examples closer together in the representation space while pushing dissimilar (negative) examples further apart.
The items in each pair can belong to the same or different modalities.
For example, [DINO](../../Models/_Basis/DINO.md) uses image-image pairs to enhance vision representation, while [CLIP](../../Models/_Basis/2021.02.26_CLIP.md) employs text-image pairs to improve language alignment within vision representation.

Currently, LMMs that only feature multimodal understanding capabilities, such as InstructBLIP~\cite{dai2023instructblip} [74] and LLaVA~\cite{liu2023llava} [255], opt for tokenizers with superior representation abilities like [CLIP](../../Models/_Basis/2021.02.26_CLIP.md), as they do not require reconstruction of the multimodal information.
Conversely, LMMs supporting multimodal generation capabilities tend to choose VQ-VAE as the tokenizer, exemplified by models like Unified-IO~\cite{lu2022unifiedio} [272], Chameleon~\cite{chameleonteam2024chameleon} [375], Emu3~\citep{Emu3} [407], among others~\citep{wang2024mio, seedllama, wang2022ofa} [128] [402] [411].

</td><td>

如图 05 所示, Tokenizer 的训练方法可以基于对应的训练目标分为四组:
- 自编码 (Auto-Encoding)
- 去噪自编码 (Denoising Auto-Encoding)
- 监督预训练 (Supervised Pretraining)
- 对比学习 (Contrastive Learning)

此处我们提供了和各种 Tokenizer 相关的核心概念的总结.

#### 自编码 (Auto-Encoding)

自编码器 (Auto-Encoder, AE) 是一种人工神经网络的类型, 旨在学习有效的数据表示.
它由两个主要组件组成: 编码器将输入数据映射到具有缩减维度的潜在空间, 解码器从此潜在表示中重构输入数据.
**自编码器的训练目标是最小化重构误差, 确保解码输出与原始输入尽可能相似**.

如[变分自编码器 (Variational Auto-Encoders, VAEs)](../../Models/_Basis/VAE.md) 等变体, 使用概率方法生成更健壮和更具信息的嵌入.

**在多模态生成模型中, 使用自编码方法训练的 Tokenizer 被用于从潜在表示中恢复多模态输入.**

一个特殊的例子是[扩散模型, 也可以视为自编码器](https://sander.ai/2022/01/31/diffusion.html), 允许以非自回归的方式生成 ([MAR](../../Models/CV/2024.06.17_MAR.md)).
离散 Token 通常由量化自编码器的潜在空间内的连续数据表示来生成 ([Discrete VAE](../../Models/_Basis/2016.09.07_Discrete_VAE.md)).

#### 去噪自编码 (Denoising Auto-Encoding)

**去噪自编码器 (Denoising Auto-Encoder, DAE)** 构建于基本的自编码器概念之上, 引入噪声到输入数据中, 并训练模型以重构原始, 无噪声版本.
这种方法鼓励模型学习健壮的特征, 能够处理数据损坏, 从而提高泛化能力.

在基于 Transformer 的模型中, 一种常见的技术称为**掩码语言建模 (Masked Language Modeling, MLM)** ([BERT](../../Models/TextLM/2018.10.11_BERT.md)), 它通过掩盖部分输入 Token 并训练模型来预测它们, 这可以视为一种特殊的去噪自编码器.
这一方法成为各个模态中的主流:
- 语言领域的 [BERT](../../Models/TextLM/2018.10.11_BERT.md);
- 视觉领域的 [BEiT](../../Models/CV/2021.06.15_BEiT.md) 和 [MAE](../../Models/CV/2021.11.11_MAE.md);
- 音频领域的 [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md).

#### 监督预训练 (Supervised Pretraining)

一些 Tokenizer 在具体任务上使用监督学习进行预训练, 旨在通过带标注数据集获得特定任务的表示.
这些模型一开始在大规模数据集上训练以捕获输入数据的具体特征.

- 在视觉模态中, 监督任务包括语义分割, 目标检测, 深度估计.
针对这些任务训练的模型, 如 SAM~\citep{sam,samclip}[197] [398], ViTDet~\citep{vitdet} [241], 和 MiDaS [334]~\citep{midas}, 随后作为 LMMs 中的 Tokenizer 使用 (如 DeepSeek-VL [269]~\citep{DeepSeek-VL}, Cambrain-1 [382] ~\citep{Cambrian-1}), 来从输入数据中提取多样化的视觉特征.

- 在音频模态中, Whisper 以弱监督的方式在 680K 小时的标注音频数据上训练. 由于其健壮且强大的语音特征提取能力, Whisper 被广泛用于语音 LLMs (如 [SALMONN](../../Models/SpokenDialogue/2023.10.20_SALMONN.md); [Qwen-Audio](../../Models/SpeechLM/ST2T/2023.11.14_Qwen-Audio.md) [WavLLM](../../Models/SpeechLM/2024.03.31_WavLLM.md)) 提取语音嵌入.

#### 对比学习 (Contrastive Learning)

**对比学习 (Contrastive Learning, CL)** 是一种自监督学习方法, 着重于通过区分正例和负例来学习表示.
核心思想是将相似的示例 (正例) 在表示空间中靠近, 而将不相似的示例 (负例) 推远.
每对中的项可以属于同一或不同模态.
- [DINO](../../Models/_Basis/DINO.md) 使用图像-图像对来增强视觉表示;
- [CLIP](../../Models/_Basis/2021.02.26_CLIP.md) 使用文本-图像对来增强视觉表示中的语言对齐.

目前, 仅具备多模态理解能力的 LMM (如 InstructBLIP~\cite{dai2023instructblip}; LLaVA~\cite{liu2023llava}) 选择使用具有更强表示能力的 Tokenizer (如 [CLIP](../../Models/_Basis/2021.02.26_CLIP.md)), 因为它们不需要重建多模态信息.

相反, 支持多模态生成能力的 LMM 则倾向于选择 VQ-VAE 作为 Tokenizer, 例如 Unified-IO~\cite{lu2022unifiedio}, Chameleon~\cite{chameleonteam2024chameleon}, Emu3~\citep{Emu3} 和 ~\citep{wang2024mio, seedllama, wang2022ofa}.

</td></tr></table>

## 2.2·Discrete Tokenization Basics: 离散分词基础

<a id="Section.2.2"></a>


<table><tr><td width="50%">

Unlike the language modality, which inherently comprises discrete symbols (e.g., tokens or words), most other modalities naturally exist in a continuous space.
To bridge the gap, the core technique is **Vector Quantization (VQ)**, which aims to map the original continuous information into a compressed, finite representation space, i.e.
discrete tokens.
The discrete tokens can have 2-dimensional or 3-dimensional structures for images and videos.
These tokens are initially linearized based on a specific order, such as left to right and top to bottom, transforming them into a 1-dimensional sequence.
This linearization allows for effective modeling using the next token prediction objective.

In this section, we will first elaborate on modern vector quantization techniques widely used as multimodal tokenizers, such as VQ-VAE (Section.2.2.1) and its variants.
Following that, we will introduce the specific optimizations of discrete tokenization in different modalities (Section.2.3).

</td><td>

不同于语言模态, 其本质由离散符号 (如 Token 或词) 组成, 大多数其他模态自然存在于连续空间中.
为了弥合这种差距, 核心技术是 **向量量化 (Vector Quantization, VQ)**, 其目标是将原始连续信息映射到压缩的有限的表示空间中, 即离散 Token.
对于图像和视频, 离散 Token 可以是二维或三维的结构.
这些 Token 最初基于特定顺序 (如左至右和上至下) 线性化, 转换为一维序列.
这种线性化允许使用 NTP 任务进行有效建模.

在本节中, 我们首先详细介绍被广泛作为多模态 Tokenizer 的现代向量量化技术, 如 VQ-VAE (Section.2.2.1) 和其变体.
随后, 我们在 Section.2.3 中介绍离散 Tokenization 在不同模态中的特定优化.

</td></tr></table>

### 2.2.1·Vector Quantization Methods: 向量量化方法


<table><tr><td width="50%">

The origins of VQ method trace back to the 1950s at Bell Laboratories, where researchers endeavored to optimize signal transmission through the development of suitable discretization procedures~\cite{Pags2015IntroductionTV}.
In essence, quantization is the process of mapping an infinite set of continuous values to a smaller, discrete set of finite values.
The primary objective of vector quantization is to reconstruct all the information in the original data as accurately as possible with a finite set of vectors, which is also called the \emph{codebook}.

**Vanilla VQ**

The original VQ-VAE proposed by \citet{Oord2017NeuralDR} is a milestone of many successive vector quantization methods.
As shown in Figure~\ref{fig:vqvae}, a VQ-VAE consists of three main components: the encoder, the quantizer, and the decoder.
The encoder comprises the input data to a compact latent space, the quantizer select the nearest code vectors from the finite codebook to approximate the continuous latents, the decoder reconstruct the input data using the discrete codes.
When training the VQ-VAE, three main loss components are crucial: reconstruction loss, codebook loss, and commitment loss~\citep{Oord2017NeuralDR}.
The reconstruction loss, often implemented as mean squared error or binary cross-entropy, ensures accurate data reconstruction by minimizing differences between input and output.
Codebook loss, or vector quantization loss, enables effective encoding by aligning encoder outputs with nearest codebook entries, ensuring discrete latent variables.
Meanwhile, commitment loss acts as a regularizer, encouraging encoder outputs to stay close to codebook entries to maintain stable learning, preventing erratic mapping.
As gradient can not pass the quantization operator (finding the nearest code), the straight-through estimator~\cite{bengio2013estimatingpropagatinggradientsstochastic} is adopted to let the gradient flow normally.

Recent advancements in vector quantization methods have focused on achieving better image reconstruction and enhancing generative capabilities.
To improve reconstruction quality, both architectural innovations and codebook designs have been proposed.
Transformer-based frameworks, such as ViT-VQGAN~\citep{yu2022vectorquantized}, Swin-MAE~\citep{xu2023swin}, Swin-Unet~\citep{cao2021swinunet}, and Efficient-VQGAN~\citep{cao2023efficientvqgan}, replace traditional CNN encoders and decoders with more robust modules like ViT~\citep{vit} and Swin-Transformer~\citep{liu2021swinTransformer,liu2022swinV2}, leading to better feature representations and reconstruction fidelity.
Additionally, several methods such as LFQ~\citep{magvit2} and FSQ~\citep{FSQ} are proposed to address the significant challenge of codebook collapse during **codebook learning**, where a large portion of code embeddings are not used when enlarging the codebook size, causing a redundancy in the codebook and limiting the expressive power of the generative model~\citep{baykal2024edvaemitigatingcodebookcollapse}.
For improved generative performance and efficiency, several approaches have been introduced.
\citet{tian2024VAR} propose Visual Autoregressive modeling, which facilitates image generation through "next-scale prediction", moving away from the traditional raster-scan "next-token prediction" used in standard VQ-VAE-based models.
RQ-Transformer~\citep{lee2022RQVAE} employs residual quantization (RQ) to precisely approximate feature maps and reduce spatial resolution.
RQ helps the RQ-Transformer to significantly reduce computational costs and effectively learn long-range interactions in inputs.
RAR~\citep{RAR} introduces a randomness annealing strategy with a permuted objective, enhancing the model's ability to learn bidirectional contexts while retaining the autoregressive framework.
TiTok~\citep{yu2024imageworth32tokens} tokenizes images into 1D latent sequences, providing a more compact latent representation that is substantially more efficient and effective than conventional techniques.
It greatly reduces the number of tokens required to encode an image compared to previous methods~\citep{cao2023efficientvqgan,yu2022vectorquantized}.

**VQ with Auxiliary Losses**

The primary goal of the vanilla VQ-VAE is to accurately reconstruct input data by minimizing the mean squared error loss.
However, this auto-encoding objective doesn't always align with human perception of the quality of reconstructed data.
For example, in the visual modality, the vanilla MSE loss often results in images with blurred details, particularly in human faces~\citep{larsen2016autoencodingpixelsusinglearned}.
To address this issue, several approaches introduce higher-level training objectives aimed at improving the overall quality of the output data.
In the realm of vision, perceptual loss~\citep{johnson2016perceptuallossesrealtimestyle} is widely used to enhance the quality of reconstructed images by leveraging a pre-trained CNN.
VQGAN~\citep{cao2023efficientvqgan} incorporates a discriminator network to enhance image fidelity by adding an adversarial training objective.
The role of the discriminator is to discern between the reconstructed and original images, while the VQ-VAE is optimized to deceive the discriminator, thereby improving the quality of the reconstructed images.
In the audio modality, it is essential to decouple the audio into its acoustic and semantic components to achieve both powerful audio reconstruction quality and LLM modeling.
[SpeechTokenizer](../../Models/SpeechCodec/2023.08.31_SpeechTokenizer.md) and [Mimi](../../Models/SpokenDialogue/2024.09.17_Moshi.md) introduce the loss of semantic distillation at the first layer of Residual VQ, using self-supervised models, such as [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) and [WavLM](../../Models/SpeechRepresentation/2021.10.26_WavLM.md).

**Residual Vector Quantization**

Residual vector quantization (RVQ) has been used for image~\citep{Lee_Kim_Kim_Cho_Han_2022} and audio~\citep{Zeghidour_Luebs_Omran_Skoglund_Tagliasacchi_2022} generation, where quantized codes are refined by storing additional quantized residuals.
\citet{lee2022RQVAE} propose the RQVAE that also introduces a residual quantization to recursively quantize the feature map in a coarse-to-fine manner, employing a fixed-size codebook to maintain both precision and code diversity.

**Product Quantization**

\citet{PO-VAE} propose product quantization (PQ), to factor the codebook into a product of smaller codebooks, allowing for high-quality quantizers without the requirement of intractably large codebooks.

**Multi-scale Quantization**

~\citet{tian2024VAR} introduce the Visual Autoregressive modeling (VAR), which develops a multi-scale quantization autoencoder that encodes images into $K$ multi-scale discrete token maps using a shared codebook.
It aids the model in generating images through "next-scale prediction," instead of the raster-scan "next-token prediction" typically used in standard VQ-VAE-based models.
The multi-scale quantization enables the model to learn visual distributions and demonstrates strong generalization capabilities.

**Finite Scalar Quantization**

To generate concise and expressive tokens using a larger token vocabulary and avoid codebook collapse, \citet{FSQ} propose finite scalar quantization (FSQ).
FSQ projects the VAE representation down to a few dimensions that can be quantized into fixed values, creating an implicit codebook.

**Look-up Free Quantization**

LFQ~\citep{yu2023language} reduces the embedding dimension of the codebook to zero, effectively replacing the codebook with an integer set.
It allows VQ-VAE to improve the quality of image reconstruction and generation by vastly increasing the vocabulary size by magnitudes.
For example, the rFID on Imagenet decreases from 2.5 to 1.4 when the LFQ vocabulary size increases from $2^10$ to $2^16$ on ImageNet dataset.

**Embedding-Free Quantization**

Maskbit~\cite{maskbit} explores an embedding-free tokenization approach that utilizes binary quantization.
It projects latent embeddings into K dimensions and then quantizes them based on their sign values to produce bit token representations.
The generated bit tokens exhibit highly structured semantic representations, which are crucial for generation tasks.

**Group Vector Quantization**

Unlike RVQ which models the information residually, Group Vector Quantization models the information across different dimensions.
In the audio domain, [HiFi-Codec](../../Models/SpeechCodec/2023.05.04_HiFi-Codec.md) proposes a group-residual vector quantization technique to reduce the number of codebooks, while [FACodec](../../Models/SpeechCodec/2024.03.05_FACodec.md) disentangles speech into prosody information, content information, and acoustic details using three-factorized vector quantizers.

</td><td>

向量量化方法的起源可以追溯到 1950 年代的贝尔实验室, 研究人员试图通过开发合适的离散过程来优化信号传输.
简单来说, 量化是将连续值的无限集映射到较小的有限值的离散集的过程.
向量量化的主要目标是用向量的有限集合 (也称为码本) 尽可能精确地重构原始数据中的所有信息.

**原始向量量化**

原始 VQ-VAE 由 \citet{Oord2017NeuralDR} 提出, 是许多后续向量量化方法的里程碑.
如图 \ref{fig:vqvae} 所示, VQ-VAE 由三个主要组件组成: 编码器, 量化器, 和解码器.
- 编码器: 将输入数据压缩到紧凑的潜在空间;
- 量化器: 从有限码本中选择最接近的编码向量来近似连续的潜在变量;
- 解码器: 使用离散编码重构输入数据.

当训练 VQ-VAE 时, 三个主要的损失项是至关重要的:
- **重构损失 (Reconstruction Loss)**: 通常采用均方误差或二元交叉熵, 通过最小化输入和输出之间的差异来确保精确的数据重构;
- **码本损失/向量量化损失 (Codebook Loss/Vector Quantization Loss)**: 通过对齐编码器输出和最近的码本元素来实现有效编码, 实现离散潜在变量;
- **承诺损失 (Commitment Loss)**: 作为正则化, 鼓励编码器输出与码本元素保持接近, 以维持稳定学习, 避免不稳定的映射.

因为梯度无法通过量化操作 (寻找最近编码), 所以采用了直通估计 (Straight-Through Estimator)~\cite{bengio2013estimatingpropagatinggradientsstochastic}, 让梯度正常流动.

在向量量化方法的近期进展聚焦于获得更好的图像重构和增强生成能力.

为了提升重构质量, 架构创新和码本设计都有所发展.
基于 Transformer 的框架 (如 ViT-VQGAN, Swin-MAE, Swin-Unet, Efficient-VQGAN) 将传统的 CNN 编码器和解码器替换为更健壮的模块 (如 ViT, Swin-Transformer), 带来更好的特征表示和重构质量.

此外, 诸如 LFQ 和 FSQ 等方法被提出用于处理在**码本学习 (Codebook Learning)**时码本坍缩的显著挑战, 即在扩大码本尺寸时大部分编码嵌入未被使用, 导致码本冗余, 限制了生成模型的表达能力.

为了提升生成性能和效率, 提出了数种方法:
- \citet{tian2024VAR} 提出了视觉自回归建模 (Visual Autoregressive modeling), 它通过 Next-Scale Prediction 促进图像生成, 而不是标准 VQ-VAE 基线模型中使用的传统光栅扫描 Next-Token Prediction.
- RQ-Transformer 采用残差量化来精确地近似特征图并减少空间分辨率.
RQ 帮助 RQ-Transformer 显著降低计算成本并有效学习输入中的长距离交互.
- RAR 引入了 Permuted 目标和随机退火策略, 增强模型能力来学习双向上下文, 同时保留自回归框架.
- TiTok 将图像离散化为一维隐变量序列, 提供了更紧凑的潜在表示, 比传统技术更效率且有效.
  与之前的方法相比, 它极大减少了编码一个图像所需的 Token 数量.

**带辅助损失的向量量化**

原始 VQ-VAE 的主要目标是通过最小化均方误差损失来精确地重构输入数据.
然而这一自编码目标并不总和人类对重构数据质量的感知对齐.
(例如, 在视觉模态中, 原始的 MSE 损失往往导致图像的细节模糊, 尤其是在人脸上.)

为了解决这一问题, 一些方法引入了高级别的训练目标, 旨在提升输出数据的整体质量.
在视觉领域中,
- 通过使用预训练 CNN, 感知损失 (Perceptual Loss) 被广泛使用以增强重构图像的质量.
- VQGAN 引入了判别器网络和添加对抗训练目标来增强图像质量.
  判别器的作用时区分重构图像和原始图像, 而 VQ-VAE 被优化以欺骗判别器, 进而提升重构图像的质量.

在音频模态中, 有必要将音频解耦为声学和语义组分, 以实现强大的音频重构质量和 LLM 建模.
- [SpeechTokenizer](../../Models/SpeechCodec/2023.08.31_SpeechTokenizer.md) 和 [Mimi](../../Models/SpokenDialogue/2024.09.17_Moshi.md) 在残差向量量化的第一层引入了语义蒸馏的损失, 使用了自监督模型 (如 [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) 和 [WavLM](../../Models/SpeechRepresentation/2021.10.26_WavLM.md)).

**残差向量量化 (Residual Vector Quantization, RVQ)**

残差向量量化被用于图像和音频生成中, 其中量化编码通过存储额外的量化残差以细化.
RQVAE 也引入了残差量化来递归地以从粗到细的方式量化特征图, 采用了固定尺寸的码本来维持精度和编码多样性.

**点积量化 (Product Quantization, PQ)**

PO-VAE 提出了点积量化 (PQ), 将码本分解为较小码本的点积, 允许高质量的量化器而无需大型码本.

**多尺度量化 (Multi-Scale Quantization, MSQ)**

VAR 引入了视觉自回归建模, 开发了多尺度量化自编码器, 使用共享码本将图像编码为 $K$ 个多尺度离散 Token 图.
它帮助模型通过 "Next-Scale Prediction" 生成图像, 而不是标准 VQ-VAE 类模型中所使用的传统光栅扫描 "Next-Token Prediction".
多尺度量化使得模型学习视觉分布并展示了强大的泛化能力.

**有限标量量化 (Finite Scalar Quantization, FSQ)**

为了生成简洁且表达能力强的 Token, 使用更大的 Token 词表并避免码本坍缩, FSQ 提出了有限标量量化.
FSQ 将 VAE 表示映射到低维, 然后量化到固定值, 创建了隐式码本.

**Look-Up Free Quantization (LFQ)**

LFQ 将码本的嵌入维度减少到 0, 使用整数集替换码本.
它允许 VQ-VAE 提升图像重构和生成质量, 通过极大地增加词表的尺寸 (数量级).
例如当 LFQ 词表尺寸从 $2^10$ 增加到 $2^16$ 时, rFID 在 ImageNet 上的值从 2.5 降低到 1.4.

**Embedding-Free Quantization (EFQ)**

Maskbit 探索了无嵌入的 Tokenization 方法, 使用了二元量化.
他将隐变量嵌入映射到 K 维, 然后基于它们的符号值量化以生成比特 Token 表示.
生成的比特 Token 展现了高度结构化的语义表示, 对于生成任务很重要.

**Group Vector Quantization (GVQ)**

和 RVQ 以残差化地建模信息不同, GVQ 采用跨不同维度的方式来建模信息.
在音频领域,
- [HiFi-Codec](../../Models/SpeechCodec/2023.05.04_HiFi-Codec.md) 提出了分组残差向量量化技术来减少码本数量;
- [FACodec](../../Models/SpeechCodec/2024.03.05_FACodec.md) 使用三个因子向量量化器将语音分解为韵律信息, 内容信息和声学信息.

</td></tr></table>

### 2.2.2·Evaluation of VQ Tokenizers


<table><tr><td width="50%">

When evaluating VQ-VAEs, two critical metrics are commonly considered: **reconstruction ability** and **generation ability**.

Reconstruction ability refers to how well the VQ-VAE can reproduce the original input data after encoding and decoding.
This metric evaluates the fidelity of the model in terms of how accurately it can reconstruct the input data from its latent representations.
L2 distance, Peak Signal-Noise Ratio (PSNR), and reconstruction Fréchet Inception Distance (rFID) are often applied to assess the reconstruction ability.

Generation ability assesses the model’s capacity to generate new, plausible samples from the learned distribution in the codebook space.
This metric evaluates the creativity and diversity of the VQ-VAE in producing new data that is consistent with the training data distribution.
To quantitatively evaluate generation ability, metrics such as the Inception Score (IS) and generation Fréchet Inception Distance (gFID)~\citep{heusel2018ganstrainedtimescaleupdate} are often used.

rFIDs are often computed between ImageNet validation images and their reconstructed images.
gFIDs are usually computed against the training set with ADM's evaluation suite~\cite{dhariwal2021diffusionmodelsbeatgans}.

</td><td>

在评估 VQ-VAE 时, 两个关键指标是常见的: **重构能力**和**生成能力**.

重构能力指的是 VQ-VAE 在编码和解码后能够多好地复现原始输入.
这一指标通过模型从潜在表示重构输入数据能有多精确的方面来评估模型的精确性.
L2 距离, PSNR 和 rFID 都是常用的评估重构能力的指标.

生成能力评估模型存在码本空间中的学习到的分布中生成新的合理的样本的能力.
这一指标评估 VQ-VAE 在生成新数据时的创造性和多样性, 这些数据与训练数据分布一致.
为了量化评估生成能力, 诸如 IS 和 gFID 等指标被广泛使用.

rFID 通常通过 ImageNet 验证集图像和它们的重构图像计算.
gFID 通常通过 ADM 评估套件~\cite{dhariwal2021diffusionmodelsbeatgans} 计算, 它与训练集进行比较.

</td></tr></table>

## 2.3·Discrete Tokenization for Different Modalities: 不同模态的离散分词

<a id="Section.2.3"></a>


<table><tr><td width="50%">

Generic quantization methods provide basic ways to convert continuous data into discrete tokens.
However, there isn't a single quantizer that works well for all modalities because each modality has unique characteristics.
Therefore, it is important to create specific tokenizers for each modality.
This section will explain the unique features of different modalities and showcase some examples of tokenizers for images, audio, and video, among others.

</details>
<br>

</td><td>

一般量化方法提供了将连续数据转换为离散 Token 的基本方法.
然而, 并没有单个量化器能够对所有模态都良好工作, 因为每个模态都有独特的特征.
因此, 为每个模态创建特定的 Tokenizer 是很重要的.
本节将介绍不同模态的独特特征, 并展示一些图像, 音频和视频等其他模态的 Tokenizer 的例子.

</td></tr></table>


### 2.3.1·Image: 图像

<table><tr><td width="50%">

Images can be tokenized into discrete symbols with the previously introduced VQ-VAE structure.
Compared to text tokens, images diverge in three fundamental aspects that significantly impact how they should be tokenized:

1. Rich Information Granularity: Unlike text, which primarily encapsulates high-level semantic meaning, images are contain with a myriad of perceptual details.
These encompass low-level visual elements such as colors, shapes, and textures, alongside more abstract concepts like objects and actions.
2. Dense Information: Images inhabit a densely packed representational realm, where each pixel, across multiple dimensions including height, width, and color channels (RGB being a common example), carries information.
This stands in stark contrast to the discreteness of text in nature, characterized by sequentially arranged words.
3. Two-Dimensional Spatial Structure: Images are inherently structured in two dimensions, spread across a grid defined by height and width.
This 2D layout differs fundamentally from the straightforward, one-dimensional sequence that characterizes textual data, introducing unique complexities in their processing and analysis.

Given these differences, bridging the gap between text and image modalities in the training of LLMs based on discrete image tokens requires a robust image tokenizer, which must balance the fusion of sufficient alignment with LLM's language ability (referred to as "representation"), the retention of rich original image information (referred to as "reconstruction"), and the efficient use of tokens given the growing inference cost of transformer decoder (referred to as "token efficiency").
These factors possess a trade-off~\citep{seedllama,seed-tokenizer, magvit2, sun2023generative}, making it crucial for the construction of an image tokenizer to maintain equilibrium among these factors.

In terms of better representation, models like ViT~\citep{vit} are commonly employed, often aligned with a text encoder through contrastive loss~\citep{[CLIP](../../Models/_Basis/2021.02.26_CLIP.md), peng2022beit}, or aligned with text modalities through generative loss~\citep{coca}.
Additionally, modules like Q-Former~\citep{li2023blip2} can also be used for image feature transformation~\citep{li2023blip2, seedllama}.
Consequently, the resultant image features integrate higher-level semantics and gradually compress high-dimensional images into lower-dimensional representations aligned with text.
While the initial arrangement of image patches follows a raster order, preserving intrinsic sequential relationships, this configuration lacks causal semantics, posing challenges for language modeling.

Regarding reconstruction ability, an image decoder is often layered atop the image encoder to reconstruct the original image from its representation, incorporating reconstruction loss into the training process~\citep{amused, seedllama, lavit, [VQGAN](../../Modules/VQ/2020.12.17_VQGAN.md)}.
Training labels typically use the original images, but with advancements in diffusion models, more research is incorporating latents for diffusion models as reconstruction labels~\citep{lavit, seedllama}.

For token efficiency, modules like selectors or mergers for image tokens are utilized to truncate their length (i.e., the number of tokens per image).
For instance, SEED-LLaMA~\citep{seedllama} compresses longer image features encoded by ViT into 32 continuous tokens using a Causal Q-Former and then discretizes them through quantization.
LaViT~\citep{lavit} first predicts whether each patch token should be selected using a shared MLP, and then compresses the image length by employing selected patches as queries and unselected patches as keys and values in cross-attention blocks~\citep{seedllama}.

Beyond these aspects, some studies also focus on the unique properties of specific image types or tasks.
For example, VQ-IMG aims to enhance the modeling capabilities of image tokenizers for faces~\citep{make-a-scene}, while LVM integrates tasks like segmentation and object detection during the training of models based on VQGAN to enrich the representation of image tokens~\citep{bai2023sequential}.
StrokeNVWA introduces a VQ-Stroke method to discretize vector graphic images into stroke tokens~\citep{strokenvwa}.


</td><td>

</td></tr></table>




### 2.3.2·Audio: 音频

<table><tr><td width="50%">

Raw audios are typically stored as 16-bit integer values with a sampling rate that exceeds tens of thousands values per second, which leads to extremely long sequences and renders next token prediction training more difficult.
Versatile quantization methodologies have been investigated for audio tokenization.
Initially aimed at audio compression, these methodologies have more recently been developed to create compact semantic and acoustic representations in the context of NTP language modeling.

As a traditional companding algorithm, $\mu$-law/A-law algorithm is commonly employed in speech generative models such as WaveNet~\citep{van2016wavenet}.
While this algorithm projects each audio frame to an 8-bit value, it does not reduce the sampling rate, thereby preserving overlong sequences.
Self-supervised learned models have shown exceptional performance in various speech-related tasks, sparking interest in clustering their speech representations for speech quantization.
The vq-wav2vec~\citep{baevski2019vq} uses either a Gumbel-Softmax or online k-means clustering to quantize the SSL-learned dense representation.
[HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) is trained with a masked prediction task, whose targets are obtained through k-means clustering of learned features from earlier iterations.
Utilizing quantized tokens learned with Self-Supervised Learning (SSL), GSLM~\cite{lakhotia2021gslm} and VQTTS~\citep{du2022vqtts} demonstrate faster speed in speech generation tasks compared with WaveNet.
Because SSL tokens are extracted with highly abstracted semantics while discarding low-level acoustic information, the reconstruction quality is relatively low, and speaker identity is lost~\citep{borsos2023audiolm}.
Neural codec models typically apply a VQ-VAE on the raw audios with residual vector quantization, exemplified by SoundStream~\citep{zeghidour2021soundstream} and EnCodec~\citep{encodec}.
They are originally designed for audio compression, have the capability to encode waveforms into discrete codes and faithfully reconstruct them back into high-quality waveforms.
Recently, they are widely used in audio generation models such as AudioLM~\citep{borsos2023audiolm}, VALL-E~\citep{wang2023neural} and their variants~\citep{han2024valler,song2024ellav, wang2023viola}, and reach new state-of-the-art performance on various tasks.
Compared with traditional $\mu$-law/A-law algorithms, codec models can efficiently reduce the length of token sequences.
It can also maintain multi-scale acoustic information indicating speaker identity compared with highly-abstracted SSL-learned discrete tokens such as [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) tokens.
Additionally, the codec models are typically off-the-shelf and lightweight.

Latest works have attempted to impose additional supervision on the discrete codes extracted by codec models.
The objective is to enhance their ability to extract and encode higher-level semantic information, thereby improving language modeling.
[SpeechTokenizer](../../Models/SpeechCodec/2023.08.31_SpeechTokenizer.md) is an RVQ-based codec model, where its first-layer codebook incorporates semantic information through the semantic distillation process, using [HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) representations as the semantic teacher.
Mimi, used by Moshi~\citep{défossez2024moshispeechtextfoundationmodel}, further improves upon this by replacing the semantic teacher from HuBERT with [WavLM](../../Models/SpeechRepresentation/2021.10.26_WavLM.md).
Additionally, it isolates the first-layer codebook from the RVQ process to achieve better semantic and acoustic disentanglement.
To enhance the compression rate, WavTokenizer~\citep{ji2024wavtokenizer} is capable of quantizing one-second audio into 75 or 40 tokens with a single quantizer.

</td><td>

</td></tr></table>

### 2.3.3·Video: 视频

<table><tr><td width="50%">

Compared to images, videos introduce an additional temporal dimension that must be considered during the tokenization process.
A straightforward strategy is to utilize an image-based VQ-VAE model to tokenize the video frame-by-frame.
This approach is employed by several multimodal foundation models, such as LVM~\cite{bai2023sequential}, LWM~\cite{liu2023world}, and Unified-IO series~\cite{lu2022unifiedio,lu2023unifiedio2}.
However, a significant drawback of frame-by-frame tokenization is its inability to compress video data over time, resulting in a high degree of token redundancy across frames—particularly in long-form videos—thereby imposing substantial computational demands~\citep{song2024moviechatdensetokensparse}.
Furthermore, using an image-based tokenizer fails to model temporal relationships between frames, leading to issues of temporal inconsistency.

To address token redundancy and enhance temporal modeling, several studies have proposed training a 3D tokenizer that compresses videos across spatial and temporal dimensions.
For example, VideoGPT~\citep{yan2021videogpt} applies a 3D-CNN architecture in the encoder and decoder of the video tokenizer.
C-ViViT~\cite{villegas2022phenaki} uses a transformer architecture to split videos into 3D cubes, which are then discretized into token IDs.

There are two additional desirable features for a video tokenizer:
**(1) Joint Image-Video Tokenization**.
The MAGVIT series~\cite{magvit2} enables tokenizing images and videos with a shared vocabulary.
To achieve this, the number of frames in an input video, $T$, must satisfy $T=1+n \times F_T$, meaning the video comprises an initial frame followed by $n$ clips, each containing $F_T$ frames.
When $n=0$, the video contains only the initial frame, thus simplifying the video to an image.
Accordingly, both the initial frame and each subsequent clip are discretized into a $(1, H', W')$ token map, where $H'$ and $W'$ are the height and weight of the token map.
**(2) Temporal Causality**.
Compared to vanilla 3D architectures, using causal 3D architecture can ensure the tokenization and detokenization of each clip depend only on the preceding clips, facilitating autoregressive modeling along the temporal dimension.

</td><td>

</td></tr></table>

### 2.3.4·More Modalities: 更多模态

<table><tr><td width="50%">

Modeling various information as discrete tokens has gone far beyond the traditional text, image, video and audio modalities.
In the computer vision field, we can unify the output spaces of tasks like object detection, semantic segmentation, and depth mapping into images.
These can then be tokenized into discrete image tokens, allowing us to train a single NTP model to handle all these tasks \cite{wang2022ofa,wang2023images,bai2023sequential}.
In **robotics and embodied AI** domain, the robots actions in response to the environments can be coded into various discrete tokens and learn the policy in NTP manner as shown in recent studies such as VIMA~\cite{jiang2023vima}, RT2~\cite{brohan2023rt2} and Locomotion NTP~\cite{humanoid}.
In **AI4Science**, by factorizing various proteins into DNA token sequences, protein language models are capable of learning from a wide array of sequences that span the evolutionary tree.
These models have demonstrated their efficacy as powerful tools for sequence design and protein engineering, as highlighted in studies ~\cite{madani2023large,ruffolo2024designing}.

</td><td>

</td></tr></table>

## 2.4·Continuous Tokenization Basics: 连续分词基础

<a id="Section.2.4"></a>

<table><tr><td width="50%">

Continuous tokens represent non-textual modalities in a continuous feature space, offering less information loss~\citep{dnd-transformer} and improved data representation compared to discrete tokens~\citep{xie2024showosingletransformerunify}.
However, their dense feature encapsulation makes direct mapping to a fixed vocabulary challenging, unlike discrete tokens.
It poses a challenge for LLMs aiming to comprehend and generate such information in a NTP manner.

To handle continuous multimodal token inputs for LLM to understand, transformations or adapters are necessary to balance data representation and text alignment.
For multimodal generation, modifying the output head to align with non-textual modality specific decoders' input feature space is also crucial.
The following subsections introduce the basic designs and change for LLMs to accommodate continuous multimodal token from multimodal understanding (\S\ref{sec: Tokenize Continuous Input}) and generation (\S\ref{sec: De-tokenize Continuous Output}) perspectives.

</td><td>

</td></tr></table>

### 2.4.1·Tokenize Continuous Input for Understanding

<table><tr><td width="50%">

To effectively integrate raw non-textual modality data into Large Language Models (LLMs), two key steps are typically undertaken: (1) encoding the data into a more suitable representation space, and (2) aligning it with the LLM’s feature space.

**Encoding**

The encoding of non-textual modality data aims to capture meaningful features and important nuances that are essential for the understanding of the data.
This can be achieved through different types of encoders such as Transformer-based encoders~\citep{li2023blip2, liu2023llava, liu2023llava15,zhu2023minigpt4, [CLIP](../../Models/_Basis/2021.02.26_CLIP.md)} or CNN-based encoders~\citep{davinci, zhang2023universal, jiang2023vima, alayrac2022flamingo}.
There's also an option to go encoder-free~\citep{kim2021vilt, fuyu}, which allows for raw data to be fed directly into the model.

Transformer-based encoders are widely used for their robust representation capabilities and generalizability~\citep{vaswani2017attention, vit}.
For a non-textual modality sample, the input is initially divided into patches and transformed into a 1D sequence, with each patch represented as a soft token.
This sequence is then processed through the Transformer's encoder layers, employing self-attention mechanisms to capture relationships between patches.
Consequently, the model produces a rich representation of the input.
Typically, there are two types of encoders: (1) unimodal encoders, designed to process information from a single modality~\citep{vit, sam, arnab2021vivit, usm, mert, prismer, [BEiT](../../Models/CV/2021.06.15_BEiT.md), liu2021swinTransformer}; and (2) multi-modal encoders, capable of integrating information from multiple modalities~\citep{[CLIP](../../Models/_Basis/2021.02.26_CLIP.md), imagebind, eva-clip, clap, anymal, imu2clip, coca}.
For instance, PaLM-E~\citep{driess2023palme}, Unified-IO-2~\citep{lu2023unifiedio}, and PaLI~\citep{pali} use ViT~\citep{vit} encoders trained solely on visual data.
Conversely, LLaVA~\citep{liu2023llava}, Emu~\citep{sun2023emu1, sun2023generative}, and Qwen-VL~\citep{QwenVL} utilize [CLIP](../../Models/_Basis/2021.02.26_CLIP.md) or EVA-CLIP~\citep{eva-clip} encoders with contrastive loss to align textual and non-textual representations.
NExT-GPT~\citep{nextgpt}, CoDi-2~\citep{tang2023codi2}, and BuboGPT~\citep{zhao2023bubogpt} employ ImageBind~\citep{imagebind} as their non-textual encoder, aligning various modalities like audio, text, and heat maps with image representations.

In comparison, CNN-based encoders are less frequently used but remain vital due to their flexibility in image resolution generalization~\citep{magvit, magvit2} and ability to capture local features~\citep{jiang2023vima}.
For example, DaVinCi~\citep{davinci} uses ResNet~\citep{resnet} as the visual encoder.
Flamingo~\citep{alayrac2022flamingo} utilizes NFNet~\citep{nfnet}, a normalizer-free ResNet, for image encoding.

Beyond encoders, Fuyu-8B~\citep{fuyu} directly processes raw image patches after a single linear projection to accommodate images of varying resolutions and aspect ratios, similar to ViLT~\citep{kim2021vilt}.
However, Fuyu-8B adds the flexibility of an any-resolution setting using a decoder-only model, benefiting from architectural simplicity but showing reduced downstream performance compared to encoder-based models.
Moreover, ImageGPT~\citep{imagegpt} trains a decoder-only generative model on raw image pixel sequences, which, despite its effectiveness in image generation and understanding, requires significant computational resources and is limited to low-resolution images.

**Input Alignment**

After encoding non-textual modality data, we obtain a meaningful representation.
However, this representation often lacks alignment with the textual embedding space of large language models, leading to a failure in properly understanding these inputs.
Although multi-modal encoders like [CLIP](../../Models/_Basis/2021.02.26_CLIP.md) have made strides in narrowing the gap, they still encounter two significant challenges: (1) the presence of redundant continuous tokens~\citep{alayrac2022flamingo, perceiver, li2023blip2}; and (2) a lack of contextual semantics, such as causal semantics, because they are typically trained only with image-caption paired data rather than image-text interleaved data or image-prompt instructional data~\citep{seed-tokenizer, gemini1, laurencon2023obelics, zhu2023multimodal}.
Therefore, it is crucial to establish a connection between the representation space of non-textual modality data and the LLM textual embedding space.
There are typically two approaches to construct such a bridge: (1) Slot-based Resampler~\citep{alayrac2022flamingo, li2023blip2}; and (2) Projection~\citep{fuyu, liu2023llava, liu2023llava15,QwenVL}.

The Slot-based Resampler compresses redundant non-textual modality tokens from the encoding stage into fewer learned query vectors, known as slots.
This is typically accomplished using multiple Transformer blocks with a cross-attention mechanism.
For instance, BLIP-2~\citep{li2023blip2} employs a Q-Former and linear projection to bridge the image encoder with the LLM backbone.
The Q-Former blocks consist of a self-attention layer on the learned queries, a cross-attention layer between the encoded image representation and the learned queries, and a feed-forward layer.
Initially, it is trained for image-text matching, image-text contrastive learning, and image-grounded text generation, followed by training for next token prediction with the frozen LLM backbone.
Another model using this approach is Flamingo~\citep{alayrac2022flamingo}, which utilizes a Perceiver Resampler~\citep{perceiver} to compress byte arrays into latent vectors in a modality-agnostic manner.
Specifically, Perceiver~\citep{perceiver} employs multiple cascaded attention mechanisms: the latents act as queries and initially cross-attend to keys and values calculated from the byte array (e.g., an image), followed by processing with a self-attention block, iterating several times.
PerceiverIO~\citep{perceiverio} enhances this with an additional cross-attention block between an output query array and the slots (i.e., the latents).
The Hierarchical Perceiver~\citep{hip} decomposes the input array into multiple groups, compresses each group, and merges the resulting latents to obtain the output array.

Compared to a slot-based resampler, projection is much simpler in architecture, involving only a single linear projection~\citep{fuyu, liu2023llava} or an Multi-layer Perceptron (MLP) ~\citep{liu2023llava15}.
For instance, LLaVA~\citep{liu2023llava} employs a linear projection to convert encoded image representations into the language embedding space.
Similarly, Fuyu-8B~\citep{fuyu} projects raw image patches onto the embedding space.
LLaVA-1.5~\citep{liu2023llava15} enhances LLaVA by substituting the linear projection with an MLP.

There are also other approaches to connect the non-textual modality encoder with the LLM backbone.
For example, Emu~\citep{sun2023emu1} leverages a Causal Transformer (i.e., C-Former) to convert the image tokens autoregressively; Emu2~\citep{sun2023generative} replaces the C-Former with mean pooling followed by a linear projection.

</td><td>

</td></tr></table>

### 2.4.2·De-tokenize Continuous Output for Generation

<table><tr><td width="50%">

The backbone of large language models is inherently designed for language generation.
Typically, their output layers function as classification heads that predict distributions over a language vocabulary.
For discrete non-textual modalities, the discrete token vocabularies can be integrated into the LLM’s original text vocabulary since token generation is still managed by the classification heads.
However, this approach does not work for continuous non-textual modalities.
To enable the generation of continuous token outputs from LLM backbones, it is essential to modify their output layers (i.e., language modeling heads) to produce representations suited for non-textual modality data.
These representations are then transformed to align with the input features of specific non-textual modality data decoders, such as a diffusion model~\citep{Rombach_Blattmann_Lorenz_Esser_Ommer_2022}.
Recent work includes MAR~\citep{MAR} and Transfusion~\citep{Transfusion}.
We will further elaborate on the decoding of continuous output in ~\S\ref{par:soft-token-output-decoding} and the transformations to the output feature in ~\S\ref{par:soft-token-output-transformation}.

**Decoding**

Unlike pure text generation, multimodal generation requires the model to decide when to switch modalities during decoding, due to their intrinsic differences.
We refer to this objective as **positioning**.
There are typically two methods to achieve this: (1) using placeholders~\citep{zheng2023minigpt5, nextgpt, koh2023GILL}; and (2) employing a non-textual modality begin-of-sentence (BOS) token~\citep{sun2023emu1, sun2023generative, dreamllm}.

Firstly, special tokens can be introduced as placeholders for non-textual modality data.
For instance, Mini-GPT5~\citep{zheng2023minigpt5} and GILL~\citep{koh2023GILL} utilize a sequence of image placeholder tokens ranging from [IMG1] to [IMGr], which can be interleaved with textual tokens, and these tokens are added to the model's vocabulary.
Likewise, NExT-GPT~\citep{nextgpt} uses 5 image placeholder tokens, along with 9 audio and 25 video placeholder tokens.
Secondly, the use of a single BOS token (sometimes accompanied by an EOS token) can simplify the process by signaling the position of non-textual modality data.
For example, DreamLLM~\citep{dreamllm} employs a special <dream> token to mark the start of modality switching, allowing a single model run to process a sequence of queries.
Emu~\citep{sun2023emu1} and Emu2~\citep{sun2023generative} use both image BOS and EOS tokens to encase encoded image features.

In addition to focusing on positioning, models must also learn to generate accurate features for non-textual modalities.
Typically, the output layers of large language models (LLMs) feature classification heads for discrete token decoding, an objective we refer to as **output representation**.
To enable continuous token outputs, modifications to these output layers are required.
Generally, there are three approaches: (1) adapting the original language modeling head to be regressive~\citep{sun2023emu1, sun2023generative}; (2) introducing a new head for dense outputs~\citep{dreamllm}; and (3) utilizing the final hidden states before the language model head~\citep{zheng2023minigpt5, koh2023GILL}.

**Output Alignment**

Typically, generated continuous tokens cannot be directly used for multimodal generation because they don't align with the input features of multimodal decoders like LDM~\citep{Rombach_Blattmann_Lorenz_Esser_Ommer_2022} and AudioLDM~\citep{audioldm}.
To address this, additional modules are introduced to convert these tokens into representations suitable for multimodal decoders, ultimately generating the final non-textual modality data.
For instance, NExT-GPT~\citep{nextgpt} employs a Transformer-based output projection, while Mini-GPT5~\citep{zheng2023minigpt5} and GILL~\citep{koh2023GILL} utilize a Q-Former-like architecture~\citep{li2023blip2} consisting of a Transformer encoder and decoder to transform continuous tokens into conditional latent features for the Stable Diffusion Model.
DreamLLM~\citep{dreamllm} uses a linear layer, whereas Emu~\citep{sun2023emu1} and Emu2~\citep{sun2023generative} directly utilize the generated continuous tokens as latents for multimodal decoders.

</td><td>

</td></tr></table>

## 2.5·Continuous Tokenization for Different Modalities: 不同模态的连续分词

<a id="Section.2.5"></a>

<table><tr><td width="50%">

While the aforementioned workflow and categorization outline a general approach to continuous multimodal tokenization, research indicates that employing modality-specific encoders, tailored to each modality, can significantly enhance performance~\citep{navit, fixres, anymal}.
Given the unique characteristics of different modalities, these approaches introduce specific inductive biases into the tokenization process.

</td><td>

</td></tr></table>

### 2.5.1·Images: 图像

<table><tr><td width="50%">

For images, specific research directions include but are not limited to: \textbf{image augmentation}, \textbf{resolution and aspect ratio} and \textbf{heterogeneous images}.

(1) Image Augmentation: This involves enhancing image representation using elements like depth, edge, and segmentation~\citep{prismer, sam, samclip}.
Prismer~\citep{prismer}, for instance, introduces features beyond traditional RGB patches, such as depth and normal patchification.
These features are compressed with a shared experts resampler before being integrated by a unified image encoder.
SAM-CLIP~\citep{samclip} leverages SAM~\citep{sam} and the CLIP text encoder for distillation training, boosting the semantic and spatial comprehension of the image encoder.

(2) Resolution and Aspect Ratio: This strategy includes support for high-resolution images, multi-resolution capabilities, and arbitrary aspect ratios~\citep{msvit, navit, fuyu, llava-uhd, ureader}.
For example, Fuyu~\citep{fuyu} uses raw pixels as image encoding inputs for the LLM backbone via linear projection, employing a special image newline token for delineating raster-ordered patches.
This enables support for various resolutions and aspect ratios.
MS-ViT~\citep{msvit} suggests varying patchification based on image region complexity, introducing a gate mechanism to mark tokens needing finer patchification, which then undergoes encoding after position encoding interpolation.

(3) Heterogeneous Images: This includes encoding methods for specific image types like vector images, diagrams, charts, and PDFs~\citep{layoutlm, textmonkey, ureader}.
Document images, for example, require detailed observation, as seen in TextMonkey~\citep{textmonkey}, which splits large document images into smaller sub-images.
Each sub-image is encoded individually, and trainable shifted attention layers are added post-frozen ViT layers for interactive representation across sub-images.
These are then compressed and fed into the LLM backbone via an image and token resampler.

</td><td>

</td></tr></table>

### 2.5.2·Audio: 音频

<table><tr><td width="50%">

Recently, MELLE~\citep{meng2024autoregressive} indicates that predicting continuous tokens in an NTP manner can generate audio with high quality and naturalness comparable to ground truth.
Traditionally, audio frames are converted from the temporal domain to the frequency domain using the Short-Time Fourier Transform (STFT)~\cite{griffin1984signal} or the Fast Fourier Transform (FFT)~\cite{duhamel1990fast}.
The magnitude of the Fourier-transformed frames is modeled as spectrogram, which is a 2D image showing how the frequency content of the signal evolves over time.

Spectrograms or other transformations of raw audio signals are additionally going through the feature selection pipeline before converting into discrete tokens.
Mel-Frequency Cepstral Coefficients (MFCCs)~\cite{furui1986speaker} extracts coefficients that represent the short-term power spectrum of sound and is one of the most common features used in speech recognition.
Mel-spectrogram~\cite{furui1986speaker} converts the spectrogram to the mel scale, which is more perceptually relevant to human hearing.
These continuous features are commonly used in audio generation tasks.

Pre-trained foundation models, typically learned in a self-supervised manner on large-scale corpora, have emerged as powerful speech and audio representation extractors~\citep{latif2023sparks}.
To obtain general speech features, wav2vec 2.0~\citep{baevski2020wav2vec} masks speech input in the latent space and addresses a contrastive task defined over quantized latent representations that are learned simultaneously.
data2vec~\citep{baevski2022data2vec} biases the query-key attention scores with a penalty proportional to their distance.
[HuBERT](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) employs an offline clustering step to provide aligned target labels for a BERT-like prediction loss, which is applied solely on the masked regions.
[WavLM](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) introduces denoising in pretraining, jointly with regular masked speech prediction, as HuBERT.
Whisper~\citep{radford2023robust}  is a speech recognition model characterized by an attention-based encoder-decoder architecture, trained on web-scale labeled speech data.
It is increasingly being employed as a foundational speech model, extending its applications beyond speech recognition tasks~\citep{[WavLLM](../../Models/SpeechLM/2024.03.31_WavLLM.md); [SALMONN](../../Models/SpokenDialogue/2023.10.20_SALMONN.md),meng24c_interspeech,meng2024llm}.

For continuous tokenization of audio, AST~\citep{ast} uses a convolution-free pure-transformer architecture to extract features for audio classification, drawing insights from ViT~\citep{vit}.
Inspired by [CLIP](../../Models/_Basis/2021.02.26_CLIP.md), CLAP~\citep{clap} introduces a contrastive language-audio pre-training task to learn text-enhanced audio representations using supervised audio and text pairs.
Fine-tuned based on a pre-trained CLIP model, Wav2CLIP~\citep{wu2022wav2clip} and AudioCLIP~\citep{guzhov2022audioclip} incorporate an additional audio encoder using supervised pairs of audio and class labels.
Audio-MAE~\citep{huang2022masked} adopts a Transformer-based encoder-decoder framework to learn audio representations.
Similar to MAE, it uses a reconstruction pre-training task where the decoder is tasked with reconstructing masked patches from the encoded information of the unmasked patches.
BEATs~\citep{chen2022beats} introduces a self-distilled tokenizer that converts continuous audio signals into discrete labels, facilitating classic mask and discrete label prediction pre-training.

</td><td>

</td></tr></table>

### 2.5.3·Video: 视频

<table><tr><td width="50%">

Video can be viewed as a sequence of images (frames) over time, making the modeling of temporal relationships between these frames a central focus.
There are two common approaches to this modeling: post-temporal fusion and full-temporal fusion.

In the case of \textbf{post-temporal fusion}, models such as CLIP4Clip~\citep{luo2022clip4clip} and CLIPBERT~\citep{lei2021less} first independently encode each frame using an image encoder.
They then employ lightweight pooling, convolution, and attention mechanisms to temporally fuse the features from all frames.
The advantage of this approach lies in its ability to leverage pre-trained image encoders, thereby reducing the computational overhead associated with adapting to video data.
However, a significant drawback is its limited capacity to adequately model features in the temporal dimension.

On the other hand, \textbf{full spatial-temporal fusion} models, like Temporal 3D ConvNets~\citep{diba2017temporal}, VideoMAE~\citep{tong2022videomae}, and ViViT~\citep{arnab2021vivit}, utilize 3D convolutions or 3D attention structures, allowing for comprehensive interaction among inputs in the spatio-temporal dimension.
This enables better modeling of dynamic changes in temporal order, effectively capturing the motion of objects and backgrounds.
However, this approach requires substantial 3D computation, prompting common strategies such as decoupling temporal and spatial self-attention~\citep{bertasius2021space, ren2023testa} and implementing sparse 3D attention~\citep{lin2022swinbert} to enhance computational efficiency.

Recent advancements, such as TimeChat~\citep{ren2024timechat} and NumPro~\citep{wu2024number}, have explored the integration of timestamp information into continuous video tokens, facilitating explicit time-vision associations for improved temporal grounding and reasoning.

</td><td>

</td></tr></table>
