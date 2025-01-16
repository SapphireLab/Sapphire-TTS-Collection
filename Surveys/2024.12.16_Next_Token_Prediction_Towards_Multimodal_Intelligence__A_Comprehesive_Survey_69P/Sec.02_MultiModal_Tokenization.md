# Sec.02·MultiModal Tokenization: 多模态分词

<details>
<summary>展开原文</summary>

Tokenization is the first and a fundamental step for multimodal sequential modeling under the next token prediction framework.
It decomposes information from various sources, such as images, videos, and audio clips, into a sequence of minimal, manageable units known as tokens for the NTP model to learn.
Table.02 provides an overview of the tokenizers used across various modalities in recent research.

Despite being derived from various modalities, these tokenization methods can all be categorized into two prototypes: **discrete tokenization**  and **continuous tokenization**.
In this section, we will initially introduce the general definition and basics techniques of training multimodal tokenizers ([Section.2.1](#Section.2.1)), then the fundamentals and applications of discrete tokens ([Section.2.2](#Section.2.2), [Section.2.3](#Section.2.3)) and continuous tokens ([Section.2.4](#Section.2.4), [Section.2.5](#Section.2.5)) in NTP framework.

</details>
<br>

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

## 2.1·Tokenization of Different Modalities: 不同模态的分词

<a id="Section.2.1"></a>

We first define the tokenization process as a function $f$ that maps a sample $x$ from the raw multimodal space $X$ to a representation $z$ in the tokenizer's output representation space $Z_f$.

$$
f(x) = z,
$$

where $x\in X$ and $z\in Z_f$.

### 2.1.1·Tokenizer Type: 分词器类型

As illustrated in Fig.04, tokenizers for multimodal information can be categorized into two types: discrete and continuous.
This classification is based on how tokens are derived from the original data.
Both tokenization methods encode the original information into a latent representation space, but they differ in their approach.

Discrete tokenization performs quantization on the latent space, utilizing a fixed-size, discrete space similar to the vocabulary of language models.
In contrast, continuous tokenization does not involve quantization, resulting in a much larger representation space.

**Discrete**

In Equation~\ref{eq:tokenization}, a discrete token implies that the representation space $Z_f$ comprises a finite number of discrete symbols.
The output space is called the codebook $C = \{c_1, c_2, ..., c_N\}$, where $c_i \in \mathbb{R}^0$, and each representation $z$ is composed of codes from this codebook, i.e., $z = \{z_1, z_2, ..., z_n\}$ with $z_i \in C$.
Language tokens are inherently discrete because they originate from a finite vocabulary.
Each word or subword unit is mapped to a unique token from this predefined set.
In contrast, modalities such as audio and images exist in continuous, high-dimensional spaces.
To process these modalities within the same framework (i.e., NTP) as for discrete language tokens, they need to be transformed into a discrete representation.

Quantization is a process that maps values from a continuous space to a discrete space, typically resulting in a much smaller representation space.
It is a default operation when a discrete representation is desired for tokenizing multimodal information.
Quantization is often combined with auto-encoder techniques to reduce the size of the latent space.
Typical examples include VQ-series tokenizers such as VQVAE~\cite{vq} and VQGAN~\cite{Esser2020TamingTF}, which inherently feature discrete representations.
Details of the quantization process are introduced in Section~\ref{sec:discrete tokens}.

**Continuous**

In contrast to discrete tokenization, continuous tokenization represents data using a continuous space where tokens are derived directly from the data's inherent properties without enforcing quantization into a predefined codebook.
In this approach, the representation space \( Z_f \) is not limited to a finite set of predetermined codes; rather, it preserves the continuous nature of the data.
Each token \( z \) is sampled from a continuous distribution, allowing for a more nuanced and flexible representation that can capture the subtleties of the input data.
Continuous tokenization is particularly advantageous for modalities that naturally exist in a continuous form and require a rich representational capacity to capture their complex patterns.
For instance, in audio and visual data, continuous representations can effectively retain fine-grained temporal and spatial information that might be lost during discrete tokenization.

### 2.1.2·Features of Tokenizers

Before diving into different tokenization techniques, we summarize the basic two features (Representation and Reconstruction) that an ideal multimodal tokenizer should possess to achieve better understanding and generation capabilities in the NTP framework.

**Representation Ability**

Effective representation encodes semantically relevant information into the latent space $Z$ while removing redundant information.
This is crucial for various downstream tasks that learn a conditional probability $P(Y|X)$ over the label space $Y$, conditioned on the multimodal input space $X$, by replacing it with $P(Y|Z)$.
Prominent tokenizers known for better representation include language-guided contrastive learning methods such as CLIP~\cite{radford2021clip} and fully self-supervised methods like DINO~\cite{caron2021emerging}.

**Reconstruction Ability**

For generating multimodal information, it is expected that the tokenization function $f$ is invertible or nearly invertible, meaning there is a detokenization function $g$ that can recover the original input from the representation space, satisfying $g(f(x)) = x$ or $g(f(x)) \approx x$.
Notable works that excel in reconstruction include Auto-Encoder (AE) series models such as Variational Auto-Encoder~\cite{kingma2022autoencoding} (VAE) and VQVAE~\cite{vq}.

It is important to note that these abilities are not mutually exclusive; their balance depends on the training techniques used.

### 2.1.3·Training Methods for Tokenizers

The training methodologies for tokenizers can be categorized into four groups, based on their respective training objectives: Auto-Encoding, Denoising Auto-Encoding, Supervised Training, and Contrastive Learning, as depicted in Figure.05.
Herein, we provide a summary of the core concepts associated with various tokenizers.

**Auto-Encoding**

Auto-Encoder (AE) is a type of artificial neural network designed to learn efficient data representations.
It consists of two main components: an encoder, which maps input data to a latent space with reduced dimensions, and a decoder, which reconstructs the input data from this latent representation.
The training goal for an Auto-Encoder is to minimize the reconstruction error, ensuring the decoded output closely resembles the original input.
Variants like Variational Auto-Encoders~\cite{kingma2022autoencoding} (VAEs) use probabilistic approaches to generate more robust and informative embeddings.
In multimodal generation models, tokenizers trained with auto-encoder methodologies are used to restore the multimodal input from the latent representation.
A special case is diffusion models~\cite{dieleman2022diffusion}, which can also be viewed as an Auto-Encoder, enabling generation in a non-autoregressive manner~\cite{li2024autoregressive-without}.
Discrete tokens are typically generated by quantizing~\citep{rolfe2017discrete} the continuous data representation within the latent space of auto-encoders.

**Denoising Auto-Encoding**

A Denoising Auto-Encoder (DAE) builds on the basic auto-encoder concept by introducing noise into the input data and training the model to reconstruct the original, noise-free version.
This approach encourages the model to learn robust features capable of handling data corruption, thereby improving its generalization capabilities.
In transformer-based models, a common technique known as Masked Language Modeling~\citep{devlin2019bert} involves masking parts of the input tokens and training the model to predict them, which can be viewed as a special type of denoising auto-encoder.
This method has become mainstream across various modalities, popularized in language by BERT~\citep{devlin2019bert}, in vision by BEiT~\citep{beit} and MAE~\citep{he2021maskedautoencodersscalablevision}, and in audio by HuBERT~\citep{hsu2021hubert}.

**Supervised Pretraining**

Some tokenizers are pretrained on specific tasks using supervised learning, aiming to acquire task-specific representations through labeled datasets.
These models are initially trained on large-scale datasets to capture specific features of the input data.
In the vision modality, supervised tasks include semantic segmentation, object detection, and depth estimation.
Models trained for these tasks, such as SAM~\citep{sam,samclip}, ViTDet~\citep{vitdet}, and MiDaS~\citep{midas}, are later used in LMMs as tokenizers, like in DeepSeek-VL~\citep{DeepSeek-VL} and Cambrain-1~\citep{Cambrian-1}, to extract diverse visual features from input data.
In the audio modality, Whisper~\citep{radford2023robust} is trained with 680,000 hours of labeled audio data in a weakly supervised manner.
Thanks to its robust and powerful speech feature extraction capabilities, Whisper is widely used in Speech LLMs~\cite{tang2023salmonn, chu2023qwen, hu2024wavllm} for extracting speech embeddings.

**Contrastive Learning**

Contrastive Learning is a self-supervised learning method that focuses on learning representations by distinguishing between positive and negative pairs.
The core idea is to bring similar (positive) examples closer together in the representation space while pushing dissimilar (negative) examples further apart.
The items in each pair can belong to the same or different modalities.
For example, DINO~\citep{caron2021emerging} uses image-image pairs to enhance vision representation, while CLIP~\citep{radford2021clip} employs text-image pairs to improve language alignment within vision representation.

Currently, LMMs that only feature multimodal understanding capabilities, such as InstructBLIP~\cite{dai2023instructblip} and LLaVA~\cite{liu2023llava}, opt for tokenizers with superior representation abilities like CLIP~\citep{radford2021clip}, as they do not require reconstruction of the multimodal information.
Conversely, LMMs supporting multimodal generation capabilities tend to choose VQVAE as the tokenizer, exemplified by models like Unified-IO~\cite{lu2022unifiedio}, Chameleon~\cite{chameleonteam2024chameleon}, Emu3~\citep{Emu3}, among others~\citep{wang2024mio, seedllama, wang2022ofa}.

## 2.2·Discrete Tokenization Basics: 离散分词基础

<a id="Section.2.2"></a>

Unlike the language modality, which inherently comprises discrete symbols (e.g., tokens or words), most other modalities naturally exist in a continuous space.
To bridge the gap, the core technique is **Vector Quantization (VQ)**, which aims to map the original continuous information into a compressed, finite representation space, i.e.
discrete tokens.
The discrete tokens can have 2-dimensional or 3-dimensional structures for images and videos.
These tokens are initially linearized based on a specific order, such as left to right and top to bottom, transforming them into a 1-dimensional sequence.
This linearization allows for effective modeling usingthe next token prediction objective.


In this section, we will first elaborate on modern vector quantization techniques widely used as multimodal tokenizers, such as VQVAE (\S~\ref{sec:vq}) and its variants.
Following that, we will introduce the specific optimizations of discrete tokenization in different modalities (\S~\ref{sec:vq app}).

### 2.2.1·Vector Quantization Methods: 向量量化方法

The origins of VQ method trace back to the 1950s at Bell Laboratories, where researchers endeavored to optimize signal transmission through the development of suitable discretization procedures~\cite{Pags2015IntroductionTV}.
In essence, quantization is the process of mapping an infinite set of continuous values to a smaller, discrete set of finite values.
The primary objective of vector quantization is to reconstruct all the information in the original data as accurately as possible with a finite set of vectors, which is also called the \emph{codebook}.

**Vanilla VQ**

The original VQVAE proposed by \citet{Oord2017NeuralDR} is a milestone of many successive vector quantization methods.
As shown in Figure~\ref{fig:vqvae}, a VQVAE consists of three main components: the encoder, the quantizer, and the decoder.
The encoder comprises the input data to a compact latent space, the quantizer select the nearest code vectors from the finite codebook to approximate the continuous latents, the decoder reconstruct the input data using the discrete codes.
When training the VQVAE, three main loss components are crucial: reconstruction loss, codebook loss, and commitment loss~\citep{Oord2017NeuralDR}.
The reconstruction loss, often implemented as mean squared error or binary cross-entropy, ensures accurate data reconstruction by minimizing differences between input and output.
Codebook loss, or vector quantization loss, enables effective encoding by aligning encoder outputs with nearest codebook entries, ensuring discrete latent variables.
Meanwhile, commitment loss acts as a regularizer, encouraging encoder outputs to stay close to codebook entries to maintain stable learning, preventing erratic mapping.
As gradient can not pass the quantization operator (finding the nearest code), the straight-through estimator~\cite{bengio2013estimatingpropagatinggradientsstochastic} is adopted to let the gradient flow normally.


Recent advancements in vector quantization methods have focused on achieving better image reconstruction and enhancing generative capabilities.
To improve reconstruction quality, both architectural innovations and codebook designs have been proposed.
Transformer-based frameworks, such as ViT-VQGAN~\citep{yu2022vectorquantized}, Swin-MAE~\citep{xu2023swin}, Swin-Unet~\citep{cao2021swinunet}, and Efficient-VQGAN~\citep{cao2023efficientvqgan}, replace traditional CNN encoders and decoders with more robust modules like ViT~\citep{vit} and Swin-Transformer~\citep{liu2021swinTransformer,liu2022swinV2}, leading to better feature representations and reconstruction fidelity.
Additionally, several methods such as LFQ~\citep{magvit2} and FSQ~\citep{FSQ} are proposed to address the significant challenge of codebook collapse during **codebook learning**, where a large portion of code embeddings are not used when enlarging the codebook size, causing a redundancy in the codebook and limiting the
expressive power of the generative model~\citep{baykal2024edvaemitigatingcodebookcollapse}.
For improved generative performance and efficiency, several approaches have been introduced.
\citet{tian2024VAR} propose Visual Autoregressive modeling, which facilitates image generation through "next-scale prediction", moving away from the traditional raster-scan "next-token prediction" used in standard VQVAE-based models.
RQ-Transformer~\citep{lee2022RQVAE} employs residual quantization (RQ) to precisely approximate feature maps and reduce spatial resolution.
RQ helps the RQ-Transformer to significantly reduce computational costs and effectively learn long-range interactions in inputs.
RAR~\citep{RAR} introduces a randomness annealing strategy with a permuted objective, enhancing the model's ability to learn bidirectional contexts while retaining the autoregressive framework.
TiTok~\citep{yu2024imageworth32tokens} tokenizes images into 1D latent sequences, providing a more compact latent representation that is substantially more efficient and effective than conventional techniques.
It greatly reduces the number of tokens required to encode an image compared to previous methods~\citep{cao2023efficientvqgan,yu2022vectorquantized}.


\paragraph{VQ with Auxiliary Losses}
The primary goal of the vanilla VQVAE is to accurately reconstruct input data by minimizing the mean squared error loss.
However, this auto-encoding objective doesn't always align with human perception of the quality of reconstructed data.
For example, in the visual modality, the vanilla MSE loss often results in images with blurred details, particularly in human faces~\citep{larsen2016autoencodingpixelsusinglearned}.
To address this issue, several approaches introduce higher-level training objectives aimed at improving the overall quality of the output data.
In the realm of vision, perceptual loss~\citep{johnson2016perceptuallossesrealtimestyle} is widely used to enhance the quality of reconstructed images by leveraging a pre-trained CNN.
VQGAN~\citep{cao2023efficientvqgan} incorporates a discriminator network to enhance image fidelity by adding an adversarial training objective.
The role of the discriminator is to discern between the reconstructed and original images, while the VQ-VAE is optimized to deceive the discriminator, thereby improving the quality of the reconstructed images.
In the audio modality, it is essential to decouple the audio into its acoustic and semantic components to achieve both powerful audio reconstruction quality and LLM modeling.
SpeechTokenizer~\citep{zhang2023speechtokenizer} and Mimi~\citep{defossez2024moshi} introduce the loss of semantic distillation at the first layer of Residual VQ, using self-supervised models, such as HuBERT~\citep{hsu2021hubert} and WavLM~\citep{chen2022wavlm}.

**Residual Vector Quantization**

Residual vector quantization (RVQ) has been used for image~\citep{Lee_Kim_Kim_Cho_Han_2022} and audio~\citep{Zeghidour_Luebs_Omran_Skoglund_Tagliasacchi_2022} generation, where quantized codes are refined by storing additional quantized residuals.
\citet{lee2022RQVAE} propose the RQVAE that also introduces a residual quantization to recursively quantize the feature map in a coarse-to-fine manner, employing a fixed-size codebook to maintain both precision and code diversity.

**Product Quantization**

\citet{PO-VAE} propose product quantization (PQ), to factor the codebook into a product of smaller codebooks, allowing for high-quality quantizers without the requirement of intractably large codebooks.

**Multi-scale Quantization**

~\citet{tian2024VAR} introduce the Visual Autoregressive modeling (VAR), which develops a multi-scale quantization autoencoder that encodes images into $K$ multi-scale discrete token maps using a shared codebook.
It aids the model in generating images through "next-scale prediction," instead of the raster-scan "next-token prediction" typically used in standard VQVAE-based models.
The multi-scale quantization enables the model to learn visual distributions and demonstrates strong generalization capabilities.

**Finite Scalar Quantization**

To generate concise and expressive tokens using a larger token vocabulary and avoid codebook collapse, \citet{FSQ} propose finite scalar quantization (FSQ).
FSQ projects the VAE representation down to a few dimensions that can be quantized into fixed values, creating an implicit codebook.

**Look-up Free Quantization**

LFQ~\citep{yu2023language} reduces the embedding dimension of the codebook to zero, effectively replacing the codebook with an integer set.
It allows VQVAE to improve the quality of image reconstruction and generation by vastly increasing the vocabulary size by magnitudes.
For example, the rFID on Imagenet decreases from 2.5 to 1.4 when the LFQ vocabulary size increases from $2^10$ to $2^16$ on ImageNet dataset.

**Embedding-Free Quantization**

Maskbit~\cite{maskbit} explores an embedding-free tokenization approach that utilizes binary quantization.
It projects latent embeddings into K dimensions and then quantizes them based on their sign values to produce bit token representations.
The generated bit tokens exhibit highly structured semantic representations, which are crucial for generation tasks.

**Group Vector Quantization**

Unlike RVQ which models the information residually, Group Vector Quantization models the information across different dimensions.
In the audio domain, HiFi-Codec~\citep{yang2023hifi} proposes a group-residual vector
quantization technique to reduce the number of codebooks, while FACodec~\cite{ju2024naturalspeech} disentangles speech into prosody information, content information, and acoustic details using three-factorized vector quantizers.

### 2.2.2·Evaluation of VQ Tokenizers

When evaluating VQVAEs, two critical metrics are commonly considered: **reconstruction ability** and **generation ability**.

Reconstruction ability refers to how well the VQVAE can reproduce the original input data after encoding and decoding.
This metric evaluates the fidelity of the model in terms of how accurately it can reconstruct the input data from its latent representations.
L2 distance, Peak Signal-Noise Ratio (PSNR), and reconstruction Fréchet Inception Distance (rFID) are often applied to assess the reconstruction ability.

Generation ability assesses the model’s capacity to generate new, plausible samples from the learned distribution in the codebook space.
This metric evaluates the creativity and diversity of the VQVAE in producing new data that is consistent with the training data distribution.
To quantitatively evaluate generation ability, metrics such as the Inception Score (IS) and generation Fréchet Inception Distance (gFID)~\citep{heusel2018ganstrainedtimescaleupdate} are often used.

rFIDs are often computed between ImageNet validation images and their reconstructed images.
gFIDs are usually computed against the training set with ADM's evaluation suite~\cite{dhariwal2021diffusionmodelsbeatgans}.

## 2.3·Discrete Tokenization for Different Modalities: 不同模态的离散分词

<a id="Section.2.3"></a>

Generic quantization methods provide basic ways to convert continuous data into discrete tokens.
However, there isn't a single quantizer that works well for all modalities because each modality has unique characteristics.
Therefore, it is important to create specific tokenizers for each modality.
This section will explain the unique features of different modalities and showcase some examples of tokenizers for images, audio, and video, among others.

### 2.3.1·Image: 图像

Images can be tokenized into discrete symbols with the previously introduced VQVAE structure.
Compared to text tokens, images diverge in three fundamental aspects that significantly impact how they should be tokenized:

1.
Rich Information Granularity: Unlike text, which primarily encapsulates high-level semantic meaning, images are contain with a myriad of perceptual details.
These encompass low-level visual elements such as colors, shapes, and textures, alongside more abstract concepts like objects and actions.
2.
Dense Information: Images inhabit a densely packed representational realm, where each pixel, across multiple dimensions including height, width, and color channels (RGB being a common example), carries information.
This stands in stark contrast to the discreteness of text in nature, characterized by sequentially arranged words.
3.
Two-Dimensional Spatial Structure: Images are inherently structured in two dimensions, spread across a grid defined by height and width.
This 2D layout differs fundamentally from the straightforward, one-dimensional sequence that characterizes textual data, introducing unique complexities in their processing and analysis.

Given these differences, bridging the gap between text and image modalities in the training of LLMs based on discrete image tokens requires a robust image tokenizer, which must balance the fusion of sufficient alignment with LLM's language ability (referred to as "representation"), the retention of rich original image information (referred to as "reconstruction"), and the efficient use of tokens given the growing inference cost of transformer decoder (referred to as "token efficiency").
These factors possess a trade-off~\citep{seedllama,seed-tokenizer, magvit2, sun2023generative}, making it crucial for the construction of an image tokenizer to maintain equilibrium among these factors.

In terms of better representation, models like ViT~\citep{vit} are commonly employed, often aligned with a text encoder through contrastive loss~\citep{radford2021clip, peng2022beit}, or aligned with text modalities through generative loss~\citep{coca}.
Additionally, modules like Q-Former~\citep{li2023blip2} can also be used for image feature transformation~\citep{li2023blip2, seedllama}.
Consequently, the resultant image features integrate higher-level semantics and gradually compress high-dimensional images into lower-dimensional representations aligned with text.
While the initial arrangement of image patches follows a raster order, preserving intrinsic sequential relationships, this configuration lacks causal semantics, posing challenges for language modeling.

Regarding reconstruction ability, an image decoder is often layered atop the image encoder to reconstruct the original image from its representation, incorporating reconstruction loss into the training process~\citep{amused, seedllama, lavit, Esser2020TamingTF}.
Training labels typically use the original images, but with advancements in diffusion models, more research is incorporating latents for diffusion models as reconstruction labels~\citep{lavit, seedllama}.

For token efficiency, modules like selectors or mergers for image tokens are utilized to truncate their length (i.e., the number of tokens per image).
For instance, SEED-LLaMA~\citep{seedllama} compresses longer image features encoded by ViT into 32 continuous tokens using a Causal Q-Former and then discretizes them through quantization.
LaViT~\citep{lavit} first predicts whether each patch token should be selected using a shared MLP, and then compresses the image length by employing selected patches as queries and unselected patches as keys and values in cross-attention blocks~\citep{seedllama}.

Beyond these aspects, some studies also focus on the unique properties of specific image types or tasks.
For example, VQ-IMG aims to enhance the modeling capabilities of image tokenizers for faces~\citep{make-a-scene}, while LVM integrates tasks like segmentation and object detection during the training of models based on VQGAN to enrich the representation of image tokens~\citep{bai2023sequential}.
StrokeNVWA introduces a VQ-Stroke method to discretize vector graphic images into stroke tokens~\citep{strokenvwa}.

### 2.3.2·Audio: 音频

Raw audios are typically stored as 16-bit integer values with a sampling rate that exceeds tens of thousands values per second, which leads to extremely long sequences and renders next token prediction training more difficult.
Versatile quantization methodologies have been investigated for audio tokenization.
Initially aimed at audio compression, these methodologies have more recently been developed to create compact semantic and acoustic representations in the context of NTP language modeling.

As a traditional companding algorithm, $\mu$-law/A-law algorithm is commonly employed in speech generative models such as WaveNet~\citep{van2016wavenet}.
While this algorithm projects each audio frame to an 8-bit value, it does not reduce the sampling rate, thereby preserving overlong sequences.
Self-supervised learned models have shown exceptional performance in various speech-related tasks, sparking interest in clustering their speech representations for speech quantization.
The vq-wav2vec~\citep{baevski2019vq} uses either a Gumbel-Softmax or online k-means clustering to quantize the SSL-learned dense representation.
HuBERT~\citep{hsu2021hubert} is trained with a masked prediction task, whose targets are obtained through k-means clustering of learned features from earlier iterations.
Utilizing quantized tokens learned with Self-Supervised Learning (SSL), GSLM~\cite{lakhotia2021gslm} and VQTTS~\citep{du2022vqtts} demonstrate faster speed in speech generation tasks compared with WaveNet.
Because SSL tokens are extracted with highly abstracted semantics while discarding low-level acoustic information, the reconstruction quality is relatively low, and speaker identity is lost~\citep{borsos2023audiolm}.
Neural codec models typically apply a VQ-VAE on the raw audios with residual vector quantization, exemplified by SoundStream~\citep{zeghidour2021soundstream} and EnCodec~\citep{encodec}.
They are originally designed for audio compression, have the capability to encode waveforms into discrete codes and faithfully reconstruct them back into high-quality waveforms.
Recently, they are widely used in audio generation models such as AudioLM~\citep{borsos2023audiolm}, VALL-E~\citep{wang2023neural} and their variants~\citep{han2024valler,song2024ellav, wang2023viola}, and reach new state-of-the-art performance on various tasks.
Compared with traditional $\mu$-law/A-law algorithms, codec models can efficiently reduce the length of token sequences.
It can also maintain multi-scale acoustic information indicating speaker identity compared with highly-abstracted SSL-learned discrete tokens such as HuBERT~\citep{hsu2021hubert} tokens.
Additionally, the codec models are typically off-the-shelf and lightweight.

Latest works have attempted to impose additional supervision on the discrete codes extracted by codec models.
The objective is to enhance their ability to extract and encode higher-level semantic information, thereby improving language modeling.
SpeechTokenizer~\citep{zhang2023speechtokenizer} is an RVQ-based codec model, where its first-layer codebook incorporates semantic information through the semantic distillation process, using HuBERT~\citep{hsu2021hubert} representations as the semantic teacher.
Mimi, used by Moshi~\citep{défossez2024moshispeechtextfoundationmodel}, further improves upon this by replacing the semantic teacher from HuBERT with WavLM~\citep{chen2022wavlm}.
Additionally, it isolates the first-layer codebook from the RVQ process to achieve better semantic and acoustic disentanglement.
To enhance the compression rate, WavTokenizer~\citep{ji2024wavtokenizer} is capable of quantizing one-second audio into 75 or 40 tokens with a single quantizer.

### 2.3.3·Video: 视频


Compared to images, videos introduce an additional temporal dimension that must be considered during the tokenization process.
A straightforward strategy is to utilize an image-based VQVAE model to tokenize the video frame-by-frame.
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

### 2.3.4·More Modalities: 更多模态

Modeling various information as discrete tokens has gone far beyond the traditional text, image, video and audio modalities.
In the computer vision field, we can unify the output spaces of tasks like object detection, semantic segmentation, and depth mapping into images.
These can then be tokenized into discrete image tokens, allowing us to train a single NTP model to handle all these tasks \cite{wang2022ofa,wang2023images,bai2023sequential}.
In **robotics and embodied AI** domain, the robots actions in response to the environments can be coded into various discrete tokens and learn the policy in NTP manner as shown in recent studies such as VIMA~\cite{jiang2023vima}, RT2~\cite{brohan2023rt2} and Locomotion NTP~\cite{humanoid}.
In **AI4Science**, by factorizing various proteins into DNA token sequences, protein language models are capable of learning from a wide array of sequences that span the evolutionary tree.
These models have demonstrated their efficacy as powerful tools for sequence design and protein engineering, as highlighted in studies ~\cite{madani2023large,ruffolo2024designing}.

## 2.4·Continuous Tokenization Basics: 连续分词基础

<a id="Section.2.4"></a>

Continuous tokens represent non-textual modalities in a continuous feature space, offering less information loss~\citep{dnd-transformer} and improved data representation compared to discrete tokens~\citep{xie2024showosingletransformerunify}.
However, their dense feature encapsulation makes direct mapping to a fixed vocabulary challenging, unlike discrete tokens.
It poses a challenge for LLMs aiming to comprehend and generate such information in a NTP manner.

To handle continuous multimodal token inputs for LLM to understand, transformations or adapters are necessary to balance data representation and text alignment.
For multimodal generation, modifying the output head to align with non-textual modality specific decoders' input feature space is also crucial.
The following subsections introduce the basic designs and change for LLMs to accommodate continuous multimodal token from multimodal understanding (\S\ref{sec: Tokenize Continuous Input}) and generation (\S\ref{sec: De-tokenize Continuous Output}) perspectives.

### 2.4.1·Tokenize Continuous Input for Understanding

To effectively integrate raw non-textual modality data into Large Language Models (LLMs), two key steps are typically undertaken: (1) encoding the data into a more suitable representation space, and (2) aligning it with the LLM’s feature space.

**Encoding**

The encoding of non-textual modality data aims to capture meaningful features and important nuances that are essential for the understanding of the data.
This can be achieved through different types of encoders such as Transformer-based encoders~\citep{li2023blip2, liu2023llava, liu2023llava15,zhu2023minigpt4, radford2021clip} or CNN-based encoders~\citep{davinci, zhang2023universal, jiang2023vima, alayrac2022flamingo}.
There's also an option to go encoder-free~\citep{kim2021vilt, fuyu}, which allows for raw data to be fed directly into the model.

Transformer-based encoders are widely used for their robust representation capabilities and generalizability~\citep{vaswani2017attention, vit}.
For a non-textual modality sample, the input is initially divided into patches and transformed into a 1D sequence, with each patch represented as a soft token.
This sequence is then processed through the Transformer's encoder layers, employing self-attention mechanisms to capture relationships between patches.
Consequently, the model produces a rich representation of the input.
Typically, there are two types of encoders: (1) unimodal encoders, designed to process information from a single modality~\citep{vit, sam, arnab2021vivit, usm, mert, prismer, beit, liu2021swinTransformer}; and (2) multi-modal encoders, capable of integrating information from multiple modalities~\citep{radford2021clip, imagebind, eva-clip, clap, anymal, imu2clip, coca}.
For instance, PaLM-E~\citep{driess2023palme}, Unified-IO-2~\citep{lu2023unifiedio}, and PaLI~\citep{pali} use ViT~\citep{vit} encoders trained solely on visual data.
Conversely, LLaVA~\citep{liu2023llava}, Emu~\citep{sun2023emu1, sun2023generative}, and Qwen-VL~\citep{QwenVL} utilize CLIP~\citep{radford2021clip} or EVA-CLIP~\citep{eva-clip} encoders with contrastive loss to align textual and non-textual representations.
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
Although multi-modal encoders like CLIP~\citep{radford2021clip} have made strides in narrowing the gap, they still encounter two significant challenges: (1) the presence of redundant continuous tokens~\citep{alayrac2022flamingo, perceiver, li2023blip2}; and (2) a lack of contextual semantics, such as causal semantics, because they are typically trained only with image-caption paired data rather than image-text interleaved data or image-prompt instructional data~\citep{seed-tokenizer, gemini1, laurencon2023obelics, zhu2023multimodal}.
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

### 2.4.2·De-tokenize Continuous Output for Generation


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

## 2.5·Continuous Tokenization for Different Modalities: 不同模态的连续分词

<a id="Section.2.5"></a>

While the aforementioned workflow and categorization outline a general approach to continuous multimodal tokenization, research indicates that employing modality-specific encoders, tailored to each modality, can significantly enhance performance~\citep{navit, fixres, anymal}.
Given the unique characteristics of different modalities, these approaches introduce specific inductive biases into the tokenization process.

### 2.5.1·Images: 图像


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

### 2.5.2·Audio: 音频

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
HuBERT~\cite{hsu2021hubert} employs an offline clustering step to provide aligned target labels for a BERT-like prediction loss, which is applied solely on the masked regions.
WavLM~\cite{chen2022wavlm} introduces denoising in pretraining, jointly with regular masked speech prediction, as HuBERT.
Whisper~\citep{radford2023robust}  is a speech recognition model characterized by an attention-based encoder-decoder architecture, trained on web-scale labeled speech data.
It is increasingly being employed as a foundational speech model, extending its applications beyond speech recognition tasks~\citep{hu2024wavllm,tang2023salmonn,meng24c_interspeech,meng2024llm}.

For continuous tokenization of audio, AST~\citep{ast} uses a convolution-free pure-transformer architecture to extract features for audio classification, drawing insights from ViT~\citep{vit}.
Inspired by CLIP~\citep{radford2021clip}, CLAP~\citep{clap} introduces a contrastive language-audio pre-training task to learn text-enhanced audio representations using supervised audio and text pairs.
Fine-tuned based on a pre-trained CLIP model, Wav2CLIP~\citep{wu2022wav2clip} and AudioCLIP~\citep{guzhov2022audioclip} incorporate an additional audio encoder using supervised pairs of audio and class labels.
Audio-MAE~\citep{huang2022masked} adopts a Transformer-based encoder-decoder framework to learn audio representations.
Similar to MAE, it uses a reconstruction pre-training task where the decoder is tasked with reconstructing masked patches from the encoded information of the unmasked patches.
BEATs~\citep{chen2022beats} introduces a self-distilled tokenizer that converts continuous audio signals into discrete labels, facilitating classic mask and discrete label prediction pre-training.

### 2.5.3·Video: 视频

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
