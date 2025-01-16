# Sec.02·MultiModal Tokenization: 多模态分词

Tokenization is the first and a fundamental step for multimodal sequential modeling under the next token prediction framework.
It decomposes information from various sources, such as images, videos, and audio clips, into a sequence of minimal, manageable units known as tokens for the NTP model to learn.
Table.02 provides an overview of the tokenizers used across various modalities in recent research.

Despite being derived from various modalities, these tokenization methods can all be categorized into two prototypes: **discrete tokenization**  and **continuous tokenization**.
In this section, we will initially introduce the general definition and basics techniques of training multimodal tokenizers ([Section.2.1](#Section.2.1)), then the fundamentals and applications of discrete tokens ([Section.2.2](#Section.2.2), [Section.2.3](#Section.2.3)) and continuous tokens ([Section.2.4](#Section.2.4), [Section.2.5](#Section.2.5)) in NTP framework.

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

## 2.3·Discrete Tokenization for Different Modalities: 不同模态的离散分词

<a id="Section.2.3"></a>

## 2.4·Continuous Tokenization Basics: 连续分词基础

<a id="Section.2.4"></a>

## 2.5·Continuous Tokenization for Different Modalities: 不同模态的连续分词

<a id="Section.2.5"></a>
