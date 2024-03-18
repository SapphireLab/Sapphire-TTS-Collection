---
PDF: 
标题: Deep Learning-Based Expressive Speech Synthesis - A Systematic Review of Approaches, Challenges, and Resources
作者:
 - Huda Barakat
 - Oytun Turk
 - Cenk Demiroglu
机构:
 - OzyeginU
代码: 
ArXiv: 
提出时间: 
出版社: 
发表期刊: 
发表时间: 
引文数量: 
被引次数: 
tags:
 - 综述
 - 语音合成
 - 情感语音
---
# Deep Learning-Based Expressive Speech Synthesis: A Systematic Review of Approaches,  Challenges, and Resources<br>基于深度学习的表现性语音合成: 方法, 挑战, 资源的系统性回顾

## A.摘要

## 1.引言

## 2.方法

## 3.监督学习方法

## 4.无监督学习方法

## 5.表现性语音合成的主要挑战

> In this section, we list and explain the most important challenges that face expressive TTS models and the main solutions that have been proposed in the literature to overcome these challenges. We then provide a summary of papers addressing each challenge in [Table 5]().

本节列出并展示表现性语音合成模型面临的最重要挑战, 以及在现有文献中提出的克服这些挑战的主要解决方案. 在表格五种提供了解决每个挑战的文献的总览.

|参考文献|信息泄露|缺少参考音频|韵律可控性|未知风格/说话人|
|---|:-:|:-:|:-:|:-:|
||√|√|√|√|
||√|√||√|
|[097](#K.Lee2021)|√||√|√|
||√|||√|
|[102](#K.Zhang2022)|√||||
|||√|√||
|[047](#C.Lu2021) [111](#D.Tan2020)|√|√|||
|[019](#S.Jo2023)|√||√||
||||√||
|||√|||


### 5.1.无关信息泄露 (Irrelevant Information Leakage)

> One main problem in unsupervised approaches that rely on having a style reference or a prompt, is the leakage of irrelevant information, like speaker or text related information, into the generated style or prosody embedding. 
> This irrelevant information within the speech style can lead to degradation in the quality of the synthesized speech. 
> As a result, many studies have investigated this problem, and several solutions have been proposed as outlined below.

在依赖风格参考或提示的无监督学习方法中, 一个主要的问题是无关信息的泄露, 如说话人或文本相关的信息进入到生成的风格或韵律嵌入.
这种语音风格中的无关信息可能导致合成语音的质量下降.
因此许多文献研究了这一问题并提出了以下几种方案.

#### 5.1.1.对抗性训练 (Adversarial Training)

> [Adversarial training](#Y.Ganin2016) is one of the widely used techniques to confront the information leakage problem.
> Typically, a classifier is trained to distinguish the type of unwanted information (such as speaker or content information) that is leaking from the prosody reference audio into the generated prosody embedding. 
> During the training process, the weights of the employed prosody encoder/extractor from the reference audio are modified with gradient inversion of the proposed classifier. 
> In other words, the classifier penalizes the prosody encoder/extractor for any undesired information in its output. 
> A **Gradient Reversal Layer (GRL)** is usually used to achieve the inversion of the classifier gradients.

对抗训练是广泛用于处理信息泄露的一种技术.
通常, 训练一个分类器用于区分从韵律参考音频泄露到生成的韵律嵌入中的不需要的信息类型 (例如说话人或内容信息).
在训练过程中, 参考音频中使用的韵律编码器或提取器的权重被分类器的梯度反转修改.
换句话说, 分类器在它的输出中对任何不需要的信息惩罚韵律编码器或提取器.
通常使用**梯度反转层 (Gradient Reversal Layer, GRL)** 来获得分类器梯度的反转.



> Several studies utilize adversarial training to prevent the flow of either speaker or content-related information from the given reference audio to the resulting prosody embedding. 
> For instance, the [VAE-TTS model](#C.Lu2021) learns phoneme-level 3-dimensional prosody codes. 
> The VAE is conditioned on speaker and emotion embeddings, besides the tone sequence and mel-spectrogram from the reference audio. 
> Adversarial training using a **Gradient Reversal Layer (GRL)** is applied to disentangle speaker and tone from the resulting prosody codes.
> Similarly, adversarial training is introduced to the style encoder of the [cross-speaker emotion transfer model](#S.Jo2023) to learn a speaker-independent style embedding, where the target speaker embedding is provided from a separate speaker encoder.

有几项研究利用对抗训练来防止说话人或内容相关信息从给定参考音频流动到生成韵律嵌入.
例如, VAE-TTS 模型学习音素级别的三维韵律编码.
用说话人和情感嵌入, 参考音频的语调序列和梅尔频谱条件化 VAE.
采用**梯度反转层**的对抗学习用于解耦韵律编码中的说话人和语调.
类似地, 跨说话人情感转移模型中的风格编码器也引入了对抗训练用于学习说话人独立的风格嵌入, 其中目标说话人嵌入由单独的说话人编码器提供.

> The [STYLER model](#K.Lee2021) employs multiple style encoders to decompose the style reference into several components, including duration, pitch, speaker, energy, and noise. 
> Both channel-wise and frame-wise bottleneck layers are added to all the style encoders to eliminate content-related information from the resulting embeddings. 
> Furthermore, as noise is encoded individually by a separate encoder in the model, other encoders are constrained to exclude noise information by employing either domain adversarial training or residual decoding.

在 STYLE 模型中使用了多个风格编码器将风格参考分解为多个成分, 包括时长, 音高, 能量和噪声.
在所有的风格编码器中添加通道级和帧级瓶颈层从导出的嵌入中排除内容相关的信息.
此外, 由于噪声在模型中通过单独的编码器编码, 通过应用领域对抗训练或残差解码来约束其他编码器以排除噪声信息.

> In [111](#D.Tan2020), prosody is modeled at the phone-level and utterance-level by two separate encoders. 
> The first encoder consists of two sub-encoders: a style encoder and a content encoder, besides two supporting classifiers. 
> The first classifier predicts phone identity based on the content embedding, while the other classifier makes the same prediction but based on the style embedding. 
> The content encoder is trained via collaborative training with the guidance of the first classifier, while adversarial training is used to train the style encoder, utilizing the second classifier.

文献 111 中韵律通过两个单独的编码器在音素级别和语调级别进行建模.
第一个编码器由两个子编码器组成: 一个风格编码器和一个内容编码器, 以及两个支持分类器.
首个分类器基于内容嵌入预测音素标识, 其他分类器则基于风格嵌入进行相同的预测.
结合第一个分类器的指导采用协作训练来训练内容编码器, 利用第二个分类器采用对抗训练来训练风格编码器.

> On the other hand, [102](#K.Zhang2022) proposes adversarial training for the style reference by inverting the gradient of an **Automatic Speech Recognition (ASR)** model. 
> The proposed model introduces a shared layer between an ASR and a reference encoder-based model. 
> Specifically, a single BiLSTM layer from the listener module of a pre-trained ASR model serves as the prior layer to the reference encoder. 
> The process starts by passing the reference Mel-spectrogram to the shared layer to produce the shared embedding as input to both the reference encoder and the ASR model. 
> A **Gradient Reversal Layer (GRL)** is employed by the ASR model to reverse its gradient on the shared layer. 
> Accordingly, the reference encoder parameters are modified so that the ASR model fails to recognize the shared embedding, and thus content leakage to the style embedding from the reference encoder is reduced.

另一方面, 文献 102 提出通过反转自动语音识别模型的梯度对风格参考进行对抗性训练.
该模型在 ASR 和基于参考编码器的模型之间引入了一个共享层.
具体地, 预训练 ASR 模型中听众模块中的单个 BiLSTM 层作为参考编码器的前一层.
将参考梅尔频谱传递到共享层用于生成共享嵌入, 作为参考编码器和 ASR 模型的输入.
ASR 模型通过使用梯度反转层来反转共享层的梯度.
因此, 参考编码器的参数被修改使得 ASR 模型无法识别共享嵌入, 从而减少从参考编码器到风格嵌入器的内容泄露.


#### 5.1.2.韵律分类器 (Prosody Classifiers)

> This is a supporting approach used by some studies to produce more discriminative prosody embeddings by passing them to a prosody classifier. 
> This method can be applied when the training data is labeled with emotion or style labels. 
> In the [two consecutive studies](#T.Li2022) [](#T.Li2021) from the same research group, an auxiliary reference encoder is proposed and located after the decoder of the [baseline TTS model](#R.Skerry2018). 
> The two reference encoders in the model are followed by emotion classifiers to further enhance the discriminative nature of their resulting embeddings.
> However, the emotion embedding that is passed to the TTS model is the output of an intermediate hidden layer of the classifiers. 
> In addition to the classification loss, an additional style loss is also applied between the two emotion embeddings from the two employed emotion classifiers.

这是某些研究中采用的支持方法, 通过将它们传递给韵律分类器用于产生更具区分性的韵律嵌入.
当训练数据带有情感或风格标签时可以采用此方法.
在同一研究小组的两项连续研究中, 给基线 TTS 模型的解码器后添加一个辅助的参考编码器.
模型中的两个参考编码器后跟着情感分类器用于进一步增强其生成嵌入的区分性.
然而传递给 TTS 模型的情感嵌入是分类器的中间隐藏层的输出.
除了分类器损失外, 还应用了来源于两个情感分类器的情感嵌入之间的附加风格损失.

> In [36], alongside the text encoder, two encoders are introduced to generate embeddings for speaker and emotion from a reference audio. 
> To further disentangle emotion, speaker, and text information, both speaker and emotion encoders are supported with a classifier to predict speaker and emotion labels, respectively. 
> Similarly, in paper [39], a model with two encoders and two classifiers is proposed to produce disentangled embeddings for speakers and emotions from a reference audio. 
> However, the paper claims that some emotional information is lost during the process of disentangling speaker identity from the emotion embedding. 
> As a result, an ASR model is introduced to compensate for the missing emotional information. 
> The emotion embedding is incorporated within a pre-trained ASR model through a **Global Context (GC)** block. 
> This block extracts global emotional features from the ASR model’s intermediate features (AIF). 
> Subsequently, a prosody compensation encoder is utilized to generate emotion compensation information from the output of the AIF layer, which is then added to the emotion encoder output.

在文献 [36] 中, 除了文本编码器之外, 还引入了两个编码器从参考音频中生成说话人和情感的嵌入.
为了进一步解耦情感, 说话人和文本信息, 说话人和情感编码器都用一个分类器进行支持, 分别用于预测说话人和情感标签.
类似地, 文献 [39] 中, 提出了一个具有两个编码器和两个分类器的模型以从参考音频中产生解耦的说话人和情感嵌入.
然而该文献声称再从情感嵌入中解耦说话人标识时丢失了一些情感信息.
因此引入了一个 ASR 模型用于补偿丢失的情感信息.
情感嵌入通过全局文本块被整合进预训练 ASR 模型中.
这个块从 ASR 模型的中间特征 (AIF) 中提取全局情感特征.
然后, 使用韵律补偿编码器从 AIF 层的输出中生成情感补偿信息, 之后加入到情感编码器的输出中.

#### 5.1.3.信息瓶颈 (Information Bottleneck)

> The information bottleneck is a technique used to control information flow via a single layer/network. 
> It helps prevent information leakage as it projects input into a lower dimension so that there is not enough capacity to model additional information and only important information is passed through it. 
> In other words, the bottleneck can be seen as a down-sampling and up-sampling filter that restricts its output and generates a pure style embedding. 
> Several prosody-reference based approaches, as in [86] [93] [97] [101] [130], have employed this technique to prevent the flow of speaker or content-related information from the reference audio to the prosody embedding.

信息瓶颈是一种通过单层或网络控制信息流的技术。
它有助于防止信息泄露，因为它将输入投射到较低维度从而没有足够的容量来建模额外信息，只有重要信息通过它。
换句话说，信息瓶颈可以视为一个下采样和上采样滤波器，它限制了输出并生成纯风格嵌入。
有几项基于韵律参考的研究已经应用了这一技术用于防止说话人或内容相关的信息从参考音频流入韵律嵌入。

> In [93], a bottleneck layer named sieve layer is introduced to the style encoder in GST-TTS to generate pure style embedding. 
> Similarly, in the multiple style encoders model STYLER [97], each encoder involves a channel-wise bottleneck block of two bidirectional-LSTM layers to eliminate content information from encoders’ output. 
> Another example is the cross-speaker-style transfer Transformer-TTS model proposed in [86] with both speaker and style embeddings as input to the model encoder. 
> The speaker-style-combined output from the encoder is then passed to a prosody bottleneck sub-network, which produces a prosody embedding that involves only prosody-related features. 
> The proposed bottleneck sub-network consists of two CNN layers, a squeeze-and-excitation (SE) block [152], and a linear layer. 
> The encoder output is then concatenated with the resulting prosody embedding and used as input to the decoder.

- [文献 093]() 在 GST-TTS 的风格编码器中引入了一个名为筛层 (Sieve Layer) 的瓶颈层用于生成纯风格嵌入.
- [文献 097]() 提出的多风格编码器模型 STYLER 中，每个编码器都包含两个双向 LSTM 层通道级别瓶颈块用于消除编码器输出中的内容信息.
- [文献 086]() 提出的跨说话人风格迁移 Transformer-TTS 模型中, 说话人和风格嵌入都作为模型编码器的输入. 编码器输出的说话人-风格组合被传递到韵律瓶颈自网络, 导出仅包含韵律相关特征的韵律嵌入. 之后将编码器的输出与产生的韵律嵌入进行拼接作为解码器的输入.

> The Copycat TTS model [130] is a prosody transfer model via VAE. 
> The model applies three techniques to disentangle the source speaker information from the prosody embedding. 
> One of these techniques is to use a temporal bottleneck encoder [153] within the reference encoder of the model. 
> The prosody embedding that is sampled from the latent space is passed to the bottleneck to reduce speaker identity-related information in the prosody embedding before it flows to the model decoder.
> Similarly, the model proposed in [101] produces a style embedding with less irrelevant style information by adding a variational information bottleneck (VIB) [154] layer to the reference encoder. 
> The idea behind this layer is to introduce a complexity constraint on mutual information(MI) between the reference encoder input and output so that it only flows out style-related information.

- [文献 130]() 提出的 Copycat TTS 是通过 VAE 进行韵律迁移的模型. 模型应用三种技术来从韵律嵌入中解耦源说话人信息. 其中之一是在模型的参考编码器中使用**时序瓶颈编码器** [文献 153](), 在传递给模型解码器之前, 从隐空间中采样的韵律嵌入被传递到瓶颈以减少其中和说话人身份相关的信息.
- [文献 101]() 通过给参考编码器添加变分信息瓶颈 (Variational Information Bottleneck, VIB) 层以生成具有更少风格无关信息的风格嵌入. 该层背后的思想是在参考编码器的输入和输出之间的互信息引入一个复杂度约束使其只流出风格相关的信息.

#### 5.1.4.实例归一化 (Instance Normalization)

> Batch normalization (BN), first introduced in [155], is utilized in deep neural networks to accelerate the training process and increase its stability. Essentially, a batch normalization layer is added before each layer in deep neural networks to adjust the means and variances of the layer inputs, as illustrated by Eq.(1):
> $$
>   IN(x) = \gamma \left[\dfrac{x-\mu(x)}{\sigma(x)}\right]+\beta, \tag{1}
> $$
> 
> where $\gamma$, $\beta$ are affine parameters learned from data and $\mu$, $\sigma$ are the mean and standard deviation which are calculated for each feature channel across the batch size.
> Instance normalization (IN) also follows equation (1); however, it calculates means and variances across spatial dimensions independently for each channel and each sample (instance). 
> In the field of computer vision, stylization approach is significantly improved by replacing (BN) layers with (IN) layers [156]. 
> Consequently, researchers in the expressive speech field have started to apply IN to extract better prosody representations. 
> For example, an instance normalization (IN) layer is used at the reference encoder in [130], at the prosody extractor in [93], and at the style encoder in [96] to remove style/prosody irrelevant features (such as speaker identity features) and enhance the learned style/prosody embedding.

**批量归一化 (Batch Normalization, BN)** 由[文献 155]() 首次提出, 用于加速深度神经网络的训练过程并提高其稳定性. 本质上, 在深度神经网络的每一层之前添加批量归一化层用于调整输入的均值和方差, 如[公式一]()所示.
其中 $\gamma$ 和 $\beta$ 是从数据中学习到的仿射参数, $\mu$ 和 $\sigma$ 是在整个批量大小上为每个特征通道计算的均值和方差.

**实例归一化 (Instance Normaliztion, IN)** 同样满足[公式一](), 然而它独立地为每个通道和每个样本 (又称实例) 在空间维度上计算均值和方差. 

在计算机视觉领域, 通过将批量归一化层替换为实例归一化层, 风格化方法得到了显著提升. 因此, 表达性语音领域的研究人员尝试应用实例归一化用于提取更好的韵律表示. 

例如以下研究都使用了实例归一化层以去除风格/韵律无关特征 (如说话人身份特征) 并增强学习到的风格/韵律嵌入.

- [文献 130]() 的参考编码器;
- [文献 093]() 的韵律提取器;
- [文献 096]() 的风格编码器.

#### 5.1.5.互信息最小化 (Mutual Information Minimization)

> For a pair of random variables, mutual information (MI)is defined as the information obtained on one random variable by observing the other. 
> Specifically, if $X$ and $Y$ are two variables, then $MI(X; Y)$ shown by Venn diagram in [Fig.10](#FIG10), can be seen as the KL-divergence between the joint distribution $(P_{XY})$ and the product of the marginals $(P_X, P_Y)$ as in [equation (2)](#EQ02). If the two random variables $X$ and $Y$ represent linguistic and style vectors, applying $MI$ minimization between these two vectors helps to produce style vectors with less information from the content vector.
> $$
>   MI(X;Y)=DL_{KL}(P_{(X,Y)}\| P_X\otimes P_Y),\tag{2}
> $$

> For example, in [137], the Mutual Information Neural Estimation algorithm (MINE) [157] is employed to estimate the mutual information between the content and style vectors. 
> The algorithm uses a neural network that is trained to maximize the lower bound of the mutual information between the style and content vectors. 
> Simultaneously, the TTS model aims to minimize the reconstruction loss, making the overall problem a max-min problem. 
> Alternatively, in [21], the CLUB method [158], which computes an upper bound as the MI estimator, is used to prevent the leakage of speaker and content information into the style embedding.

> A new approach is proposed in [117] for MI estimation and minimization to reduce content/speaker information transfer to the style embedding in a VAE based approach. 
> Typically, the model needs to estimate MI between latent style embeddings and speaker/content embeddings. 
> To avoid the exponentially high statistical variance of the finite-sampling MI estimator, the paper suggests using a new algorithm for information divergence named Rényi divergence. 
> Two variations from the Rényi divergence family are proposed, including minimizing the Hellinger distance and minimizing the sum of Rényi divergences.

## 6.数据集与开源代码

## 7.评价指标

## 8.讨论

## 9.结论

## R.参考文献

- 019 [<a id="S.Jo2023">Cross-Speaker Emotion Transfer by Manipulating Speech Style Latents]()
- 032 [<a id="T.Li2021">Controllable Emotion Transfer for End-to-End Speech Synthesis</a>]()
- 034 [<a id="T.Li2022">Cross-Speaker Emotion Disentangling and Transfer for End-to-End Speech Synthesis</a>]()
- 047 [<a id="C.Lu2021">Multi-Speaker Emotional Speech Synthesis with Fine-Grained Prosody Modeling</a>]()
- 074 [<a id="R.Skerry2018">Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron</a>]()
- 090 [<a id="Y.Ganin2016">Domain-Adversarial Training of Neural Networks</a>]()
- 097 [<a id="K.Lee2021">STYLER: Style Factor Modeling with Rapidity and Robustness via Speech Decomposition for Expressive and Controllable Neural Text To Speech</a>]()
- 102 [<a id="K.Zhang2022">Joint and Adversarial Training with ASR for Expressive Speech Synthesis</a>]()
- 111 [<a id="D.Tan2020">Fine-Grained Style Modeling, Transfer and Prediction in Text-to-Speech Synthesis via Phone-Level Content-Style Disentanglement</a>]()