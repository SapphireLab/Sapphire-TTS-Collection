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



## 6.数据集与开源代码

## 7.评价指标

## 8.讨论

## 9.结论

## R.参考文献

- 019 [<a id="S.Jo2023">Cross-Speaker Emotion Transfer by Manipulating Speech Style Latents]()
- 047 [<a id="C.Lu2021">Multi-Speaker Emotional Speech Synthesis with Fine-Grained Prosody Modeling</a>]()
- 090 [<a id="Y.Ganin2016">Domain-Adversarial Training of Neural Networks</a>]()
- 097 [<a id="K.Lee2021">Styler: Style Factor Modeling with Rapidity and Robustness via Speech Decomposition for Expressive and Controllable Neural Text To Speech</a>]()
- 102 [<a id="K.Zhang2022">Joint and Adversarial Training with ASR for Expressive Speech Synthesis</a>]()
- 111 [<a id="D.Tan2020">Fine-Grained Style Modeling, Transfer and Prediction in Text-to-Speech Synthesis via Phone-Level Content-Style Disentanglement</a>]()