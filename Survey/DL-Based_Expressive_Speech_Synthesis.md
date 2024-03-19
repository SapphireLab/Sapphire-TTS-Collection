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

对于一对随机变量, 互信息 (Mutual Information, MI) 被定义为通过观察另一个随机变量获得的当前随机变量的信息.
具体地, 若 $X$ 和 $Y$ 为两个变量, 那么 $MI(X;Y)$ 如图十显示的韦恩图所示, 可以视为联合概率分布 $P_{XY}$ 和边际概率分布 $(P_X,P_Y)$ 的乘积之间的 KL 散度. 若两个随机变量 $X$ 和 $Y$ 分别表示语言向量和风格向量, 对这两个向量应用互信息最小化将有助于生成具有更少来自内容向量的信息的风格向量.
$$
    MI(X;Y)=DL_{KL}(P_{(X,Y)}\| P_X\otimes P_Y),\tag{2}
$$

- 文献 [137] 将文献 [157] 的互信息网络估计算法 (Mutual Information Neural Estimation, MINE) 应用于估计内容向量和风格向量之间的互信息. 该算法使用一个神经网络, 被训练以最大化风格向量和内容向量之间互信息的下界. 同时, TTS 模型最小化重构损失, 使得整体变成了最大-最小问题.
- 文献 [021] 将文献 [158] 使用 CLUB 方法计算一个上界作为 MI 估计量, 用于防止说话人和内容信息泄露到风格嵌入中.
- 文献 [117] 提出了一种用于互信息估计的新方法并在基于 VAE 方法中最小化用于减少内容/说话人信息转移到风格嵌入中. 通常模型需要估计隐风格嵌入和说话人/内容嵌入之间的互信息. 为了避免有限采样互信息估计其的指数高统计方差, 作者建议使用一种新算法用于信息散度, 名为 Rényi 散度. Rényi 散度导出两种变体, 包括最小化 Hellinger 距离和最小化 Rényi 散度之和.

Fig. 10 Venn diagram of two random variables X and Y where P(X)and P(Y) represent their entropies, P(X|Y) is the conditional entropy of X given Y and P(Y|X) is the conditional entropy of Y given X, H(X,Y)is the joint entropy of X and Y and MI(X,Y) is their mutual information


#### 5.1.6.波形转向量特征 (Wav2Vec Features)

> Wav2Vec [142] model converts speech waveform into context-dependent vectors/features. The model is trained via self-supervised or in-context training algorithms which are explained in [Section 4.4](#Sec4.4.). Features generated by wav2vec and similar models such as HuBERT [159] provide better representations of speech and its lexical and non-lexical information. Therefore, these models are utilized nowadays in different speech processing tasks such as speech recognition, synthesis, and downstream emotion detection.

Wav2Vec 模型将语音波形转化为上下文相关的向量或特征. 该模型通过自监督或上下文训练算法进行训练. 由 Wav2Vec 和类似的模型如 HuBERT 生成的特征提供了更好的语音及其词汇和非词汇信息的表示. 因此这些模型现在被用于不同的语音处理任务例如语音识别, 合成和下游情感检测.

> Some studies such as [23] [120] use Wav2vec 2.0 as a feature extractor to provide input to the reference encoder instead of spectrum features or raw audio waveform. Figure 11 illustrates the framework of the wav2vec technique and how it is utilized as a feature extractor with TTS models. The wav2vec model converts the continuous audio features into quantized finite set of discrete representations called tokens. This is done using a quantization module that maps the continuous feature vectors into a discrete set of tokens from a learned codebook. As those tokens are more abstract, they reduce the complexity of the features by retaining important features while filtering out all the irrelevant information. Because of that abstraction, it is harder to reconstruct audio from the wav2vec features, which means leakage of linguistic content into feature vectors is significantly lower compared to other features such as MFCCs.

一些研究使用 Wav2Vec 2.0 作为特征提取器, 为参考编码器提供输入, 而不是原始音频波形的频谱特征. 图十一展示了 Wac2Vec 技术及其作为特征提取器如何与 TTS 模型相结合. Wav2Vec 模型将连续音频特征转化为名为 token 的离散表示的量化有限集. 这通过使用量化模块将连续特征向量映射到取自学习好的码本的离散 token 集合. 当这些 token 越抽象, 它们通过保留重要特征并过滤掉所有不相关的信息来减低特征的复杂度. 由于这种抽象性, 很难从 Wav2Vec 特征中重构音频, 这意味着与其他特征如 MFCC 相比, 语言内容泄漏到特征向量将明显下降.

#### 5.1.7.正交性损失 (Orthogonality Loss)

> Studies [34] [39] propose a model with two separate encoders to encode speaker and emotion information through speaker and emotion classification loss, along with gradient inversion of the emotion classification loss in the speaker encoder. Additionally, to disentangle the source speaker information from the emotion embedding, the emotion embedding is made orthogonal to the speaker embedding with an orthogonality loss shown in equation (3). An ablation study in [34] showed that applying an orthogonality constraint helped the encoders learn both speaker-irrelevant emotion embedding and emotion-irrelevant speaker embedding.
> $$
    \mathcal{L}_{orth} = \sum_{i=1}^n \|S_i-e_i\|_F^2\tag{3}
> $$
> 
> where $\|\cdot\|_F$ is the Frobenius norm, $e_i$ is the emotion embedding and $S_i$ is the speaker embedding.

一些研究提出了具有两个独立编码器的模型, 并且通过说话人和情感分类损失, 以及说话人编码器中的情感分类损失的梯度反转来编码说话人和情感信息. 此外, 为了从情感嵌入中解耦源说话人的信息, 情感嵌入通过公式三所示的正交性损失与说话人嵌入正交. 
$$
    \mathcal{L}_{orth} = \sum_{i=1}^n \|S_i-e_i\|_F^2\tag{3}
$$

其中 $\|\cdot\|_F$ 是 F 范数, $e_i$ 是情感嵌入, $S_i$ 是说话人嵌入.

消融实验表明应用正交性约束有助于编码器学习与说话人无关的情感嵌入和与情感无关的说话人嵌入.

### 5.2.无参考音频推理 (Inference without Reference Audio)

> A main drawback of the unsupervised approaches ([Section 4](#Sec4)) is that they require a reference audio for the desired prosody or style of the generated speech. However, prosody references are not always available for the desired speaker, style, or text. Besides, using prosody reference introduces the leakage problem as discussed in [Section 5.1](). As a result, different techniques have been proposed that enable unsupervised expressive speech synthesis without prosody references. Some techniques utilize the reference audio at training phase while at inference phase speech synthesis can be done with or without a reference audio. Other techniques depend on input text only to generate prosody embedding at both training and inference phases. In the following three sections, we will describe techniques for inference without reference audio applied with each of the three main unsupervised ETTS approaches. In [Section 5.2.4](), we will discuss some ETTS approaches that are based on text only. Then in Table 4, we summarize main approaches that are used to extract text-based features with related papers links.

第四节中总结的无监督方法的一个主要缺点是, 它们需要一个参考音频用于所生成语音的期望韵律或风格. 然而所需说话人/风格/文本的韵律参考并不总是能够获得.
此外使用韵律参考会引入第5.1节讨论的泄露问题. 
因此出现了各种技术使得无监督表达性语音合成能够无韵律参考.
一些技术在训练阶段使用参考音频, 而推理阶段语音合成就可以用或不同参考音频.
其他技术只依赖于输入文本用于在训练和推理阶段生成韵律嵌入.
在以下三小节中将描述无参考音频的推理技术, 应用于三种主要的无监督表达性语音合成方法中.
在第四小节将讨论一些仅依赖于文本的表达性语音合成方法.
表格四总结了相关文献中用于提取基于文本的特征的主要方法.

#### 5.2.1.无参考音频的直接参考编码 Direct Reference Encoding without Reference Audio 

> In several studies, prosody predictors are trained jointly with the proposed reference encoder to bypass the requirement for reference audio at inference time. The prosody predictors are trained to predict either the prosody embeddings generated by reference encoders[50] [96] [111] [116], or the acoustic features used as input to reference encoders [37] [63]. As input to these prosody predictors, most studies utilize the phoneme embeddings[37] [63] [96] [111].

在一些研究中, 韵律预测器与所提出的参考编码器联合训练, 以绕过推理时对参考音频的要求. 韵律预测器被训练用于预测参考编码器生成的韵律嵌入或用作参考编码器输入的声学特征.
作为这些韵律预测器的输入, 大多数研究使用音素嵌入.

> Alternatively, features extracted from input text can also be used as input for prosody predictors. In [50], the prosody predictor has a hierarchical structure that utilizes contextual information at both the sentence and paragraph levels to predict prosody embeddings. The input features for this predictor are in the form of 768-dimensional phrase embeddings extracted by the pre-trained language model XLNet [160]. Sentence embeddings are initially predicted from the input features using an attention network. Then a second attention network is used to predict the paragraph-level prosody embedding.

或者从输入文本提取的特征也可以作为韵律预测器的输入.
文献 [050] 中韵律预测其有一个层次结构, 利用句子和段落级别的上下文信息来预测韵律嵌入. 这一预测器的输入特征是使用预训练语言模型 XLNet 提取的 768 维的短语嵌入形式. 首先通过注意力网络从输入特征中预测句子嵌入, 然后第二个注意力网络用于预测段落级别的韵律嵌入.

> Furthermore, in [33], emotion is modelled at three levels: global, utterance, and syllable (local). The model employs three prosody encoders, each with a predictor trained to predict the corresponding prosody embedding based on input text. The global-level predictor functions as an emotion classifier, where the output of its final soft-max layer serves as the global emotion embedding. The emotion label’s embedding is used as the ground truth for this emotion classifier. Both the utterance and local prosody encoders receive level-aligned mel-spectrograms as input and produce utterance prosody embedding and local prosody strength embedding, respectively. Similarly, two prosody predictors are used to predict utterance and local-level embeddings based on the output from the text encoder of the TTS model.

文献 [033] 中情感被建模在三个级别: 全局, 语调和音节 (局部). 模型应用三个韵律编码器, 每个编码器都有一个预测器训练成基于输入文本预测对应的韵律嵌入. 全局预测器作为情感分类器, 其最终的 softmax 层输出作为全局情感嵌入. 情感标签的嵌入作为这一情感分类器的真实值. 语调和局部韵律编码器都接收级别对齐的梅尔频谱作为输入, 并分别生成语调韵律嵌入和局部韵律强度嵌入. 类似地, 两个韵律预测器基于 TTS 模型的文本编码器的输出预测语调和局部级别嵌入.

> In contrast, the prosody predictor proposed in paper [44] learns multiple mixed Gaussian distributions model (GMM) for prosody representations. Therefore, the final outputs of the prosody predictor involve three parameters: mean, variance, and weight of multiple mixed Gaussian distributions from which prosody representations can be sampled at inference time. As input, the predictor receives two phoneme-level sequences including embeddings from the text encoder and embeddings from a pre-trained language model. Similar work is proposed in [95] where only phoneme embeddings are used as input to the prosody predictor. GMM in both studies is modeled via the mixture density network [161].

文献 [044] 提出的韵律预测器学习多重混合高斯分布模型用于韵律表示. 因此韵律预测器的最终输出包含三个参数: 均值, 方差和多混合高斯分布的权重. 从这些分布中可以在推理时采样韵律表示. 作为输入, 预测器接收两个音素级别的序列包括来自文本编码器的嵌入和来自预训练语言模型的嵌入.

文献 [095] 提出的相似工作, 只使用音素嵌入作为韵律预测器的输入.

两项工作的 GMM 都通过混合密度网络进行建模.

#### 5.2.2.无参考语音的VAE类方法 (VAE‑Based Approaches without Reference Audio)

> Sampling from the latent space without reference audio results in less controllability of style. In addition, it can also introduce naturalness degradation and inappropriate contextual prosody with regard to the input text [68] [129]. 
> Therefore, to avoid sampling the latent space without a reference, authors of [131] proposed utilizing the same prosody embedding of the most similar training sentence to input sentence at inference time. The selection process is based on measuring cosine similarity between sentences’ linguistic features. Three methods are proposed for extracting sentence linguistic information including 
> (1) calculating the syntactic distance between words in the sentence using constituency trees [162], 
> (2) averaging the contextual word embeddings (CWE) for the words in the sentence using BERT, and 
> (3) combining the previous two methods.

在没有参考音频的情况下从隐空间采样会导致风格的可控性降低. 此外它还会引入自然性退化和和不适合输入文本的语境韵律.
因此为了避免在没有参考的情况下从隐空间中采样, 文献 [131] 提出推理时用和输入句子最相似的训练句子的相同韵律嵌入. 选择过程基于句子语言特征之间的余弦相似度. 三种方法用于提取句子语言信息:
1. 使用句法树计算句子中单词的句法距离;
2. 使用 BERT 计算句子中单词的上下文词嵌入 (Contextual Word Embeddings, CWE) 的平均值;
3. 结合前两种方法.

> Other studies approach the problem in alternative ways, seeking to enhance the sampling process either through refining the baseline model structure or by incorporating text-based components into the baseline.
> Regarding the improvement of the baseline structure, study [68] suggests the combination of multiple variational autoencoders to generate latent variables at three distinct levels: utterance-level, phrase-level, and word-level. Furthermore, they apply a conditional prior (CP) to learn the latent space distribution based on the input text embedding. To account for dependencies within the input text, they employ Autoregressive (AR) latent converters to transform latent variables from coarser to finer levels.
> An alternative approach is proposed in [126] by replacing the conventional VAE encoder with a residual encoder that leverages phoneme embedding and a set of learnable free parameters as inputs. With this modified structure, the model learns a latent distribution that represents various prosody styles for a specific sentence (i.e.,the input text), in addition to capturing potential global biases within the applied dataset (represented by the free parameters). At the same time, with this modification, the problem of speaker and content leakage into prosody embedding is addressed.

其他研究以不同的方式解决这个问题, 寻求通过改进基线模型结构或向基线模型中添加基于文本的组件来增强采样过程.
关于基线结构的改进:
- 文献 [068] 建议组合多个变分自编码器用于生成三个不同级别的隐变量: 语调级, 短语级和单词级. 此外应用了条件先验基于输入文本嵌入学习隐空间分布. 为了考虑输入文本内的依赖性, 他们应用自回归隐转化器将隐变量从粗糙级别转化为精细级别.
- 文献 [126] 提出了一种替代方法, 通过将传统 VAE 替换为一个利用音素嵌入和一组可学习自由参数作为输入的残差编码器. 采用这一结构, 模型为一个具体句子即输入文本学习一个隐分布表示各种韵律风格, 同时捕捉应用数据集的潜在全局偏差 (由自由参数表示). 同时说话人和内容泄露到韵律嵌入的问题也得到解决.

> Various studies propose training a predictor for the latent prosody vectors based on features extracted from the input text [35] [47]. The proposed model in [47] generates fine-grained prosody latent codes of three dimensions at phoneme-level. These prosody codes are then used to guide the training process of a prosody predictor that receives phoneme embeddings as input, in addition to emotion and speaker embeddings as sentence-level conditions. In [35], the predicted mean values of the latent space distribution are employed as prosody codes. Similarly, a prosody predictor is trained to predict these prosody codes using two text-based inputs, including sentence-level embeddings from a pre-trained BERT model and contextual information considering BERT embeddings of a few of surrounding k sentences given the current sentence.

多项研究提出基于从输入文本中提取的特征训练一个隐韵律向量的预测器.
- 文献 [047] 提出的模型在音素级别生成细粒度的三维韵律隐代码. 这些韵律代码之后用于指导韵律预测器的训练. 预测器接收音素嵌入作为输入, 此外情感和说话人嵌入作为句子级别条件.
- 文献 [035] 中隐空间分布的预测均值作为韵律编码. 类似地训练一个韵律预测器使用两个基于文本的输入包括来自预训练 BERT 模型的句子级别嵌入和考虑当前句子周围的 k 个句子的 BERT 嵌入的上下文信息用于预测这些韵律编码.

> Alternatively, study [129] proposed training a sampler, i.e., Gaussian parameters, to sample the latent space using features extracted from the input text. Three different structures are investigated for the sampler based on the input features it receives. The applied text-based features include BERT representations of a sentence (semantic information), the parsing tree of the sentence (syntactic information) after it is fed to a graph attention network, and the concatenation of outputs from the previous two samplers.

文献 [129] 提出训练一个采样器, 即高斯参数, 使用输入文本提取的特征对隐空间进行采样. 根据采样器接收的输入特征, 研究了三种不同的结构. 应用基于文本的特征包括句子的 BERT 表示 (语义信息), 图注意力网络输出的句子的解析树 (语法信息) 和前两个采样器的输出的拼接.

#### 5.2.3 GST‑Based Approaches without Reference Audio 

There are GST-TTS models that utilize text-based fea-tures from pre-trained language models such as BERT to guide expressive speech synthesis at inference time without a reference. In [59], the training dataset is labeled with short phrases that describe the style of the utter-ance and are known as style tags. A pre-trained Sentence BERT (SBERT) model is used to produce embeddings for each style tag as input to a style tag encoder. The style embedding from the GST-TTS model is used as ground truth for the style tag encoder. During inference, either a reference audio or a style tag can be used to generate speech.
Alternatively, pre-trained language models are used to extract features from input text and train a prosody pre-dictor to predict the style embedding based on these text-based features [17] [46] [50] [73] [91] [94]. In [94], the baseline model [75] is extended with a prosody predictor module that extracts time-aggregated features from the output of the baseline text encoder. Two pathways are suggested for the targets of the predictor output: either using the weights of the GSTs or the final style embedding. Simi-larly, in [73], two prosody predictors are investigated,using different inputs from a pre-trained multi-language BERT model. While the first predictor utilizes BERT embeddings for the sub-word sequence of input text, the other predictor employs only the CLS token from the sentence-level information extracted by the BERT model.
Both inputs provide rich information for the predictors to synthesize prosodic speech based solely on input text.
The multi-scale GST-TTS proposed in [50] which employs three style encoders, also introduces three style predictors that employ hierarchical context encoders(HCE). The input to the first predictor is the BERT sub word-level semantic embedding sequence. The atten-tion units in the HCE, however, are used to aggregate the resulting context embedding sequence from lower level as input to higher-level predictors. Additionally, the output of higher-level predictor is used to condition the lower-level predictor. BERT embeddings are also used in [46] but at word-level and are passed as input to the pro-posed prosody predictor. The style embedding which is generated via word-level GSTs is used to guide the pros-ody predictor during model training.
A Context-aware prosody predictor is proposed in [17]which considers both text-side context information and speech-side style information from preceding speech.
This predictor comprises two hierarchical components:a sentence encoder and a fusion context encoder. The context-aware input to the predictor includes word-level embeddings from XLNet [160] for each word in the cur-rent sentence, as well as the N preceding and following sentences. The sentence encoder focuses on learning low-level word meanings within each sentence, while the fusion context encoder captures high-level contex-tual semantics between the sentences. Additionally, style embeddings from previous sentences are integrated into the fusion context encoder input to account for speech-side information.
In [91] Speech emotion recognition model (SER) is employed as a style descriptor to learn the implicit con-nection between style features and input text. Deep style features for both synthesized speech and reference speech are obtained from a small intermediate fully con-nected layer of a pre-trained SER model during training.
The extracted style features are compared where an addi-tional loss is introduced to the GST-TTS model loss. At inference time only text is used to synthesize expressive speech.

#### 5.2.4 ETTS approaches based only on text 

This category involves approaches that depend solely on input text to obtain prosody-related representations/embeddings during TTS model training. Several features related to speech prosody have been proposed by vari-ous studies for extraction from input text and subsequent transmission to a DNN-based module to generate pros-ody representations. For instance, the features extracted by the pre-trained language models can capture both semantic and syntactic relationships with the input text,making them effective representations for prosody. In[83], input text word-level embeddings are extracted by the Embeddings from Language Models (ELMo) model[163] and used to generate context-related embed-dings via a context encoder. Similarly, in [29], BERT is employed to extract embeddings for utterance sentences and pass them to a specific context-encoder to aggregate these embeddings and form a final context vector.
Other studies, such as [30] [40] [54], utilize graph repre-sentations of input text, which can also reflect semantic and syntactic information about the given text. In [30],the graphical representations of prosody boundaries in Chinese text are passed to a graph encoder based on Graph Neural Networks (GNN) to generate prosodic information for the input text. The prosody boundaries of the Chinese language can be manually annotated or predicted using a pre-trained model. In contrast, [54]combines BERT-extracted features for input text with its graph dependency tree to produce word-level pros-ody representations. Specifically, the input text is passed through both BERT and a dependency parsing model to extract the dependency tree for word-level BERT embed-dings. A Relational Gated Graph Network (RGGN) is used to convert this dependency tree into word-level semantic representations upon which the decoder of the TTS model is conditioned.
Different text-based features have been extracted from input text to obtain prosody (style) embeddings in [40].The paper utilizes an emotion lexicon to extract word-level emotion features, including VAD (valence, arousal,dominance) and BE5 (joy, anger, sadness, fear, disgust).Additionally, the [CLS] embedding by BERT for each utterance is also extracted. The obtained features are then passed to a style encoder to produce a style embedding.
Other models under this category train a prosody encoder/predictor jointly with an autoregressive TTS model such as Tacotron 2, to encode some prosody related features utilizing text-based features. The trained encoder is then used at inference time to encode pros-ody-related features based on input text to the TTS model. The text-based input to these prosody encoders in most of the studies is the text’s character/phoneme embeddings [20] [48] [71] [72] [103], while some studies use features extracted from the input text [64] [125]. For instance, [125] employs four ToBI (Tones and Break Indi-ces) features as word-level prosody tags that are com-bined with the phoneme embedding as input to the TTS model. A ToBI predictor is jointly trained to predict four ToBI features based on grammatical and semantic infor-mation extracted from the input text using a self-super-vised language representation model ELECTRA [164].In addition to the previously mentioned features, sev-eral other prosodic features are also proposed as the output of the prosody predictors in other studies. For example, the prosody predictor in [103] predicts a set of utterance-wise acoustic features, including log-pitch, log-pitch range, log-phone duration, log-energy, and spec-tral tilt. In [48], the proposed pitch predictor outputs a continuous pitch representation, which is converted into discrete values using Vector Quantization (VQ) [149].Furthermore, studies [20] [71] propose predicting the three prosody-related features, i.e., F0, energy, and dura-tion, either by a single acoustic features predictor (AFP)[71] or via three separated predictors [20].Another type of emotion embedding is sentiment fea-ture embedding, which is utilized to produce expressive speech by extracting sentiment information from the input text. This is demonstrated in work [135], where the Stanford Sentiment Parser is used to generate vec-tor embeddings or sentiment probabilities based on the tree structure of the sentence. To synthesize expressive speech, different combinations of probabilities and vec-tor embeddings (for individual words or word-context)are added to the linguistic features as inputs to the TTS model.

### 5.3 Prosody Controllability 韵律可控性

Text-to-speech is a one-to-many mapping problem, i.e.,for one piece of text there could be many valid prosody patterns because of speaker-specific variations. Accord-ingly, providing a kind of controllability over prosody-related features in synthesized speech is essential for generating expressive speech with different variations.
However, it’s not always easy to mark-up prosody or even to define boundaries between prosody events, i.e., dura-tion boundaries can vary depending on segmentation,pitch contour prediction is error-prone, and prosody fea-tures may not always correlate well with what listeners perceive.
Several studies in literature have addressed the control-lability issue in terms of selecting an emotion/style class or intensity level and adjusting prosody-related features at different speech levels. In this section, we discuss stud-ies considering prosody controllability.

文本到语音转换是一个一一对应的问题，即对于一段文本，由于说话人特定的变化，可能存在许多有效的韵律模式。因此，在合成的语音中提供对韵律相关特征的控制对于生成具有不同变化的表达性语音至关重要。
然而，标记韵律或定义韵律事件之间的界限并不总是容易的，即持续时间界限可能会根据分段而变化，音高轮廓预测容易出错，韵律特征并不总是与听众感知相关。
在文献中，有几项研究从选择情感/风格类别或强度级别以及在不同的语音级别调整韵律相关特征的角度解决了可控性问题。在本节中，我们将讨论考虑韵律可控性的研究。

#### 5.3.1 Modeling‑specific prosody styles 

This group of studies provides individual representa-tions of expressive styles/emotions, enabling the control of prosody in synthesized speech by offering the ability to select from available representations or adjust their values. In some studies [55] [70] [116], style is modeled at a single speech/text level, while in other studies [68] [79] [133] a multi-level or hierarchical model of expressive styles is used to allow for a better capture of prosody var-iation in expressive speech.
In single-level prosody modeling approaches, [55] is one of the early studies that extends a baseline with fine-grained control over the speaking style/prosody of syn-thesized speech. The proposed modification involves adding an embedding network with temporal structure to either the speech-side or text-side of the TTS model.
Accordingly, the resulting prosody embedding is of vari-able length, and it is used to condition input to either encoder or decoder based on the position of the embed-ding network. Speech-side prosody embedding provides adjustment of prosody at frame-level, while text-side prosody embedding enables phoneme-level prosody control.
Single-level prosody embeddings can be converted into discrete embeddings as in [70] [116]. Discrete pros-ody representations are easier to control and analyze and provide a better interpretation of prosodic styles.

这些研究提供了表达风格/情感的个体表示，使得在合成语音中控制语调成为可能，通过提供从可用表示中选择或调整其值的能力。在某些研究中，风格在单个语音/文本级别建模，而在其他研究中，使用多级或分层模型来更好地捕捉表达性语音中的语调变化。

在单级语调建模方法中，[55]是早期研究之一，它扩展了基线，对合成语音的说话风格/语调进行细粒度控制。提出的修改涉及在TTS模型的语音侧或文本侧添加具有时间结构的嵌入网络。

因此，得到的语调嵌入是可变长度的，并且根据嵌入网络的位置用于条件输入到编码器或解码器。语音侧语调嵌入提供帧级语调调整，而文本侧语调嵌入允许音素级语调控制。

单级语调嵌入可以转换为离散嵌入，如[70] [116]。离散语调表示更容易控制和分析，并提供语调风格的更好解释。


In [116], a word-level prosody embedding is proposed based on decision trees and a GMM. A word-level refer-ence encoder is first used to obtain word-level prosody embedding from reference audio. A binary decision tree is employed to cluster embeddings with their identities based on their phonetic information. Prosody embed-dings of words in each leaf node will differ only in their prosodies. Then prosody embeddings of each leaf can be clustered via a GMM model where clusters represent prosody tags. If the applied GMM consists of five com-ponents and a tree of ten leaf nodes, a set of 50 prosody tags is produced. At inference time, prosody tags can be selected manually or via a prosody predictor that is trained to select appropriate prosody tags based on input text.
In [70], an audiobook speech synthesis model is pro-posed. The model uses a character-acting-style extrac-tion module based on ResCNN [165] to extract different character acting styles from the input speech. Discrete character-level styles are obtained via vector quantization(VQ) [149], which maps them to a codebook, limiting the number of styles. At inference, the discrete character-act-ing-styles are predicted via a style predictor. The charac-ter-level style predictor uses both character embeddings from Skip-Gram [166] and text-based features from RoB-ERTa [167] as input.
在[116]中，基于决策树和GMM提出了一个词级语调嵌入。首先使用词级参考编码器从参考音频中获取词级语调嵌入。使用二叉决策树根据其音素信息对嵌入进行聚类。每个叶节点中单词的语调嵌入仅在语调上有所不同。然后，可以通过GMM模型对每个叶的语调嵌入进行聚类，其中聚类表示语调标签。如果应用的GMM包含五个组件和一个十个叶节点的树，则会产生50个语调标签。在推理时，语调标签可以手动选择或通过训练以根据输入文本选择适当语调标签的语调预测器选择。

在[70]中，提出了一种有声读物语音合成模型。该模型使用基于ResCNN [165]的角色扮演风格提取模块从输入语音中提取不同的角色扮演风格。通过矢量量化(VQ) [149]获得离散角色级风格，将其映射到码本，限制风格数量。在推理时，通过风格预测器预测离散角色扮演风格。角色级风格预测器使用来自Skip-Gram [166]的角色嵌入和来自RoBERTa [167]的文本特征作为输入。



Regarding multi-level prosody modeling, some stud-ies propose enhancing prosody control in the baseline models [74] [75] [77] by modifying their single-level pros-ody modeling to multiple levels. For instance, [133] pro-poses a hierarchical structure of [75] with multiple GST layers. Three GST layers are employed in the proposed model, each consisting of 10 tokens, which were found to yield better token interpretation. Tokens of the first and second layers were found to learn different speak-ers and styles, but these representations were not easily interpreted. Interestingly, the tokens in the third layer were able to generate higher quality samples with more distinct and interpretable styles. Specifically, third-layer styles exhibit clear differences in their features, includ-ing pitch, stress, speaking rate, start offset, rhythm, pause position, and duration.
Model in [77] is further extended in [68] with three VAEs to generate three different levels (utterance, phrase,and word) of latent variables with varying time resolu-tions. Acoustic features and linguistic features are passed as input to the three VAEs. Initially, a conditional prior(CP) is applied to learn a distribution for sampling utter-ance-level latent variables based on linguistic features from the input text. The generated latent variables are passed to other levels via auto-regressive (AR) latent converters that convert latent variables from coarser-level to finer-level with input text condition. In fact, the utterance-level latent variables can be used to control the generated speech styles, regardless of latent variables of other levels, as they are predicted based on the utterance-level latent variables.




The Controllable Expressive Speech Synthesis (ConEx)model in [79] proposes modeling prosody at two levels,utterance-level (global) and phone-level (local), using reference encoders [74]. However, the global prosody embedding is used to condition the local prosody embed-ding, resulting in an integrated prosody embedding.
The local embeddings are 3D vectors that are converted into discrete local prosody embeddings (codes) via vec-tor quantization (VQ) [149]. At inference time, the integrated prosody embedding is predicted by an auto-regressive (AR) prior model trained to predict categori-cal distributions for each of the discrete codes utilizing global prosody embedding and the phoneme embed-ding as inputs. While global prosody embedding can be obtained from training samples or from an audio refer-ence, local prosody embeddings for a given global pros-ody embedding are achieved via the AR prior model.
Fine-grained prosody control can be achieved by select-ing a specific phoneme to start adjusting prosody from.
The AR prior model will first generate the top k pros-ody options for this phoneme. Then, the local prosody sequence will be generated autoregressively for each of the first top k options by the AR prior model.

#### 5.3.2 Modeling‑specific prosody features 

This group of studies provides individual representations of prosody-related features. Control over prosody of the synthesized speech is provided via selecting or adjust-ing a specific representation of a specific prosody-related feature. Some studies in this direction model prosody features at the global or utterance-level [97] [128], while other studies propose modeling at fine-grained lev-els [48] [63] [71] [122] [138], such as phoneme, syllable, or word-level.
The STYLER model [97], for example, employs multi-ple style encoders to factor speech style into several com-ponents, including duration, pitch, speaker, energy, and noise. This structure enables STYLER to generate con-trollable expressive speech by adjusting each of the indi-vidually modeled features. Furthermore, with the explicit noise encoding, other encoders can be constrained to exclude noise information as a style factor, and thus the model can generate clean speech even with noisy refer-ences. Adjusting the style factors, various styles of speech can be generated from STYLER.

这些研究提供了与韵律相关的特征的个体表示。通过选择或调整特定韵律相关特征的特定表示来控制合成的语音的韵律。在这个方向上，一些研究在全局或句子级别建模韵律特征[97] [128]，而其他研究则提出在细粒度级别建模[48] [63] [71] [122] [138]，例如音素、音节或单词级别。

例如，STYLER模型[97]使用多个风格编码器将语音风格分解为几个组成部分，包括持续时间、音高、说话者、能量和噪声。这种结构使STYLER能够通过调整每个单独建模的特征来生成可控的表达性语音。此外，通过显式噪声编码，其他编码器可以被约束以排除噪声信息作为风格因素，因此模型可以生成干净的语音，即使使用嘈杂的参考。通过调整风格因素，可以从STYLER生成各种风格的语音。


Adjusting several features at fine-grained levels can be a difficult task. For example, FastSpeech2 [6] pro-vides fine-grained control over pitch range, duration,energy, which are modeled at the phone-level (phone-wise), and it is not easy to adjust these features to achieve a specific prosodic output. Raitio and Seshadri[128] improves FastSpeech2 with an utterance-wise(coarse-grained) prosody model using an additional variance adaptor. That second variance adaptor is the same as the original one, but it models five features at the utterance-level: pitch, pitch range, duration, energy,and spectral tilt. These features are then concatenated with the corresponding output of the first variance adaptor. Such utterance-wise prosody model enables easier control of prosody while still allowing modifica-tion at the phone-level. To control high-level prosody,a bias is added to the corresponding utterance-wise prosody predictions. A phone-level prosody control is achieved by directly modifying the phone-wise features.
Fine-grained control over a specific prosody-feature can also be required specially for strong speaking styles.
To that end, in [71], a predictor is proposed to predict F0, energy, and duration features at the phoneme-level.
During inference, the predicted features are generated based on the input text alone; however, they can also be provided externally and modified as desired.
Furthermore, two prosody modeling levels are pro-posed in [63]: the local level (word-level) and global level (utterance-wise). The global prosody embedding is the emotion embedding obtained by a reference-based encoder. The local prosody embedding is obtained from a predictor of the F0 features at the word-level with global prosody embedding and the phoneme embed-ding as inputs. Both embeddings are then passed to a multi-style encoder to form the final multi-style pros-ody embedding. Therefore, modifying the predicted F0values can provide control of prosody at the utterance,word, and phoneme levels.

在细粒度级别调整多个特征可能是一项困难的任务。例如，FastSpeech2[6]在音素级别（音素级）提供对音高范围、持续时间和能量的细粒度控制，并且不容易调整这些特征以实现特定的韵律输出。Raitio和Seshadri[128]通过使用额外的变异适配器改进FastSpeech2，该变异适配器在句子级别（粗粒度）使用韵律模型。第二个变异适配器与原始变异适配器相同，但在句子级别建模五个特征：音高、音高范围、持续时间、能量和频谱倾斜。然后，将这些特征与第一个变异适配器的相应输出连接起来。这种句子级别的韵律模型允许更容易地控制韵律，同时仍然允许在音素级别进行修改。为了控制高级别韵律，向相应的句子级别韵律预测添加偏差。通过直接修改音素级特征来实现音素级韵律控制。

对于强烈的说话风格，可能需要对特定韵律特征进行细粒度控制。为此，[71]中提出了一种预测器来预测音素级别的F0、能量和持续时间特征。在推理过程中，根据输入文本生成预测特征；但是，它们也可以从外部提供并根据需要进行修改。此外，[63]中提出了两个韵律建模级别：局部级别（单词级别）和全局级别（句子级别）。全局韵律嵌入是从基于参考的编码器获得的情绪嵌入。局部韵律嵌入是从具有全局韵律嵌入和音素嵌入作为输入的预测器获得的音素级别的F0特征。然后，将这两个嵌入传递到一个多风格编码器，以形成最终的多风格韵律嵌入。因此，修改预测的F0值可以提供对句子、单词和音素级别的韵律的控制。

More flexibility in controlling the F0 feature is pro-vided in the controllable deep auto-regressive model(C-DAR) model [138] which allows for F0 contour adjustment by the user. To achieve this goal, three strat-egies are used: 1) context awareness by conditioning the model on the preceding and following speech dur-ing training, 2) conditioning the model on some ran-dom segments of ground truth F0, and 3) predicting F0 values in reverse order. Additionally, several text-based features are used as input to the model, includ-ing word embeddings derived from BERT, V/UV label,one-hot vector for the nearby punctuation, and pho-neme encodings. At inference, F0 values specified by the user are used as alternatives for the ground truth F0 segments, and the model predicts the rest of the utter-ance’s F0 contour through context awareness.
Discrete fine-grained representations for prosody features as in [48] [122] are also useful to limit the number of the obtained representations. Both studies [48] [122]utilize VQ [149] to map each prosody embedding to the closest discrete representation from a predefined code-book. In [48], a pitch predictor is used to predict charac-ter-level continuous pitch representation using character embeddings from the text encoder as input. Zhang et al.[122], however, produces syllable-level prosody embed-dings from a reference encoder that takes F0, intensity,and duration features from reference audio as input. The resulting prosody embeddings are then mapped to a pre-defined codebook to extractb discrete prosody codes.

在可控深度自回归模型（C-DAR）模型[138]中提供了对F0特征的更多控制灵活性，该模型允许用户调整F0轮廓。为了实现这一目标，使用了三种策略：1）通过在训练期间将模型条件化在先前和随后的语音上来实现上下文感知，2）将模型条件化在地面真F0的一些随机片段上，以及3）以相反的顺序预测F0值。此外，将几个基于文本的特征用作模型的输入，包括从BERT派生的词嵌入、V/UV标签、附近标点符号的一热向量和音素编码。在推理时，用户指定的F0值被用作地面真F0片段的替代品，并且模型通过上下文感知预测整个句子的F0轮廓。

与[48] [122]中一样，离散细粒度表示对于韵律特征也很有用，可以限制获得的表示的数量。两项研究[48] [122]都使用VQ[149]将每个韵律嵌入映射到预定义码本中最近的离散表示。在[48]中，使用音高预测器来预测使用文本编码器作为输入的字符嵌入的字符级连续音高表示。然而，张等人[122]从参考编码器产生音节级韵律嵌入，该参考编码器将参考音频的F0、强度和持续时间特征作为输入。然后，将得到的韵律嵌入映射到一个预定义的码本，以提取离散的韵律代码。


Resulting prosody codes in [48] represent the pitch and other suprasegmental information that can be adjusted via a specific bias value to generate speech with differ-ent pitch accents. The codes in [122], can be interpreted as representing some prosody features such as pitch and duration. The prosody variation at the syllable-level can be manually controlled by assigning each syllable the desired prosody code from the codebook.
In [125], ToBI features, which involve a set of con-ventions used for transcribing and annotating speech prosody, are used. The applied ToBI features are four word-level tags: pitch accents, boundary tones, phrase accents, and break indices. The extracted ToBI tags are used as input to TTS model. Simultaneously, a ToBI pre-dictor is trained to predict these prosody tags based on grammatical and semantic information extracted from the input text using a self-supervised language model.
The resulting model had the ability to control the stress,intonation, and pause of the generated speech to sound natural, utilizing only ToBI tags from the text-based predictor.

在[48]中，得到的韵律代码表示音高和其他超音段信息，可以通过特定的偏置值进行调整，以生成具有不同音高重音的语音。在[122]中，代码可以解释为表示一些韵律特征，如音高和持续时间。可以通过将每个音节分配从码本中所需的韵律代码来手动控制音节级别的韵律变化。

在[125]中，使用了ToBI特征，这是一种用于转录和注释语音韵律的约定集。应用的ToBI特征是四个单词级别的标签：音高重音、边界音、短语重音和停顿索引。提取的ToBI标签用作TTS模型的输入。同时，训练了一个ToBI预测器来预测这些韵律标签，基于使用自监督语言模型从输入文本中提取的语法和语义信息。

最终模型能够仅使用文本预测器中的ToBI标签来控制生成的语音的强调、语调和停顿，使其听起来自然。


#### 5.3.3 Modeling prosody strength 

This group of studies focus on regulating the strength of emotion or prosody. For instance, [61] utilizes the distance between emotion embeddings and the neutral emotion embeddings to identify scalar values for emotion intensity. It proposes a phoneme-level emotion embed-ding and a fine-grained emotion intensity. The emo-tion embedding is first obtained via a reference encoder.
The emotion intensity is then generated by an intensity extractor that takes the emotion embedding as input. The intensity extractor produces intensity as a scalar value based on the distance between the emotion embedding and the centroid of a pre-defined cluster for neutral emo-tion embeddings. The resulting emotion intensity values are quantized into pseudo-labels that serve as the index for an intensity embedding table.
Another method for learning emotion strength values in an unsupervised manner is by using ranking functions.
Studies [27] [31] [33] [64] utilize a ranking function-based method named relative attributes [89] for this purpose. In[33], prosody is modeled at three levels: global-level rep-resentation by emotion embedding, utterance-level rep-resented by prosody embedding from a reference-based encoder, and the local-level represented by emotion strength. The study trains an emotion strength extractor at the syllable-level based on input speech utilizing the ranking function. Simultaneously, a predictor of emo-tion strength is trained based on features extracted from input text via BERT model. Besides changing emotion label and emotion reference audio, the model provides manual control of the emotion strength values in the syn-thesized speech.
Alternatively, the reference encoder in [31] functions as a ranking function to learn a phoneme-level emotion strength (descriptor) sequence. The proposed ranking function [89] receives its input from fragments of target reference audio obtained via a forced alignment model to phoneme boundaries. The OpenSMILE [139] tool is then used to extract 384-dimensional emotion-related features from these reference speech fragments as input to the ranking function. Similarly, the proposed ranking func-tion in [27] takes a set of acoustic features extracted from the input speech via OpenSMILE tool but at the utter-ance-level as input. The ranking function leverages the difference between neutral samples and samples associ-ated with each emotion class in the dataset. The training process is formulated as solving a max-margin optimiza-tion problem. The resulting emotion strength scalars can be manually adjusted or predicted based on text or refer-ence speech.
In [64], both emotion class and emotion strength value are obtained via a joint emotion predictor based only on the input text. The input to the predictor is features extracted from input text via the Generative Pre-trained Transformer (GPT)-3 [88]. Emotion class and emotion strength are the two outputs of the predictor where the former is represented as a one-hot encoded vector and the latter is presented as a scalar value. Emotion labels and emotion strength values which are also obtained via[89], are used as ground truth for predictor training.
Another ranking method is proposed in [19] using the ranking support vector machine. The model generates style embedding and speaker embedding via two separate encoders. Both style and speaker embeddings at infer-ence time are represented by centroids of each single speaker and style embeddings. However, a linear SVM is trained with the model to provide the ability for style embedding adjustment. The proposed SVM model is trained to classify between neutral emotion and a specific emotion embedding, where the learned hyperplane is utilized to move(scale) the style vectors in a direction towards/opposite to the hyperplane.
Another type of control that contributes to generat-ing speech with a better representation of local prosodic variation is introduced in [124]. The proposed model suggests an unsupervised approach to obtain word-level prominence and phrasal boundary strength features. For this purpose, continuous wavelet transform (CWT) [168]is utilized to extract continuous estimates of word promi-nence and boundary information from the audio signal.
First, the three prosodic signals f0, energy, and duration are extracted and combined as input to the CWT. Then,the combined signal is decomposed via CWT into scales that represent prosodic hierarchy. Word and phrase-level prosody are then obtained by following ridges or valleys across certain scales. The continuous word prominence and boundary estimates are achieved via the integration of the resulting lines aligned with the textual informa-tion. With manually identified intervals, the continuous values of prominence and boundary strength are then discretized.

#### 5.3.4 Prosody clustering 

In this section, methods for selecting the appropri-ate prosody embedding for the referenced-based ETTS models are described. To begin with, clustering methods are utilized in [57] [58] to generate representative pros-ody embeddings for each emotion class when the GST-TTS model is trained with a labeled dataset. Initially,the resulting emotion embeddings are clustered in a 2d space. In [57], the centroid of each cluster is used as the weights of the GSTs to generate emotion embedding for each emotion class. In [58], the weight vector that repre-sents each emotion cluster is obtained by considering the inter and intra distances between emotion embedding clusters. Specifically, an algorithm is used for minimizing each embedding distance to the target emotion cluster and maximizing its distance to other emotion clusters.
Similarly, clustering algorithms are applied in [112] [113] to achieve discrete prosody embeddings but for two specific prosody-related features. The two studies employ K-means algorithm to cluster F0 and duration features extracted for each phoneme. The centroids of the clus-ters are then used as discrete F0 and duration values/tokens for each phoneme. work [112] applies a balanced clustering method with duration features to overcome degradation in voice quality that appeared in [113] dur-ing duration control. Moreover, to keep phonetic and prosodic information separate during training, an atten-tion unit is introduced to map prosody tokens to decoder hidden states and generate prosody context vectors. The resulting discrete tokens for F0 and duration features provide a fine-grained level of control over prosody by changing the corresponding prosodic tokens for each phoneme.
In [105], a cross-domain SER model with the GST-TTS model is proposed to obtain emotion embeddings for an unlabeled dataset. The cross-domain SER model is trained using two datasets including: 1) an SER data-set (source) labeled with emotions, and 2) a TTS data-set (target) that is not labeled. Simultaneously, the SER model trains an emotion classifier that generates soft labels for the unlabeled TTS dataset. These soft labels are then used to train an extended version of the baseline in[74] with an emotion predictor. In the training process,the weights of the style tokens layer are passed as input to the predictor, which employs the learned soft labels as ground truth values. At inference time, weights vectors for each emotion class are averaged to obtain the emo-tion class embedding. However, since the predicted labels for the TTS dataset are soft labels, and thus not entirely reliable, only the top K samples with the highest posterior probabilities are selected.

### 5.4 Speech synthesis for unseen speakers and unseen styles

Building a speech synthesis model that supports mul-tiple speakers or styles can be achieved by training TTS model with a multi-speaker multi-style dataset. How-ever, generating speech for an unseen speaker or style is a challenging task for which several solutions have been proposed in the literature. A popular approach is to fine-tune the averaged TTS model with some samples from the unseen target speaker or style. The fine-tuning pro-cess may require a single sample from the unseen speaker or style (referred to as one-shot models) or a few samples(referred to as few-shot models). There are also models that do not require any fine-tuning steps, and these are known as zero-shot TTS models.
For instance, the fine-tuning process proposed in [112]focused on sentences used in the process to ensure pho-netic coverage, meaning that each phoneme should appear at least once in these sentences. The proposed model requires about 5 minutes of recordings from the unseen target speaker to clone the voice and allow for manipulation of some voice features (such as F0 and duration) by the model at the phoneme-level.
Another approach to address the problem of unseen data is to employ specific structures in the TTS model,as proposed in [52] [96] [97] [107]. As an example, in [107],a cycle consistency network is proposed with two Vari-ational Autoencoders (VAEs). The model incorporates two training paths: a paired path and an unpaired path.
The unpaired path refers to training scenarios where the reference audio differs from the output (target) speech in terms of text, style, or speaker. Two separate style encod-ers are utilized in the model, with one dedicated to each path. This structure facilitates style transfer among intra-speaker, inter-speaker, and unseen speaker scenarios.
In [52], the U-net structure proposed for the TTS model supports one-shot speech synthesis for unseen styles and speakers. The U-net structure is used between the style encoder and the mel decoder of the TTS model,with an opposite flow between them. Both the style encoder and decoder consist of multiple modules with the main building unit as ResCnn1D and instance nor-malization (IN) layers. The decoder receives phoneme embedding and produces the Mel-spectrogram as out-put. In parallel, the style encoder receives the reference audio and produces its linguistic content with guidance from the content (text) encoder. The style encoder mod-ules produce latent variables, i.e., mean, and standard deviation, for the hidden inputs in the IN layers. These latent variables are used to bias and scale the normal-ized hiddens of the corresponding module layers in the decoder.
A separate encoder (reference encoder) has been used in [96] to extract speaker-related information besides the prosody encoder (extractor) that encodes prosody fea-tures into the prosody embedding. A prosody predictor is also trained to predict the prosody embedding based on the phoneme-embedding. While the instance nor-malization (IN) layer is utilized by the prosody extractor to remove global (speaker) information and to keep pros-ody-related information, the speaker encoder is designed with a special structure (Conv2D layers, residual blocks(GLU with fully connected layers), and a multi-head self-attention unit) for better extraction of speaker informa-tion. Moreover, instead of concatenation or summation with the decoder input, the speaker embedding is adap-tively affine transformed to the different FFT blocks of the decoder through a Speaker-Adaptive Linear Modu-lation (SALM) network that is inspired by Feature-wise Linear Modulation (FiLM) [141]. The speaker encoder and conditioning of decoder blocks with speaker embed-ding allow the model to generate natural speech for unseen speakers with only a single reference sample(zero-shot).The attention unit used in seq2seq TTS models aims at mapping the different length between text and audio pairs. However, it can get unstable when the input is not seen during training [97]. The STYLER model has addressed this issue by using a linear compression or expansion of the audio to match the text’s length via a method named Mel Calibrator. With this simplification of the alignment process as a scaling method, the unseen data robustness issue is alleviated and all audio-related style factors become dependent only on the audio.
Similarly, in [119], the Householder Normalizing Flow[169] is incorporated into the VAE-based baseline model[77]. The Householder normalizing flow applies a series of easily invertible affine transformations to align the VAE’s latent vectors (style embeddings) with a full covari-ance Gaussian distribution. As a result, the correlation among the latent vectors is improved. Generally, this architecture enhances the disentanglement capability of the baseline model and enables it to generate embedding for unseen style with just a single (one-shot) utterance of around one second length.
The Multi-SpectroGAN TTS model proposed in [98]is a multi-speaker model trained based on adversarial feedback. The model supports the generation of speech for unseen styles/speakers by introducing adversarial style combination (ASC) during the training process.
Style combinations result from mixing/interpolating style embeddings from different source speakers. The model is then trained with adversarial feedback using mixed-style mel-spectrograms. Two mixing methods are employed:binary selection or manifold mix-up via linear combina-tion. This training strategy enables the model to generate more natural speech for unseen speakers.
Lastly, recent TTS models based on in-context learn-ing [18] [22] [25] all share the capability to perform zero-shot speech synthesis, as explained in Section 4.4. In fact, the in-context training strategy underlies the ability of these models to synthesize speech given only a style prompt with the input text. Specifically, the synthesis process treats the provided prompt/reference as part of the desired output speech. Therefore, the model’s goal is to predict the rest of this speech in the same style as the given part (prompt) and with the input text. In Table 5 we list papers addressing each challenge.

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