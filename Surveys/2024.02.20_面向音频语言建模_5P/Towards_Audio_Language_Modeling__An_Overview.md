# Towards Audio Language Modeling: An Overview

作者列表

- [Haibin Wu](../../Authors/Haibin_Wu.md) 
- [Xuanjun Chen](../../Authors/Xuanjun_Chen.md)
- [Yi-Cheng Lin](../../Authors/Yi-Cheng_Lin.md) 
- [Kai-wei Chang](../../Authors/Kai-wei_Chang_(张凯为).md)
- [Ho-Lam Chung](../../Authors/Ho-Lam_Chung.md) 
- [Alexander H. Liu](../../Authors/Alexander_H._Liu.md) 
- [李宏毅](../../Authors/Hung-Yi_Lee_(李宏毅).md)

## Abstract·摘要

> Neural audio codecs are initially introduced to compress audio data into compact codes to reduce transmission latency.
> Researchers recently discovered the potential of codecs as suitable tokenizers for converting continuous audio into discrete codes, which can be employed to develop audio language models (LMs).
> Numerous high-performance neural audio codecs and codec-based LMs have been developed.
> The paper aims to provide a thorough and systematic overview of the neural audio codec models and codec-based LMs.

最初引入神经音频编解码器是用于压缩音频数据为紧凑编码以降低传输延迟.
最近研究人员发现编解码器作为将连续音频转换为离散代码的分词器的潜力, 这可以用于开发音频语言模型.
许多高性能的神经音频编解码器和基于编解码器的语言模型已经被开发出来.
本文旨在提供有关神经音频编解码器模型和基于编解码器的语言模型的全面综述.

## 1.Introduction·引言

> Neural audio codec models were first introduced to compress audio for efficient data transmission. 
> The encoder converts the audio into codec codes, which are then transmitted.
> The receiver then uses the codec decoder to reconstruct the audio using the received codes.

神经音频编解码模型首次引入用于压缩音频以高效数据传输.
编码器将音频转换为编解码器编码, 然后传输.
接收方使用编解码器的解码器使用收到的编码重构音频.

> Language modeling has proven to be highly successful in the field of Natural Language Processing (NLP). 
> Audio data encompasses not only textual content but also rich information about speaker timbre, emotion, and general audio, offering deeper possibilities for language model applications.
> Researchers, especially those in large companies with significant computational resources, recently leverage the potential of neural codecs ([Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md), [SoundStorm (2023)](../../Models/Speech_LLM/2023.05.16_SoundStorm.md), [AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md), [AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md), [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md), [SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), [Descript-Audio-Codec (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md)) as suitable tokenizers for converting continuous audio into discrete codes, which can be employed to develop audio language models (LMs) ([AudioLM (2022)](../../Models/Speech_LLM/2022.09.07_AudioLM.md), [AudioPaLM (2023)](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md), [MusicLM (2023)](../../Models/Speech_LLM/2023.01.26_MusicLM.md), [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [VALL-E X (2023)](../../Models/Speech_LLM/2023.03.07_VALL-E_X.md), [VioLA (2023)](../../Models/Speech_LLM/2023.05.25_VioLA.md), [UniAudio (2023)](../../Models/Speech_LLM/2023.10.01_UniAudio.md), [LauraGPT (2023)](../../Models/Speech_LLM/2023.10.07_LauraGPT.md), [SpeechX (2023)](../../Models/Speech_LLM/2023.08.14_SpeechX.md), [MusicGen (2023)](../../Models/Speech_LLM/2023.06.08_MusicGen.md), [AudioGen (2022)](../../Models/Speech_LLM/2022.09.30_AudioGen.md)).
> The current codec-based language models and codec models are summarized in [Fig.01](#Fig.01). 
> These findings promptly garnered the community’s attention, sparking a fervor for developing codecs tailored to audio language modeling. 
> Numerous high performance neural audio codec models and audio LMs have been developed.

语言建模已经被证明在自然语言处理领域是高度成功的.
音频数据不仅包含文本内容还有关于说话人音色, 情感, 以及一般音频的丰富信息, 为语言模型应用提供了更深层次的可能性.
研究人员, 尤其是那些拥有大量计算资源的大型公司的研究人员, 最近开始利用神经编解码器的潜力, 将其作为合适的分词器把连续的音频转化为离散编码, 从而用于开发音频语言模型.
当前基于编解码器的语言模型和编解码器模型总结在[图 01](#Fig.01) 中.
这些发现迅速引起了社区的关注, 激发了开发专门为音频语言建模定义的编解码器的热潮.
许多高性能的神经音频编解码器模型和音频语言模型已经被开发出来.

<a id="Fig.01"></a>
<center>

![Timeline of Current Neural Codec Models and Codec-Based Language Models](../_Images/2024.02.20_Fig.01.png)

> Fig.01: Timeline of Current Neural Codec Models and Codec-Based Language Models

</center>

> An ideal codec should maintain content while preserving paralinguistic and speaker-related information. 
> Similarly, a universal audio language model should be able to generalize across various audio types, such as speech, music, and general audio, covering a wide range of applications. 
> The arms race in developing codecs and audio LMs is still ongoing.

一个理想的编解码器应该在保持内容的同时, 保留伴随语言和说话人相关的信息.
类似地, 一个通用的音频语言模型应该能够跨多种音频类型进行泛化, 例如语音, 音乐和一般音频, 从而覆盖广泛的应用场景.
开发编解码器和音频语言模型的竞赛仍在继续.

> Given the significant advancements in codecs and audio language models over the past three years as shown in [Fig.01](#Fig.01), there has yet to be a comprehensive review comparing them and providing inspiration to the community.
> In this study, we aim to fill this research gap by thoroughly reviewing and comparing various existing neural codec models and audio codec-based language models.
> Firstly, we specifically conduct an in-depth analysis of six representative open-source neural codec models to cover their training methodologies, implementation settings, and training data. 
> Secondly, we expand our analysis to include eleven diverse codec-based language models, examining how they utilize the codecs and the tasks to which they can be applied. 
> Through this comprehensive review, we aim to offer the community insights into the diverse methodologies and potential directions in the field of neural codecs and codec-based language modeling.

鉴于过去三年中编解码器和音频语言模型所取得的显著进步 (如[图 01](#Fig.01) 所示), 目前尚未有一篇全面的综述来比较它们并为社区提供启发.
在本研究中, 我们的目标是通过详尽回顾和比较各种现有的神经编解码器模型和基于编解码器的音频语言模型.
首先, 我们特别地对六个代表性的开源神经编解码器模型进行深入分析, 涵盖它们的训练方法, 实现细节和训练数据.
其次, 我们将分析扩展到包括十一个不同的基于编解码器的语言模型, 探讨它们如何利用编解码器和可以应用的任务.
通过这一全面的回顾, 我们期望为社区提供关于神经编解码器和基于编解码器的语言建模领域中多样化方法和潜在方向的见解.

## 2.Comprehensive Comparison for Neural Audio Codec Models·神经音频编解码器模型的综合比较

> Codec models aim to compress and decompress speech signals efficiently.
> Traditional codecs are developed based on psycho-acoustics and speech synthesis [21], [22].
> Recently, the neural codec models demonstrated highly effective for compression and signal reconstruction, outperforming traditional codecs.
> Considering the broad spectrum of codec models within the research community, each trained with its distinct configurations and training techniques, there is a clear need for a thorough examination that covers the training methodologies, implementation settings, and training data employed across these codec models.
> The six codec models have distinct training details, resulting in a collection of fifteen different codec models, as summarized in [Tab.01](#Tab.01).

编解码器模型的目标是高效地压缩和解压缩语音信号.
传统的编解码器是基于心理声学和语音合成开发的.
近期, 神经编解码器模型在压缩和信号重建方面表现出了极高效率, 超越了传统编解码器.
考虑到研究社区中编解码器模型的广泛范围, 每个模型都使用其独特的配置和训练技术进行训练, 显然需要进行一次彻底的检查, 以涵盖这些编解码器模型所采用的训练方法, 实现设置和训练数据.
以下六个编解码器模型有着不同的训练细节, 从而形成一个包含十五个不同编解码器模型的集合, 如[表 01](#Tab.01) 所示.

<a id="Tab.01"></a>

![](../_Images/2024.02.20_Tab.01.png)

> Tab.01: Codec Information Comparison. 
> `A-F` represents different neural codec models, where 
> - `A` is [SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), 
> - `B` is [AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md), 
> - `C` is [AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md), 
> - `D` is [Descript-Audio-Codec (DAC) (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md), 
> - `E` is [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md),
> - `F` is [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md). 
> 
> $n_c$ represents the codebook number, `SR` represents the Sample Rate, and `BPS` represents the bit rate in unit Bits Per Second.

### 2.1.Brief Method Overview for Codecs·编解码器的简要方法概述

> [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) stands as one of the pioneering implementations of neural codec models, embodying a classic neural codec architecture comprising encoder, quantizer, and decoder modules.
> It utilizes the streaming [SEANets (2020)](../../Models/_Basis/2020.09.04_SEANet.md) as its encoder and decoder.
> The quantizer incorporates a speech enhancement system with a **Residual Vector Quantization (RVQ)** ([SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md), [MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md)) bottleneck to obtain parallel token streams.
> During training, the model parameters are optimized using a combination of reconstruction and adversarial loss.
> [SoundStorm (2023)](../../Models/Speech_LLM/2023.05.16_SoundStorm.md) is an improved version of [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) to achieve both efficiency and high-quality audio generation.
> It accomplishes this by employing an architecture specifically tailored to the hierarchical structure of audio tokens.
> Moreover, it pioneers a parallel, non-autoregressive decoding scheme, which relies on confidence-based strategies for residual vector-quantized token sequences.

[SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 作为神经编解码器模型的先驱之一, 体现了一个经典的神经编解码器架构, 由编码器, 量化器和解码器模块组成.
它使用了流式 [SEANets (2020)](../../Models/_Basis/2020.09.04_SEANet.md) 作为编码器和解码器.
量化器集成了一个带有残差向量量化瓶颈的语音增强系统以获得并行的 Token 流.
在训练时, 模型参数使用重构损失和对抗损失的结合进行优化.

#TODO ?
[SoundStorm (2023)](../../Models/Speech_LLM/2023.05.16_SoundStorm.md) 是 [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 的改进版本, 实现了高效和高质量音频生成.
该模型通过采用专门针对音频 Token 的层次结构的架构来实现这一点.
此外, 它首创了一种并行的, 非自回归的解码方案, 该方案依赖于基于置信度的策略来处理残差向量量化后的 Token 序列.

> [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) builds upon a framework similar to [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md).
> Nonetheless, it further augments its capabilities by integrating supplementary LSTM [25] layers and harnessing a [Transformer (2017)](../../Models/_Basis/2017.06.12_Transformer.md)-based language model to model the RVQ codes, thereby amplifying its sequence modeling performance.
> Then, there is a stream of work aimed at making codec models more general and powerful.
> [AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md) represents an enhanced version of [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), implementing a group convolution mechanism to facilitate the real-time operation of the streamable network while also harnessing the capabilities of [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) to effectively generate high-fidelity audio at a high sampling rate of 48 kHz. 

[Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 建立在类似于 [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 的框架之上.
然而, 它通过整合追加的 LSTM 层和利用基于 [Transformer (2017)](../../Models/_Basis/2017.06.12_Transformer.md) 的语言模型来建模 RVQ 编码以进一步增强其能力, 从而扩大其序列建模性能.
因此, 有一系列工作旨在使编解码器模型更加通用和强大.
[AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md) 是 [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 的增强版本, 应用了组卷积机制, 以促进流式网络的实时操作, 同时利用 [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) 的能力, 在 48k Hz 的高采样率下有效地生成高保真度的音频.

> In the [AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) model, a novel technique known as group-residual vector quantization is presented.
> It employs multiple parallel RVQ groups.
> This technique is specifically tailored for generation tasks.
> It aims to enhance the reconstruction performance while using a limited number of codebooks, consequently achieving an impressively low bit rate per second (BPS).
> This low BPS is of utmost significance as it effectively addresses the challenge of lengthy speech tokens in speech language modeling, resulting in reduced sequence lengths.

[AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) 模型提出了一个新的技术, 组残差向量量化.
它采用了多个并行的 RVQ 组.
这一技术是专门为生成任务而设计的.
其目的是在使用有限的码本的同时提高重构性能, 因此实现了惊人的低码率每秒 (BPS).
这种低码率对于长语音 Token 在语音语言建模中的挑战是至关重要的, 能使得序列长度减少.

> [SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) is a unified speech tokenizer designed for speech language models.
> It implements an Encoder-Decoder architecture enhanced with RVQ.
> By integrating both semantic and acoustic tokens, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) hierarchically separates various aspects of speech information across different RVQ layers.
> Specifically, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) is designed to regularize the first RVQ layer to highlight semantic information by learning the [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) tokens.
> Using such techniques can enhance the disentanglement of information across different RVQ layers.

[SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 是为语音语言模型设计的统一语音分词器.
它采用了 RVQ 增强的编码器-解码器架构.
通过整合语义 Token 和声学 Token, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 逐层地将语音信息分离到不同的 RVQ 层.
具体来说, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 被设计成通过学习 [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) Token 正则化第一个 RVQ 层, 以突出语义信息.
采用这样的技术可以增强不同 RVQ 层之间的信息解耦.

> [Descript-Audio-Codec (DAC) (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md), a universal neural codec model, distinguishes itself through its exceptional ability to maintain high-fidelity audio quality across a wide spectrum of data types, encompassing general audio, music, and speech.
> It accomplishes this feature by employing a number of training techniques, such as periodic activation functions [29], enhanced residual vector quantization using factorized and L2-normalized codes, random quantizer dropout to preserve audio reconstruction quality, as well as refining adversarial and reconstruction loss during the training process.
> The authors highlight the crucial importance of the periodic activation function among the employed techniques.

[Descript-Audio-Codec (DAC) (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) 是一种通用神经编解码器模型, 其独特的能力能够在广泛的数据类型范围内 (一般音频, 音乐和语音) 保持高保真度的音频质量.
它通过采用多种训练技术来实现这一特性, 例如周期激活函数, 使用因子化和 L2 归一化编码的增强残差向量量化, 随机量化器失活以保持音频重建质量, 以及在训练过程中细化对抗损失和重构损失.
作者强调了周期性激活函数在所采用技术中的关键重要性.

> Unlike most models focusing on the time domain, [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) proposes a frequency-domain codec.
> The authors claim they can achieve comparable performance with fewer parameters and lower computation complexity.
> Meanwhile, it also finds that incorporating semantic information in the codec tokens improves speech quality at low bit rates.

与大部分模型关注时域不同, [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 提出了频域编解码器.
作者声称他们可以用较少参数和较低计算复杂度获得相当的性能.
同时还发现将语义信息整合到编解码器 Token 中可以提高低码率下的语音质量.

### 2.2.Comparison from Methodology Angles·方法论角度的比较

> We compare several techniques proposed by these codecs in [Tab.02](#Tab.02).
> The abbreviation `A-F` represents different codec models.
> Please refer to [Tab.01](#Tab.01) for the corresponding model full name.

我们比较这些编解码器提出的数种技术, 如[表 02](#Tab.02) 所示.
缩写 `A-F` 表示不同的编解码器模型. 
请参考[表 01](#Tab.01) 以获取相应模型的全称.

<a id="Tab.02"></a>

![](../_Images/2024.02.20_Tab.02.png)

> Tab.02: Comparison between codec implementation strategy. 
> - `SEM` represents codec including Semantic Tokens. 
> - `SNAKE` represents the codec model that employs Snake Activation. 
> - `MRG` represents codec has Multiple Residual Groups.
> - `NOISY` represents codec utilizes Noisy data in training. 
> - `LM` represents the model including Language Model training. 
> - `KM` represents codec uses K-Means to cluster samples as initialization of VQ codebook.

#### 2.2.1.Discriminators·判别器

> The design of discriminators constitutes a pivotal element within codec models.
> [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) initially introduces the **Multi-Scale-STFT Discriminator (MS-STFTD)**.
> In contrast to the **Multi-Scale Discriminator (MSD)** proposed in [MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md), which captures long-term dependencies, the **Multi-Period Discriminator (MPD)** proposed in [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) exhibits a capacity to discern more nuanced periodic details.
> Consequently, [AudioDec](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md) replaces the conventionally employed **STFTD** with a [HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md)-based MPD, observing an enhancement in audio quality within their model.
> [AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) integrates prior research efforts by incorporating the **MS-STFTD** from [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) and both [HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md)-based **MPD** and **MSD**.
> Both [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) and [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) adopt identical discriminators to [AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md), with [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) offering a unified interface adaptable to any combination of these three discriminator types.
> [DAC](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) identifies that employing MSD and MPD alone generates audio displaying blurriness and artifacts.
> To address this, they propose the application of a **Multi-Scale Multi-Band STFT Discriminator (MS-MB-STFT-D)** to improve phase modeling and mitigate aliasing artifacts.

判别器的设计是编解码器模型中的关键元素.
[Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 最初引入了 **多尺度短时傅里叶变换判别器 (Multi-Scale-STFT Discriminator, MS-STFTD)**.
与 [MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md) 提出的**多尺度判别器 (Multi-Scale Discriminator, MSD)** 用于捕获长期依赖不同, [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) 提出的**多周期判别器 (Multi-Period Discriminator, MPD)** 具有更强的能力来辨别更微妙的周期细节.
因此, [AudioDec](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md) 将传统使用的**短时傅里叶变换判别器 (STFTD)** 替换为基于 [HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) 的多周期判别器, 观察其模型中的音频质量提升.

[AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) 整合了先前研究的努力, 即混合了 [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 的 **MS-STFTD**, 基于 [HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) 的 **MPD** 和 **MSD**.

[SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 和 [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 都采用了和 [AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) 相同的判别器, 其中 [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 提供了一个统一的接口, 可以适应这三种判别器的任意组合.

[DAC](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) 发现, 仅仅使用 **MSD** 和 **MPD** 生成的音频会出现模糊和伪影.
为了解决这一问题, 他们提出了一个**多尺度多频带的 STFT 判别器 (Multi-Scale Multi-Band STFT Discriminator, MS-MB-STFT-D)** 用于提升相位建模和减轻混叠伪影.

#### 2.2.2 Quantizer·量化器

> [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) utilizes semantic tokens from [HuBERT L9](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) as a teacher for the RVQ process.
> This guidance enables the disentanglement of content information into the first layer of the tokenizer, while paralinguistic information is retained in subsequent layers.
> [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) seeks to integrate semantic information by combining, adding, or residualizing the audio codec with semantic tokens.
> The study reveals that including semantic tokens enhances audio quality, particularly with the residual inclusion method.
> Additionally, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) and [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) utilize K-means to cluster samples in the first mini-batch for initializing the VQ codebook, leading to improved code utilization.
> [DAC](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) follows the approach of [BigVGAN (2022)](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md), employing snake activation [29] for trainable control over the frequency of periodic signals.
> [AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) employs multiple RVQ codebooks (multiple residual groups) to represent intermediate features.
> They demonstrate that using multiple residual groups achieves good reconstruction performance while employing only a few codebooks.
> [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) trains an additional small [Transformer](../../Models/_Basis/2017.06.12_Transformer.md) model for entropy coding over the quantized units, which reduces bandwidth and accelerates encoding and decoding.

[SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 利用从 [HuBERT L9](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) 提取的语义 Token 作为教师来指导 RVQ 过程.
这一指导使得内容信息在分词器的第一层被分离, 而其他语言相关的信息在后续层中被保留.

[FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 试图通过结合, 添加或残差化音频编解码器和语义 Token 来融合语义信息.
研究表明包含语义 Token 可以提高音频质量, 特别是使用残差化包含方式.

此外, [SpeechTokenizer](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 和 [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 利用 K-means 对第一个小批量样本进行聚类, 以便初始化 VQ 码本, 进而提高码本利用率.

[DAC](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) 遵循 [BigVGAN (2022)](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md) 的方法, 应用蛇形激活来实现对周期信号频率的可训练控制.


[AcademiCodec](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) 采用多个 RVQ 码本 (多个残差组) 来表示中间特征.
他们证明使用多个残差组可以在仅使用少量码本的情况下实现良好的重建性能.

[Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 训练了一个额外的小型 [Transformer](../../Models/_Basis/2017.06.12_Transformer.md) 模型来对量化单元进行熵编码, 这可以减少带宽并加速编码和解码.

### 2.3.Implementation Details·实现细节

> We compare the codebook number, training data, sampling rate, and bit rate per second in [Tab.01](#Tab.01).
> From the training data perspective, [SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), [AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md), and [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) utilize only English speech dataset.
> [AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) incorporates bilingual speech datasets, including [AISHELL](../../Datasets/AISHELL.md) for Chinese and [LibriTTS](../../Datasets/LibriTTS.md) and [VCTK](../../Datasets/VCTK.md) for English.
> Both [DAC (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md), and [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) encompass diverse modality data, including speech, music, and audio, in the training data.

我们在[表 01](#Tab.01) 中比较了码本数量, 训练数据, 采样率, 和每秒比特率.
从训练数据的角度看:
- [SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), [AudioDec (2023)](../../Models/Speech_Neural_Codec/2023.05.26_AudioDec.md), 和 [FunCodec (2023)](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 只使用了英语语音数据集.
- [AcademiCodec/HiFi-Codec (2023)](../../Models/Speech_Neural_Codec/2023.05.04_HiFi-Codec.md) 包含了双语语音数据集, 包括 [AISHELL](../../Datasets/AISHELL.md) 用于中文, [LibriTTS](../../Datasets/LibriTTS.md) 和 [VCTK](../../Datasets/VCTK.md) 用于英语.
- [DAC (2023)](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) 和 [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 在训练数据中包含了多种模态数据, 包括语音, 音乐, 以及音频.

## 3.Current Codec-Based Speech Language Models·现有的基于编解码器的语音语言模型

> As shown in [Fig.02](#Fig.02), the process of neural codec-based audio language modeling begins by converting context information, such as text and MIDI, into context codes, while simultaneously encoding the audio into codec codes.
> These context and codec codes are then employed in the language modeling phase to generate the desired target codec code sequence.
> Subsequently, the target codec code sequence is passed to the codec decoder to produce the audio output.
> The entire pipeline embodies an audio-to-audio modeling approach.

<a id='Fig.02'></a>

![](../_Images/2024.02.20_Fig.02.png)

> Fig.02: Codec-Based Language Modeling.

如[图 02](#Fig.02) 所示, 基于神经编解码器的音频语言建模过程首先将内容信息 (文本和 MIDI) 转化为内容编码, 同时将音频编码为编解码器编码.
这些内容和编解码器编码之后在语言建模阶段用于生成所需的目标编解码器代码序列.
之后, 目标编解码器编码序列传递到编解码器解码器以产生音频输出.
整个流程体现了一种音频到音频的建模方法.

### 3.1.Overview of Codec-Based Language Models·编解码器语言模型概览

> [AudioLM (2022)](../../Models/Speech_LLM/2022.09.07_AudioLM.md) is the pioneering model in introducing codec codes for language modeling, utilizing a hierarchical approach that encompasses two distinct stages.
> The first stage generates semantic tokens using a self-supervised [W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) model.
> These tokens are then leveraged in the second stage as conditioning elements to create acoustic tokens using a [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) neural codec.

[AudioLM (2022)](../../Models/Speech_LLM/2022.09.07_AudioLM.md) 是引入编解码器编码进行语言建模的先驱模型, 采用包含两个不同的阶段的分层方法.
第一个阶段使用自监督 [W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) 模型生成语义 Token.
第二个阶段使用这些 Token 作为条件化元素, 使用 [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 神经编解码器生成音频 Token.

> [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [VALL-E X (2023)](../../Models/Speech_LLM/2023.03.07_VALL-E_X.md), and [SpeechX (2023)](../../Models/Speech_LLM/2023.08.14_SpeechX.md), all originate from Microsoft and are neural codec language models trained to generate discrete codes derived from [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), based on textual or acoustic inputs.
> [VALL-E](../../Models/Speech_LLM/2023.01.05_VALL-E.md) can generate high-quality personalized speech with only a 3-second enrollment recording from an unseen speaker.
> Furthermore, [VALL-E X](../../Models/Speech_LLM/2023.03.07_VALL-E_X.md) can produce high-quality speech in the target language with just a single speech utterance in the source language as a prompt.
> Additionally, [SpeechX (2023)](../../Models/Speech_LLM/2023.08.14_SpeechX.md) introduces a unified framework to address not only zero-shot TTS but also various types of speech transformation tasks, including speech enhancement and speech editing.

[VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [VALL-E X (2023)](../../Models/Speech_LLM/2023.03.07_VALL-E_X.md) 和 [SpeechX (2023)](../../Models/Speech_LLM/2023.08.14_SpeechX.md) 都来自于 Microsoft, 是经过训练的神经编解码器语言模型用于根据文本或声学输入, 生成基于 [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 的离散编码.

- [VALL-E](../../Models/Speech_LLM/2023.01.05_VALL-E.md) 可以仅凭一个未见过的说话人的三秒输入语音生成高质量的个性化语音.
- [VALL-E X](../../Models/Speech_LLM/2023.03.07_VALL-E_X.md) 则能够仅凭源语言的单个语音发言作为提示, 生成目标语言的高质量语音.
- [SpeechX](../../Models/Speech_LLM/2023.08.14_SpeechX.md) 引入了一个统一框架, 不仅可以解决零样本文本转语音, 还能处理各种类型的语音转换任务, 包括语音增强和语音编辑.

> What sets [VioLA (2023)](../../Models/Speech_LLM/2023.05.25_VioLA.md), [AudioPaLM (2023)](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md), and [LauraGPT (2023)](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) apart is their dual capability to generate both text and audio.
> [VioLA](../../Models/Speech_LLM/2023.05.25_VioLA.md) tries to tackle the question “Is one decoder-only generative model all you need for speech recognition, synthesis, and translation?” by employing language modeling that integrates both text tokens and audio tokens (extracted by [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md)), along with the use of task IDs and language IDs.
> [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md) constructs a unified vocabulary comprising both text and audio tokens.
> It is a decoder-only, autoregressive model capable of processing and generating both text and speech.
> Additionally, [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md)’s initialization stems from [PaLM-2 (2023)](../../Models/LLM/2023.05.17_PaLM2.md), a text-only language model.
> [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md)’s approach to audio tokenization resembles that of [AudioLM](../../Models/Speech_LLM/2022.09.07_AudioLM.md).
> Moreover, [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md) adopts and extends the [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) model to [SoundStorm (2023)](../../Models/Speech_LLM/2023.05.16_SoundStorm.md).
> [LauraGPT (2023)](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) is a versatile language model built on a decoder-only text-based language model, [Qwen-2B (2023)](../../Models/LLM/2023.09.28_Qwen.md).
> [LauraGPT](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) has the capability to process both audio and text inputs, generating outputs in either modality.
> [LauraGPT](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) encodes input audio into continuous representations using a [Conformer](../../Models/_Basis/2020.05.16_Conformer.md) encoder and decodes output audio using [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) discrete codes.
> The authors claim this specific audio features design for inputs and outputs will result in improved performance for speech generation using some preliminary experimental results.

[VioLA](../../Models/Speech_LLM/2023.05.25_VioLA.md), [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md), 和 [LauraGPT](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) 的不同指出在于它们具有生成文本和音频的双重能力.
- [VioLA](../../Models/Speech_LLM/2023.05.25_VioLA.md) 试图解决 "是否只需要一个仅用解码器的生成模型就能完成语音识别, 合成和翻译?" 的问题, 通过采用结合 文本 Token 和音频 Token (由 [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 提取), 任务 ID 和语言 ID 进行语言建模.
- [AudioPaLM](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md) 构造了一个统一的词汇表, 包括文本和音频 Token. 它是一个仅使用解码器, 自回归模型, 能够处理和生成文本和语音. 此外, 其初始化来自一个仅文本的语言模型 [PaLM-2 (2023)](../../Models/LLM/2023.05.17_PaLM2.md). 其音频分词方式与 [AudioLM](../../Models/Speech_LLM/2022.09.07_AudioLM.md) 类似. 此外, 还采用并扩展了 [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 到 [SoundStorm](../../Models/Speech_LLM/2023.05.16_SoundStorm.md).
- [LauraGPT](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) 是建立在仅使用解码器的基于文本的语言模型 [Qwen-2B (2023)](../../Models/LLM/2023.09.28_Qwen.md) 的多功能语言模型, 它具有处理音频和文本输入的能力, 生成的双模态输出. 它采用 [Conformer](../../Models/_Basis/2020.05.16_Conformer.md) 编码器编码输入音频, 并使用 [FunCodec](../../Models/Speech_Neural_Codec/2023.09.14_FunCodec.md) 离散编码解码为输出音频. 作者声称特定的音频特征设计, 对于输入和输出, 将会提升语音生成的性能, 并给出了一些初步的实验结果.

> [UniAudio (2023)](../../Models/Speech_LLM/2023.10.01_UniAudio.md) utilizes language modeling to generate a wide range of audio types, including speech, sounds, music, and singing, using textual or acoustic tokens as inputs.
> [UniAudio](../../Models/Speech_LLM/2023.10.01_UniAudio.md) stands out for its ability to enhance autoregressive prediction speed by introducing a multi-scale Transformer model [35], which employs a large global Transformer to predict the first-layer codec codes and a small local Transformer to predict the codec codes for the subsequent codec layers.
> The codec model in [UniAudio](../../Models/Speech_LLM/2023.10.01_UniAudio.md) is revised from [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md).

[UniAudio (2023)](../../Models/Speech_LLM/2023.10.01_UniAudio.md) 利用语言建模生成广泛的音频类型, 包括语音, 声音, 音乐, 歌唱, 输入可以是文本或声学 Token.
[UniAudio](../../Models/Speech_LLM/2023.10.01_UniAudio.md) 的突出之处在于它通过引入多尺度 Transformer 模型来提高自回归预测速度, 该模型使用一个大的全局 Transformer 来预测第一层编解码器编码, 以及一个小的局部 Transformer 来预测后续编解码器层的编码.
[UniAudio](../../Models/Speech_LLM/2023.10.01_UniAudio.md) 中的编解码器模型是基于 [Encodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 改进的.

> Additionally, there are other codec-based language models designed for sound modeling.
> [AudioGen (2022)](../../Models/Speech_LLM/2022.09.30_AudioGen.md) trained a [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) model to get audio tokens and subsequently trained a language model to utilize textual features as conditions for generating audio tokens.
> [MusicLM (2023)](../../Models/Speech_LLM/2023.01.26_MusicLM.md) follows a training strategy similar to [AudioLM](../../Models/Speech_LLM/2022.09.07_AudioLM.md) but extends its scope to encompass music features.
> It approaches the task of conditional music generation through a hierarchical sequence-to-sequence modeling approach.
> Initially, it utilizes music tokens from [Mulan](../../Models/_Basis/2022.08.26_MuLan.md)  to generate semantic tokens from the [W2V-BERT](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) model.
> Subsequently, it employs both music tokens and semantic tokens to generate acoustic features through [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md).
> [MusicGen (2023)](../../Models/Speech_LLM/2023.06.08_MusicGen.md) is a music language model designed to work with [EnCodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) discrete tokens.
> It accepts textual descriptions or melodic features as input conditions to generate tokens, which can be reconstructed to high-fidelity music.

此外有其他一些基于编解码器的语言模型为声音建模设计.
- [AudioGen (2022)](../../Models/Speech_LLM/2022.09.30_AudioGen.md) 训练了一个 [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 模型来获得音频 Token, 然后训练了一个语言模型来利用文本特征作为条件生成音频 Token.
- [MusicLM (2023)](../../Models/Speech_LLM/2023.01.26_MusicLM.md) 采取与 [AudioLM](../../Models/Speech_LLM/2022.09.07_AudioLM.md) 类似的训练策略, 但扩展了其范围, 包括音乐特征. 
  它采用层次序列到序列的建模方式来解决指导式音乐生成问题. 
  最初, 它利用 [Mulan](../../Models/_Basis/2022.08.26_MuLan.md) 中的音乐 Token 来生成语义 Token, 并利用 [W2V-BERT](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) 模型来生成音频特征. 
  随后, 它采用音乐 Token 和语义 Token 来生成声学特征, 并使用 [SoundStream](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md).
- [MusicGen (2023)](../../Models/Speech_LLM/2023.06.08_MusicGen.md) 是为 [EnCodec](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 离散 Token 设计的音乐语言模型. 
  它接受文本描述或旋律特征作为输入条件, 生成 Token, 这些 Token 可以用于高保真的音乐生成.

> Another branch of speech language modeling aims to utilize discrete units obtained by quantizing self-supervised speech representations.
> While these discrete units contain rich acoustic and linguistic information [37], they lack speaker and paralinguistic information [38].
> This research direction focuses on modeling the semantics of speech, with the optional use of encoders to learn about speaker characteristics and prosody.
> Pioneering work is speech-resynthesis [38], which utilizes these discrete units in conjunction with prosody and speaker encoders to encode speech into low-bitrate codes.
> These codes can then be resynthesized into a speech signal with a decoder to achieve low-bitrate transmission.
> Additionally, these discrete units can be regarded as “pseudo-text,” serving as a foundation for training textless speech language models.
> Notable examples include [GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md), [pGSLM (2021)](../../Models/Speech_LLM/2021.09.07_pGSLM.md), [dGSLM (2022)](../../Models/Speech_LLM/2022.03.30_dGSLM.md), and [TWIST (2023)](../../Models/Speech_LLM/2023.05.22_TWIST.md).
> By engaging in the pre-trained task of next-token prediction, these speech LMs perform spoken language modeling and can conduct the task of speech continuation.

语音语言建模的另一个分支旨在利用由自监督语音表示量化获得的离散单元.
这些离散单元包含丰富的声学和语言学信息, 它们缺少说话人和伴随语言的信息.
这一研究方向注重于建模语音的语义, 可选地使用编码器来学习说话人特征和韵律.
开创性的工作是语音重合成, 利用这些离散单元结合韵律和说话人编码器将语音编码器为低码率编码.
这些编码随后可以通过解码器被重新合成为语音信号, 实现低比特率传输.
此外, 这些离散单元可以被视为伪文本, 作为训练无文本语音语言模型的基础.
显著的例子包括 [GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md), [pGSLM (2021)](../../Models/Speech_LLM/2021.09.07_pGSLM.md), [dGSLM (2022)](../../Models/Speech_LLM/2022.03.30_dGSLM.md), 和 [TWIST (2023)](../../Models/Speech_LLM/2023.05.22_TWIST.md).
通过采用预训练任务的下一标记预测, 这些语音语言模型可以进行口语语言建模, 并能够进行语音续写任务.

> In the field of speech translation, recent advancements have been made possible through these discrete units. 
> [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation]() pre-trained a Unit mBART combined with a [Wav2Vec 2.0 (2020)](../../Models/Speech_Representaion/2020.06.20_Wav2Vec2.0.md) encoder to directly predict the translated discrete units.
> [UnitY (2022)](../../Models/_Basis/2022.12.15_UnitY.md) further incorporates text modality to enhance speech translation.
> The Seamless models ([SeamlessM4T](../../Models/_Basis/2023.08.22_SeamlessM4T.md), [Seamless](../../Models/_Basis/2023.12.08_Seamless.md)) integrate the [UnitY](../../Models/_Basis/2022.12.15_UnitY.md) framework to perform expressive and streaming speech-to-text and speech-to-speech translation.
> With the development of these powerful speech LMs, researchers have begun to explore the use of prompting on speech LMs for various speech processing tasks, including prompt tuning, in-context learning, and instruction tuning.

在语音翻译领域, 最近的进展已经成为可能, 例如 [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation]() 预训练了一个 Unit mBART 与 [Wav2Vec 2.0 (2020)](../../Models/Speech_Representaion/2020.06.20_Wav2Vec2.0.md) 编码器, 直接预测翻译后的离散单元.
[UnitY (2022)](../../Models/_Basis/2022.12.15_UnitY.md) 进一步整合了文本模态以增强语音翻译.
[SeamlessM4T](../../Models/_Basis/2023.08.22_SeamlessM4T.md), [Seamless](../../Models/_Basis/2023.12.08_Seamless.md) 整合 [UnitY](../../Models/_Basis/2022.12.15_UnitY.md) 框架用于实现表达性和流式语音转文本和语音转语音翻译.
随着这些强力语音语言模型, 研究人员开始探索在语音语言模型对于各种语音处理任务的提示使用, 包括
- 提示调参: [An Exploration of Prompt Tuning on Generative Spoken Language Model for Speech Processing Tasks](), [SpeechPrompt v2](../../Models/_Basis/2023.03.01_SpeechPrompt_v2.md) [SpeechGen](../../Models/_Basis/2023.06.03_SpeechGen.md)
- 上下文学习: [An Exploration of In-Context Learning for Speech Language Model]()
- 指令调参: [Towards General-Purpose Text-Instruction-Guided Voice Conversion](), [Dynamic-Superb](../../Evaluations/2023.09.18_Dynamic-SUPERB.md)

### 3.2.Comparison for Codec-Based Audio Language Models

> In [Tab.03](#Tab.03), we compare the inputs, outputs, and downstream tasks of different codec-based language models.
> We also summarize that the downstream tasks conducted by different codec-based language models: 
> - Speech Continuation (SC), 
> - Piano Continuation (PC), 
> - Audio Continuation (AC), 
> - Text-to-Speech (TTS), 
> - Music Generation (MG), 
> - Stereophonic Generation (SG), 
> - Speech to Speech Translation (S2ST), 
> - Automatic Speech Recognition (ASR), 
> - Spoken Language Understanding (SLU), 
> - Automated Audio Captioning (AAC), 
> - Speech to Text Translation (S2TT), 
> - Machine Translation (MT), 
> - Speech Enhancement (SE), 
> - Speech Removal (SR), 
> - Target Speaker Extraction (TSE), 
> - Speech Editing (SPED), 
> - Voice Conversion (VC), 
> - Singing Voice Synthesis (SVS), 
> - Text-to-Sound (TTSO), 
> - Text-to-Music (TTM), 
> - Audio Editing (AUED), 
> - Speech Dereverb (SD), 
> - Instructed TTS (ITTS).
> 
> Finally, we show the codec models adopted by different LMs.

<a id="Tab.03"></a>

![](../_Images/2024.02.20_Tab.03.png)

> Tab.03: Codec-Based Language Models Comparison. 
> `T` means Text, 
> `Aud` means Audio, 
> `P` means Phoneme,
> `M` means MIDI.

在[表 03](#Tab.03), 我们对比了不同基于编解码器的语言模型的输入, 输出和下游任务.
我们还总结了由不同基于编解码器的语言模型引导的下游任务:
语音续写, 钢琴续写, 音频续写, 文本转语音, 音乐生成, 立体声生成, 语音转语音, 自动语音识别, 口语理解, 自动音频字幕, 语音转文本, 机器翻译, 语音增强, 语音去除, 目标说话人提取, 语音编辑, 声音转换, 歌唱声音合成, 文本转声音, 文本转音乐, 音频编辑, 语音反混响, 指导式文本转语音.

最后我们展示了不同语言模型采用的编解码器模型.

## 4.Conclusions

> The paper fills the research blank to review the neural codec models and LMs built upon them.
> We hope the comprehensive review and comparisons can inspire future research works to boost the development of neural codec models and codec-based LMs.

本文填补了当前研究的空白, 对神经编解码器模型和基于其上的语言模型进行了回顾.
我们希望这一全面的回顾和比较能够启发未来的研究工作, 推动神经编解码器模型和基于编解码器的语言模型的发展.