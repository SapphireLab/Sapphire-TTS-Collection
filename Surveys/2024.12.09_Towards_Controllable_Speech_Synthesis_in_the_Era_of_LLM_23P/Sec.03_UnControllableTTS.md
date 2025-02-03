# 3·Uncontrollable TTS: 非可控文本转语音

#仅一些简单的总结

<table><tr><td width="50%">

The development of Uncontrollable Text-To-Speech (UC-TTS) systems represents a significant shift from traditional, linguistics-based synthesis to modern, data-driven deep learning techniques.
This shift highlights the integration of both local and global information to produce speech with human-like quality and naturalness.
This survey explores UC-TTS evolution, emphasizing the role of local and global information in enhancing speech fidelity and expressiveness.

In the context of UC-TTS, "uncontrollable" refers to the absence of explicit control mechanisms for speech features such as emotion, timbre, and speaker style.
Despite this lack of explicit control, the goal is to achieve natural, fluid speech while minimizing issues like mispronunciations and omissions.

</td><td>

**非可控文本转语音 (Uncontrollable Text-To-Speech, UC-TTS)** 系统的发展代表着从传统基于语言学的合成到现代数据驱动深度学习技术的重要转变.
这一转变强调了集成局部和全局信息以产生类似人类质量和自然度的语音.

本综述探讨了 UC-TTS 的演进, 强调了局部信息和全局信息在增强语音保真度和表现力的作用.

在 UC-TTS 的语境中, "非可控"一词指的是缺乏对语音特征 (如情感, 音色, 说话人风格) 显式的控制机制.
尽管缺乏显式的控制, 但目标是实现自然流畅的语音, 同时避免如发音错误和漏读等问题.

</td></tr></table>

## A·Early Approaches: Statistical Models: 早期方法: 统计模型

<table><tr><td width="50%">

Early Text-To-Speech (TTS) systems relied on statistical models such as Hidden Markov Models (HMMs) ([Yoshimura et al. (1999) [64]](../../Models/_Early/Simultaneous_Modeling_of_Spectrum_Pitch_&_Duration_in_HMM-Based_Speech_Synthesis.md); [Tokuda et al. (2000) [65]](../../Models/_Early/Speech_Parameter_Generation_Algorithms_for_HMM-Based_Speech_Synthesis.md)) and early neural network-based parametric methods ([Statistical Parametric Speech Synthesis Using DNNs [111]](../../Models/_Early/2013.05.26_Statistical_Parametric_Speech_Synthesis_Using_DNNs.md); [DBLSTM-RNN [112]](../../Models/_Early/DBLSTM-RNN.md)).
These models operated at the frame level, using acoustic models and vocoders for text-to-speech conversion.
Notable contributions from [Tokuda et al. [173]](../../Models/_Early/2013.04.09_Speech_Synthesis_Based_on_Hidden_Markov_Models.md) employed HMMs for statistical parametric synthesis, focusing on local features like phonemes, accents, and prosody to improve speech naturalness.

While robust, these statistical methods were limited by their reliance on pre-segmented data, leading to oversimplified assumptions about speech dynamics.
Local linguistic features were well-modeled, but the global phonetic context was often overlooked, resulting in speech that sounded monotone and lacked emotional depth, as noted by [Zen et al. [174]](../../Models/_Early/2009.01.14_Statistical_Parametric_Speech_Synthesis.md).

</td><td>

早期文本转语音系统依赖于
- 统计模型, 如隐马尔科夫模型 (Hidden Markov Models, HMMs)
  - [Yoshimura et al. (1999) [64]](../../Models/_Early/Simultaneous_Modeling_of_Spectrum_Pitch_&_Duration_in_HMM-Based_Speech_Synthesis.md);
  - [Tokuda et al. (2000) [65]](../../Models/_Early/Speech_Parameter_Generation_Algorithms_for_HMM-Based_Speech_Synthesis.md);
- 早期基于神经网络的参数方法
  - [Statistical Parametric Speech Synthesis Using DNNs [111]](../../Models/_Early/2013.05.26_Statistical_Parametric_Speech_Synthesis_Using_DNNs.md);
  - [DBLSTM-RNN [112]](../../Models/_Early/DBLSTM-RNN.md).

这些模型运行在帧级上, 使用声学模型和声码器进行文本到语音的转换.

值得提及的是, [Tokuda et al. [173]](../../Models/_Early/2013.04.09_Speech_Synthesis_Based_on_Hidden_Markov_Models.md) 使用 HMMs 进行统计参数化语音合成, 着重于局部特征, 如音素, 语调, 和语调, 以提高语音自然度.

尽管这类统计方法很健壮, 但受限于预分割的数据, 导致了对语音动态的过度简化假设.
局部语言特征被很好地建模, 但全局音素上下文往往被忽视, 导致产生单调且缺乏情感深度的语音, 如 [Zen et al. [174]](../../Models/_Early/2009.01.14_Statistical_Parametric_Speech_Synthesis.md) 所述.

</td></tr></table>

## B·Sequence-to-Sequence Models: 序列到序列模型

<table><tr><td width="50%">

The emergence of sequence-to-sequence models represents a significant breakthrough by removing the need for explicit linguistic features, thereby enabling the capture of the nuances and idiosyncrasies of human speech.
Models such as [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md) and [Tacotron 2 [175]](../../Models/Acoustic/2017.12.16_Tacotron2.md) utilize recurrent neural networks (RNNs) with attention mechanisms to effectively model the complex, nonlinear nature of speech sequences.
These innovations allow for precise tuning of speech parameters, enhancing prosody and rhythm by modeling entire utterances rather than isolated phonetic units.

Building on these advancements, [DeepVoice3 [176]](../../Models/Acoustic/2017.10.20_DeepVoice3.md) introduces a fully convolutional sequence-to-sequence architecture that significantly accelerates training speed compared to RNN-based models.
This approach achieves training times an order of magnitude faster, enabling scalability to handle large datasets.
Additionally, the use of a position-augmented attention mechanism in Deep Voice 3 enhances the naturalness of synthesized speech, achieving competitive mean opinion scores, especially when paired with advanced neural vocoders like WaveNet.
This development not only improves training efficiency but also enhances the scalability and naturalness of text-to-speech systems.

</td><td>

序列到序列模型的出现代表着一种显著突破, 移除了对显式语言特征的需要, 使得能够捕获人类语音的细微差别和癖好.
诸如 [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md) 和 [Tacotron 2 [175]](../../Models/Acoustic/2017.12.16_Tacotron2.md) 之类的模型利用带有注意机制的循环神经网络 (RNNs) 模型来有效地建模语音序列的复杂的非线性特性.
这些创新能够对语音参数进行精确调整, 通过建模整个句子而不是孤立的音素单元来增强语调和韵律.

基于这些进步, [DeepVoice3 [176]](../../Models/Acoustic/2017.10.20_DeepVoice3.md) 提出了一个完全卷积的序列到序列架构, 与基于 RNN 的模型相比显著加快了训练速度.
这种方法能够以 10 倍以上的速度训练, 这使得它能够处理大型数据集.
此外, 在 DeepVoice3 中引入的位置增强注意力机制进一步增强了合成语音的自然度, 可以获得竞争力的平均意见得分, 尤其是与 WaveNet 等先进的神经声码器配合时.
这一发展不仅提高了训练效率, 而且也增强了文本到语音系统的可扩展性和自然度.

</td></tr></table>

## C·Transformer-Based Models: 基于 Transformer 的模型

<table><tr><td width="50%">

Transformer-based architectures advanced the field by enabling computational parallelization and effectively capturing long-range dependencies.
Models like [Transformer TTS [177]](../../Models/Acoustic/2018.09.19_TransformerTTS.md) overcame RNN challenges, such as gradient vanishing, by using efficient training paradigms.
Self-attention mechanisms allowed simultaneous modeling of local phonetic details and global prosodic contexts, resulting in more sophisticated and human-like speech synthesis.

Although transformers improved contextual information incorporation, challenges remained in preserving local phonetic precision.
To address these, techniques such as relative position encodings and localized attention were integrated ([Transformer [124]](../../Models/_Transformer/2017.06.12_Transformer.md)).

</td><td>

基于 Transformer 的架构通过计算并行化和有效捕获长距离依赖, 推动了该领域的发展.
诸如 [Transformer TTS [177]](../../Models/Acoustic/2018.09.19_TransformerTTS.md) 之类的模型采用高效的训练范式来克服 RNN 的挑战, 如梯度消失.
自注意力机制允许同时建模局部音素细节和全局韵律上下文, 产生更复杂更类似人类的语音合成.

尽管 Transformer 提升了上下文信息的整合, 但仍然存在局部音素精度的问题.
为了解决这些问题, 诸如相对位置编码和局部注意力等技术被集成到了模型中 ([Transformer [124]](../../Models/_Transformer/2017.06.12_Transformer.md)).

</td></tr></table>

## D·Advanced Architectures: Integrating Flow and Diffusion Models: 集成 Flow 和 Diffusion 模型

<table><tr><td width="50%">

Recent advancements have shifted towards integrating global information within end-to-end architectures to enhance speech naturalness and coherence.
Flow-based models like [Glow-TTS [133]](../../Models/Acoustic/2020.05.22_Glow-TTS.md) and [Flow-TTS [132]](../../Models/Acoustic/2020.04.09_Flow-TTS.md) exemplify this by employing invertible transformations that maintain the balance between local precision and global coherence.
These architectures enable the synthesis of high-fidelity speech by modeling complex dependencies across the entire utterance, thus improving the overall fluidity and naturalness of the generated speech.

Moreover, the introduction of diffusion models in TTS, such as [WaveGrad 2 [178]](../../Models/Vocoder/2021.06.17_WaveGrad2.md), highlights the shift towards models that can iteratively refine speech output.
These models use score matching and diffusion processes to generate speech directly from phoneme sequences, effectively capturing both local nuances and overarching global patterns.
The iterative nature of these models allows for adjustments that enhance the quality of the synthesized audio, accommodating variations in speech without explicit control over specific attributes.

The integration of adversarial training and variational autoencoders (VAEs) further exemplifies the evolution towards incorporating global information.
Systems like [VITS [159]](../../Models/E2E/2021.06.11_VITS.md) leverage these techniques to enhance expressiveness and naturalness by learning complex mappings between text and speech.
This approach allows the model to manage variations in prosody and rhythm inherently derived from the textual input, aligning with the objectives of UC-TTS to produce diverse and natural speech outputs.

The evolution from HMMs to advanced architectures in UC-TTS exemplifies progress toward synthesizing speech that is both expressive and precise.
The interplay of local and global information is crucial for enhancing speech quality and customizability.
Future UC-TTS research aims to produce high-fidelity, customizable speech by harmonizing deep contextual insights with precise local adjustments, meeting diverse user needs and communication contexts.

</td><td>

近期的进展已经转向在端到端架构中集成全局信息以增强语音自然度和连贯性.

基于流的模型, 如 [Glow-TTS [133]](../../Models/Acoustic/2020.05.22_Glow-TTS.md) 和 [Flow-TTS [132]](../../Models/Acoustic/2020.04.09_Flow-TTS.md), 展示了这一点, 通过使用可逆变换来保持局部精度和全局连贯性的平衡.
这些架构通过对整个句子建模复杂依赖进行高保真语音合成, 这有助于提高合成语音的流畅性和自然度.

此外, 扩散模型的引入, 如 [WaveGrad 2 [178]](../../Models/Vocoder/2021.06.17_WaveGrad2.md), 也突出了转向迭代式精细语音输出的模型的趋势.
这些模型使用得分匹配和扩散过程来直接从音素序列生成语音, 有效捕获局部细微差别和全局模式.
这些模型的迭代性质使得一些调整能够增强合成音频的质量, 适应语音的变化而无需对特定属性进行显式控制.

对抗训练和变分自编码器的集成进一步证明了集成全局信息的进步.
如 [VITS [159]](../../Models/E2E/2021.06.11_VITS.md) 等系统利用这些技术通过学习复杂的文本-语音映射来增强表达性和自然度.
这一方法允许模型管理文本输入中隐含的韵律和节奏, 与 UC-TTS 的目标一致, 生成多样化和自然的语音输出.

从 HMM 到 UC-TTS 中的先进架构的演进, 证明了向生成具有丰富表达性和精确度的语音的进展.
局部和全局信息之间的交互对于增强语音质量和可定制性至关重要.
未来 UC-TTS 研究的目标是通过结合深度上下文见解和精细的局部调整, 实现高保真度、可定制的语音输出, 同时满足广泛的用户需求和通信环境.

</td></tr></table>