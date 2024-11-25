# 3·Representations of Spoken Dialogue Models: 口语对话模型中的表示

<details>
<summary>展开原文</summary>

Representations play a critical role in spoken dialogue systems as they determine how the spoken dialogue system comprehends, processes, and generates speech signals.
Additionally, they serve as a bridge between speech and other modalities, thereby directly influencing the system’s performance, functionality, and range of applications.
Compared to text and visual representations, speech representations possess a unique complexity.
Text representations primarily rely on a well-defined symbolic system, conveying meaning through structured elements like vocabulary and syntax.
Visual representations, on the other hand, focus on capturing spatial relationships and visual features in images.
In contrast, speech signals contain both dynamic acoustic features (such as timbre, prosody and emotion) and rich semantic content, requiring representations that not only capture temporal variations but also preserve an understanding of the underlying meaning.

The unique nature of speech has led to the development of two types of representation models.
The representations obtained by these two modeling approaches are often classified as semantic tokens and acoustic tokens.
**One category (semantic) is prediction-based modeling**, these models are trained for representation learning by predicting future frames in an autoregressive manner ([VQ-APC [35]](../../Models/_Basis/2020.05.17_VQ-APC.md); [Shain et al. [187]](../../Models/_Full/Acquiring_Language_from_Speech_by_Learning_to_Remember_and_Predict.md)) or by using surrounding frames to predict masked frames ([Audio ALBERT [31]](../../Models/SpeechRepresentation/2020.05.18_Audio_ALBERT.md); [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md); [Mockingjay [133]](../../Models/SpeechRepresentation/2019.10.25_Mockingjay.md)).
This approach tends to prioritize capturing linguistic information within speech, making it particularly useful for recognition and understanding tasks.
**The other category (acoustic) focuses on speech compression and reconstruction** ([WavTokenizer [90]](../../Models/Speech_Neural_Codec/2024.08.29_WavTokenizer.md);  [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md); [DAC [113]](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md); [SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)).
These models quantify speech features (which are downsampled from raw waveforms by one encoder) into a series of discrete tokens, then use one decoder to upsample these discrete tokens into the speech, calculating the reconstruction loss against the original signal.
By this approach, we can get discrete acoustic tokens with impressive compression rates and high-fidelity acoustic information, making it more suitable for tasks such as speech synthesis and emotion analysis.

In the spoken dialogue systems, as illustrated in Fig.02, different spoken dialogue models employ various approaches for representation selection.
In the following part, we will enumerate the commonly used speech representations in spoken dialogue models from both the input and output perspectives.
At the end of this section, we will thoroughly discuss the advantages and limitations of these representations, as well as the future trends in the development of representations used in spoken dialogue models.

</details>
<br>

表示 (Representations) 在口语对话系统中扮演者至关重要的角色, 因为它们决定了语音对话系统如何理解, 处理和生成语音信号.
此外, 它们作为语音和其他模态之间的桥梁, 因此直接影响系统的性能, 功能和应用范围.
和文本表示与视觉表示相比, 语音表示具有独特的复杂性.
- 文本表示主要依赖于定义明确的符号系统, 通过结构化元素 (如词汇和句法) 传达含义.
- 视觉表示侧重于捕捉图像中的空间关系和视觉特征.
- 语音信号包含动态声学特征 (如音色, 韵律和情感) 和丰富的语义内容, 要求相应的表示不仅捕获时序变化而且还能保留对底层含义的理解.

语音的独特性促使了两种表示模型的发展.
由这两种建模方法获得的表示通常被分类为语义 Token 和声学 Token.

1. 语义类是基于预测的建模. 这些模型被训练以进行表示学习, 如以自回归方式预测未来帧 ([VQ-APC [35]](../../Models/_Basis/2020.05.17_VQ-APC.md); [Shain et al. [187]](../../Models/_Full/Acquiring_Language_from_Speech_by_Learning_to_Remember_and_Predict.md)) 或使用周围帧来预测被掩膜的帧 ([Audio ALBERT [31]](../../Models/SpeechRepresentation/2020.05.18_Audio_ALBERT.md); [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md); [Mockingjay [133]](../../Models/SpeechRepresentation/2019.10.25_Mockingjay.md)).
   这种方法倾向于优先捕捉语音中的语言信息, 特别适用于识别和理解任务.
2. 声学类专注于语音压缩和重建 ([WavTokenizer [90]](../../Models/Speech_Neural_Codec/2024.08.29_WavTokenizer.md);  [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md); [DAC [113]](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md); [SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)).
   这些模型将语音特征量化为一系列离散 Token (通过编码器对原始波形进行下采样), 然后使用解码器来上采样这些离散 Token 为语音, 计算和原始信号之间的重构损失.
   通过这种方法, 我们可以获得具有惊人压缩率和高保真度声学信息的离散声学 Token, 更适合例如语音合成和情感分析等任务.

![](Images/Fig.02.png)

在口语对话系统中, 如图 2 所示, 不同口语对话模型对于表示选择采用不同的方法.
在接下来的部分, 我们将枚举口语对话模型中常用的语音表示, 既包括输入端, 也包括输出端.
在最后, 我们将详细讨论这些表示的优势和局限性, 以及在口语对话模型中使用的语音表示的未来趋势.

## 3.1·Speech Representations at the Inputs: 输入端的语音表示

### Semantic: 语义

<details>
<summary>原文</summary>

To enhance language models' ability to understand speech representations and align multimodal data at input, using pretrained models such as [Wav2Vec [184]](../../Models/SpeechRepresentation/2019.04.11_Wav2Vec.md), [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md), [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md), and [WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) to extract high-level semantic features from speech has become a core strategy for many spoken dialogue systems.

</details>
<br>

为了增强语言模型对理解语音表示的能力和在输入时对齐多模态数据, 使用预训练模型, 如 [Wav2Vec [184]](../../Models/SpeechRepresentation/2019.04.11_Wav2Vec.md), [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md), [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md), [WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) 来从语音中提取高级语义特征已经成为许多口语对话系统的核心策略.

#### Wav2Vec 系列

<details>
<summary>原文</summary>

[Wav2Vec [184]](../../Models/SpeechRepresentation/2019.04.11_Wav2Vec.md) is a foundational work in the field of speech representation learning, pioneering the extraction of self-supervised speech representations from unlabeled speech data.
This approach has driven technological advancements in tasks such as speech recognition, speaker identification, and other speech processing applications.
Wav2Vec employs a multi-layer, one-dimensional convolutional neural network directly on raw speech waveforms to progressively extract temporal speech features.
Training is accomplished through contrastive learning: the model selects a "correct" target (from the current speech frame) alongside several "incorrect" targets (negative samples).
By learning to distinguish positive samples from negatives, the model effectively learns to represent speech features in latent space.
As an improved version of Wav2Vec, [Wav2Vec 2.0 [10]](../../Models/SpeechRepresentation/2020.06.20_Wav2Vec2.0.md) introduces the Transformer architecture and masked modeling.
Wav2Vec 2.0 quantizes the latent speech representations extracted by the CNN and then uses a Transformer to model semantic information, similar to [BERT [45]](../../Models/TextLM/2018.10.11_BERT.md).
It also employs a contrastive learning objective, requiring the model to distinguish the correct quantized representations from multiple candidate representations.
[ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md) aims to incorporate emotional expression in conversational interactions, choosing Wav2Vec 2.0 for its proven capability to encode rich prosodic information, beneficial for speech emotion recognition ([Li et al. (Survey) [123]](../2022.10.05__Survey__Exploration_of_A_Self-Supervised_Speech_Model__A_Study_on_Emotional_Corpora/Main.md)).
Specifically, ParalinGPT uses Wav2Vec 2.0’s intermediate layer (the 12th layer) for frame-by-frame feature extraction, as this layer has shown optimal results in linear probing tasks for emotion analysis.
Additionally, ParalinGPT applies mean pooling and a linear feature projector to extract utterance embeddings.

</details>
<br>

[Wav2Vec [184]](../../Models/SpeechRepresentation/2019.04.11_Wav2Vec.md) 是语音表示学习领域的基础工作, 开创了从无标注语音数据中提取自监督语音表示.
这一方法推动了语音识别, 说话人识别和其他语音处理应用等任务的技术进步.
Wav2Vec 采用多层一维卷积神经网络直接在原始语音波形上进行逐步提取时序语音特征.
训练通过对比学习完成: 模型选择一个 "正确" 目标 (当前语音帧), 以及多个 "错误" 目标 (负样本).
通过学习区分正样本和负样本, 模型有效地在潜在空间中学习如何表示语音特征.

作为 Wav2Vec 的改进版本, [Wav2Vec 2.0 [10]](../../Models/SpeechRepresentation/2020.06.20_Wav2Vec2.0.md) 引入了 Transformer 架构和掩模建模.
Wav2Vec 2.0 量化了由 CNN 提取的潜在语音表示, 然后使用 Transformer 来建模语义信息, 类似于 [BERT [45]](../../Models/TextLM/2018.10.11_BERT.md).
它还采用对比学习目标, 要求模型从多个候选表示中区分出正确量化表示.

[ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md) 旨在将情感表达纳入对话互动, 选择具有编码丰富韵律信息方面的能力的 Wav2Vec 2.0, 有益于语音情感识别 ([Li et al. (Survey) [123]](../2022.10.05__Survey__Exploration_of_A_Self-Supervised_Speech_Model__A_Study_on_Emotional_Corpora/Main.md)).
具体来说, ParalinGPT 使用 Wav2Vec 2.0 中间层 (第 12 层) 进行逐帧特征提取, 因为这一层在情感分析线性探测任务中表现出了最佳结果.
此外, ParalinGPT 应用平均池化和线性特征映射器来提取发言嵌入.

#### XLS-R

<details>
<summary>原文</summary>

[XLS-R [9]](../../Models/SpeechRepresentation/2021.11.17_XLS-R.md) is a multilingual self-supervised speech representation model based on the Wav2Vec 2.0 architecture.
It extends and optimizes Wav2Vec 2.0 to support a broader range of languages, particularly low-resource languages.
During cross-lingual training, XLS-R employs multilingual data augmentation and denoising techniques, enhancing the model's adaptability when processing speech in various languages.
[USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md) uses XLS-R to obtain continuous intermediate representations at 50Hz, followed by a quantizer ([Seamless [14]](../../Models/_Basis/2023.12.08_Seamless.md)) with $K$=10000 to generate speech tokens.

</details>
<br>

[XLS-R [9]](../../Models/SpeechRepresentation/2021.11.17_XLS-R.md) 是一种基于 Wav2Vec 2.0 架构的多语言自监督语音表示模型.
它扩展并优化了 Wav2Vec 2.0 以支持更广泛的语言范围, 特别是低资源语言.
在跨语言训练中, XLS-R 采用多语言数据增强和降噪技术, 增强了模型在处理多种语言语音时的适应性.

[USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md) 使用 XLS-R 获得 50Hz 的连续中间表示, 然后使用 $K$=10000 的量化器 ([Seamless [14]](../../Models/_Basis/2023.12.08_Seamless.md)) 生成语音 Token.

#### HuBERT

<details>
<summary>原文</summary>

[HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) is a commonly used unsupervised learning model that performs K-Means clustering on the MFCC ([Zheng et al. [251]](../../Models/_Full/Comparison_of_Different_Implementations_of_MFCC.md)) features of speech to assign pseudo-labels to each frame.
It uses a convolutional encoder to generate a sequence of features at a 20ms frame rate from 16kHz sampled speech.
Finally, it randomly masks a portion of features from consecutive frames as input to the [Transformer [201]](../../Models/_Transformer/2017.06.12_Transformer.md).
HuBERT generates masked content based on surrounding context, enabling it to capture temporal and semantic information within speech and gain a deeper understanding of contextual details.
Spoken dialogue systems, such as [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md), [SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md), [PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md), [IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md), widely use HuBERT as their speech encoder.
E-Chat extracts the weighted sum of the 24 layers from the HuBERT to serve as speech embeddings, and incorporates an additional set of weighted parameters to extract emotion embeddings, thereby enabling emotion-aware capabilities.
SpeechGPT applies K-Means clustering to quantize the continuous features extracted from HuBERT, converting them into discrete unit sequences.
These discrete units are then integrated into the vocabulary of the large language model, enabling direct alignment between the text and speech modalities.
To more effectively integrate the language model with speech streams, PSLM adds an additional embedding layer after extracting features with HuBERT.
IntrinsicVoice uses HuBERT as the speech tokenizer, grouping speech tokens to reduce sequence length.
An embedding layer then converts these tokens into dense embeddings, which are subsequently mapped into the language model's embedding space using a trainable speech adapter.
[Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) extracts semantic features using HuBERT, employing a K-Means model with 500 units as the basic unit.
It trains a feedforward quantizer with data augmentation techniques ([Gat et al. [64]](../../Models/_Full/2022.09.30_Augmentation_Invariant_Discrete_Representation_for_Generative_Spoken_Language_Modeling.md)) to produce discrete speech tokens.
In the [Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md), HuBERT is used and the cluster number K is set to 500.
Notably, when continuous representations are clustered into discrete units, they primarily capture content information, which can be leveraged for modeling and understanding.
This process first extracts 25Hz frame-level continuous representations from the 11-th layer of the HuBERT model, assigns each frame to its closest cluster index, and then de-duplicates consecutive identical indices to shorten the sequence.

</details>
<br>

[HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) 是一种常用的无监督学习模型, 对语音的 MFCC ([Zheng et al. [251]](../../Models/_Full/Comparison_of_Different_Implementations_of_MFCC.md)) 特征进行 K-Means 聚类, 为每帧分配伪标签.
它使用卷积编码器从 16kHz 采样率的语音以 20ms 的帧率生成特征序列.
最后, 它随机掩膜连续帧的一部分特征作为 [Transformer [201]](../../Models/_Transformer/2017.06.12_Transformer.md) 的输入.
HuBERT 基于周围的上下文生成掩膜内容, 使其能够捕获语音中的时序和语义信息并获得上下文细节的更深刻理解.

口语对话系统, 例如 [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md), [SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md), [PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md), [IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) 广泛使用了 HuBERT 作为它们的语音编码器.
- E-Chat 提取 HuBERT 的 24 层权重之和作为语音嵌入, 并结合一组额外的权重参数来提取情感嵌入, 从而实现情感感知的能力.
- SpeechGPT 对 HuBERT 提取的连续特征进行 K-Means 聚类来量化, 将其转化为离散单元序列.
  这些离散单元随后整合到大语言模型的词表中, 实现文本和语音模态之间的直接对齐.
- 为了更有效地将集成语言模型和语音流, PSLM 在 HuBERT 提取特征后添加了额外的嵌入层.
- IntrinsicVoice 使用 HuBERT 作为语音分词器, 将语音 Token 进行分组以减少序列长度. 然后嵌入层将这些 Token 转换为密集的嵌入, 并通过可训练的语音适配器映射到语言模型的嵌入空间中.
- [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) 使用 HuBERT 提取语义特征, 使用500 个基本单元的 K-Means 模型作为基础单元.
  它采用数据增强技术 ([Gat et al. [64]](../../Models/_Full/2022.09.30_Augmentation_Invariant_Discrete_Representation_for_Generative_Spoken_Language_Modeling.md)) 训练了一个前馈量化器, 以生成离散语音 Token.
- [Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md) 使用 HuBERT, 聚类数 K=500.

值得注意的是, 当连续表示被聚类到离散单元时, 它们主要捕获内容信息, 这可以用于建模和理解.
这一过程首先从 HuBERT 的第 11 层提取 25Hz 帧级连续表示, 将每帧分配到最近的聚类索引, 然后消除连续相同的索引来缩短序列.

#### Whisper

<details>
<summary>原文</summary>

[Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md), based on the classic encoder-decoder architecture, has gained widespread attention in the field of speech recognition.
The encoder transforms input speech into high-level feature representations, while the decoder generates the corresponding text output from these representations.
Pretrained on large-scale data across various speech environments with text as the target, Whisper demonstrates strong capabilities in extracting semantic information from speech.
[Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md), [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md) use Whisper’s encoder to convert speech into continuous representations, which are then combined with text representations and fed into the large language model.
[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md), [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md), and [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) follow a similar approach, connecting a speech adapter after the Whisper encoder.
Their shared objective is to map speech representations into the text embedding space of the large language model, enhancing the model's ability to understand speech by forcibly aligning them through vocabulary expansion.

</details>
<br>

[Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) 是基于经典的编码器-解码器架构的模型, 在语音识别领域获得了广泛关注.
编码器将输入语音转换为高级特征表示, 而解码器则从这些表示生成相应的文本输出.
在大规模的各种语音环境的数据上以文本为目标进行预训练, Whisper 展示了从语音提取语义信息的强大能力.
- [Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md), [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md) 使用 Whisper 的编码器将语音转换为连续表示, 然后与文本表示结合并输入到大型语言模型中.
- [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md), [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md), [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) 采用类似的方法, 在 Whisper 编码器之后连接了一个语音适配器.
  它们共同的目标是将语音表示映射到大语言模型的文本嵌入空间, 通过词表扩展强制对齐从而增强模型理解语音的能力.

#### WavLM

<details>
<summary>原文</summary>

[WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) is a pretrained model designed for comprehensive speech processing tasks, playing a critical role in advancing speech technology.
Specifically, WavLM employs a masked speech denoising and prediction framework, where some inputs consist of simulated noise or overlapping speech with masked sections.
The goal is to predict pseudo-labels of the original speech in the masked areas.
This approach enables the model to learn ASR-related information through masked speech prediction, while also gaining knowledge relevant to non-ASR tasks through speech denoising modeling.
The masking and prediction pipeline for speech frames in WavLM is similar to that of HuBERT.
However, WavLM introduces an additional gated relative position bias to enhance the model's sensitivity to temporal information in speech.
[SpeechVerse [41]](../../Models/SpeechLM/2024.05.14_SpeechVerse.md) leverages the pretrained WavLM Large as its backbone speech encoder, encoding all intermediate layer features from WavLM to capture various forms of semantics and achieve better generalization performance.
To address the significant length disparity between speech features and text tokens, SpeechVerse applies a learnable convolutional module for downsampling the speech features.

</details>
<br>

[WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) 是一种为全面语音处理任务设计的预训练模型, 在推动语音技术发展中发挥了关键作用.
具体来说, WavLM 采用了掩膜语音降噪和预测框架, 其中一些输入包含模拟噪声或带有掩码部分的叠加语音.
目标是预测掩膜区域中的原始语音的伪标签.
这种方法使得模型能够通过掩膜语音预测学习到与 ASR 相关的信息, 同时也通过语音降噪建模获得其他和非 ASR 任务相关的知识.
WavLM 中语音帧的掩码和预测流程和 HuBERT 类似.
然而, WavLM 引入额外的门控相对位置偏置来增强模型对语音中时序信息的敏感性.
- [SpeechVerse [41]](../../Models/SpeechLM/2024.05.14_SpeechVerse.md) 利用预训练的 WavLM Large 作为其语音编码器的骨干, 将 WavLM 的所有中间层特征编码到语义表示中, 实现更好的泛化性能.
  为了处理语音特征和文本 Token 之间的显著的长度差异, SpeechVerse 采用了一个可学习的卷积模块来对语音特征进行下采样.

#### $S^3$ Tokenizer

<details>
<summary>原文</summary>

[CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md) proposes using a supervised automatic speech recognition module to generate a supervised semantic speech($S^3$) tokenizer.
Unlike a standard ASR model, the $S^3$ tokenizer splits the encoder into two parts and introduces a vector quantization layer in between.
The first encoder converts the mel spectrogram into context-aware representations, while the second encoder transforms discrete speech units into continuous hidden states.
Finally, a Transformer-based ASR decoder predicts the posterior probabilities of text labels.
Through supervision in multilingual ASR tasks, the $S^3$ tokenizer can convert speech into semantically consistent tokens that facilitate both speech understanding and generation.
[OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md) uses the $S^3$ tokenizer to extract discrete speech tokens, which are then directly fed into a text-speech pre-trained Transformer.

</details>
<br>

[CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md) 提出使用一个监督的自动语音识别模块生成监督语义语音分词器 (Supervised Semantic Speech ($S^3$) Tokenizer).
和标准的 ASR 模型不同, $S^3$ 分词器将编码器分为两个部分, 并在这两个部分之间引入向量量化层.
第一个编码器将梅尔频谱转化为上下文感知的表示, 而第二个编码器将离散语音单元转换为连续隐藏状态.
最后, 基于 Transformer 的 ASR 解码器预测文本标签的后验概率.
通过在多语言 ASR 任务上的监督学习, $S^3$ 分词器能够将语音转换为语义上一致的 Token, 这有助于促进语音理解和生成.
- [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md) 使用 $S^3$ 分词器提取离散语音 Token, 并直接将其输入到文本语音预训练的 Transformer 中.

#### SPIRAL

<details>
<summary>原文</summary>

[SPIRAL [85]](../../Models/SpeechRepresentation/2022.01.25_SPIRAL.md) aims to learn representations from speech data that are robust to noise and perturbations.
It uses a teacher-student network, where various perturbations—such as noise addition, gain adjustment, and time-frequency warping—are applied to the speech input of the student model.
The teacher model then guides the student model to produce consistent representations despite these perturbations.
[EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md) utilizes the SPIRAL’s architecture as a speech encoder to process speech, and employs the [finite scalar quantization [149]](../../Modules/VQ/FSQ.md) to discretize these features.
This process aligns speech with the text vocabulary, allowing for a more natural integration into the LLM.

</details>
<br>

[SPIRAL [85]](../../Models/SpeechRepresentation/2022.01.25_SPIRAL.md) 旨在从语音数据中学习对噪声和扰动健壮的表示.
它使用了教师-学生网络, 其中对学生模型的语音输入应用了各种扰动, 例如加噪, 增益调整, 以及时频扭曲.
教师模型随后引导学生模型在这些扰动下生成一致的表示.
- [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md) 利用 SPIRAL 的架构作为语音编码器处理语音, 并采用[有限标量量化 (Finite Scalar Quantization, FSQ)[149]](../../Modules/VQ/FSQ.md) 来离散这些特征.
这一过程将语音和文本词表对齐, 从而更自然地集成到 LLM 中.

#### Others

<details>
<summary>原文</summary>

Some spoken dialogue systems do not use pre-trained representation models; instead, they process input features by stacking fundamental modules.
[VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) initially decomposes the speech signal using mel filter banks, mimicking the nonlinear perception of sound in humans.
It then processes the input features with a 4-layer CNN downsampling module followed by a 24-layer Transformer.
To align with the subsequent language model, VITA employs a simple 2-layer MLP as an adapter.
[Freeze-Omni [213]](../../Models/SpeechLM/2024.11.01_Freeze-Omni.md) utilizes a chunk-wise streaming speech encoder to transform input speech features into high-dimensional representations.
An adapter module then maps these high-dimensional representations into the embedding space of the main LLM, ensuring a quick, low-latency response to the input speech.
The speech encoder module consists of several downsampling convolutional layers and Transformer blocks, while the adapter includes only a few downsampling convolutional layers.
Downsampling layers are used to reduce the frame rate of speech features, increase the LLM's processing speed during the prefill phase, and minimize latency.

</details>
<br>

一些口语对话系统并没有使用预训练的表示模型, 它们通过堆叠基础模块来处理输入特征.
- [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) 首先使用梅尔滤波器组分解语音信号, 模拟人类对声音的非线性感知.
  然后, 使用四层卷积神经网络下采样模块和 24 层 Transformer 来处理输入特征.
  为了和后续的语言模型对齐, VITA 采用了一个简单的两层 MLP 作为适配器.
- [Freeze-Omni [213]](../../Models/SpeechLM/2024.11.01_Freeze-Omni.md) 使用分块流式语音编码器将输入语音特征转换为高维表示.
  然后, 适配器模块将这些高维表示映射到主 LLM 的嵌入空间, 确保对输入语音的快速, 低延迟响应.
  语音编码器模块由数个下采样卷积层和 Transformer 块组成, 而适配器只包含几个下采样卷积层.
  下采样层用于减少语音特征的帧率, 在预填充阶段提高 LLM 的处理速度, 并减少延迟.

### Acoustic: 声学

<details>
<summary>原文</summary>

Considering that semantic features are insufficient to capture the emotion, timbre, and style of speech, some representation models, such as [Emotion2Vec [143]](../../Models/Speech_Representaion/2023.12.23_Emotion2Vec.md), attempt to extract acoustic information through self-supervised training.
Others focus on reconstruction objectives to ensure high-fidelity speech, including models like [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)).

</details>
<br>

考虑到语义特征不足以捕获语音的情感, 音色和风格, 一些表示模型如 [Emotion2Vec [143]](../../Models/Speech_Representaion/2023.12.23_Emotion2Vec.md) 尝试通过自监督训练提取声学信息. 其他模型则着重于重建目标以确保高保真语音, 包括 [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) 等.

#### EnCodec

<details>
<summary>原文</summary>

[EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) is a straightforward, streaming, convolution-based encoder-decoder architecture.
Raw speech is downsampled through a series of convolutional layers, mapping it to latent feature representations.
Residual vector quantization ([SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)) then discretizes the encoder’s continuous latent features.
The quantization objective is to map continuous features to a predefined set of discrete tokens (known as a "codebook") for subsequent compression and transmission.
The decoder restores the discrete features to a waveform close to the original speech through a series of de-convolution layers.
[LauraGPT [50]](../../Models/SpeechLM/2023.10.07_LauraGPT.md) employs an enhanced version of EnCodec as its speech encoder with specific modifications: (1) adding a reconstruction loss in the magnitude spectral domain to improve mid-to-high frequency signal quality; (2) stacking five strided convolutional blocks with strides of (8, 5, 4, 2, 2) to address the challenges of long sequence lengths, resulting in a token rate of 25Hz per token group; and (3) using 32 quantizers with structured dropout in the Residual Vector Quantization (RVQ) module, each with a vocabulary size of 1024.
This revision increases speech quality by incorporating more quantizers while preserving most information in the shallow quantizers.
LauraGPT ultimately selects the output from the first quantizer layer as the speech token, balancing performance with sequence length efficiency.
The remaining quantizers are used only during the training of the encoder-decoder model.

</details>
<br>

[EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 是一种直接, 流式, 卷积式的编码器-解码器架构.
原始语音通过一系列卷积层进行下采样, 将其映射到潜在特征表示.
残差向量量化 ([SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)) 随后将编码器的连续潜在特征离散化.
量化目标是将来连续特征映射到离散 Token 的预定义集合 (称为 "码本") 进行后续压缩和传输.
解码器通过连续的反卷积层将离散特征恢复为接近原始语音的波形.
- [LauraGPT [50]](../../Models/SpeechLM/2023.10.07_LauraGPT.md) 采用了 EnCodec 的增强版本作为其语音编码器, 进行了特定修改:
  (1) 在幅度谱域中添加重构损失以提高中到高频信号质量;
  (2) 堆叠五个步长为 (8, 5, 4, 2, 2) 的卷积块以处理长序列长度的挑战, 实现每 Token 组的码率为 25Hz;
  (3) 在 Residual Vector Quantization (RVQ) 模块中使用 32 个量化器, 其词汇大小为 1024, 并采用结构性 Dropout.
  这种改进通过引入更多量化器来提高语音质量, 同时保留了浅层量化器中的大部分信息.
  LauraGPT 最终从第一个量化器层的输出中选择语音 Token, 在性能与序列长度效率之间取得平衡.
  剩余的量化器仅在编码器-解码器模型训练时使用.

#### SpeechTokenizer

<details>
<summary>原文</summary>

[SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) unifies semantic and acoustic tokens, hierarchically decomposing different aspects of speech information across various RVQ layers.
It is built on the framework of RVQ-GANs, following the same pattern as [SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) and [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md).
Notably, SpeechTokenizer has substituted the two-layer LSTM, originally following the convolution blocks in the EnCodec encoder, with a two-layer BiLSTM to augment the semantic modeling ability.
SpeechTokenizer uses HuBERT as a semantic teacher, given HuBERT’s proven capacity to encode substantial content information ([Mohamed et al. (Survey) [155]](../2022.05.21_Self-Supervised_Speech_Representation_Learning__A_Review/Main.md)).
During training, it introduces two types of distillation: continuous representation distillation and pseudo-label prediction.
For continuous representation distillation, SpeechTokenizer employs the 9th layer HuBERT representation or the average representation across all HuBERT layers as semantic teachers.
The training objective is to maximize the cosine similarity at the dimension level across all timesteps between the outputs of RVQ first layer and semantic teacher representations.
For pseudo-label prediction, SpeechTokenizer adopts HuBERT units as the target label.
In dialogue systems, SpeechGPT-Gen uses SpeechTokenizer RVQ-1 to process raw speech, primarily enhancing the large language model's ability to model the semantics of speech.

</details>
<br>

[SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 统一了语义和声学 Token, 在不同的 RVQ 层上分层次地分解语音信息地不同方面.
它建立在 RVQ-GANs 框架之上, 遵循与 [SoundStream [238]](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) 和 [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) 相同的模式.
值得注意的是, SpeechTokenizer 替换了 EnCodec 编码器中原本跟随卷积块地两层 LSTM 为两层 BiLSTM, 以增强语义建模能力.
鉴于 HuBERT 在编码大量内容信息方面的能力 ([Mohamed et al. (Survey) [155]](../2022.05.21_Self-Supervised_Speech_Representation_Learning__A_Review/Main.md)), SpeechTokenizer 使用 HuBERT 作为语义教师.
在训练时, 它引入了两种蒸馏方法: 连续表示蒸馏和伪标签预测.
- 连续表示蒸馏: SpeechTokenizer 采用 HuBERT 第 9 层表示或 HuBERT 所有层的平均表示作为语义教师.
  训练目标是在所有时间步上最大化 RVQ 第一层输出与语义教师表示之间的维度级余弦相似度.
- 伪标签预测: SpeechTokenizer 采用 HuBERT 单元作为目标标签.

- 在对话系统中, SpeechGPT-Gen 使用 SpeechTokenizer RVQ-1 处理原始语音, 主要增强了大型语言模型对语音语义建模能力.

#### Mimi

Taking inspiration from previous work on SpeechTokenizer, Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) uses distillation to transfer non-causal, high-level semantic information into the tokens produced by a causal model, allowing for streaming encoding and decoding of semantic-acoustic tokens.
To improve the ability of Mimi to encode speech into compact representations while reconstructing high-quality speech, Transformer modules are added in the encoder and decoder.
Mimi uses WavLM to distill RVQ-1, enriching it with semantic information.
Notably, performing distillation significantly enhances the speech discrimination capability of the first quantizer; however, it can also negatively impact speech quality.
Mimi hypothesizes that this is due to distilling semantic information into the first level of a single RVQ: As higher-order quantizers operate on the residual of the first one, the latter needs to trade speech quality for phonetic discriminability.
Mimi addresses this issue by introducing a split-RVQ approach.
Instead of using a single 8-level RVQ, it extracts semantic information into a simple VQ and applies a parallel 7-level RVQ, combining their outputs at the end.
This removes the constraint that acoustic information must be preserved in the residuals of the semantic quantizer.
After careful design, Mimi serves as the speech encoder in [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md), this approach enhances the model's ability to capture both semantic and acoustic details.

#### Emotion2Vec

[Emotion2Vec [143]](../../Models/Speech_Representaion/2023.12.23_Emotion2Vec.md) is a versatile speech emotion representation model designed to extract emotional features from speech.
During the pre-training phase, Emotion2Vec conducts online distillation with a teacher network and a student network.
When a specific downstream task is performed, Emotion2Vec is frozen and a lightweight downstream model is trained.
Emotion2Vec introduces an utterance-level loss to control global emotion and employs a frame-level loss to build a frame-wise pretext task, enabling it to learn contextual emotions.
[Spoken-LLM [127]](../../Models/SpeechLM/2024.02.20_Spoken-LLM.md) uses features extracted by Emotion2Vec as input for the large language model, aiming to enable the model to understand and respond to emotions.

## 3.2·Speech Representations at the Outputs: 输出端的语音表示

### Semantic: 语义

At the output stage, Most spoken dialogue systems choose to autoregressively model semantic tokens, such as $S^3$ tokens ([CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)) and [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) units.
It is worth noting that these semantic tokens lack acoustic conditioning and therefore require a vocoder ([HiFi-GAN [108]](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md); [Polyak et al. [166]](../../Models/_Full/2021.04.01_Speech_Resynthesis_from_Discrete_Disentangled_Self-Supervised_Representations.md)) or decoder, which futher takes semantic discrete units as input to synthesize speech consistent with the speakers encountered during training.

#### $S^3$ Tokenizer

[OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md); uses the LLM to autoregressively predict $S^3$ tokens at the speech output stage.
When converting discrete tokens back into speech, it adopts the same optimal transport conditional flow matching model (OT-CFM) as used in [CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md).
OT-CFM transforms the speech token sequence into Mel spectrogram, which is then used to generate the final speech with the [HiFi-GAN [108]](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) vocoder.

#### HuBERT

Speech tokens extracted by the pre-trained [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) are widely used as generation targets for large language models in the spoken dialogue systems.
[SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md) and [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) use [LLaMA [200]](../../Models/TextLM/2023.02.27_LLaMA.md) to autoregressively predict a sequence of units and are trained with a HuBERT unit-based [HiFi-GAN [108]](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) to decode the speech signal from discrete representations.
[PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md) introduces an additional speech projection layer after the Transformer layers to process the hidden states, obtaining semantic tokens via the softmax layler.
The speech decoder in [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) operates in a non-autoregressive manner, taking the output hidden states of the large language model as input to generate a discrete HuBERT unit sequence corresponding to the speech response.
The discrete units can be converted into waveform with an additional unit-based vocoder ([Polyak et al. [166]](../../Models/_Full/2021.04.01_Speech_Resynthesis_from_Discrete_Disentangled_Self-Supervised_Representations.md)).
[IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) introduces Group-Former to enhance the large language model’s capability in sequence modeling.
When the large language model predicts the $<speech>$ token, the global embedding is passed through a projection layer and delivered, along with a set of learnable queries, to the group model, which then predicts units.
IntrinsicVoice uses [HiFi-GAN [108]](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md), a non-autoregressive neural vocoder that efficiently generates high-fidelity waveforms, for speech detokenization to reduce overall latency.
[Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md) also uses a [HiFi-GAN [108]](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md)-based model to convert discrete units back into waveforms, utilizing model checkpoints from the [textlesslib [102]](../../Models/Toolkits/2022.02.15_textless-lib.md) library.

#### Others

[USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md) does not generate speech directly from input speech; instead, it first transcribes the speech, generates the response text, and then produces corresponding speech token in an end-to-end pipeline.
By inserting text-related tasks between speech input and output, the model benefits from both pre-trained LLMs and [chain-of-thought [218]](../../Models/_Basis/CoT.md) reasoning in the intermediate modality.
Since each stage in the pipeline processes all input and output tokens generated by the previous stage.
USDM is more robust to transcription errors and better able to produce contextually relevant spoken responses compared to a cascaded approach with separate modules.
USDM uses the [Voicebox [117]](../../Models/SpeechLM/2023.06.23_VoiceBox.md) architecture to train a unit-to-speech model for reconstructing speech from units.
[EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md) generates a response in the form of speech units when given an image or speech input, which is then converted into an output waveform using the U2S detokenizer.
The U2S detokenizer follows the VAE architecture: it uses a speech unit encoder to convert the predicted speech units into continuous embeddings, combines these with style embeddings predicted by the large language model to determine duration, and finally reconstructs the speech waveform through the decoder.

### Acoustic: 声学

Many spoken dialogue systems choose to directly generate tokens from acoustic representation models, such as [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), and Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)).
These acoustic tokens are then upsampled into the raw waveform through the frozen codec decoder directly.

#### EnCodec

[LauraGPT [50]](../../Models/SpeechLM/2023.10.07_LauraGPT.md) uses [Qwen-1.8B [11]](../../Models/TextLM/2023.09.28_Qwen.md) to predict speech tokens.
When synthesizing speech, it conditions the predictor not only on the speech tokens predicted by the LLM but also on text and speech inputs.
Such text and speech conditionings allow the model to generate high-quality speech signals by leveraging the diverse information in prompt and noisy speeches, which is lacked in the discrete tokens (output from the first quantizer of the Encodec).
The predicted speech tokens and conditioning inputs are delivered together to the codec vocoder.
An encoder-only Transformer models these inputs into dense embeddings, which are then reconstructed into speech by the codec decoder.

#### SNAC

[SNAC [193]](../../Models/Speech_Neural_Codec/2024.10.18_SNAC.md) encodes speech into hierarchical tokens, similar to [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) and [DAC [113]](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md), by introducing quantization at different time resolutions to form a multi-scale discrete representation of speech.
In this approach, shallow RVQ layers have a lower sampling frequency, covering a broader time span, while deeper RVQ layers sample at higher frequencies.
SNAC introduces modest enhancements over RVQ-GAN by incorporating residual noise blocks, deep convolutions, and local window attention.
The Mini-Omni ([Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md)) series continues the parallel generation method introduced by [MusicGen [40]](../../Models/SpeechLM/2023.06.08_MusicGen.md), utilizing [SNAC [193]](../../Models/Speech_Neural_Codec/2024.10.18_SNAC.md) as the speech encoder, which comprises seven complementary token layers.
In a single step, it generates eight tokens, including text, while maintaining a one-step delay between layers.
Furthermore, Mini-Omni and Mini-Omni 2 incorporates a batch approach that involves two samples: one requiring both text and speech responses and the other necessitating a text-only response.
By discarding the text token from the first sample and embedding the output from the second sample into the first, it effectively transfer the model’s text-based capabilities to speech tasks, significantly enhancing reasoning abilities with minimal resource overhead.

#### SpeechTokenizer

On the output side, SpeechGPT-Gen synthesizes speech tokens using [flow matching [131]](../../Models/Diffusion/2022.10.06_Flow_Matching.md).
Flow matching effectively models the transformation from a simple prior distribution to complex data distributions, yielding promising results in speech generation.
[SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md) applies flow matching for perceptual modeling, generating speech tokens that align with those of [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md).
Specifically, given speech $S$, semantic representation $V_1$, perceptual representation $V_{2:8}$ and the complete information representation $V_{1:8} = V_1 + V_{2:8}$ extracted by SpeechTokenizer, perceptual modeling refers to predicting the complete representation $V_{1:8}$ given the prompt speech a and the semantic representation $V_1$.
SpeechGPT-Gen synthesizes response speech by concatenating the output of [SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md) with the prompt speech and using a flow matching model.

#### Mimi

Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) has eight codebooks at a frame rate of 12.5Hz, which requires 100 autoregressive steps to generate one second speech.
This results in high computational costs and incompatibility with streaming inference.
To address these issues, [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) proposes the RQ-Transformer, comprising a temporal Transformer and a deep Transformer.
The RQ-Transformer breaks down a flattened sequence of length $K \cdot S$ into $S$ timesteps for a large temporal Transformer which produces a context embedding used to condition a smaller depth Transformer over $K$ steps.
This allows scaling to longer sequences by increasing $S$ or to a higher depth by increasing $K$ than modeling the flattened sequence with a single model.

#### TiCodec

[TiCodec [177]](../../Models/Speech_Neural_Codec/2023.09.15_TiCodec.md) is a decoupled codec model which can separate the time-varying and time-invariant information in speech and quantize them separately.
Inspired by [VALL-E [209]](../../Models/SpeechLM/2023.01.05_VALL-E.md), [Freeze-Omni [213]](../../Models/SpeechLM/2024.11.01_Freeze-Omni.md) uses a token-based speech decoder which contains NAR prefill and AR generate stage to achieve speech output capabilities.
The speech decoder mainly consists of the NAR decoder, the AR decoder, and the frozen decoder of a codec model ([TiCodec [177]](../../Models/Speech_Neural_Codec/2023.09.15_TiCodec.md)).
Both the NAR decoder and AR decoder are built upon transformer blocks.
The NAR decoder is used to model the semantic features from the output of LLM, and then the AR decoder generates speech tokens based on the output of the NAR decoder.
Finally, the decoder of the codec model converts the speech tokens into a speech stream.

## 3.3·Discussions about Representation used in Spoken Dialogue Systems: 讨论

### 3.3.1·Semantic Representation vs Acoustic Representation: 语义表示与声学表示

Current dialogue systems typically choose different approaches for the understanding (input) and generation (output) sides based on task requirements.
For example, [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) uses semantic representations ([HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md)) consistently on both ends, while [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) uses semantic representations ([Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md)) on the input side and acoustic representations ([SNAC [193]](../../Models/Speech_Neural_Codec/2024.10.18_SNAC.md)) on the output side.
Each combination offers unique advantages and trade-offs, and a consensus on a unified speech representation approach has yet to be reached in practical applications.

We revisited the differences between semantic and acoustic representations, as shown in Table~\ref{comparison_of_rep}.
Benefiting from specific task objectives, models such as [Wav2Vec [184]](../../Models/SpeechRepresentation/2019.04.11_Wav2Vec.md), [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md), [WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md), and [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) focus on extracting semantic information embedded within the spoken content.
This inherent advantage allows speech to be directly mapped into the embedding space of large language models (LLMs), facilitating alignment with other modalities and fully leveraging the LLM’s strengths.
In contrast, acoustic representations extracted by models like [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) and [DAC [113]](../../Models/Speech_Neural_Codec/2023.06.11_Descript-Audio-Codec.md) are less conducive to LLM understanding, which is why [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) and Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) opt for semantic distillation.
In addition, semantic representations offer higher compression rates.
By configuring various downsampling parameters in convolutional layers, models like HuBERT and Whisper easily achieve frame rates of 25Hz to 50Hz.
[Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md), for instance, uses 25Hz HuBERT units, meaning that only 25 tokens are needed to represent one second of speech.
In contrast, acoustic features are designed with compression and reconstruction in mind, where the constraints of signal transmission make extreme compression and high-quality reconstruction challenging to achieve simultaneously.
Although Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) has achieved a frame rate of 12.5Hz, its use of 8 codebooks means that autoregressively predicting one second of speech requires 100 steps.
Finally, in certain scenarios, semantic representations hold distinct advantages.

However, we must acknowledge that purely semantic representations fall short in naturalness and expressiveness, especially in tasks involving emotional expression or complex speech dynamics, where acoustic representations provide more nuanced information.
For instance, [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) cannot extract prosodic and stylistic features as effectively as [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) or [Emotion2Vec [143]](../../Models/Speech_Representaion/2023.12.23_Emotion2Vec.md).
Notably, using acoustic representations allows for flexible handling of various data types—speech, audio, music, and sound—making dialogue systems more unified and versatile.
Moreover, when acoustic representations are used as the output of a language model, they can seamlessly connect to the codec decoder for speech synthesis.
In contrast, dialogue systems using semantic features often require separately trained vocoders ([Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md); [USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md)) or rely on additional text-to-speech toolkits ([LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md)).
This gap is crucial for dialogue systems, as the resulting latency directly impacts the user experience.

Given the unique advantages of semantic and acoustic features across different tasks, future research may shift toward integrating these features.
A valuable perspective is that models like [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) and Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) have already attempted to distill semantic representations from [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) or [WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md) into RVQ-1, ensuring a balanced representation of both semantic and acoustic information in the system.
With technological advancements, we look forward to more unified and refined modeling approaches.
A promising direction would be to design new training objectives for speech tokenizers, exploring both data-driven and objective-driven methods, thus avoiding the need for additional pre-trained models.
As spoken dialogue Systems are still evolving, exploring more robust hybrid representations is indeed valuable.

### 3.3.2·Continuous Representation vs Discrete Representation: 连续表示与离散表示

There is still no consensus on whether to use continuous or discrete representations in the spoken dialogue systems.
Considerations on the input side mainly depend on the type of representation model chosen by the system.
Some systems ([Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md)) use models like [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) or [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) to extract continuous speech representations, which requires adding a speech adapter and an additional training phase focused on modality alignment.
Another systems ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) use models like [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) or Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) to extract discrete speech representations, adding speech tokens directly to the LLM’s vocabulary, thereby shifting the training burden onto the LLM itself.
Despite the different approaches, the key is to enable large language models to effectively understand speech features.
For autoregressive models, using discrete inputs may appear more manageable; however, whether this truly outperforms continuous inputs in terms of performance remains to be explored.

Language models trained with next-token prediction objectives tend to favor discrete modalities.
Using discrete features on the output side naturally supports simple codec decoders ([Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [Freeze-Omni [213]](../../Models/SpeechLM/2024.11.01_Freeze-Omni.md)) for reconstructing high-fidelity speech, enhancing speech quality and acoustic control while enabling an end-to-end system.
In contrast, continuous features may require additional text-to-speech toolkits ([VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md)) or vocoders ([LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md)), resulting in a cascaded pipeline and making it difficult to preserve detailed acoustic information.
Another notable advantage of using discrete representations as output is the ability to quickly feed them into the input of the next dialogue round, as demonstrated in [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md);.
In the field of computer vision, a range of work ([Transfusion [256]](../../Models/CV/2024.08.20_Transfusion.md); [Show-o [221]](../../Models/_Basis/2024.08.22_Show-o.md)) has emerged that combines discrete and continuous representations, aiming to fully integrate these modes without information loss, and has already achieved success in certain areas.
These approaches may provide valuable insights for the next generation of spoken dialogue systems.

### 3.3.3·Single-Layer Quantizer vs Multi-Layer Quantizer: 单层量化器和多层量化器

As previously mentioned regarding compression rates, the number of quantizers must be carefully considered when using the speech codec.
Currently, dialogue systems commonly use multi-layer quantizers, such as those in [EnCodec [43]](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md), [SNAC [193]](../../Models/Speech_Neural_Codec/2024.10.18_SNAC.md) and Mimi ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)).
This inevitably introduces generation latency, as residual vector quantization requires each quantizer’s input to depend on the output of the previous quantizer.
[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) and [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md) adopt an approach similar to [MusicGen [40]](../../Models/SpeechLM/2023.06.08_MusicGen.md), introducing delayed steps to enable parallel generation across multiple quantizers.
[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) proposes splitting the RVQ, allowing the eight VQs to generate independently in parallel.
These strategies help mitigate latency issues to some extent but still fall short of the efficiency achieved with semantic representations.

Recently, research on single-layer quantizers has shown promising breakthroughs.
Models like [WavTokenizer [90]](../../Models/Speech_Neural_Codec/2024.08.29_WavTokenizer.md), [Single-Codec [119]](../../Models/Speech_Neural_Codec/2024.06.11_Single-Codec.md), and [BigCodec [224]](../../Models/Speech_Neural_Codec/2024.09.09_BigCodec.md) advocate using a single VQ to discretize speech, achieving competitive results in both reconstruction and generation tasks.
Notably, [WavTokenizer [90]](../../Models/Speech_Neural_Codec/2024.08.29_WavTokenizer.md) has already achieved an impressive compression rate of 40Hz.
Integrating a single-layer quantizer with dialogue systems is promising, as it allows for rapid extraction of speech features on the input side and significantly reduces the burden of autoregressive modeling.

### 3.3.4·With Text Guidance vs Without Text Guidance: 文本引导与无文本引导

In practice, researchers have found direct speech-to-speech generation challenging ([Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md)) due to complex mapping relationships, so intermediate texts are often generated to achieve higher generation quality.
Current end-to-end dialogue systems commonly adopt one of two strategies: one ([LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md)) generates the hidden states corresponding to the text response first, which are then post-processed to obtain speech tokens; the other ([Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) generates text and speech tokens in parallel.
These approaches leverage the text modeling capabilities of large language models, essentially guiding the synthesis of semantically consistent speech by first generating text.
However, this comes at the expense of response speed.

Although directly performing speech-to-speech generation presents challenges such as increased model complexity and inference difficulty, we believe it remains a promising direction for future research.
One approach is to retrain large spoken language models to adapt to specific speech representations.
However, this faces challenges related to data resources, as large-scale and high-quality conversational datasets remain scarce.
Additionally, this method cannot completely eliminate text prompts and requires multi-stage training, starting with text-speech pairs to allow the model to progressively acquire conversational capabilities.
Another approach could begin with speech codecs, as demonstrated by SpeechTokenizer and Mimi’s extensive work in semantic distillation.
We envision a novel speech codec that aligns text and speech during the encoding phase, thereby reducing the generation burden on large language models.
By aligning speech representations with the text representation space earlier in the process, the autoregressive modeling would no longer require text guidance, giving rise to an entirely new paradigm for conversational systems.
