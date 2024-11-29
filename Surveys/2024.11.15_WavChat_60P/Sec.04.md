# 4·Training Paradigm of Spoken Dialogue Model: 口语对话模型的训练范式

<details>
<summary>展开原文</summary>

Existing text-based large language models have demonstrated strong contextual understanding and reasoning abilities in the field of natural language processing, such as [GPT-4 [1]](../../Models/TextLM/2023.03.15_GPT-4.md), [LLaMA 3.1 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md), and [Qwen-2 [228]](../../Models/TextLM/Qwen2.md).
Due to their training on large-scale corpora, these models achieve exceptional accuracy when handling complex contexts.
To further expand the capabilities of large language models, some research ([EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md); [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md)) has explored enabling them to understand other modalities, thereby building multimodal interaction abilities.
The spoken dialogue model, also known as the speech-text dialogue model, allows users to interact with LLMs naturally and straightforwardly through speech.
However, the transition from text intelligence to speech intelligence involves two inherent hurdles: one core issue is the insufficient amount of speech data compared to the massive datasets used for pre-training text-based large language models.
For instance, [LLaMA 3.1 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md) uses 800 billion training tokens, and [Qwen-2 [228]](../../Models/TextLM/Qwen2.md) is trained on over 7 trillion tokens, whereas pure speech pre-training data often amounts to hundreds of thousands or millions of hours.
For example, [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)'s pre-training speech data comprises 7 million hours, and the amount of labeled speech data is even smaller, making it difficult to support LLMs in achieving powerful speech intelligence comparable to text.
Another challenge is that speech information density is not as compact as text.
Text commonly uses [byte-pair encoding (BPE)[186]](../../Models/_Basis/2015.08.31_BPE.md) (~~[A New Algorithm for Data Compression [62]]~~) to compress it into a tight token space, whereas the speech modality includes not only semantic information but also acoustical information, which is less dense.
This undoubtedly increases the difficulty for LLMs to learn.
Understanding and generating the inherent knowledge of the speech modality more effectively is a significant challenge.

Consequently, existing spoken dialogue models aim to build upon text-based LLMs by incorporating the speech modality into these large language models.
[SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) support speech-in and speech-out capabilities for LLMs, forming the foundation of basic speech dialogue capabilities.
Some of the latest advanced approaches ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md);[SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md)) attempt to transition from traditional turn-based spoken dialogue systems to full-duplex systems, aiming to simulate the natural spontaneity of human conversation.
While these advancements are promising, achieving low latency and natural interaction in full-duplex systems remains a significant challenge.
Moreover, enhancing LLMs to effectively handle the speech modality—mastering both speech comprehension and generation—while maintaining robust natural language text processing capabilities, is hindered by the limited size of labeled speech datasets.
These datasets are far smaller compared to the vast amounts of pure text data available, which risks diminishing the models' original text processing capabilities.
Thus, building a truly end-to-end conversational model that meets real-world requirements necessitates careful consideration of model architecture, training paradigms, and training data.
Overall, we believe that several key aspects are crucial in the training paradigm of spoken dialogue models: aligning speech-text modalities to ensure consistent understanding, designing multi-stage training strategies for gradual adaptation, and optimizing training structures and inference paradigms for efficient performance.

</details>
<br>

现有的基于文本的大语言模型已经在自然语言处理领域展现出了强大的上下文理解和推理能力, 如 [GPT-4 [1]](../../Models/TextLM/2023.03.15_GPT-4.md), [LLaMA 3.1 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md) 和 [Qwen-2 [228]](../../Models/TextLM/Qwen2.md).
由于它们在大规模语料库上进行训练, 这些模型在处理复杂上下文时取得了卓越的准确性.
为了进一步扩展大语言模型的能力, 一些研究 ([EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md); [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md)) 探索了让大语言模型理解其他模态的可能性, 从而构建多模态交互能力.

口语对话模型, 也称为语音-文本对话模型, 允许用户通过语音以自然且直接的方式与 LLMs 进行交互.
然而, 从文本智能到语音只能的转变, 涉及到两个内在的障碍:
一个核心的问题是相比于预训练的基于文本的大语言模型所使用的海量数据集相比, 缺乏足够的语音数据.
- [LLaMA 3.1 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md) 使用 8000 亿个训练 Token, [Qwen-2 [228]](../../Models/TextLM/Qwen2.md) 在超过 7 万亿个 Token 上训练, 而纯语音预训练数据往往只占数十万或数百万小时.
- [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) 的预训练语音数据包含 7 百万小时, 而标注语音数据却很少, 使得它很难支持 LLMs 在语音智能方面取得与基于文本的模型相当的能力.

另一个挑战是语音信息密度不如文本紧凑.
文本通常使用[字节对编码 (BPE) [186]](../../Models/_Basis/2015.08.31_BPE.md) (~~[A New Algorithm for Data Compression [62]]~~) 将其压缩到紧密的 Token 空间中, 而语音模态不仅包含语义信息, 还包含声学信息, 其密度较低.
这无疑增加了 LLMs 学习的难度.

更有效地理解和生成语音模态的内在知识是一个重大挑战.

因此, 现有的口语对话模型旨在通过将语音模态引入大语言模型中, 从而在这些基于文本的大语言模型基础上进行构建.
- [SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) 支持 LLMs 的语音输入和输出功能, 形成了基本的语音对话能力的基础.
- 一些最新的先进方法 ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md);[SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md)) 试图从传统的基于轮次的口语对话系统转变为全双工系统, 旨在模拟人类对话的自然随意性.

尽管这些进展前景广阔, 但在全双工系统中实现低延迟和自然交互的能力仍然是一个重要挑战.
此外, 增强 LLMs 以有效地处理语音模态, 即掌握语音理解和生成, 同时保持稳健的自然语言文本处理能力, 受到标注语音数据集规模有限的阻碍.
与大量可用的纯文本数据相比, 这些数据集的规模要小得多, 这可能会削弱模型原有的文本处理能力.

因此, 构建一个真正满足现实需求的端到端对话模型, 需要在模型架构, 训练范式, 训练数据方面进行仔细的考虑.

总的来说, 我们认为口语对话模型的训练范式中有几个关键方面至关重要:
- 使语音-文本模态一致, 以确保一致理解;
- 设计多阶段训练策略, 逐步适应;
- 优化训练结构和推理范式, 以实现高效性能.

## 4.1·Architecture Paradigm about Modal Alignment of Speech and Text: 语音和文本模态对齐的架构范式

<details>
<summary>展开原文</summary>

To enable large language models (LLMs) to handle both speech input and output, a significant amount of prior work ([AudioPaLM [179]](../../Models/SpeechLM/2023.06.22_AudioPaLM.md); [LLaMA3 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) has focused on adapting text-based foundation models into robust spoken dialogue models.
Based on different architectural paradigms, these approaches can be broadly categorized into five types, as shown in Figure.05.

</details>
<br>

![](Images/Fig.05.png)

为了使得大语言模型能够处理语音输入和输出, 大量先前工作 ([AudioPaLM [179]](../../Models/SpeechLM/2023.06.22_AudioPaLM.md); [LLaMA3 [52]](../../Models/TextLM/2024.07.31_LLaMA3.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) 都集中在将基于文本的基础模型转化为健壮的口语对话模型.
基于不同的架构范式, 这些方法可以大致分为五类, 如图 05 所示.

### Text-Output Only Method: 仅输出文本的方法

<details>
<summary>展开原文</summary>

These systems ([Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md); [Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md); [LTU-AS [67]](../../Models/SpeechLM/2023.09.25_LTU-AS.md); [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md); [SALMONN [198]](../../Models/SpeechLM/2023.10.20_SALMONN.md); [WavLLM [80]](../../Models/SpeechLM/WavLLM.md); [SpeechVerse [41]](../../Models/SpeechLM/2024.05.14_SpeechVerse.md); [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md)) maintain the text-based LLM’s foundational structure unchanged, **using an audio encoder and adaptor to map speech input into the LLM's pre-trained text latent space directly**.
This method of direct embedding alignment, combined with a multi-task training strategy, equips the LLM with the ability to 'listen,' thus enabling it to understand and process speech modality inputs effectively and perform exceptionally well in various audio understanding tasks.
Nevertheless, the output remains text-based, which necessitates the use of an external text-to-speech (TTS) system ([XTTS [21]](../../Models/SpeechLM/2024.06.07_XTTS.md); [CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)) to generate speech output.
[LTU-AS [67]](../../Models/SpeechLM/2023.09.25_LTU-AS.md) uses [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) and the Time and Layer-Wise Transformer (TLTR) as its audio encoder, allowing it to recognize both speech and audio events.
[Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md) scales up audio-language pre-training to cover over 30 tasks and various audio types, facilitating universal audio understanding abilities.
It employs a unified encoder for all audio inputs, bridging the gap between audio and textual modalities, and uses the large language model [Qwen-7B [11]](../../Models/TextLM/2023.09.28_Qwen.md) as its foundational component.
[Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md) simplifies the pre-training process by utilizing natural language prompts for different data and tasks, with [DPO [170]](../../Modules/RLHF/DPO.md) optimizing the model’s performance in terms of factuality and adherence to desired behavior.
[SALMONN [198]](../../Models/SpeechLM/2023.10.20_SALMONN.md) employs dual auditory encoders: a speech encoder from the Whisper model and a non-speech [BEATs [28]](../../Models/SpeechRepresentation/2022.12.18_BEATs.md) audio encoder.
The auditory features from these two encoders are complementary, making them suitable for general audio inputs that contain both speech and non-speech information.
These inputs are then connected to a well-trained LLM using Q-former style attention to generate responses.
[VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) implements a duplex solution through two independent modules: one generates text responses to user queries, while the other continuously monitors environmental input to selectively provide updated interaction content, although it still requires an external TTS system.
All the aforementioned methods frequently overlook paralinguistic information, including emotion, prosody, and non-verbal elements, rendering them insufficient for scenarios that involve emotional speech dialogue.
[ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md) utilizes an ASR model to obtain text and a speech encoder to extract emotion embeddings, thereby more accurately simulating both the linguistic content and paralinguistic attributes of spoken responses.
[E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md) employs a [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) speech encoder to extract speech and emotion features, using a connection module to map these features to the textual space within the LLM decoder.
Although these approaches have explored emotional responses within spoken dialogue systems, they require additional systems to synthesize speech from text and suffer from high latency, making real-time dialogue challenging to achieve.

</details>
<br>

这些系统 ([Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md); [Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md); [LTU-AS [67]](../../Models/SpeechLM/2023.09.25_LTU-AS.md); [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md); [SALMONN [198]](../../Models/SpeechLM/2023.10.20_SALMONN.md); [WavLLM [80]](../../Models/SpeechLM/WavLLM.md); [SpeechVerse [41]](../../Models/SpeechLM/2024.05.14_SpeechVerse.md); [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md)) 保持了基于文本的大语言模型的基础架构不变, **使用音频编码器和适配器将语音输入直接映射到大语言模型预训练的文本潜在空间**.
这种直接嵌入对齐的方法, 和多任务训练策略相结合, 使得大语言模型具备了听的能力, 从而能够有效地理解和处理语音模态输入, 并在各种音频理解任务中表现出色.
然而, 输出仍然是基于文本的, 这需要使用外部的文本转语音系统 ([XTTS [21]](../../Models/SpeechLM/2024.06.07_XTTS.md); [CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)) 来生成语音输出.
- [LTU-AS [67]](../../Models/SpeechLM/2023.09.25_LTU-AS.md) 使用 [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) 与时间和层级 Transformer (TLTR) 作为音频编码器, 使其能够识别语音和音频事件.
- [Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md) 将音频-语言预训练扩展到覆盖超过三十个任务和各种音频类型, 促进了通用音频理解能力. 它对所有的音频输入采用了统一的编码器, 弥合了音频和文本模态之间的差距, 并使用大语言模型 [Qwen-7B [11]](../../Models/TextLM/2023.09.28_Qwen.md) 为基础组件.
- [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md) 通过为不同数据和任务使用自然语言提示来简化预训练过程, 并使用 [DPO [170]](../../Modules/RLHF/DPO.md) 优化模型在真实性和遵循期望行为方面的表现.
- [SALMONN [198]](../../Models/SpeechLM/2023.10.20_SALMONN.md) 采用了双听觉编码器: 来自 Whisper 模型的语音编码器和非语音的 [BEATs [28]](../../Models/SpeechRepresentation/2022.12.18_BEATs.md) 音频编码器. 这两个编码器的听觉特征互补, 使其适合用于包含语音和非语音信息的通用音频输入. 这些输入随后通过 Q-Former 风格注意力机制连接到一个经过良好训练的 LLM 以生成响应.
- [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) 通过两个独立的模块实现全双工方案: 一个模块生成对用户查询的文本响应, 另一个持续监控环境输入以有选择地提供更新的交互内容, 尽管它仍然需要额外的 TTS 系统.

上述提及的方法经常忽略副语言信息, 包括情感, 韵律, 和非语言元素, 这使得它们在涉及情感语音对话的场景中表现不足.

- [ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md) 使用 ASR 模型获取文本, 并使用语音编码器提取情感嵌入, 从而更准确地模拟语音响应的语言内容和副语言属性.
- [E-Chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md) 采用 [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) 语音编码器提取语音和情感特征, 使用连接模块将这些特征映射到 LLM 解码器中的文本空间.

尽管这些方法已经探索了口语对话系统的情感响应, 它们要求额外的系统来从文本合成语音并面临高延迟问题, 使得实时对话难以实现.

### Chain-of-Modality (CoM) Method: 模态链方法

<details>
<summary>展开原文</summary>

This method tokenizes speech into discrete tokens and extends the LLM’s vocabulary to handle both speech input and output.
To address alignment issues between speech and text modalities, Recent works ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md); [Spectron [156]](../../Models/SpeechLM/2023.05.24_Spectron.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md)) utilize a prompting approach called Chain-of-Modality (CoM), which first generates response text autoregressively before producing the corresponding speech.
This technique allows the text LLM's output to guide speech generation, thereby enhancing the quality of the response content.
However, it is not suitable for live interactions, as the model must complete the entire text response before beginning speech generation, leading to increased response latency.
[SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md) and [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md) employ the [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) model as a speech token extractor, breaking down speech generation into the prediction of semantic tokens followed by acoustic tokens.
[Spectron [156]](../../Models/SpeechLM/2023.05.24_Spectron.md) performs speech continuation by predicting spectrograms frame-by-frame, optimizing the LLM with a combination of cross-entropy loss for text and reconstruction loss for speech frames.
[EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md), on the other hand, utilizes the FSPIRAL ([SPIRAL [85]](../../Models/SpeechRepresentation/2022.01.25_SPIRAL.md)) architecture for its speech encoder to capture phonetic and tonal information, which is then discretized using [finite scalar quantization (FSQ) [149]](../../Modules/VQ/FSQ.md).
Its speech response procedure is divided into three primary steps:
1) transcribing user instructions into text,
2) generating textual responses based on these instructions, and
3) producing style labels and response speech units from the textual responses.

This process enables EMOVA to facilitate emotional speech dialogue.

</details>
<br>

这种方法将语音分词为离散 Token 并扩展大语言模型的词表以处理语音输入和输出.
为了处理语音和文本模态之间的对齐问题, 近期工作 ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md); [Spectron [156]](../../Models/SpeechLM/2023.05.24_Spectron.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md)) 利用了名为模态链 (Chain-of-Modality, CoM) 的提示方法, 首先自回归地生成响应文本然后生成相应的语音.
这种技术允许文本大语言模型的输出来引导语音生成, 从而增强响应内容的质量.
然而, 这不适合现场交互, 因为模型必须完成整个文本响应才能开始语音生成, 这导致响应延迟增加.
- [SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md) 和 [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md) 采用 [SpeechTokenizer [249]](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) 模型作为语音 Token 提取器, 将语音生成分解为语义 Token 预测和音频 Token 预测.
- [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md) 使用 FSPIRAL ([SPIRAL [85]](../../Models/SpeechRepresentation/2022.01.25_SPIRAL.md)) 架构作为其语音编码器, 捕捉语音的音素和音调信息, 然后使用 [finite scalar quantization (FSQ) [149]](../../Modules/VQ/FSQ.md) 对其离散化.
  其语音响应过程分为三个主要步骤: (1) 将用户指令转化为文本, (2) 根据这些指令生成文本响应, (3) 从文本响应生成样式标签和响应语音单元.
  这种过程使得 EMOVA 能够促进情感语音对话.

### Interleaving Text and Speech Tokens: 交错文本和语音 Token

<details>
<summary>展开原文</summary>

Some earlier models ([AudioPaLM [179]](../../Models/SpeechLM/2023.06.22_AudioPaLM.md), [VoxtLM [145]](../../Models/SpeechLM/2023.09.14_VoxtLM.md)) employed supervised training methods, using specific input and output sequences, and trained on mixed speech-text tasks, including text-to-speech (TTS), automatic speech recognition (ASR), and speech-to-speech translation.
[Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) leverages the temporal alignment between speech and its transcription, continuing training on a pre-trained text-based LLM using alternating text and speech tokens.
This significantly improves the model’s performance in both speech understanding and generation.
However, it employs discrete [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) units as speech representations, which results in some loss of paralinguistic information.
[USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md) continues pretraining [Mistral-7B [22]](../../Models/TextLM/Mistral-7B.md) with interleaved speech-text data to capture multimodal semantics.
For dialogue finetuning, it constructs templates using both speech and transcripts of user input as instruction data.

</details>
<br>

一些早期的模型 ([AudioPaLM [179]](../../Models/SpeechLM/2023.06.22_AudioPaLM.md), [VoxtLM [145]](../../Models/SpeechLM/2023.09.14_VoxtLM.md)) 采用了监督训练方法, 使用具体的输入和输出序列, 并在混合语音-文本任务上进行训练, 包括文本到语音 (TTS), 自动语音识别 (ASR), 以及语音到语音的翻译.
- [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) 利用语音和其转写之间的时序对齐, 使用交替文本和语音 Token 继续训练预训练的基于文本的大语言模型 (LLM).
  这种方法显著提升了模型在语音理解和生成中的性能.
  但是, 它采用离散的 [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) 单元作为语音表示, 这导致部分丢失了副语音信息.
- [USDM [106]](../../Models/SpeechLM/2024.02.08_USDM.md) 继续使用语音-文本数据对 [Mistral-7B [22]](../../Models/TextLM/Mistral-7B.md) 进行预训练, 以捕捉多模态语义.
  为了对话微调, 它使用用户输入的语音和转写作为指令数据构造模板.

### Parallel Generation of Text and Speech: 并行生成文本和语音

<details>
<summary>展开原文</summary>

[PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md) proposes generating speech and text tokens in parallel to reduce latency; however, this approach may compromise response quality.
Additionally, this method still relies on speech recognition for input ([Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md)), which introduces further delay.
[LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) introduces a novel streaming speech decoder that can simultaneously generate text responses and discrete speech unit sequences, significantly reducing latency and meeting real-time interaction needs.
[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) and [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) adopt similar approaches, introducing dual streams that generate both speech tokens and corresponding text tokens simultaneously on the assistant side, facilitating the transfer of the pre-trained LLM’s textual capabilities to the speech modality, enabling the model to directly engage in reasoning through speech.
The key difference lies in how speech-text alignment is handled: [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) uses explicit alignment information to supervise the model’s learning, while [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) allows the LLM to learn implicit alignment information.
On the input side, Mini-Omni feeds continuous speech embeddings from the Whisper encoder ([Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md)) into the LLM, enhancing the model's ability to understand spoken instructions without requiring text input.
However, inconsistencies between speech input and output introduce additional computational overhead, increasing latency in multi-turn dialogue scenarios.
In contrast, Moshi allows users to input speech without relying on text, and generates both text and speech tokens in parallel on the assistant side.
Moshi further extends its architecture to model several speech streams in parallel, allowing for conceptually and practically simple handling of full-duplex dialogues with arbitrary dynamics.

</details>
<br>

- [PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md) 提出并行地生成语音和文本 Token 来减少延迟, 然而这种方法可能会损害响应的质量.
  此外, 这种方法仍然依赖于输入的语音识别 ([Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md)), 这引入了进一步的延迟.
- [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) 引入了新颖的流式语音编码器, 以同时生成文本响应的离散语音单元序列, 显著减少延迟并适应实时交互的需求.
- [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) 和 [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) 采用相似方法, 引入双流架构在助手端同时生成语音 Token 和相应的文本 Token, 从而促进了预训练大语言模型的文本能力向语音模态的迁移, 使得模型能够直接通过语音进行推理.

关键区别在于如何处理语音-文本对齐:
- [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) 使用显式的对齐信息来监督模型的学习;
- [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) 允许 LLM 学习隐式的对齐信息.

在输入侧, Mini-Omni 将 [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md) 编码器生成的连续语音嵌入输入到 LLM 中, 增强了模型理解语音指令的能力, 而无需文本输入.
然而, 语音输入和输出之间的不一致性引入了额外的计算开销, 增加了多轮对话场景中的延迟.
相比之下, Moshi 允许用户直接输入语音而不依赖于文本, 并在助手端并行生成文本和语音 Token.
Moshi 进一步扩展了架构以并行建模多个语音流, 从而能够从概念上和实践上简单地处理具有任意动态的全双工对话.

### Speech-to-Speech Generation: 语音转语音生成

<details>
<summary>展开原文</summary>

This approach aims to remove the dependency on intermediate text, thereby reducing latency and making the system closer to real-time interaction.
[SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md) achieves real-time full-duplex interaction through time chunking methods, integrating time information into LLMs to enable synchronous operation with the real-world clock.
[IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) utilizes a specific model to generate multiple speech tokens in a single step, effectively reducing speech token sequences to lengths comparable to text sequences while producing high-quality audio.
[Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md) utilizes a pre-trained self-supervised [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) model with K-means clustering ([TWIST [74]](../../Models/SpeechLM/2023.05.22_TWIST.md)) to convert continuous speech representations into discrete units.
It employs [LoRA [79]](../../Modules/LoRA/2021.06.17_LoRA.md) adapter fine-tuning on a pre-trained [TWIST [74]](../../Models/SpeechLM/2023.05.22_TWIST.md) to produce multiple speech continuations from a given prompt and uses semantic metrics to generate preference data for [Direct Preference Optimization (DPO) [170]](../../Modules/RLHF/DPO.md).
Experimental results indicate that integrating the preference optimization method significantly improves the semantic comprehension of the Spoken LLM.

</details>
<br>

这种方法旨在移除对中间文本的依赖, 从而减少延迟并使系统更接近实时互动.
- [SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md) 通过时间分块方法实现实时全双工交互, 将时间信息集成到 LLMs 中, 以便与真实世界时钟同步运行.
- [IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) 使用特定模型在单步生成多个语音 Token, 有效地将语音 Token 序列长度与文本序列长度相当, 同时产生高质量的音频.
- [Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md) 使用预训练的自监督 [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) 模型和 K-means 聚类 ([TWIST [74]](../../Models/SpeechLM/2023.05.22_TWIST.md)) 将连续语音表示转换为离散单元. 它采用在预训练的 [TWIST [74]](../../Models/SpeechLM/2023.05.22_TWIST.md) 上微调的 [LoRA [79]](../../Modules/LoRA/2021.06.17_LoRA.md) 适配器来从给定的提示生成多个语音延续, 并使用语义度量来生成偏好数据用于[直接偏好优化 (DPO) [170]](../../Modules/RLHF/DPO.md). 实验结果表明, 将偏好优化方法集成到 Spoken LLM 中可以显著提高语义理解能力.

## 4.2·Multi-Stage Training Strategy: 多阶段训练策略

<details>
<summary>展开原文</summary>

This section primarily discusses the training process of the Spoken Dialogue Model, building upon previous work on spoken dialogue systems.
Generally, this process consists of four stages: text LLM pre-training, modality adaptation and alignment post-training, followed by supervised fine-tuning, and optionally, preference optimization.
The primary goal in training most spoken dialogue systems is to preserve the model's original capabilities while integrating the speech modality for voice interaction into the LLM.
The diagram of multi-stage training can be referred to in Figure.06.

</details>
<br>

本节主要基于之前在口语对话系统方面的工作来讨论口语对话模型的训练过程.
通常, 该过程包括四个阶段:
- 文本 LLM 预训练
- 模态适配
- 对齐后训练
- 监督微调
- 以及可选的偏好优化

训练大多数口语对话系统的主要目标是保留模型的原始能力, 同时将语音模态集成到 LLM 中, 以实现语音交互.

多阶段训练的流程图可参考图 06。

![](Images/Fig.06.png)

### 4.2.1·Text LLM Pre-Training

The goal is to develop a text-intelligent LLM model capable of handling complex contexts and possessing knowledge reasoning abilities, thus preparing it for integration with speech-intelligent LLMs.
Most spoken dialogue systems utilize pre-trained large language models as foundational models rather than pre-training with separate text data themselves.
A series of approaches ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md); [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md)) use the LLaMA model and its variants as their foundational language model.
On the other hand, ([LauraGPT [50]](../../Models/SpeechLM/2023.10.07_LauraGPT.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md)) employ the Qwen ([Qwen [11]](../../Models/TextLM/2023.09.28_Qwen.md) [Qwen-2 [228]](../../Models/TextLM/Qwen2.md)) family of large language models as their backbone.
Meanwhile, [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) employs an RQ-Transformer for hierarchical autoregressive modeling of speech, utilizing a unique structure that involves pre-training a text-only language model with datasets from the internet (e.g., [Wikipedia](https://dumps.wikimedia.org/) and [StackExchange](https://archive.org/details/stackexchange/)).
The collected data was filtered using a comprehensive preprocessing pipeline to ensure quality and relevance, which included deduplication to remove redundant entries, language identification to retain text in the desired language, and quality filtering to exclude low-quality or irrelevant content based on criteria such as coherence and completeness.
[VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) utilizes [Mixtral 8x7B1 [95]](../../Models/TextLM/Mixtral.md), a representative LLM with a sparse mixture of experts (SMoE) architecture, and performs pure-text instruction tuning for its extended Chinese vocabulary.

### 4.2.2·Modality Adaptation and Alignment Post-training

This phase explores strategies to adapt text-based large language models (LLMs) for speech modality input, focusing on aligning text and audio modalities effectively.
The primary goal is to enhance the models' ability to understand and generate speech by bridging the gap between these two modalities.
Common approaches include multimodal training techniques, leveraging unlabeled speech corpora, and employing multi-task learning frameworks.
These methods typically involve fine-tuning existing LLMs with speech-related tasks and integrating speech-specific modules, such as speech adaptors and decoders, to facilitate seamless interaction between text and speech modalities.
Different training tasks for modality adaptation and alignment are shown in Figure.07.
[Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) continuously pretrain on text LLM checkpoints using interleaved text and speech tokens to improve the model's performance in speech understanding and generation.
[LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) adopts a two-stage training strategy: the first stage jointly trains a speech adaptor and LLM with speech input and text responses, while the second stage uses the same dataset to train a streaming speech decoder independently.
Consequently, this LLM primarily possesses the capability for speech input understanding, with speech generation handled by a separate decoder module.
[SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md), [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md), and [VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md) utilize unlabeled speech corpora to train models in a next-token prediction task.
In the first phase, VITA focuses on training the audio encoder and connector, while in the second phase, it optimizes both the connector and the LLM model through multimodal training.
Although capable of processing speech input, it outputs only text.
[Spectron [156]](../../Models/SpeechLM/2023.05.24_Spectron.md) addresses the alignment issue between text and speech representations by jointly supervising multiple objectives.
[IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) employs a two-stage training approach, constructing multiple cross-modal tasks from a single dataset to enable the model to better learn the semantic consistency between speech and text.
[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md), [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md), and [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md); adopt similar methodologies, commencing with supervised multi-task fine-tuning of the text LLM backbone to achieve speech-text modality alignment and develop a multimodal LLM ([Jin et al. (Survey) [99]](../2024.05.17_Efficient_Multimodal_Large_Language_Models__A_Survey/Main.md); [Li et al. (2024) [120]](../2024.08.16_A_Survey_on_Benchmarks_of_Multimodal_Large_Language_Models/Main.md)) using Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) tasks.
Notably, Mini-Omni divides the training of various modules into three phases: the first phase utilizes data from speech recognition and synthesis to enhance the model’s abilities in these aspects, training only the ASR and TTS adapters.
The second phase focuses exclusively on enhancing the model’s text capabilities when given speech inputs, updating only the LLM parameters while freezing other modules.
Through these two training phases, the original language LLM’s capabilities are maximally preserved, while adapting to speech modality input and output, thereby addressing the primary modality alignment tasks.

### 4.2.3·Supervised Fine-tuning or Dialogue Dataset Fine-tuning

During this stage, most models use instruction-following datasets or dialogue data for supervised fine-tuning of the LLM, enhancing natural conversational abilities.
([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md)) propose a two-stage instruction-tuning process that includes cross-modal instruction fine-tuning and chain-of-modality instruction fine-tuning.
Ultimately, the model follows the A-T-T-A method to achieve end-to-end speech input and output.
[EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md) employs a similar chain-of-modality concept to construct instruction-tuning datasets, empowering it to respond accurately to speech instructions.
[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md), [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md), [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md), and [SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md) utilize spoken dialogue datasets for fine-tuning, endowing the models with conversational interaction capabilities.
Remarkably, Moshi constructs a more natural and realistic dialogue dataset that incorporates elements such as noise and overlap, enabling the model to learn authentic multi-stream interactions.
OmniFlatten fine-tunes the speech-text LLM using interleaved and serialized dialogues across three stages to progressively train the model in acquiring half-duplex and full-duplex communication capabilities.
Similarly, SyncLLM employs a three-stage training procedure that predominantly uses synthetic spoken dialogue data along with a relatively small amount of real-world spoken dialogue data to develop a full-duplex voice agent.

### 4.2.4·Preference Optimization and Reinforcement Learning

The research on leveraging preference optimization to align a spoken dialogue model with human preferences is virtually absent.
Recently, [Seed-TTS [5]](../../Models/SpeechLM/2024.06.04_Seed-TTS.md); [SpeechAlign [243]](../../Models/SpeechLM/2024.04.08_SpeechAlign.md); [UNO [23]](../../Modules/RLHF/2024.06.02_UNO.md) adopted preference optimization for Text-to-Speech (TTS) models to align speech synthesis quality with human preferences but not for spoken dialogue models.
[Align-SLM [129]](../../Models/SpeechLM/2024.11.04_Align-SLM.md) pioneers the integration of [Direct Preference Optimization (DPO) [170]](../../Modules/RLHF/DPO.md) in textless Spoken Language Models (SLMs) to enhance semantic understanding.
It transforms continuous speech into discrete units using a pre-trained Hubert model and K-means clustering.
LoRA fine-tuning on a Spoken LLM generates multiple speech continuations from prompts.
Semantic metrics create preference data offline, making DPO training efficient and stable, eliminating the need for an external reward model.
Coupled with [curriculum learning [15]](../../Models/_Basis/Curriculum_Learning.md), Align-SLM progressively refines preference data selection, optimizing semantic feedback, and improving SLM performance.

## 4.3·Training Frameworks and Generation Strategies: 训练框架和生成策略

Recent advanced methods in spoken dialogue models employ a variety of innovative techniques to achieve more natural speech output and lower latency.
In this part, we explore various approaches that exemplify these advancements:

### LLaMA-Omni

[LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) adds a streaming speech decoder that operates after the LLM.
This decoder runs in a non-autoregressive manner, taking the output hidden states from the LLM as input and generating the discrete unit sequence corresponding to the speech response.
To model the variable-length mapping between input and output, LLama-Omni employs an upsample factor, denoted as $\lambda$, along with [Connectionist Temporal Classification (CTC) loss [69]](../../Models/ASR/CTC.md).
This ensures that the model can generate speech responses simultaneously with text responses.
Additionally, a predefined chunk size is set to further enable vocoder streaming synthesis of speech waveforms, facilitating real-time interaction and reducing latency.

### Mini-Omni

[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) selects [SNAC [193]](../../Models/Speech_Neural_Codec/2024.10.18_SNAC.md), a music-grade encoder, to discretize one second of audio into hundreds of tokens, which significantly increases the burden on the LLM for modeling speech tokens.
Delay Pattern language model decoding strategies are often applied in modeling multiple parallel streams of acoustic tokens in speech tasks like [MusicGen [40]](../../Models/SpeechLM/2023.06.08_MusicGen.md), [VoiceCraft [163]](../../Models/SpeechLM/2024.03.25_VoiceCraft.md), and [Parler-TTS [140]](../../Models/SpeechLM/2024.02.02_Parler-TTS.md).
Compared with traditional sequential step decoding, this strategy can effectively reduce the time steps required for LLM decoding and generating speech tokens.
Inspired by this, Mini-Omni innovatively applies text-instructed delayed parallel generation to address the issue of long SNAC codebook sequences, simultaneously producing audio and text tokens.
This effectively leverages and preserves the original capabilities of the language model.
Moreover, Mini-Omni proposes a Batch Parallel Decoding method.
Specifically, it generates two samples in parallel for a single input: the first predicts text tokens, and the second predicts both text and speech tokens simultaneously.
The text output from the first sample is embedded into the corresponding positions of the second sample, while the second sample's text output is discarded.
This further enhances the model’s reasoning capabilities during dialogue, maximizing the transfer of its text-based abilities.

### IntrinsicVoice

[IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) introduces a speech encoder and a streaming vocoder for the tokenization and detokenization of speech, and a GroupFormer for modeling speech and text sequences.
This architecture integrates a large language model (LLM) with a GroupModel.
Specifically, it uses a pre-trained [HuBERT [78]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md) encoder and its corresponding K-Means quantizer ([TWIST [74]](../../Models/SpeechLM/2023.05.22_TWIST.md)) to process speech inputs into discrete units.
These units are organized into a grouped token sequence through a group partition operation.
The grouped tokens are then passed through an embedding layer and adaptor module to map these embeddings into the LLM's embedding space.
The context embeddings output by the LLM are processed through a linear layer and concatenated with a specified number of learnable queries.
This input is fed into a smaller non-autoregressive transformer encoder model, dubbed the "GroupModel," to predict a group of speech tokens in one step.
The introduction of GroupFormer effectively improves the model's ability to handle sequences within a group, mitigates the modality gap between speech and text, accelerates inference speed, and alleviates issues associated with long-sequence modeling.

### Moshi

[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) introduces a mini codec model with 8 codebooks at a frame rate of 12.5 Hz for speech representation, where one second corresponds to 100 speech tokens.
It adopts an RQ-Transformer consisting of a Temporal Transformer and a smaller Depth Transformer as the backbone network for the LLM, hierarchically modeling multi-codebook audio tokens.
Similar architectures have appeared in prior research, such as [UniAudio [232]](../../Models/SpeechLM/2023.10.01_UniAudio.md) and [MegaByte [237]](../../Models/SpeechLM/2023.05.12_MegaByte.md).
The Depth Transformer models sub-sequence tokens conditioned on temporal context predicted by the Temporal Transformer.
Given the smaller size of the Depth Transformer, sub-sequence generation can almost be viewed as parallel generation.
This allows the model to scale to longer sequences by extending the temporal modeling capacity of the Temporal Transformer or to achieve greater depth by enhancing the hierarchical modeling capabilities of the Depth Transformer, rather than modeling the flattened sequence with a single model.

### SyncLLM

[SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md) employs an auto-regressive transformer decoder for full-duplex dialogue, integrating time synchronization to align speech units with the real-world clock.
It predicts interleaved speech tokens for both dialogue partners, maintaining timing with speaker tags.
The model is trained on deduplicated HuBERT token sequences to enhance semantic fidelity while managing latency by anticipating user responses.
Interpolation reconstructs token sequences to fit expected structures, facilitating seamless speech synthesis.

### Text-guided generation

Some end-to-end methods like ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [SpeechGPT-Gen [244]](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md); [Spectron [156]](../../Models/SpeechLM/2023.05.24_Spectron.md); [EMOVA [25]](../../Models/SpeechLM/2024.09.26_EMOVA.md)) use chain-of-thought reasoning, which allows guiding speech generation with the output of an underlying text LLM.
However, this is fundamentally incompatible with live interactions, as the model needs to produce an entire answer as text before it starts speaking.
Later methods ([LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md)) can accept user speech input and simultaneously output speech and text, ensuring high-quality responses while significantly reducing latency.
[LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md) utilizes a streaming decoder to generate text and speech tokens in parallel.
[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) is restructured to transfer language reasoning abilities to streaming audio output through a text-audio parallel decoding approach.
[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) details a novel feature, the Inner Monologue, which consists of joint modeling of the textual and speech modalities on the system side to improve the quality of interactions.

### W/o text-guided generation

Other methods achieve speech-to-speech generation without relying on text stream generation.
[IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md) introduces a novel GroupModel that predicts a group of speech tokens in one step based on global context embeddings.
[SyncLLM [203]](../../Models/SpeechLM/2024.09.23_SyncLLM.md) predicts interleaved chunks of token sequences at each time step, allowing the model to handle all conversational cues such as backchannels, overlaps, interruptions, etc.

## 4.4·Discussions about Training Paradigm in Spoken Dialogue Models: 讨论

### 4.4.1·Text and Speech Modality Alignment: 文本和语音模态对齐

In spoken dialogue systems, the alignment between speech and text modalities is a crucial stage.
To preserve the textual intelligence of large language models (LLMs) as much as possible, nearly all current methodologies ([SpeechGPT [242]](../../Models/SpeechLM/2023.05.18_SpeechGPT.md); [PSLM [154]](../../Models/SpeechLM/2024.06.18_PSLM.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md)) incorporate a post-training phase utilizing speech-text paired data when developing spoken dialogue models.
This may involve either expanding the vocabulary to treat speech tokens as an extension of the original vocabulary or using speech adaptors to map speech embeddings to the original text latent space of the LLM, and designing multi-task training objectives to achieve alignment between text and speech modalities.
For example, data from speech recognition and speech synthesis can be used to train the model's speech recognition and synthesis capabilities.
Although this is an effective strategy, its implementation can still lead to a certain degree of catastrophic forgetting in LLMs due to the large volume of pre-trained text corpora and the imbalance with paired speech-text data, which can harm the model's text-based capabilities.
Therefore, precise parameter design and customized optimization strategies are needed to mitigate this issue as much as possible, as demonstrated by approaches like [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md).

This raises a consideration: during the training phase of spoken dialogue models, is it feasible to directly utilize speech data for adaptation to text-based LLMs, thereby eliminating the necessity for speech-text paired data? This is because unlabeled speech data is abundant and easily accessible, making it convenient and beneficial for training the speech intelligence of LLMs.
This approach would require us to obtain a pre-aligned speech representation with the text modality.
Perhaps we can consider further exploration and experimentation in the speech tokenizer component, such as directly mapping the semantic discrete units of speech onto the text token space to achieve enforced alignment.

### 4.4.2·Different Temporal Alignment Methods in Spoken Dialogue Models: 不同时序对齐方法

In speech and text modalities, there is often a significant mismatch in sequence lengths.
Even when some speech tokenizers ([WavTokenizer [90]](../../Models/Speech_Neural_Codec/2024.08.29_WavTokenizer.md); [Single-Codec [119]](../../Models/Speech_Neural_Codec/2024.06.11_Single-Codec.md)) employ extreme sequence compression methods, a length gap remains between the two.
Temporal alignment information between speech and text has been explored in tasks like Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) as demonstrated by models such as [Whisper [169]](../../Models/SpeechLM/2022.12.06_Whisper.md), [FastSpeech2 [176]](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md), and [VITS [107]](../../Models/E2E/2021.06.11_VITS.md).
Recently, some spoken dialogue systems have utilized temporal alignment information to enhance model performance, yielding promising results.
For instance, [Spirit-LM [158]](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) uses interleaving text and speech tokens for continual pre-training on the LLaMA base model, significantly boosting the model’s performance in speech understanding and generation.
Experimental visualizations demonstrate that the similarity between text and speech features is notably higher in models trained with interleaved token sequences compared to those trained without this approach.
This indicates that providing the model with explicit fine-grained temporal alignment information can effectively enhance modality alignment and improve the performance of LLMs.

[Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md) achieves parallel generation of text and speech by padding text tokens to match the length of speech tokens, allowing the LLM to implicitly learn the alignment information between speech and text tokens.
This can be viewed as a form of sentence-level temporal alignment information, a method also utilized in recent speech synthesis work ([F5-TTS [30]](../../Models/Diffusion/2024.10.09_F5-TTS.md)).
[Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md), on the other hand, uses word-level speech-text temporal alignment information and special marker tokens to achieve similar parallel generation capabilities.
The difference lies in that Mini-Omni fully allows the LLM to implicitly learn the alignment, whereas Moshi provides word-level alignment priors first, and then lets the model learn finer-grained alignments.

Exploring the impact of introducing different levels of temporal alignment priors on the training effectiveness of spoken dialogue models, such as sentence-level, word-level, or phoneme-level, is an intriguing area of research.
Understanding how these various alignment strategies affect model performance can guide the development of more efficient and accurate systems.
For instance, sentence-level alignment might offer a broader contextual understanding, while word-level or phoneme-level alignments could provide more detailed synchronization between speech and text, potentially leading to improvements in nuanced tasks like speech synthesis and understanding.

### 4.4.3·Reinforcement Learning (RL) in Spoken Dialogue Models: 强化学习

Reinforcement Learning (RL) has proven to be an effective learning paradigm in text and image processing ([PPO [185]](../../Models/_Basis/PPO.md); [Policy Gradient [196]](../../Models/_Basis/PG.md); [Diffusion-DPO [204]](../../Models/CV/2023.11.21_Diffusion-DPO.md)).
Recent research has shown that [Direct Preference Optimization (DPO) [170]](../../Modules/RLHF/DPO.md) can be extended to music and speech generation ([MusicRL [36]](../../Models/SpeechLM/2024.02.06_MusicRL.md); [SpeechAlign [243]](../../Models/SpeechLM/2024.04.08_SpeechAlign.md)).
[MusicRL [36]](../../Models/SpeechLM/2024.02.06_MusicRL.md) uses Reinforcement Learning from Human Feedback (RLHF) to improve music generation by fine-tuning a pretrained model for better text adherence and audio quality.
By collecting extensive human feedback, MusicRL creates a more refined and subjective music generation system.
[Seed-TTS [5]](../../Models/SpeechLM/2024.06.04_Seed-TTS.md) explores RL methods, comparing external reward models like REINFORCE with simpler methods like DPO.
The study highlights using REINFORCE to enhance speaker similarity and emotion controllability in the Seed-TTS system.
[Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md) uses DPO to align with human preferences by optimizing responses based on human-annotated data.
This enhances its ability to follow audio instructions accurately and intelligently respond to complex audio inputs, improving its performance in audio-centric tasks.
However, in the dialogue system field, reinforcement learning techniques based on human feedback ([Huang et al (Survey) [82]](../2023.11.09_A_Survey_on_Hallucination_in_Large_Language_Models__Principles_Taxonomy_Challenges_and_Open_Questions/Main.md)) are rarely applied.
Considering the diversity of inputs and outputs in large language models, exploring the incorporation of reinforcement learning strategies such as [Proximal Policy Optimization (PPO) [185]](../../Models/_Basis/PPO.md) can be beneficial.
Additionally, considering the performance metrics for evaluating spoken dialogue systems, designing targeted reinforcement learning strategies and feedback functions to enhance different objectives is also a direction worth exploring.
