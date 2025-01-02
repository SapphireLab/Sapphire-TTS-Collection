# 6·Training Resources and Evaluation: 训练资源和评估

## 6.1·Training Resources: 训练资源

<details>
<summary>展开原文</summary>

Training a spoken dialogue system is a complex, multi-stage process, with each stage relying on specific datasets to achieve distinct training objectives and enhance system performance.
This section provides an in-depth analysis of the training resources about the spoken dialogue models, showcasing the data collection and processing methods at each stage and illustrating how these elements contribute to the system's intelligence.
It further reveals how key steps, from foundational architecture to fine-tuning, shape the intelligent development of dialogue systems.

To address the limitations of existing training spoken dialogue data and leverage the knowledge and reasoning capabilities of mature text-based models, many approaches involve **Continue Training** on pre-trained text language models.
This training paradigm encompasses nearly all data types required to build a spoken dialogue system.
The following sections focus on analyzing data acquisition and processing methods under this training flow, covering the following core stages: **Text Language Model Pre-training**, **Post-Train for Audio Modal Adaption**, **Post-Train for Dual-Stream Audio Processing**, **Enhancing Conversational Abilities and Instruction Tuning**.
We have listed commonly used datasets for training in Table.02.
However, current spoken dialogue models lack exploration in music and sound.

To support future development in spoken dialogue systems, we provide a list of common music and sound datasets in the appendix A as a reference.
Appendix A: This section lists commonly used music and sound datasets.
These datasets cover different modalities, including environmental sounds, music, and emotional sounds, and provide some help for the development of future voice dialogue systems.
The table 4 shows the basic information of each dataset, including the dataset name, number of samples, dataset link, and modality type.

</details>
<br>

训练一个口语对话系统是一个复杂的多阶段过程, 每个阶段都依赖于特定的数据集以实现不同的训练目标和增强系统性能.
本节提供了关于口语对话模型的训练资源的深入分析, 展示了每个阶段的数据收集和处理方法, 并说明这些元素如何促进系统的智能.
这进一步揭示了关键步骤, 从基础架构到微调, 如何塑造对话系统的智能发展.

为了解决现有训练口语对话数据的局限性, 并充分利用成熟的基于文本的模型的知识和推理能力, 许多方法都采用了**继续训练**预训练文本语言模型.
这种训练范式涵盖了构建口语对话系统所需的所有数据类型.
下面的各小节着重分析了在这种训练流程下的数据采集和处理方法, 涵盖以下核心阶段:
- 文本语言模型预训练;
- 音频模态后训练;
- 双流音频处理后训练;
- 增强对话能力和指令调优.

我们在表格 02 中列出了常用的数据集.
然而, 当前的口语对话模型缺乏对音乐和声音的探索.
为了支持口语对话系统的未来发展, 我们提供了音乐和声音数据集的列表作为参考.
附录 A: 本节列出了常用的音乐和声音数据集.
这些数据集涵盖了不同的模态, 包括环境声音, 音乐, 和情绪声音, 并为未来的语音对话系统的开发提供一些帮助.
表格 04 显示了每个数据集的基本信息, 包括数据集名称, 样本数量, 数据集链接, 和模态类型.

<details>
<summary>Table.02: Datasets Used in the Various Training Stages</summary>

|Stage                  |Task          |Dataset          |Size                       |URL                                                         |Modality     |
|---                    |---           |---              |---                        |---                                                         |---          |
|Modal Alignment        |Mandarin ASR  |AISHELL-1        |170 hrs                    |https://www.openslr.org/33/                                 |Text + Speech|
|Modal Alignment        |Mandarin ASR  |AISHELL-2        |1k hrs                     |https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell2 |Text + Speech|
|Modal Alignment        |Mandarin TTS  |AISHELL-3        |85 hrs/88/035 utt.218 spk. |https://www.aishelltech.com/aishell_3                       |Text + Speech|
|Modal Alignment        |TTS           |LibriTTS         |585 hrs                    |https://www.openslr.org/60/                                 |Text + Speech|
|Modal Alignment        |ASR           |TED-LIUM         |452 hrs                    |https://lium.univ-lemans.fr/ted-lium3/                      |Text + Speech|
|Modal Alignment        |ASR           |VoxPopuli        |1.8k hrs                   |https://github.com/facebookresearch/voxpopuli               |Text + Speech|
|Modal Alignment        |ASR           |Librispeech      |1000 hrs                   |https://www.openslr.org/12                                  |Text + Speech|
|Modal Alignment        |ASR           |MLS              |44.5k hrs                  |https://www.openslr.org/                                    |Text + Speech|
|Modal Alignment        |TTS           |Wenetspeech      |22.4k hrs                  |https://wenet.org.cn/WenetSpeech/                           |Text + Speech|
|Modal Alignment        |ASR           |Gigaspeech       |40k hrs                    |https://github.com/SpeechColab/GigaSpeech                   |Text + Speech|
|Modal Alignment        |ASR           |VCTK             |300 hrs                    |https://paperswithcode.com/dataset/voice-bank-demand        |Text + Speech|
|Modal Alignment        |TTS           |LJSpeech         |24 hrs                     |https://keithito.com/LJ-Speech-Dataset/                     |Text + Speech|
|Modal Alignment        |ASR           |Common Voice     |2500 hrs                   |https://commonvoice.mozilla.org/zh-CN                       |Text + Speech|
|Modal Alignment        |Audio Caption |Wavcaps          |400k clips                 |https://github.com/XinhaoMei/WavCaps                        |Text + Speech|
|Modal Alignment        |ASR           |LibriLight       |60k hrs                    |https://github.com/facebookresearch/libri-light             |Text + Speech|
|Modal Alignment        |ASR           |PeopleSpeech     |30k hrs                    |https://huggingface.co/datasets/MLCommons/peoples_speech    |Text + Speech|
|Modal Alignment        |Mandarin ASR  |KeSpeech         |1542 hrs                   |https://github.com/KeSpeech/KeSpeech                        |Text + Speech|
|Modal Alignment        |TTS           |Emilia           |101k hrs                   |https://huggingface.co/datasets/amphion/Emilia-Dataset      |Text + Speech|
|Dual-Stream Processing |Instruction   |Alpaca           |52000 items                |https://huggingface.co/datasets/tatsu-lab/alpaca            |Text + TTS   |
|Dual-Stream Processing |Instruction   |Moss             |-                          |https://huggingface.co/fnlp/moss-moon-003-sft               |Text + TTS   |
|Dual-Stream Processing |Instruction   |BelleCN          |-                          |https://github.com/LianjiaTech/BELLE/tree/main              |Text + TTS   |
|Dual-Stream Processing |Dialogue      |UltraChat        |1.5 million                |https://github.com/thunlp/UltraChat                         |Text + TTS   |
|Dual-Stream Processing |Instruction   |Open-Orca        |-                          |https://huggingface.co/datasets/Open-Orca/OpenOrca          |Text + TTS   |
|Dual-Stream Processing |Noise         |DNS              |2425 hrs                   |https://github.com/microsoft/DNS-Challenge                  |Noise data   |
|Dual-Stream Processing |Noise         |MUSAN            |-                          |https://www.openslr.org/17/                                 |Noise data   |
|Conversation Fine-Tune |Dialogue      |Fisher           |964 hrs                    |https://catalog.ldc.upenn.edu/LDC2004T19                    |Text + Speech|
|Conversation Fine-Tune |Dialogue      |GPT-Talker       |-                          |https://github.com/AI-S2-Lab/GPT-Talker                     |Text + Speech|
|Conversation Fine-Tune |Instruction   |INSTRUCTS2S-200K |200k items                 |https://github.com/ictnlp/LLaMA-Omni                        |Text + TTS   |
|Conversation Fine-Tune |Instruction   |Open Hermes      |900k items                 |https://ollama.com/library/openhermes                       |Text + TTS   |

</details>
<br>

<details>
<summary>Table.04: Music and Non-Speech Sound Datasets</summary>

|Dataset                  |Size                            |URL                                                        |Modality                   |
|---                      |---                             |---                                                        |---                        |
|ESC-50                   |2000 clips (5s each)            |https://github.com/karoldvl/ESC-50                         |Sound                      |
|UrbanSound8K             |8732 clips (<=4s each)          |https://urbansounddataset.weebly.com/urbansound8k.html     |Sound                      |
|AudioSet                 |2000k+ clips (10s each)         |https://research.google.com/audioset/                      |Sound                      |
|TUT Acoustic Scenes 2017 |52630 segments                  |https://zenodo.org/record/400515                           |Sound                      |
|Warblr                   |10000 clips                     |https://warblr.net/                                        |Sound                      |
|FSD50K                   |51197 clips (total 108.3 hours) |https://zenodo.org/record/4060432                          |Sound                      |
|DCASE Challenge          |varies annually                 |http://dcase.community/                                    |Sound                      |
|IRMAS                    |6705 audio files (3s each)      |https://www.upf.edu/web/mtg/irmas                          |Music                      |
|FMA                      |106574 tracks                   |https://github.com/mdeff/fma                               |Music                      |
|NSynth                   |305979 notes                    |https://magenta.tensorflow.org/datasets/nsynth             |Music                      |
|EMOMusic                 |744 songs                       |https://cvml.unige.ch/databases/emoMusic/                  |Music                      |
|MedleyDB                 |122 multitrack recordings       |https://medleydb.weebly.com/                               |Music                      |
|MagnaTagATune            |25863 clips (30s each)          |https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset |Music                      |
|MUSDB                    |150 songs                       |https://paperswithcode.com/dataset/musdb18                 |Music                      |
|M4Singer                 |700 songs                       |https://github.com/M4Singer/M4Singer                       |Music                      |
|Jamendo                  |600k songs                      |https://www.jamendo.com/?language=en                       |Music                      |
<!-- |% VIVAE                  |1085 clips                      |https://zenodo.org/records                                 |emotion vocalizations      | -->

</details>
<br>

### 6.1.1·Training Resources about Text LLM Pre-training: 文本大语言模型预训练资源

<details>
<summary>展开原文</summary>

Text Language Model pre-training serves as the foundational stage for spoken dialogue models.
Through unsupervised learning on large-scale text data, the model acquires knowledge of vocabulary, grammar, and contextual relationships, gaining essential knowledge and reasoning capabilities.
Most spoken dialogue systems are built upon pre-existing open-source text language models (such as [LLaMA [200]](../../Models/TextLM/2023.02.27_LLaMA.md), [PaLM2 [6]](../../Models/TextLM/2023.05.17_PaLM2.md), etc).
Although we does not delve into this stage in detail, it provides a solid foundation for the model’s natural language understanding and generation capabilities.

</details>
<br>

文本语言模型预训练是口语对话模型的基础阶段.
通过在大规模文本数据上进行无监督训练, 模型获得词汇语法上下文关联等知识, 增进了必要的知识和推理能力.
大多数口语对话系统都是建立在开源文本语言模型基础之上的 (如[LLaMA [200]](../../Models/TextLM/2023.02.27_LLaMA.md), [PaLM2 [6]](../../Models/TextLM/2023.05.17_PaLM2.md) 等).
虽然我们并未详细探讨这一阶段, 但它为模型的自然语言理解和生成能力提供了坚实的基础.

### 6.1.2·Training Resources about Post-Train for Audio Modal Alignment: 音频模态对齐后训练资源

<details>
<summary>展开原文</summary>

After establishing a text-based foundational model, the system possesses essential knowledge and reasoning abilities.
In this stage, we introduce the audio modality, enabling the text language model to understand and generate speech while minimizing any potential loss of textual knowledge.
This process is known as **modal adaption** or **modal alignment**.
This multimodal structure incorporates an audio encoder with a codebook, helping the model recognize linguistic, emotional, and tonal information in speech.
The audio decoder supports the generation of natural and fluent speech output, while audio signal embeddings and special token types (e.g., speaker-distinguishing tokens for Synchronous LLM, task-distinguishing tokens for OmniFlatten, and state tokens for VITA) are added to the vocabulary of the text language model.

The primary goal at this stage is to align information from different modalities into a unified space or representation, allowing the model to correlate and comprehend such information.
Consequently, the model is often trained on cross-modal tasks such as TTS, ASR, and audio captioning.
The datasets used include numerous paired audio and text samples to ensure effective conversion between modalities.
Commonly used TTS and ASR datasets include [AISHELL-3 [190]](../../Datasets/2020.10.22_AISHELL-3.md), [LibriTTS [240]](../../Datasets/2019.04.05_LibriTTS.md), [TED-LIUM [178]](../../Datasets/TED-LIUM.md), [VoxPopuli [207]](../../Datasets/2021.01.02_VoxPopuli.md), [LibriSpeech [160]](../../Datasets/2015.04.19_LibriSpeech.md), [MLS [168]](../../Datasets/2020.12.07_MLS.md), [Wenetspeech [241]](../../Datasets/2021.10.07_WenetSpeech.md), [Gigaspeech [24]](../../Datasets/2021.06.13_GigaSpeech.md), [VCTK [202]](../../Datasets/2012.08.00_VCTK.md), [LJSpeech [88]](../../Datasets/2017.07.05_LJSpeech.md); [Common Voice [8]](../../Datasets/CommonVoice.md), and others.
For audio captioning, [Wavcaps [147]](../../Datasets/WavCaps.md) are frequently used.
Some speech datasets require ASR model transcription to generate corresponding text.

In this phase, the emphasis is placed on capturing and generating audio features and aligning them with text in vector space, rather than focusing on dialogue functionality.
Therefore, the data typically consists of single-channel audio, which can be used after resampling.
Notably, in some works, it is essential to ensure word-level alignment between text tokens and audio tokens (e.g., Spirit-LM, Moshi, and OmniFlatten), achievable through tools like the Whisper-timestamped package or other alignment tool.
In Moshi, to prevent catastrophic forgetting, half of the training time is allocated to text data, highlighting the importance of balancing text and audio data during training.

</details>
<br>

在建立了基于文本的基座模型后, 系统拥有了必要的知识和推理能力.
然后在这一阶段引入音频模态, 使得文本语言模型能够理解和生成语音, 同时最小化任何文本知识的潜在损失.
这一过程被称为**模态适应**或**模态对齐**.

这种多模态结构整合了带有码本的音频编码器, 帮助模型识别语音中的语言, 情感, 音调信息.
音频解码器支持自然且流畅的语音输出的生成, 同时音频信号嵌入和特殊 Token 类型加入到文本语言模型的词表中.
例如:
- 同步语言模型 SyncLLM 中用于区分发言人的 Token.
- OmniFlatten 中用于区分任务的 Token.
- VITA 中的状态 Token.

这一阶段的主要目标是将来源于不同模态的信息对齐到统一的空间或表示, 使得模型能够关联和理解这些信息.
因此, 模型通常在跨模态任务上训练, 如 TTS, ASR, 音频描述等.
使用的数据集包括大量成对的音频和文本样本, 以确保有效的模态转换.

常用的 TTS 和 ASR 数据集包括:
- [AISHELL-3 [190]](../../Datasets/2020.10.22_AISHELL-3.md)
- [LibriTTS [240]](../../Datasets/2019.04.05_LibriTTS.md)
- [TED-LIUM [178]](../../Datasets/TED-LIUM.md)
- [VoxPopuli [207]](../../Datasets/2021.01.02_VoxPopuli.md)
- [LibriSpeech [160]](../../Datasets/2015.04.19_LibriSpeech.md)
- [MLS [168]](../../Datasets/2020.12.07_MLS.md)
- [Wenetspeech [241]](../../Datasets/2021.10.07_WenetSpeech.md)
- [Gigaspeech [24]](../../Datasets/2021.06.13_GigaSpeech.md)
- [VCTK [202]](../../Datasets/2012.08.00_VCTK.md)
- [LJSpeech [88]](../../Datasets/2017.07.05_LJSpeech.md)
- [Common Voice [8]](../../Datasets/CommonVoice.md)

对于音频描述, [Wavcaps [147]](../../Datasets/WavCaps.md) 经常被使用.
一些语音数据集需要 ASR 模型的转录才能生成相应的文本.

在这一阶段, 重点放在了捕获和生成音频特征, 并将它们与文本在向量空间中对齐, 而不是关注对话功能.
因此, 数据通常由单通道音频组成, 可以在重采样后使用.
值得注意的是, 在一些工作中, 确保文本 Token 和音频 Token 之间的词级对齐至关重要 (如 Spirit-LM, Moshi, 和 OmniFlatten), 这可以通过类似 Whisper-timestamped 包或其他对齐工具实现.
在 Moshi 中为了防止灾难性以往, 训练时间的一半用于文本数据, 强调了在训练中平衡文本和音频数据的重要性.

### 6.1.3·Training Resources about Post-Train for Dual-Stream Dialogue Processing: 双流对话处理后训练资源

<details>
<summary>展开原文</summary>

To ensure that the model possesses the ability to “listen while speaking”.
Most research such as [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md) and [OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md) has implemented a dual audio-stream model: one audio stream generates model output, while the other captures user audio.
The objective of this training phase is to enable the model’s dual-stream processing without requiring complex human-computer interaction modeling.
Consequently, text dialogue data can be converted to speech and processed into dual-track audio format.
However, text dialogue data typically contains content unsuitable for TTS conversion to speech (such as code, formulas, URLs) or long, formal dialogue passages that do not align with spoken language, as real dialogue is often more concise.
Therefore, when synthesizing from text dialogue data, it is necessary to preprocess the text data.
High-quality, open-source text dialogue data is first collected, including datasets like [Alpaca [144]](../../Datasets/Alpaca.md), Moss, BelleCN, [ultraChat [46]](../../Datasets/ultraChat.md), and [Open-Orca [124]](../../Datasets/Open-Orca.md).
To ensure suitability for speech synthesis (TTS), heuristic rules are applied to filter out samples with high proportions of non-text elements (such as code, mathematical expressions), samples exceeding 200 words, and samples containing rare symbols.

After filtering the text, TTS models ([CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)) are used to synthesize speech for each turn in the dialogues.
For consistent voice effects, the model audio stream maintains a uniform voice, while the user audio stream is sampled with varied voices to enhance the model's robustness.
The synthesized dialogue audio is arranged using simulation strategies to achieve natural timing, such as turn-taking, well-timed interruptions, and pauses to maintain fluency and naturalness.
The final dialogue audio is organized in dual-channel format: the conversation begins with a user utterance, followed by alternating user and assistant turns.
After each user turn, the assistant responds immediately; upon completion of the assistant’s turn, a sampled pause length is introduced to simulate the natural rhythm of alternating dialogue.
To better simulate real scenarios, further data augmentation can be applied.
For example, random gain adjustments can be applied to the user audio stream, and background noise randomly selected from datasets like [MUSAN [194]](../../Datasets/2015.10.28_MUSAN.md) and [DNS [174]](../../Datasets/DNS.md) can be added to the user audio channel (OmniFlatten).
To simulate echo effects from a user’s microphone, portions of the audio stream can be scaled down and added to the user’s audio stream with random delays between 100 to 500 milliseconds, along with reverberation-like enhancements, helping the model adapt to real-world environments.

</details>
<br>

为了确保模型具有 "说话时听取" 的能力, 大多数研究都实现了双音频流模型: 一个音频流生成模型输出, 另一个捕获用户音频.
这一训练阶段的目标是使模型的双流处理能力无需复杂的人机交互建模.
因此, 文本对话数据可以转化为语音并处理为双轨音频格式.
然而, 文本对话数据通常包含不适合通过 TTS 转换为语音的内容 (如代码, 公式, 链接) 或长正式对话段落, 不能与口语语言对齐, 因为真正的对话往往更简洁.
因此, 当从文本对话数据合成语音时, 预处理文本数据是必要的.

首先收集高质量的开源文本对话数据, 包括 [Alpaca [144]](../../Datasets/Alpaca.md), Moss, BelleCN, [ultraChat [46]](../../Datasets/ultraChat.md), 和 [Open-Orca [124]](../../Datasets/Open-Orca.md).
为了确保适合进行语音合成, 启发式规则被应用于过滤掉具有高比例非文本元素的样本 (如代码, 数学表达式), 长度超过 200 个单词的样本, 以及包含罕见符号的样本.

经过过滤的文本, 使用 TTS 模型 ([CosyVoice [49]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)) 合成每一轮对话的语音.
为了保持一致的语音效果, 模型音频流保持统一的声音, 而用户音频流采用不同声音采样, 以增强模型的鲁棒性.
合成的对话音频使用仿真策略排列, 以实现自然的时机变化, 如轮次交换, 适当的中断, 以及停顿以保持流畅性和自然度.

最终的对话音频以双通道格式组织: 对话开始时, 用户发出一句话, 随后交替出现用户和助手的轮次.
在每个用户轮次之后, 助手立即回应; 助手完成一轮之后, 引入随机暂停长度, 模拟交替对话的自然节奏.

为了更好地模拟真实场景, 可以进行进一步的数据增强.
例如随机增益调整可以用于用户音频流, 背景噪声可以随机从数据集选择 (如 [MUSAN [194]](../../Datasets/2015.10.28_MUSAN.md) 和 [DNS [174]](../../Datasets/DNS.md)), 并添加到用户音频通道 (OmniFlatten).
为了模拟来自用户麦克风的回声效果, 音频流的一部分可以缩小, 并与用户音频流相加, 随机延迟在 100 到 500 毫秒之间, 并添加类似于混响的增强, 使模型适应真实环境.

### 6.1.4·Training resources about Enhancing Conversational Abilities and Instruction Tuning: 增强对话能力和指令微调资源

While the foundational model has been established, there remains a gap between this and a complete dialogue system.
The above model utilizes non-overlapping dialogue audio, where one party remains silent while the other speaks, failing to fully simulate real conversational dynamics.
Some speech datasets, such as [Generative Expressive Conversational Speech Synthesis [137]](../../Datasets/2024.07.31_NCSSD.md) and Fisher, contain dialogues from real-world settings, providing a basis for modeling interruptions and backchannels scenarios in voice dialogue systems.

Currently, there is no suitable dataset for real-world speech instructions.
Most approaches use synthetic methods based on text instruction data to perform **instruction tuning** in this stage.
Common text instruction datasets include Open Hermes and moss-002-sft-data, though they face similar challenges as text dialogue data, such as unsuitability for TTS conversion and inconsistency with spoken language conventions.
Following the synthetic processes provided by Moshi and Llama-Omni, this aims to generate instruction data in the format of (SpeechInstruction, TextInstruction, TextResponse, SpeechResponse).

The first method is synthetic generation from scratch.
Contexts and summaries are first generated by sourcing high-quality text data from sources like Wikipedia and StackExchange, producing thematic paragraphs as the dialogue foundation, referred to as “context.” Based on these contexts, dialogue summaries are generated.
Next, a specific prompt template guides the generation of complete dialogues, including context and requesting dialogues around the theme with roles as user and system.
The model is prompted to exhibit knowledge on the topic and include interruptions (backchannels) and brief turn-taking, simulating the natural flow of conversation.
To enhance dialogue diversity, additional instructions involving speech emotion and role-playing can be generated, requesting dialogues in specific tones or styles.
Furthermore, dialogues containing spelling errors or misinformation are synthesized to train the system in handling scenarios where user clarification or repetition is required.
Single-turn interactions on basic mathematics, grammar, and factual questions are also generated to ensure the system can handle simple factual tasks.
Finally, scenarios involving ethical or NSFW requests are created to train the system in declining to answer under such conditions.

The second method involves filtering and refining existing text instruction datasets.
Initially, open-source text language models paraphrase text instructions to match spoken language traits, adding fillers like “uh” and “um” to mimic natural speech tone, while converting numbers and symbols into spoken language to ensure the instructions are concise and conversational.
Generated text responses are also optimized to meet TTS output requirements, removing lengthy expressions and complex grammatical structures to make content clear and concise for TTS output.
After adjusting the instruction and response text, a TTS system converts the text to audio.

## 6.2·Evaluation: 评估

Fair and comprehensive evaluation of spoken dialogue models presents a multifaceted challenge.
On the one hand, the field of spoken dialogue still lacks publicly available test sets, comprehensive evaluation metrics, and established benchmarks.
On the other hand, assessing the performance of spoken dialogue systems requires consideration from multiple perspectives.
Basic aspects include the quality of generated speech, robustness, dialogue naturalness and accuracy, as well as response speed and generation time.
Beyond these, more advanced evaluations are needed to assess multi-turn dialogue capabilities (such as long-form speech editing), interaction abilities, and the system's proficiency in audio and music understanding and generation.
Given these requirements, and in line with the comprehensive expectations for spoken dialogue systems outlined in Section \ref{section21}, we will evaluate these systems from two angles: common evaluations and advanced evaluations.
Specifically, we will assess eleven key factors: speech generation quality, text intelligence, speech intelligence, audio and music generation, audio and music understanding, multilingual capability, context learning, interaction capability, streaming latency, multimodal capability, and the safety of dialogue systems.
Finally, we will list the current benchmarks and summarize the common conclusions derived from them.

### 6.2.1·Common Evaluation: 常规评估

#### Text Intelligence: 文本智能

As shown in Fig.04 (a), text intelligence refers to the fundamental understanding and generation capabilities of the spoken dialogue model.
When evaluating text intelligence, the focus is solely on the semantic content generated by the model, without considering other aspects such as timbre, emotion, or style.
In practical evaluations of this kind, some spoken dialogue models output only text ([LLaSM [119]](../../Models/SpeechLM/2023.08.30_LLaSM.md); [SALMONN [198]](../../Models/SpeechLM/2023.10.20_SALMONN.md); [Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md); [Qwen2-Audio [33]](../../Models/SpeechLM/2024.07.15_Qwen2-Audio.md); [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md)), while others generate both text and speech ([Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [Mini-Omni [222]](../../Models/SpeechLM/2024.08.27_Mini-Omni.md); [Mini-Omni2 [223]](../../Models/SpeechLM/2024.10.15_Mini-Omni2.md)), or only speech ([OmniFlatten [246]](../../Models/SpeechLM/2024.10.23_OmniFlatten.md)).
Regardless of the output format, we are concerned only with the generated text or the transcribed text from the speech when evaluation the text intelligence in the spoken dialogue models.
There are typically two categories of metrics and benchmarks used to assess text intelligence, MT-Metrics and Acc-Metrics.
The details are outlined as follows:

##### ACC-Metrics: 准确率度量

A common approach to evaluating text intelligence is to use benchmarks typically [197] [125] [239] [38] [181] [26] [255] [153] [215] [58] ~\cite{talmor2018commonsenseqa, liang2022holistic, zellers2019hellaswag, clark2018think, sakaguchi2021winogrande, chen2021evaluating, zhong2023agieval, mishra2021cross, wang2022super, feng2022mmdialog} employed for large language models, such as the classic [MMLU [75]](../../Evaluations/2020.09.07_MMLU.md) and GSM-8K [39] \cite{cobbe2021training}.
These benchmarks often include complex multiple-choice questions, which assess the model's reasoning abilities through Acc-Metrics.
Acc-Metrics refers to metrics that measure recognition accuracy, such as accuracy, F-score, and Mean Average Precision (mAP).
It is noteworthy that these benchmarks often evaluate the text-based intelligence of spoken dialogue models from various perspectives.
For example, [MMLU [75]](../../Evaluations/2020.09.07_MMLU.md) and GSM-8K \cite{cobbe2021training} are more focused on LLM's core knowledge, Flan \cite{longpre2023flan, wei2021finetuned} and Self-instruct \cite{wang2022self} are more focused on LLM's instruction following capability, CoQA \cite{reddy2019coqa} [175] and OpenAssistant  [112] \cite{kopf2024openassistant} are more focused on LLM's conversational capability.
These benchmarks often contain questions and corresponding answers.
Most of these questions are close-ended questions with short answers, so that they can have good generalization ability, any model that can generate text answers can be evaluated with these benchmarks and accuracy and F-Score can be easily adopted as the evaluation metrics.

##### MT-Metrics: 机器翻译度量

With the development of the LLMs, LLMs can follow instructions to accomplish many complex problems, so the scope of the evaluation was further expanded to include open-ended questions.
These open-ended questions often lack standard answers, therefore it's difficult to measure them by common ACC-Metrics.
A common approach is to measure the grammatical similarity between generated and reference utterances using the metrics used to measure grammatical similarity in mechanical translation (e.g.
[BLEU [161]](../../Evaluations/BLEU.md), [METEOR [13]](../../Evaluations/METEOR.md), ROUGE [126] \cite{lin2004rouge}).
We collectively refer to these evaluation metrics as \textbf{MT-Metrics}.
However, these metrics have certain limitations since one meaning has many different ways to convey.
So there are some metrics like BertScore [247] \cite{zhang2019bertscore} focus on evaluating the semantic similarity between two sentences.
And there are also been some methods utilizing LLM to judge the effectiveness of the responses which focusing on human preference \cite{zheng2023judging, liu2023g}. [252] [138]
The results of these large model-based especially GPT4o-based ratings of evaluation metrics demonstrated a high degree of correlation with human.

#### Speech Quality: 语音质量

Speech quality is one of the fundamental aspects for evaluating the performance of spoken dialogue systems, as it is closely tied to the experience of users.
There are two common dimensions for assessing speech quality: the clarity and naturalness (expressiveness and prosody) of the generated audio, and the robustness of the generated speech, such as the presence of missing or extra words.
The former is typically evaluated by using subjective MOS (Mean Opinion Score) ratings, while the latter is commonly assessed by using WER (Word Error Rate) or CER (Character Error Rate) metrics.

#### Streaming Latency: 流式延迟

In addition to evaluating the quality of text understanding and generated speech, the speed at which a spoken dialogue system generates speech responses is also crucial.
This necessitates the ability to stream both the comprehension and generation of speech in real time, achieving an effect of generating speech while speaking ([IntrinsicVoice [248]](../../Models/SpeechLM/2024.10.09_IntrinsicVoice.md); [Moshi [44]](../../Models/SpeechLM/2024.09.17_Moshi.md); [LLaMA-Omni [57]](../../Models/SpeechLM/2024.09.10_LLaMA-Omni.md)).
To assess the streaming performance of a model, one typically measures the time taken to generate the first token of speech (i.e., the waiting time after the user finishes speaking) and calculates the overall Real-Time Factor (RTF) of the spoken dialogue model's response.
The RTF value is obtained by dividing the total duration of the speech segment generated by the model by the time taken by the model to generate that response.

### 6.2.2·Advanced Evaluation: 高级评估

#### Speech Intelligence: 语音智能

Evaluating the speech intelligence of spoken dialogue systems is one of the key aspects.
The definition of speech intelligence in spoken dialogue systems is discussed in detail in Section \ref{section212}.
Given that speech intelligence encompasses a wide range of application scenarios, we address the evaluation separately for the understanding and generation components during the assessment.

##### Understanding: 理解

Ordinary cascaded spoken dialog models based on ASR getting text input will loss many paralinguistic information like speaking style, accent, emotion, etc.
Thus many spoken dialogue models ([E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md); [ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md); [Spoken-LLM [127]](../../Models/SpeechLM/2024.02.20_Spoken-LLM.md)) devoted into helping dialog models understand the paralinguistic information.
Evaluating this capability can start from two aspects: a) the accuracy of the paralinguistic information's understanding, b) the ability of \textbf{automatically} generating appropriate and coherent content responses and acoustic information based on the varying acoustic input.
\textbf{For the former}, since the classes of the paralinguistic information are always limited, for example, sentiments are generally categorized as neural, negative, positive.
So the researchers always use Accuracy or F-Score to evaluate the models' paralinguistic information understanding capability.
Recently, there are many studies 66 19 167 \cite{goel2024audio, busso2008iemocap, poria2018meld, [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md); [Spoken-LLM [127]](../../Models/SpeechLM/2024.02.20_Spoken-LLM.md); firdaus2020meisd, busso2016msp} 59 20 available for researchers to use in identifying speech emotions in the dialogue scenes.
In addition to recognizing speech emotions, recent benchmarks 7 234 \cite{ao2024sd, yang2024air} has also begun to investigate the influence of speaker age, accent, and other factors on the evaluation of spoken dialogue models.
\textbf{For the latter}, recent work ([E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md)) has increasingly focused on the possibility of generating appropriate content responses based on acoustic information from the input.
The current evaluation methods usually transcript the output audio into text through Automatic Speech Recognition and then evaluate the relevance between generated content and the reference content in the internal dataset.
Evaluations are usually conducted in text, so commonly used evaluation metrics are as the same as in the section \ref{eval_text}, like BLEU and METEOR, which are used to measure the similarity between two sentences.
Currently, there is limited research exploring whether spoken dialogue models can autonomously generate appropriate acoustic responses based on varying acoustic information, making it a promising area for future investigation.

##### Generation: 生成

In the generation component, evaluating the speech intelligence of spoken dialogue systems primarily focuses on controllability, i.e., the ability of the dialogue model to respond in a user-specified style and timbre in the zero-shot scenarios.
There are various dimensions to assess style, such as pitch, speech rate, energy, emotion, and accent, among others.
ACC-metrics can be used to evaluate whether the spoken dialogue model can generate speech in the desired style.
Additionally, the evaluation of voice cloning capabilities within the model can borrow metrics from the zero-shot TTS domain~\cite{[VALL-E [209]](../../Models/SpeechLM/2023.01.05_VALL-E.md); shen2023naturalspeech,ji2024mobilespeech,wang2024ham} 189 91 210, using speaker similarity indices ([WavLM [27]](../../Models/SpeechRepresentation/2021.10.26_WavLM.md))
Currently, there are few models that explore the generation of speech intelligence in spoken dialogue systems, and this area warrants further refinement and exploration in future work.

#### Audio Understanding and Generation: 音频理解与生成

In real-world scenarios, the broader definition of speech modality encompasses not only clear human speech but also a wide range of natural sounds such as dog barking and bird chirping, all of which can be considered forms of audio.
Evaluating the ability of spoken dialogue models to understand and generate such audio is a critical aspect of assessing the model’s performance.

##### Audio Understanding: 音频理解

On the audio comprehension side, various sub-tasks are commonly employed to measure a system's capacity to understand audio, including tasks such as Audio Captioning (AudioCap) 105 \cite{kim2019audiocaps}, Sound Event Detection (SED) 152 \cite{mesaros2021sound}, audio classification, and audio-motivated creative writing ([Qwen-Audio [34]](../../Models/SpeechLM/2023.11.14_Qwen-Audio.md)), among others.
The core of these tasks lies in evaluating the model’s ability to process and interpret the complex acoustic information embedded within the audio.
For tasks like audio classification and SED, which involve fixed outputs, evaluation is relatively straightforward, typically using objective metrics such as accuracy or Mean Average Precision (mAP).
However, for the AudioCap task, the problem is generally open-ended, meaning there are no fixed answers.
As a result, existing evaluation methods are primarily based on measuring the similarity between the generated text and the reference text, using traditional metrics such as [BLEU [161]](../../Evaluations/BLEU.md) and [METEOR [13]](../../Evaluations/METEOR.md), or newer evaluation approaches involving large language models such as GPT-4o \cite{zheng2023judging}. 252
In the case of audio-motivated creative writing, where the objective is to generate inventive descriptions from a given audio input, evaluation typically relies on subjective measures, given the divergent nature of the creative process involved.

##### Audio Generation: 音频生成

Additionally, on the audio generation side, producing high-quality audio should be considered an advanced capability for a conversational spoken dialogue model.
However, as most current spoken dialogue systems lack the ability to generate audio, this remains an area for further exploration in the future end-to-end spoken dialogue systems.
The evaluation of generated audio can draw from methods used in the text-to-audio domain~\cite{huang2023make2,[Make-An-Audio [83]](../../Models/Diffusion/2023.01.30_Make-An-Audio.md)}.81
Typically, such evaluations focus on the quality of the generated audio itself, using metrics such as Mean Opinion Score (MOS) and the similarity between generated and target audio.
Objective evaluation metrics for audio similarity often include Fréchet Distance (FD), Inception Score (IS), Kullback-Leibler (KL) divergence, Fréchet Audio Distance (FAD), and CLAP score.
Specifically, Fréchet Audio Distance (FAD) 104 \cite{kilgour2018fr} is adapted from the Fréchet Inception Distance (FID) to the audio domain and serves as a reference-free perceptual metric that quantifies the distance between the generated and ground-truth audio distributions.
The Inception Score (IS) is an effective metric that evaluates both the quality and diversity of generated audio.
KL divergence is computed at the paired sample level between generated and ground-truth audio, based on the label distribution and averaged to produce a final result.
Fréchet Distance (FD) evaluates the similarity between the generated and ground-truth audio distributions.
FD, KL, and IS are built upon the PANNs model \cite{kong2020panns} 110, which takes mel-spectrograms as input.
In contrast, FAD uses VGGish \cite{hershey2017cnn} 76 as an audio classifier, processing raw audio waveforms as input.
The CLAP score, adapted from the CLIP score 77 \cite{hessel2021clipscore}, is a reference-free metric used to assess audio-text alignment and strongly correlates with human perception.

#### Music Understanding and Generation: 音乐理解与生成

In advanced spoken dialogue models, the evaluation of music modality understanding and generation follows a methodology similar to that used for audio modality.
Unlike Audio Understanding, which only requires a general description of the events that occur in the audio, Music Understanding requires appreciating the style and genre of music, understanding its keys, themes, and other rich information.
For classification, emotion recognition tasks in music, common metrics such as accuracy can be used.
For music captioning task, MusicCaps ([MusicLM [2]](../../Models/SpeechLM/2023.01.26_MusicLM.md)) offers a general dataset for evaluating a model's music understanding capability.
For music analysis, Nsynth \cite{engel2017neural} [56] provides rich note data information.
In terms of evaluation for music generation, subjective Mean Opinion Score (MOS) assessments or measures of similarity between generated music and target music are commonly used.

#### Multilingual Capability: 多语言能力

The ability to speak multiple languages is also required for a spoken dialogue model, but most current models ([LTU [68]](../../Models/SpeechLM/2023.05.18_LTU.md); [Spoken-LLM [127]](../../Models/SpeechLM/2024.02.20_Spoken-LLM.md); [ParalinGPT [128]](../../Models/SpeechLM/2023.12.23_ParalinGPT.md); [LLaSM [119]](../../Models/SpeechLM/2023.08.30_LLaSM.md); [BLSP [208]](../../Models/SpeechLM/2023.09.02_BLSP.md); [E-chat [227]](../../Models/SpeechLM/2023.12.31_E-chat.md)) only focus on English and Chinese.
A naive idea is to directly evaluate spoken dialogue models' capability in speech-to-speech or speech-to-text translation tasks \cite{ jia2022cvss,wang2020covost1}. 94 206
These evaluations can be done with common machine learning metrics like [BLEU [161]](../../Evaluations/BLEU.md) or BertScore \cite{zhang2019bertscore}. 161 147
However, evaluating the capability of translation is insufficient to measure the model's multilingual conversational ability, and further exploration is still needed in this area of evaluation.
Explicitly requiring a spoken dialogue model to perform speech translation is not a typical use case in conversational scenarios.
In most cases, when a user asks a question in a different language or with a distinct accent, the model is expected to automatically respond in the same language that the user is using.
In this context, it seems more reasonable to evaluate the accuracy of the model’s generated speech in terms of language identification, combined with subjective human assessments, as a more intuitive and appropriate evaluation method.

#### Context Learning: 上下文学习

The context learning capability is crucial for maintaining the coherence of an entire conversation.
Similar to a memory function, the challenge lies in how to preserve this capability when relying solely on speech.
Typically, the evaluation of a spoken dialogue model's context learning ability depends on specific long-duration dialogue test sets, after which standard MT-Metrics or Acc-Metrics used in text intelligence evaluations can be applied.
For instance, a model's context learning capability can be assessed by evaluating its QA performance based on the given context \cite{lipping2022clotho}.132
However, it is important to note the relevance of editing scenarios in long-duration spoken dialogues.
In real spoken dialogue scenarios, the users will modify some certain key information, the model needs to promptly understand and respond accordingly, e.g., the users offer wrong information for solving a problem and modify the condition in the next dialog.
So how to evaluate the model's online understanding ability is still needed further study.

#### Interaction Capability: 交互能力

Interactive ability is also an essential metric for assessing the advanced capabilities of spoken dialogue systems.
As illustrated in Fig.04 (b), basic interactive ability refers to the system's capacity to allow users to interrupt the conversation at any time.
In this context, it is crucial to evaluate whether the spoken dialogue model can promptly comprehend the user's new input and halt its current response.
This is commonly measured using accuracy.
Furthermore, it is important to assess whether the model can generate a coherent and appropriate response based on the new input, which ties back to previous evaluation standards related to text and speech intelligence.

In addition, in real-world scenarios, beyond basic interruptions, various discourse markers such as "okay", "haha" are often used to indicate interaction.
Current spoken dialogue systems ([dGSLM [157]](../../Models/SpeechLM/2022.03.30_dGSLM.md)) typically track the frequency of these markers as a standard evaluation metric.
Looking ahead, it may be valuable to assess whether future spoken dialogue models can effectively and appropriately interrupt human speakers, which could also represent a key dimension for evaluation the interaction capability.

#### Multimodal Capability: 多模态能力

Spoken dialogue models primarily focus on the audio modality for both input and output.
However, considering the close coupling between video and audio modalities in practical applications of dialogue systems, recent advancements in spoken dialogue models have incorporated the understanding of video and images in the input stage~\cite{[VITA [61]](../../Models/SpeechLM/2024.08.09_VITA.md); [Ocean-Omni [122]](../../Models/SpeechLM/2024.10.11_Ocean-Omni.md); park2024let} , indicate that future spoken dialogue models need to simultaneously understand visual information and audio information to achieve real-time Audio-Visual Understandings.
The evaluation of such models generally still focuses on the evaluation of dialogue quality, that is, whether the generated dialogue and the reference dialogue are similar.
Therefore, this aspect can still be evaluated using metrics such as [BLEU [161]](../../Evaluations/BLEU.md) and [METEOR [13]](../../Evaluations/METEOR.md) to assess sentence semantic similarity.
However, research in this area also focuses on the understanding of visual information, and how to evaluate the model's correct understanding of real-time visual information in dialogue is also a difficulty, still can be a future benchmark direction.

#### Security: 安全性

Security is also an integral part of the evaluation, how to ensure that the output of the model complies with ethical and social norms is a critical aspect.
Spoken dialogue models may encounter security issues such as harmful content generation, privacy pitfalls, bias, and adversarial attacks.
There has been considerable research progress in evaluating text modalities \cite{dong2024attacks}.
The commonly used metric is to evaluate the attack success rate of injection attacks and so on.
However, there are relatively few evaluation methods in the field of speech modality.
How to construct a dataset for attacking spoken dialogue models, avoid poisoning of speech data, and evaluate the model's speech defense capabilities as benchmarks are required further research in the field of spoken dialogue model evaluation in the future.

## 6.3·Benchmark: 基准

We list the common benchmarks for evaluating voice dialogue systems in the table\ref{table:benchmark}, and briefly introduce each benchmark in this section.

#### VoiceBench

[VoiceBench [29]](../../Evaluations/2024.10.22_VoiceBench.md)'s Key evaluation dimensions include general knowledge, instruction-following ability, and safety compliance.
The benchmark incorporates both synthetic and real spoken instructions to simulate diverse speaker styles, environmental conditions, and content variations.
It challenges systems with tasks involving accent adaptability, handling noisy environments, and robustness against content irregularities such as grammatical errors, disfluencies, and mispronunciations.
Additionally, it explores the systems' resilience under varying speaker characteristics (age, pitch, and speaking speed) and environmental challenges like reverberation, background noise, and far-field effects.

#### SUPERB

The benchmark [SUPERB [235]](../../Evaluations/2021.05.03_SUPERB.md) evaluates speech processing models across multiple dimensions, including content recognition, speaker modeling, semantic understanding, and paralinguistic analysis.
Tasks in content recognition cover phoneme recognition, automatic speech recognition, keyword spotting, and query-by-example spoken term detection, focusing on transcription and content detection accuracy.
Speaker modeling involves tasks like speaker identification, automatic speaker verification, and speaker diarization to assess speaker-related features.
Semantic understanding includes intent classification and slot filling, testing models' ability to infer high-level meaning directly from raw audio.
Paralinguistic analysis focuses on emotion recognition, capturing models' ability to interpret affective cues from speech.
The evaluation framework uses publicly available datasets and conventional metrics to provide a standardized testbed for assessing generalizability and task-specific performance.

#### AudioBench

[AudioBench [205]](../../Evaluations/2024.06.23_AudioBench.md) evaluates spoken dialogue models across three primary dimensions: speech understanding, audio scene understanding, and voice (paralinguistic) understanding.
It encompasses eight distinct tasks and leverages 26 datasets, including seven newly developed datasets.
The evaluation emphasizes models' ability to handle instruction-following tasks conditioned on audio signals, addressing aspects such as speech recognition accuracy, environmental sound interpretation, and paralinguistic feature extraction (e.g., emotion, gender, accent).


#### AirBench

AIR-Bench~\cite{yang2024air} assesses the capabilities of Spoken dialogue models to understand and interact based on various audio types, including human speech, natural sounds, and music.
It consists of two primary components: a foundation benchmark with 19 specific audio tasks and over 19,000 single-choice questions, and a chat benchmark featuring more than 2,000 open-ended audio-prompted questions.
The foundation benchmark evaluates fundamental skills such as speech recognition, acoustic scene classification, and music genre identification, focusing on specific subtasks to diagnose model weaknesses.
The chat benchmark tests the models' ability to handle complex, real-world audio-based queries, including mixed audio with varying loudness and temporal offsets.
AIR-Bench introduces a novel audio mixing strategy to simulate complex real-world scenarios and employs GPT-4-based evaluation to judge model-generated hypotheses against reference answers.

#### SpokenWOZ

SpokenWOZ~\cite{si2024spokenwoz} evaluates task-oriented dialogue (TOD) systems in spoken scenarios, addressing challenges unique to spoken conversations, such as incremental processing, disfluencies, incomplete utterances, and Automatic Speech Recognition (ASR) noise.
It introduces novel metrics to assess performance in tasks like cross-turn slot detection and reasoning slot detection, which require integrating information across multiple turns and reasoning from implicit cues.
The benchmark encompasses multi-domain, human-to-human dialogues with diverse speech characteristics, testing systems on both textual and auditory inputs through large-scale annotated datasets with over 200,000 utterances and 249 hours of audio

#### SD-EVAL

SD-Eval~\cite{ao2024sd} evaluates spoken dialogue models across multiple dimensions, focusing on both spoken understanding and response generation beyond textual content.
It assesses models' abilities to process three key types of information embedded in speech: content (e.g., linguistic meaning), paralinguistic cues (e.g., emotion, accent, age), and environmental context (e.g., background sounds).
The benchmark consists of four sub-tasks—emotion, accent, age, and environment—constructed from diverse datasets and totaling 7,303 utterances spanning 8.76 hours.

#### SuperCLUE

SuperCLUE evaluates spoken dialogue systems across four main dimensions: voice interaction, general capabilities, scenario applications, and response speed.
Key metrics include interruption recognition, speech tone adjustment, semantic understanding, naturalness of speech, and memory accuracy.
Additionally, it measures real-time data retrieval, reasoning ability, compliance with commands, and multilingual translation accuracy.
Scenario-specific applications like emotional counseling, health consultations, and customer service are assessed for precision and effectiveness.
The final aspect is response timeliness, focusing on latency and delay management.However, this benchmark is not open source and focuse on Mandarine ability


#### MMAU

MMAU~\cite{sakshi2024mmau} evaluates spoken dialogue models across multiple dimensions, encompassing 27 distinct tasks divided into reasoning and information extraction categories.
It assesses models on their ability to comprehend and reason about speech, sound, and music by leveraging advanced cognitive skills and domain-specific knowledge.
Key evaluated areas include temporal event reasoning, speaker role mapping, emotional tone interpretation, eco-acoustic knowledge, phonemic stress pattern analysis, and melodic structure interpretation.
It examines not just basic recognition or transcription capabilities but also models' proficiency in complex reasoning, contextual understanding, and the ability to extract and apply world knowledge.
Additionally, MMAU scrutinizes performance consistency across varying difficulty levels, testing systems' depth of reasoning and robustness in real-world audio scenarios.
