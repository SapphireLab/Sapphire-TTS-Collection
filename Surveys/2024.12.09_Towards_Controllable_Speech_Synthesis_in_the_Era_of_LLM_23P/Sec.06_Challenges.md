# 6·Challenges and Future Directions: 挑战与未来方向

<details>
<summary>展开原文</summary>

In this section, we elaborate on current challenges for fully controllable TTS and discuss promising future directions.

</details>
<br>

在本节中, 我们详细说明完全可控 TTS 的当前挑战并讨论有望的未来方向.

## A·Challenges: 挑战

<details>
<summary>展开原文</summary>

Controllable TTS aims to synthesize speech while allowing precise control over speech characteristics such as pitch, duration, energy, prosody, speaking style, and emotion.
While significant progress has been made, achieving truly controllable TTS remains a complex task due to the multifaceted nature of human speech and the technical challenges in modeling and synthesizing it.
In this section, we delve into the primary challenges and analyze their underlying reasons.

</details>
<br>

可控 TTS 的目标是合成语音的同时允许对语音特征进行精确控制, 如音高, 时长, 能量, 韵律, 说话风格, 以及情感.
虽然已经取得了显著进展, 但实现真正可控 TTS 仍然是一个复杂的任务, 因为人类语音的多面性和模型和合成技术的困难性.
在本节中, 我们深入分析主要的挑战及其潜在原因.

### Controllable Granularity: 控制粒度

<details>
<summary>展开原文</summary>

A critical challenge in controllable TTS is determining what aspects of speech should be controlled and how to control speech characteristics at a specific granularity.
Different applications require varying levels of control granularity.
For instance, audiobook narration may need sentence-level control of emotion, while conversational AI like ChatGPT may require word or phoneme-level control over prosody.
Moreover, the emotion, prosody, and other characteristics of human speech are often intricately intertwined and can manifest across varying levels of granularity.
Additionally, achieving fine-grained control requires high-resolution annotations and sophisticated models capable of handling subtle variations without compromising synthesis quality.

Although some LLM-based TTS methods such as [VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md) can control various aspects of speech through attribute descriptions, determining the appropriate level of granularity for control and devising methods to achieve precise control at a specific granularity or to enable multiscale and fine-grained control remains a significant challenge.

</details>
<br>

可控 TTS 的关键挑战之一是**确定语音的什么方面应该被控制**, 以及**如何以特定粒度控制语音特征**.

不同应用需要不同级别的控制粒度.
- 有声书的叙述可能需要句子级的情感控制;
- 对话式 AI 如 ChatGPT 可能需要词或音素级的韵律控制;

此外, 人类语音的情感, 韵律, 以及其他特征通常是复杂而相互交织的, 可以在不同粒度上表现出来.

尽管一些基于 LLM 的 TTS 方法可以通过属性描述来控制语音的各种方面 (如 [VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md)), 但确定合适的控制粒度级别, 设计精确控制在具体力度的方法和/或实现多尺度和细粒度控制仍然是一个重要的挑战.

### Feature Extraction & Representation: 特征提取与表示

<details>
<summary>展开原文</summary>

Achieving fully controllable TTS needs good feature disentanglement.
Accurately extracting meaningful and disentangled speech features like pitch contours, energy patterns, emotion variation, and prosodic elements from training data is difficult.
The reason is that speech features are interdependent and context-sensitive, making it hard to isolate specific attributes for control.
For example, altering pitch often affects prosody, emotion, and naturalness to some extent.
To tackle this, several methods [^269] [^270] [^271] utilize pre-trained models for different speech recognition tasks (e.g., pitch, energy, and duration prediction, gender classification, age estimation, and speaker verification) to supervise feature extraction.
For example, [NaturalSpeech3](../../Models/Diffusion/2024.03.05_NaturalSpeech3.md) factorizes speech into separate feature subspaces to capture different speech attributes.

However, these methods are limited to coarse or high-level feature disentanglement, leaving a significant gap in fully disentangled control.
On the other hand, selecting suitable representations (e.g., continuous variables like mel-spectrograms or latent embeddings like tokens) for controllable attributes is non-trivial because representations must be both interpretable for humans and expressive enough for TTS models.
For example, transformer-based models are good at processing discrete tokens, while GAN and Diffusion-based models excel in modeling continuous representations.

</details>
<br>
要实现完全可控 TTS 需要良好的特征解耦.
**要从训练数据中精确提取有意义且解耦的语音特征 (如音高轮廓, 能量模式, 情感变化, 韵律元素) 是困难的**.
原因是语音特征是相互依赖的, 且对上下文敏感, 使其难以单独控制特定属性.
- 例如, 改变音高往往会影响一定程度的韵律, 情感, 以及自然度.

为了解决这一问题, 一些方法 [^269] [^270] [^271] 使用用于不同语音识别任务 (如音高, 能量, 以及时长预测, 性别分类, 年龄估计, 以及说话人验证) 的预训练模型来监督特征提取.
- 例如, [NaturalSpeech3](../../Models/Diffusion/2024.03.05_NaturalSpeech3.md) 将语音分解到不同特征子空间, 以捕捉不同语音属性.

然而这些方法仅限于粗糙或高级别的特征解耦, 而离完全解耦的控制仍然存在巨大差距.

另一方面, **选择合适的表示 (即如梅尔频谱的连续变量或如 Token 的潜在嵌入) 来实现可控属性也是困难的**.
原因是表示必须对人类来说可理解, 对 TTS 模型又具有足够表达能力.
- 例如, 基于 Transformer 的模型擅长处理离散 Token, 而基于 GAN 和扩散的模型擅长处理连续表示.

[^269]: Disentangling Style and Speaker Attributes for TTS Style Transfer
[^270]: Generalizable Zero-Shot Speaker Adaptive Speech Synthesis with Disentangled Representations
[^271]: Effective and Direct Control of Neural TTS Prosody by Removing Interactions between Different Attributes

### Scarcity of Datasets: 数据集稀缺

<details>
<summary>展开原文</summary>

High-quality, diverse, and appropriately annotated datasets are essential for training controllable TTS systems.
However, such datasets are scarce and difficult to construct.
To achieve controllable TTS, training data must encompass a wide range of styles, emotions, accents, and prosodic variations to enable versatile control because limited diversity in datasets can restrict the model's ability to generalize across unseen styles or emotions.
Although there are some TTS datasets, such as [LibriTTS [272]](../../Datasets/2019.04.05_LibriTTS.md), [GigaSpeech [258]](../../Datasets/2021.06.13_GigaSpeech.md), and [TextrolSpeech [247]](../../Datasets/2023.08.28_TextrolSpeech.md), their diversity is still not enough for fully controllable TTS due to the lack of corpus of diverse content such as comedies, thrillers, cartoons, etc.
Constructing large-scale datasets with rich diversity is also expensive and time-consuming.

Another obstacle is that creating datasets with fine-grained, attribute-specific annotations is labor-intensive and costly.
Besides, manual annotation of speech attributes requires expert knowledge and is prone to inconsistencies and errors, particularly for subjective qualities like emotion.
Currently, most datasets provide only coarse labels, such as gender, age, or a limited range of emotions.
While some datasets, such as [SpeechCraft [262]](../../Datasets/2024.08.24_SpeechCraft.md), include natural language descriptions of speech attributes, no existing dataset offers fine-grained variations and annotations within the speech of the same speakers.
Available datasets for controllable TTS are summarized in Table.05.

</details>
<br>

高质量, 多样以及具有适当标注的数据集对于训练可控 TTS 系统至关重要.
然而, 这样的数据集是稀缺的, 且难以构造.
为了实现可控 TTS, 训练数据必须包含大范围的风格, 请感, 口音和韵律变化, 以实现多样化控制, 因为数据集的有限多样性会限制模型生成未见过的风格或情感的能力.

虽然存在一些 TTS 数据集, 如 [LibriTTS [272]](../../Datasets/2019.04.05_LibriTTS.md), [GigaSpeech [258]](../../Datasets/2021.06.13_GigaSpeech.md), 和 [TextrolSpeech [247]](../../Datasets/2023.08.28_TextrolSpeech.md), 但它们的多样性仍然不足以实现完全可控 TTS, 因为它们没有包含如喜剧, 悬疑, 卡通等丰富的内容.
制作具有丰富多样性的数据集也是昂贵且耗时.

另一个障碍是, 要构造具有细粒度, 属性特定的标注的数据集是耗时费力的, 且代价高昂.
除此之外, 手动标注语音属性要求专业知识, 且容易出现不一致和错误, 特别是对于情感等主观性质.
目前, 绝大多数数据集仅提供粗糙的标签, 如性别, 年龄, 或有限的情感.
虽然一些数据集, 如 [SpeechCraft [262]](../../Datasets/2024.08.24_SpeechCraft.md), 包含了自然语言描述的语音属性, 但目前还没有提供与同一说话者的语音具有细粒度的变化和标注的数据集.

可控 TTS 可用的数据集总结如表格 05.

|Dataset|Hours|#Speakers|Pitch|Energy|Speed|Age|Gender|Emotion|Emphasis|Accent|Top.|Description|Environment|Dialogue|Language|Release Time|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Taskmaster-1 [254]](../../Datasets/2019.09.01_Taskmaster-1.md)| / | /|  |  |  |  |  |  |  |  |  |  |  | √ | en|2019.09 |
|[Libri-Light [255]](../../Datasets/2019.12.17_Libri-Light.md)| 60,000 | 9,722 | | | | | | | | | √ | | | | en | 2019.12 |
|[AISHELL-3 [256]](../../Datasets/2020.10.22_AISHELL-3.md)| 85 | 218 | | | | √ | √ | | | √ | | | | | zh | 2020.10 |
|[ESD [257]](../../Datasets/2020.10.28_ESD.md)|29|10|  |  |  | |  | √ |  |  |  |  |  |  |en,zh |2021.05 |
|[GigaSpeech [258]](../../Datasets/2021.06.13_GigaSpeech.md)| 10,000 | / | |  |  |  |  |  |  |  | √ |  |  |  | en | 2021.06 |
|[WenetSpeech [259]](../../Datasets/2021.10.07_WenetSpeech.md)| 10,000 | / |  |  |  |  |  |  |  | |√ | | | | zh | 2021.07 |
|[PromptSpeech [101]](../../Datasets/PromptSpeech.md)| / | / | √ | √ | √ | |  | √ |  |  |  | √ |  |  | en | 2022.11 |
|[DailyTalk [260]](../../Datasets/DailyTalk.md)| 20 | 2|  |  |  |  |  | √ |  |  | √ |  |  | √ | en|2023.05 |
|[TextrolSpeech [247]](../../Datasets/2023.08.28_TextrolSpeech.md)| 330 | 1,324 | √ | √ | √ |  | √ | √ |  |  |  | √ |  |  | en  | 2023.08|
|[VoiceLDM [231]](../../Datasets/VoiceLDM_Data.md)| /| /| √ |  | |  | √ | √ |  |  |  | √ | √ |  |en | 2023.09|
|[VccmDataset [106]](../../Datasets/2024.06.03_VccmDataset.md)| 330| 1,324| √ | √ | √ |  | √ | √ |  |  |  | √ |  |  | en|2024.06|
|[MSceneSpeech [261]](../../Datasets/2024.07.19_MSceneSpeech.md)| 13 | 13 |  |  |  |  | |  |  |  | √ |  |  | | zh | 2024.07 |
|[SpeechCraft [262]](../../Datasets/2024.08.24_SpeechCraft.md)| 2,391 | 3,200 | √ | √ | √ | √ | √ | √ | √ | | √ | √ | | | en,zh | 2024.08 |

### Generalization Ability: 泛化能力

<details>
<summary>展开原文</summary>

The ability of a TTS system to generalize effectively is crucial for producing natural, high-quality speech across a wide range of conditions, such as unseen speakers, languages, or topics.
However, achieving robust generalization remains a significant challenge for modern TTS methods due to various factors.

Zero-shot controllable TTS ([MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); [^92]) aims to synthesize speech for unseen speakers with various speech customization such as emotion using minimal reference audio, which can offer flexibility for personalized voice generation.
However, it faces significant challenges, including capturing unique speaker characteristics from limited data, accurately reproducing prosody and style, and disentangling speaker identity from other audio attributes like emotion or noise.

Multilingual generalization ([XTTS [250]](../../Models/SpeechLM/2024.06.07_XTTS.md); [SANE-TTS [273]](../../Models/E2E/2022.06.24_SANE-TTS.md)) in TTS refers to the ability to synthesize natural and intelligible speech across multiple languages, including those not seen during training.
This capability is essential for applications like cross-lingual communication, multilingual virtual assistants, and speech synthesis for low-resource languages [^274].
Multilingual generalization still faces many challenges such as linguistic diversity and mismatch and the scarcity of data.
Cross-lingual speaker generalization is another hurdle, as preserving speaker identity across languages can lead to artifacts.

Domain adaptation [^275] in TTS refers to tailoring a pre-trained TTS model to generate speech for a specific domain or context, such as medical terminology and conversational speech.
One challenge is that many specialized domains lack sufficient high-quality annotated data for fine-tuning.
Besides, adapting prosody, intonation, and speaking style to match domain-specific requirements such as comic dialogue is complex.
Failing to capture domain-specific nuances can make speech sound unnatural or inconsistent with the target context.

</details>
<br>

[^92]: Zero-Shot Multi-Speaker Text-To-Speech with State-of-The-Art Neural Speaker Embeddings
[^274]: Low-Resource Languages: A Review of Past Work and Future Challenges
[^275]: A Brief Review of Domain Adaptation

TTS 系统的泛化能力对于在各种条件下生成自然高质量的语音十分重要, 如未见过的说话人, 语言, 或话题.
然而由于多种因素现代 TTS 方法在实现鲁棒泛化方面仍然面临重大挑战.

零样本可控 TTS ([MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); [^92]) 的目标是用最少的参考音频为未见过的说话人合成具有多种语音定制 (如情感) 的语音, 从而为个性化语音生成提供灵活性.
然而, 它面临重大挑战, 包括从有限数据中捕捉独特的说话人特征, 准确再现韵律和风格, 并从其他音频属性 (如情感或噪声) 中分离说话人身份.

多语言泛化 ([XTTS [250]](../../Models/SpeechLM/2024.06.07_XTTS.md); [SANE-TTS [273]](../../Models/E2E/2022.06.24_SANE-TTS.md)) 是 TTS 的一种能力, 它能够跨越多种语言, 包括在训练过程中未见过的语言, 生成自然且有意义的语音.
这一能力对于跨语言交流, 多语言虚拟助手, 以及低资源语言的语音合成等应用至关重要 [^274].
多语言泛化仍然面临诸多挑战, 如语言多样性和差异, 以及缺乏足够的数据.
跨语言说话人泛化也是另一个障碍, 因为跨语言保持说话人身份会导致伪影.

TTS 的领域自适应 [^275] 指的是调整预训练 TTS 模型来生成特定领域或上下文的语音, 例如医学术语和对话语音.
一个挑战是许多专业领域缺乏足够的高质量标注数据以进行微调.
此外, 调整韵律, 语调和说话风格以匹配领域特定要求 (如卡通对话) 也很复杂.
如果不捕捉领域特定细微差别, 则语音可能变得不自然或与目标上下文不一致.

### Efficiency: 效率

<details>
<summary>展开原文</summary>

Efficiency in controllable TTS systems is a critical requirement for practical applications, as these models aim to offer fine-grained control over various speech attributes such as prosody, emotion, style, and speaker identity.
However, achieving such control often comes at the cost of increased computational complexity, larger model sizes, and longer inference times, creating significant challenges.

High latency is a major issue, as existing controllable TTS models ([MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); [PromptTTS [101]](../../Models/Acoustic/2022.11.22_PromptTTS.md); [PromptTTS2 [102]](../../Models/Acoustic/2023.09.05_PromptTTS2.md); [VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md)) often necessitate autoregressive processes to synthesize speech.
The inference time of these models ranges from several to tens of seconds.
This can be especially problematic for real-time applications like live broadcasting or interactive systems.
Additionally, the challenge of balancing granularity and efficiency arises, as finer controls demand higher-resolution data and more precise models, leading to increased resource requirements and inefficient training and inference.

Another major obstacle lies in the trade-off between model complexity and performance.
State-of-the-art controllable TTS systems often rely on large neural networks such as LLMs with billions of parameters, which provide superior naturalness and expressiveness but demand significant computational resources.
Simplifying these architectures can lead to quality degradation, including artifacts, unnatural prosody, or limited expressiveness.
Therefore, designing light-weight controllable TTS models is significantly tricky.

</details>
<br>

可控 TTS 系统的效率对于实际应用是关键要求, 这些模型旨在提供对话式语音的细粒度控制, 如韵律, 情感, 风格, 以及说话人身份.
然而, 实现这种控制往往会导致计算复杂度增加, 模型大小增加, 以及推理时间增加, 这会带来巨大的挑战.

高延迟是主要问题, 如现有的可控 TTS 模型通常需要自回归过程来合成语音.
这些模型的推理时间范围从几秒到几十秒不等.
这对于实时应用 (如直播或交互式系统) 变得更是问题.
此外, 平衡粒度和效率的挑战也随之而来, 因为更细致的控制需要更高分辨率的数据和更精确的模型, 这会导致资源需求增加和训练推理效率低下.

另一个主要障碍是模型复杂度和性能之间的权衡.
SoTA 可控 TTS 系统通常依赖于大的神经网络, 如 LLMs, 它们提供卓越的自然度和表现力, 但却需要大量的计算资源.
简化这些架构可能会导致质量下降, 包括伪影, 不自然的韵律或表达力受限.
因此, 设计轻量可控 TTS 模型是十分复杂的.

## B·Future Directions: 未来方向

<details>
<summary>展开原文</summary>

In this survey, we conduct a comprehensive investigation and analysis of existing TTS methods, particularly on controllable TTS technologies.
While these methods show great potential in real-world applications, there are still some limitations that need to be addressed.
Based on our observations, we outline several promising future directions as follows:

</details>
<br>

在本调查中, 我们构造了对现有 TTS 方法的全面的调查和分析, 特别是可控 TTS 技术.
虽然这些方法在实时应用中展示出了巨大的潜力, 但仍有一些限制需要解决.
基于我们的观察, 我们提出了一些有前途的未来方向如下:

### Fine-Grained Speech Synthesis by Natural Language Description: 基于自然语言描述的细粒度语音合成

<details>
<summary>展开原文</summary>

Using natural language description to synthesize human speech with fine-grained control over various audio attributes is currently underexplored.
Most of the existing works can only control a fixed number of attributes of the synthesized speech.
Although a few works show great control of emotion, timbres, pitch, gender, and styles, e.g., [VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md) and [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md), they can frequently synthesize unwanted speech clips.
Users need to synthesize multiple times to get satisfactory speech.

</details>
<br>

使用自然语言描述来实现细粒度控制的音频属性并合成人类语音, 目前还处于空白地带.
现有的大多数工作只能控制合成语音的固定数量的属性.
虽然一些工作已经展示出了对情感, 音色, 音高, 性别, 风格等音频属性的精细控制, 如 [VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md) 和 [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md), 但它们往往会生成不想要的语音片段.
用户需要多次合成才能获得令人满意的语音.

### Fine-Grained Speech Editing by Natural Language Description: 基于自然语言描述的细粒度语音编辑

<details>
<summary>展开原文</summary>

Speech or audio editing has been studied for a long time.
However, existing methods usually train conditional models and adjust a fixed number of conditional inputs to modify the attributes of synthesized speech, thus lacking fine-grained manipulations [^94] [^95]
Therefore, how to learn disentangled speech representations for speech attributes while supporting editing by using natural language description is worthy of investigation.

</details>
<br>

语音或音频编辑已经研究了很长一段时间.
然而, 现有的方法通常训练条件模型并调整固定数量的条件输入来修改合成语音的属性, 这导致粒度不够细致的操作 [^94] [^95]
因此, 如何学习分离的语音表示来支持自然语言描述的语音属性编辑, 值得进一步研究.

[^94]: Editspeech: A Text Based Speech Editing System Using Partial Inference and Bidirectional Fusion
[^95]: Editts: Score-Based Editing for Controllable Text-To-Speech

### Expressive Multi-Modal Speech Synthesis: 表达性多模态语音合成

<details>
<summary>展开原文</summary>

Synthesizing speech from multi-modal data such as texts, images, and videos is an appealing research topic due to its various applications in the industry such as storytelling, filming, and gaming.
Although there are several related works on this task ([^6] [^24] [^276] [^277]), few of them can fully extract useful information from multi-modal data.
Particularly, synthesizing engaging speech and expressive voiceover for complex visual content sees great opportunities in the future.

</details>
<br>

从多模态数据 (如文本, 图像, 视频) 中合成语音是一项有吸引力的研究主题, 因为它在工业界有着各种应用, 如故事讲述, 电影拍摄, 游戏.
虽然有一些相关工作 ([^6] [^24] [^276] [^277]), 但很少有工作可以从多模态数据中完全提取有用的信息.
特别是, 合成富有情感的语音和富有表现力的配音, 对于复杂的视觉内容, 具有巨大的机会.

[^6]: Speech-Driven Cartoon Animation with Emotions (2001)
[^24]: Face2speech: Towards multi-speaker text-to-speech synthesis using an embedding vector predicted from a face image
[^276]: Seeing Your Speech Style: A Novel Zero-Shot Identity-Disentanglement Face-Based Voice Conversion
[^277]: Visualtts: TTS with Accurate Lip-Speech Synchronization For Automatic Voice Over

### Natural and Emotional Conversational TTS: 自然和情感对话式语音合成

<details>
<summary>展开原文</summary>

Speech conversational TTS have come out for several decades but remained as cascaded systems for a long time and cannot generate natural and emotional speech.
These systems are not context-aware, making the synthesized speech sound robotic.
With the advent of LLMs, existing TTS technologies were directly introduced by simply synthesizing speech from the text generated by LLMs ([LLaMA-Omni [33]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md)).
However, context-aware conversational TTS with rich emotion and good naturalness has not been well studied.

</details>
<br>

语音对话式 TTS 已经存在了几十年, 但长期以来仍然是级联系统, 无法生成自然和情感语音.
这些系统不具有上下文感知能力, 导致合成语音听起来像机器人.
随着 LLMs 的出现, 现有的 TTS 技术被直接引入, 只需从 LLMs 生成的文本合成语音 ([LLaMA-Omni [33]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md)).
然而, 具有丰富情感和良好自然度的上下文感知对话式 TTS, 却没有得到充分研究.

### Zero-Shot Long Speech Synthesis with Emotion Consistency: 情感一致的零样本长语音合成

<details>
<summary>展开原文</summary>

Zero-shot TTS emerged in recent years to achieve voice cloning and speech style imitation without fine-tuning, making them more practical in real scenarios ([MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md); [F5-TTS [194]](../../Models/Diffusion/2024.10.09_F5-TTS.md)).
However, synthesizing long speech with rich emotion and style variation in a zero-shot setting remains challenging due to the lack of rich speech information in short reference audio clips.
Addressing this issue will make a big step towards fully controllable zero-shot TTS.

</details>
<br>

零样本 TTS 于近年出现以实现声音克隆和语音风格模仿而无需微调, 因此在实际场景中更加实际 ([MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md); [F5-TTS [194]](../../Models/Diffusion/2024.10.09_F5-TTS.md)).
然而, 在零样本设置中合成具有丰富情感和风格变化的长语音, 仍然存在挑战, 因为短引用音频片段中缺乏丰富的语音信息.
解决这一问题将迈出重要一步, 迈向完全可控的零样本 TTS.

### Efficient TTS by Natural Language Description: 自然语言描述的高效文本转语音

<details>
<summary>展开原文</summary>

Synthesizing speech with natural language description usually involves training large language encoders and bridge nets between the two modalities which can bring about much more computation overhead compared to previous TTS methods.
The inference time is also relatively slow, e.g., existing methods usually take tens of seconds to synthesize a short speech audio clip of less than 10 seconds ([CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md), [Prompttts++ [104]](../../Models/Acoustic/PromptTTS++.md)).
Therefore, efficient text and speech modeling and interaction is critical for natural language description-based TTS systems.

</details>
<br>

使用自然语言描述来合成语音通常涉及训练大型语言编码器和两个模态之间的桥接网络, 这会带来比以前 TTS 方法更多的计算开销.
推理时间也相对较慢, 例如, 现有的方法通常需要几十秒才能合成 10 秒以下的短语音音频片段 ([CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md), [Prompttts++ [104]](../../Models/Acoustic/PromptTTS++.md)).
因此, 基于自然语言描述 TTS 系统的高效文本和语音建模和交互至关重要.
