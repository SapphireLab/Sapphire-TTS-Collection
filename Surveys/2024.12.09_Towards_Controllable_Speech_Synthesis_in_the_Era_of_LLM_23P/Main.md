# Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey

<details>
<summary>基本信息</summary>

- 标题: "Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey"
- 作者:
  - 01 Tianxin Xie,
  - 02 Yan Rong,
  - 03 Pengfei Zhang,
  - 04 Li Liu
- 链接:
  - [ArXiv](https://arxiv.org/abs/2412.06602)
  - [Publication]()
  - [Github](https://github.com/imxtx/awesome-controllabe-speech-synthesis)
  - [Demo]()
- 文件:
  - [ArXiv](2412.06602v1__Survey__Towards_Controllable_Speech_Synthesis_in_the_Era_of_LLMs.pdf)
  - [Publication] #TODO

</details>

## Abstract: 摘要

<details>
<summary>展开原文</summary>

Text-to-speech (TTS), also known as speech synthesis, is a prominent research area that aims to generate natural-sounding human speech from text.
Recently, with the increasing industrial demand, TTS technologies have evolved beyond synthesizing human-like speech to enabling controllable speech generation.
This includes fine-grained control over various attributes of synthesized speech such as emotion, prosody, timbre, and duration.
Besides, advancements in deep learning, such as diffusion and large language models, have significantly enhanced controllable TTS over the past several years.
In this paper, we conduct a comprehensive survey of controllable TTS, covering approaches ranging from basic control techniques to methods utilizing natural language prompts, aiming to provide a clear understanding of the current state of research.
We examine the general controllable TTS pipeline, challenges, model architectures, and control strategies, offering a comprehensive and clear taxonomy of existing methods.
Additionally, we provide a detailed summary of datasets and evaluation metrics and shed some light on the applications and future directions of controllable TTS.
To the best of our knowledge, this survey paper provides the first comprehensive review of emerging controllable TTS methods, which can serve as a beneficial resource for both academic researchers and industry practitioners.

</details>
<br>

文本转语音 (Text-to-Speech, TTS), 也称为语音合成 (Speech Synthesis) 是旨在从文本生成自然听感的人类语音的重要研究领域.

近年来随着工业需求的增加, TTS 技术已经从合成类似人类的语音演进为实现可控语音生成.
这包括对合成的语音的各种属性进行细粒度的控制, 例如情感, 韵律, 音色以及时长.

除此之外, 深度学习的最新进展例如扩散和大语言模型, 已经在过去几年中显著地增强了可控 TTS.

在本文中, 我们对可控 TTS 进行了全面地调查, 涵盖从基础控制技术到利用自然语言模型的方法, 旨在为研究的当前状态提供清晰的理解.

我们研究了一般可控 TTS 的流程, 挑战, 模型架构和控制策略, 提供现有方法的全面且清晰的分类.

据我们所知, 本文是新兴可控 TTS 方法的首个全面综述, 这对于学术研究者和工业实践者都可以提供有益的参考.

## 1·Introduction: 引言

<details>
<summary>展开原文</summary>

Speech synthesis, also broadly known as text-to-speech (TTS), is a long-time developed technique that aims to synthesize human-like voices from text ([^1] [^2]), and it has extensive applications in our daily lives, such as health care ([^3] [^4]), personal assistants [^5], entertainment ([^6] [^7]), and robotics ([^8] [^9]).
Recently, TTS has gained significant attention with the rise of large language model (LLM)-powered chatbots, such as ChatGPT [^10] and [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md), due to its naturalness and convenience for human-computer interaction.
Meanwhile, the ability to achieve fine-grained control over synthesized speech attributes, such as emotion, prosody, timbre, and duration, has become a hot research topic in both academia and industry, driven by its vast potential for diverse applications.

</details>
<br>

语音合成 (Speech Synthesis), 也称为文本转语音 (Text-to-Speech, TTS), 是一项长期发展的技术, 旨在从文本合成类似人类的声音 ([^1] [^2]), 并且在日常生活中有广泛的应用, 如医疗保健 ([^3] [^4]), 个性化助手 [^5], 娱乐 ([^6] [^7]), 以及机器人 ([^8] [^9]).
近期, 随着大语言模型驱动的聊天机器人 (如 ChatGPT [^10] 和 [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md)) 的兴起, TTS 获得了广泛关注, 因为它在人机交互中的自然性和便利性.
同时, 对合成语音的属性进行细粒度的控制, 如情感, 韵律, 音色, 以及时长, 已经成为学术界和工业界热门研究话题, 其潜在的多样化应用吸引了广泛的注意力.

[^1]: Website - Speech Synthesis - Wikipedia [1] https://en.wikipedia.org/wiki/Speech_synthesis
[^2]: Book - An Introduction to Text-to-Speech Synthesis (1997)
[^3]: Speech Technology for Healthcare: Opportunities, Challenges, and State of The Art (2020)
[^4]: Speech Synthesis from Neural Decoding of Spoken Sentences (2019)
[^5]: Alexa vs. Siri vs. Cortana vs. Google Assistant: A Comparison of Speech-Based Natural User Interfaces (2018)
[^6]: Speech-Driven Cartoon Animation with Emotions (2001)
[^7]: Comic-Guided Speech Synthesis (2019)
[^8]: Spoken Language Interaction with Robots: Recommendations for Future Research (2022)
[^9]: Towards Expressive Speech Synthesis in English on A Robotic Platform (2006)
[^10]: Website - Introducing ChatGPT - OpenAI https://openai.com/index/chatgpt/

Deep learning [^12] has made great progress in the past decade due to exponentially growing computational resources like GPUs [^13], leading to the explosion of numerous great works on TTS ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md); [FastSpeech [15]](../../Models/TTS2_Acoustic/2019.05.22_FastSpeech.md); [Deep Voice [16]](../../Models/TTS0_System/2017.02.25_DeepVoice.md); [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)).
These methods can synthesize human speech with better quality ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md)) and can achieve fine-grained control of the generated voice [18] [19] [20] [21] [22] ~\cite{zhang2019learning,wang2018style,um2020emotional,li2021controllable,zhang2023iemotts}.
Besides, some recent works synthesize speech given multi-modal input, such as face images [23] [24] ~\cite{wang2022residual,goto2020face2speech}, cartoons ~\cite{wang2019comic}, and videos [25] ~\cite{choi2023diffv2s}.
Moreover, with the fast development of open-source LLMs [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md); [26] [Qwen [27]](../../Models/TextLM/2023.09.28_Qwen.md); [28] [29] ~\cite{jiang2023mistral,,bi2024deepseek,glm2024chatglm}, some researchers propose to synthesize fine-grained controllable speech with natural language description [30] [31] [32]~\cite{hao2023boosting,gao2024emo,neekhara2024improving}, coining a new way to generate custom speech voices.
Meanwhile, powering LLMs with speech synthesis has also been a hot topic in the last few years [LLaMA-Omni [33]](../../Models/Omni/2024.09.10_LLaMA-Omni.md); [34] [35]~\cite{zhang2023speechgpt,zhang2024intrinsicvoice}.
In recent years, a wide range of TTS methods has emerged, making it essential for researchers to gain a comprehensive understanding of current research trends, particularly in controllable TTS, to identify promising future directions in this rapidly evolving field.
Consequently, there is a pressing need for an up-to-date survey of TTS techniques.
While several existing surveys address parametric-based approaches [36] [37] [38] [39] [40] [41] ~\cite{klatt1987review,dutoit1997high,breen1992speech,olive1985text,king2014measuring,Zen20091039} and deep learning-based TTS [42] [43] [44] [45] [46] [47] [48] ~\cite{tan2021survey,ning2019review,kaur2023conventional_survey,mattheyses2015audiovisual_survey,triantafyllopoulos2023overview_survey,mu2021review,mehrish2023review}, they largely overlook the controllability of TTS.
Additionally, these surveys do not cover the advancements in recent years, such as natural language description-based TTS methods.

[^12]: Deep Learning (2015)
[^13]: GPU Computing (2008)

This paper provides a comprehensive and in-depth survey of existing and emerging TTS technologies, with a particular focus on controllable TTS methods.
Fig.~\ref{fig:sec1_summary} demonstrates the development of controllable TTS methods in recent years, showing their backbones, feature representations, and control abilities.
The remainder of this section begins with a brief comparison between this survey and previous ones, followed by an overview of the history of controllable TTS technologies, ranging from early milestones to state-of-the-art advancements.
Finally, we introduce the taxonomy and organization of this paper.

### A·Comparison with Existing Surveys: 与现有综述的比较

Several survey papers have reviewed TTS technologies, spanning early approaches from previous decades~\cite{klatt1987review,dutoit1997high,tabet2011speech,king2014measuring} [49] to more recent advancements~\cite{ning2019review,tan2021survey,zhang2023survey} [50].
However, to the best of our knowledge, this paper is the first to focus specifically on controllable TTS.
The key differences between this survey and prior work are summarized as follows:

#### Different Scope: 不同范围

Klatt \etal\cite{klatt1987review} provided the first comprehensive survey on formant, concatenative, and articulatory TTS methods, with a strong emphasis on text analysis.
In the early 2010s, Tabet \etal\cite{tabet2011speech} and King \etal\cite{king2014measuring} explored rule-based, concatenative, and HMM-based techniques.
Later, the advent of deep learning catalyzed the emergence of numerous neural-based TTS methods.
Therefore, Ning \etal\cite{ning2019review} and Tan \etal\cite{tan2021survey} have conducted extensive surveys on neural-based acoustic models and vocoders, while Zhang \etal\cite{zhang2023survey} presented the first review of diffusion model-based TTS techniques.
However, these studies offer limited discussion on the controllability of TTS systems.
To address this gap, we present the first comprehensive survey of TTS methods through the lens of controllability, providing an in-depth analysis of model architectures and strategies for controlling synthesized speech.

#### Close to Current Demand: 接近当前需求

With the rapid development of hardware (\ie GPUs) and artificial intelligence techniques (\ie transformers, LLMs, diffusion models) in the last few years, the demand for controllable TTS is becoming increasingly urgent due to its broad applications in industries such as filmmaking, gaming, robots, and personal assistants.
Despite this growing need, existing surveys pay little attention to control methods in TTS technologies.
To bridge this gap, we propose a systematic analysis of current controllable TTS methods and the associated challenges, offering a comprehensive understanding of the research state in this field.

#### New Insights & Directions: 新见解与方向

This survey offers new insights through a comprehensive analysis of model architectures and control methods in controllable TTS systems.
Additionally, it provides an in-depth discussion of the challenges associated with various controllable TTS tasks.
Furthermore, we address the question: ``Where are we on the path to fully controllable TTS technologies?'', by examining the relationship and gap between current TTS methods and industrial requirements.
Based on these analyses, we identify promising directions for future research on TTS technologies.

Table \ref{tab:sec1_survey_comparison} summarizes representative surveys and this paper in terms of main focus and publication year.

### B·The History of Controllable TTS: 可控 TTS 的历史

Controllable TTS aims to control various aspects of synthesized speech, such as pitch, energy, speed/duration, prosody, timbre, emotion, gender, or high-level styles.
This subsection briefly reviews the history of controllable TTS ranging from early approaches to the state-of-arts in recent years.

#### Early Approaches: 早期方法

Before the prevalence of deep neural networks (DNNs), controllable TTS technologies were built primarily on rule-based, concatenative, and statistical methods.
These approaches enable some degree of customization and control, though they were constrained by the limitations of the underlying models and available computational resources.
1) Rule-based TTS systems [51] [52] [53] [54] ~\cite{rabiner1968digital,allen1987mitalk,purcell2006adaptive,Klatt1980971}, such as formant synthesis, were among the earliest methods for speech generation.
These systems use manually crafted rules to simulate the speech generation process by controlling acoustic parameters such as pitch, duration, and formant frequencies, allowing explicit manipulation of prosody and phonetic details through rule adjustments.
2) Concatenative TTS [55] [56] [57] [58] ~\cite{wouters2001control,bulut2002expressive,bulyko2001joint,Stylianou200121}, which dominated the field in the late 1990s and early 2000s, synthesize speech by concatenating pre-recorded speech segments, such as phonemes or diphones, stored in a large database~\cite{Hunt1996373} [59].
These methods can modify the prosody by manipulating the pitch, duration, and amplitude of speech segments during concatenation.
They also allow limited voice customization by selecting speech units from different speakers.
3) Parametric methods, particularly HMM-based TTS [60] [61] [62] [63] [64] [65]~\cite{nose2007style,ling2009integrating,nose2009hmm,yoshimura1998duration,yoshimura1999simultaneous,tokuda2000speech}, gained prominence in the late 2000s.
These systems model the relationships between linguistic features and acoustic parameters, providing more flexibility in controlling prosody, pitch, speaking rate, and timbre by adjusting statistical parameters.
Some HMM-based systems also supported speaker adaptation [66] [67]~\cite{yamagishi2009robust,yamagishi2009analysis} and voice conversion [68] [69]~\cite{masuko1997voice,wu2006voice}, enabling voice cloning to some extent.
Besides, emotion can also be limitedly controlled by some of these methods~\cite{yamagishi2003modeling,yamagishi2005acoustic,nose2007style,lorenzo2015emotion} [70] [71] [72].
In addition, they required less storage compared to concatenative TTS and allowed smoother transitions between speech units.

#### Neural-Based Synthesis: 基于神经网络合成

Neural-based TTS technologies emerged with the advent of deep learning, significantly advancing the field by enabling more flexible, natural, and expressive speech synthesis.
Unlike traditional methods, neural-based TTS leverages DNNs to model complex relationships between input text and speech, facilitating nuanced control over various speech characteristics.
Early neural TTS systems, such as WaveNet [73]~\cite{van2016wavenet} and Tacotron [74] ~\cite{wang2017tacotron} laid the groundwork for controllability.
1) Controlling prosody features like rhythm and intonation is vital for generating expressive and contextually appropriate speech.
Neural-based TTS models achieve prosody control through explicit conditioning or learned latent representations~\cite{shen2018tacotron2,[FastSpeech [15]](../../Models/TTS2_Acoustic/2019.05.22_FastSpeech.md),ren2020fastspeech2,lancucki2021fastpitch,wang2024maskgct} [75] [76] [77] [78].
1) Speaker control has also gained significant improvement in neural-based TTS through speaker embeddings or adaptation techniques [79] [80] [81] [82]~\cite{fan2015multi,huang2022meta,chen2020multispeech,casanova2022yourtts}.
2) Besides, emotionally controllable TTS [83]~\cite{lei2022msemotts,gao2024emo,um2020emotional,zhang2023iemotts,neekhara2024improving} has become a hot topic due to the strong modeling capability of DNNs, enabling the synthesis of speech with specific emotional tones such as happiness, sadness, anger, or neutrality.
These systems go beyond producing intelligible and natural-sounding speech, focusing on generating expressive output that aligns with the intended emotional context.
1) Neural-based TTS can also manipulate timbre (vocal quality)~\cite{wang2024maskgct,elias2021paralleltacotron,wang2023neural,[NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md),shen2023naturalspeech2,ju2024naturalspeech3} [84] [85] [86] [87] and style (speech mannerisms)~\cite{li2022styletts,li2024styletts2,huang2022generspeech} [88] [89] [90], allowing for creative and personalized applications.
These techniques lead to one of the most popular research topics, \ie zero-shot TTS (particularly voice cloning)~\cite{casanova2022yourtts,wang2024maskgct,jiang2023megavoic,cooper2020zero} [91] [92].
1) Fine-grained content and linguistic control also become more powerful~\cite{peng2024voicecraft,tan2021editspeech,tae2021editts,seshadri2021emphasis} [93] [94] [95] [96].
These methods can emphasize or de-emphasize specific words or adjust the pronunciation of phonemes through speech editing or generation techniques.

Neural-based TTS technologies represent a significant leap in the flexibility and quality of speech synthesis.
From prosody and emotion to speaker identity and style, these systems empower diverse applications in fields such as entertainment, accessibility, and human-computer interaction.

#### LLM-Based Synthesis: 基于大语言模型合成

Here we pay special attention to LLM-based synthesis methods due to their superior context modeling capabilities compared to other neural-based TTS methods.
LLMs, such as GPT~\cite{brown2020gpt3,achiam2023gpt4}, T5~\cite{raffel2020t5} [97] [98] [99], and PaLM [100]~\cite{chowdhery2023palm}, have revolutionized various natural language processing (NLP) tasks with their ability to generate coherent, context-aware text.
Recently, their utility has expanded into controllable TTS technologies~\cite{guo2023prompttts,leng2023prompttts2,zhou2024voxinstruct,shimizu2024promptttspp,[CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)} [101] [102] [103] [104].
For example, users can synthesize the target speech by describing its characteristics, such as: ``A young girl says `I really like it, thank you!' with a happy voice'', making speech generation significantly more intuitive and user-friendly.
Specifically, an LLM can detect emotional intent in sentences (\eg ``I’m thrilled'' → happiness, ``This is unfortunate'' → sadness).
The detected emotion is encoded as an auxiliary input to the TTS model, enabling modulation of acoustic features like prosody, pitch, and energy to align with the expressed sentiment.
By leveraging LLMs' capabilities in understanding and generating rich contextual information, these systems can achieve enhanced and fine-grained control over various speech attributes such as prosody, emotion, style, and speaker characteristics~\cite{yang2024instructtts,gao2024emo,ji2024controlspeech} [105] [106].
Integrating LLMs into TTS systems represents a significant step forward, enabling more dynamic and expressive speech synthesis.

### C·Organization of This Survey: 本文结构

This paper first presents a comprehensive and systematic review of controllable TTS technologies, with a particular focus on model architectures, control methodologies, and feature representations.
To establish a foundational understanding, this survey begins with an introduction to the TTS pipeline in [Section 2](Sec.02.md).
While our focus remains on controllable TTS, [Section 3](Sec.03.md) examines seminal works in uncontrollable TTS that have significantly influenced the field's development.
[Section 4](Sec.04.md) provides a thorough investigation into controllable TTS methods, analyzing both their model architectures and control strategies.
[Section 5](Sec.05.md) presents a comprehensive review of datasets and evaluation metrics.
[Section 6](Sec.06.md) provides an in-depth analysis of the challenges encountered in achieving controllable TTS systems and discusses future directions.
[Section 7](#Sec.07) explores the broader impacts of controllable TTS technologies and identifies promising future research directions, followed by the conclusion in [Section 8](#Sec.08).

## 2·[TTS Pipeline: TTS 流程](Sec.02.md)

## 3·[Uncontrollable TTS: 非可控 TTS](Sec.03.md)

## 4·[Controllable TTS: 可控 TTS](Sec.04.md)

## 5·[Datasets & Evaluation: 数据集和评估](Sec.05.md)

## 6·[Challenges & Future Directions: 挑战与未来方向](Sec.06.md)

## 7·Discussion: 讨论

<a id="Sec.07"></a>

### A·Impacts of Controllable TTS: 可控 TTS 的影响

#### Applications: 应用

Controllable TTS systems allow fine-grained manipulation of speech attributes such as pitch, emotion, speaking rate, and style, enabling a wide range of applications across industries.
One major application is in virtual assistants and customer support systems, where controllable TTS ensures tailored and context-aware responses.
For instance, a virtual assistant can speak in a calm tone for technical support but switch to an enthusiastic tone when presenting promotional offers, enhancing user experience.

In the entertainment industry, controllable TTS is invaluable for creating dynamic voiceovers, audiobooks, and gaming characters.
It enables precise adjustments in tone and delivery, allowing audiobooks to reflect character emotions and gaming characters to exhibit personality traits that align with their roles.
Similarly, in education, TTS systems can adapt speaking styles to suit different learners, such as adopting a slow, clear tone for language learning or an engaging, storytelling style for children’s content.

Controllable TTS is also transformative in assistive technologies, where it can generate speech that reflects the user’s intended emotion or personality.
This is particularly impactful for individuals with speech impairments, enabling more expressive and natural communication.

In content localization, controllable TTS systems adjust speech characteristics to suit regional and cultural preferences, ensuring a natural fit for global audiences.
Additionally, it finds applications in human-computer interaction research, enabling adaptive dialogue systems that modify speech dynamically based on user mood or environment.
By offering this flexibility, controllable TTS systems enhance accessibility, personalization, and engagement across various domains.

#### Deepfakes: 深度伪造

A deepfake is a type of synthetic media in which a person in an existing image, video, or audio recording is replaced with someone else’s likeness or voice.
This technology uses deep learning, particularly GANs~\cite{zhang2022deepfake}, to create highly realistic, but fabricated, content.
While deepfakes are most commonly associated with video manipulation, such as face swapping~\cite{nirkin2019fsgan}, they can also apply to audio, enabling the creation of synthetic speech that mimics a specific person’s voice, which is well known as voice cloning.

Voice cloning, especially few-shot~\cite{arik2018neural} and zero-shot TTS~\cite{wang2023neural,wang2024maskgct}, poses a significant threat to systems that rely on voice authentication, such as banking, customer service, and other identity verification processes.
With a convincing clone of someone’s voice, attackers could potentially impersonate individuals to gain unauthorized access to sensitive information or accounts.

To address these concerns, it’s essential to establish robust security protocols, consent-based regulations, and public awareness around voice cloning.
Furthermore, advancements in detecting voice clones are equally important to help distinguish genuine voices from synthesized ones, protecting both individuals and organizations from potential misuse.

### B·Limitation of this Survey: 本综述的局限性

Although we conduct a comprehensive survey of controllable TTS in this paper, there are some limitations we want to address in the future:
1) A unified benchmark method will be provided to evaluate controllable TTS methods.
2) Detailed analysis and control strategies of each specific speech attribute will be provided in an updated version of this paper.
3) The methodologies for feature disentanglement are crucial for controllable TTS but are not adequately discussed.

## 8·Conclusions: 结论

<a id="Sec.08"></a>

<details>
<summary>展开原文</summary>

In this survey paper, we first elaborate on the general pipeline for controllable TTS, followed by a glimpse of uncontrollable TTS methods.
Then we comprehensively review existing controllable methods from the perspectives of model architectures and control strategies.
Popular datasets and commonly used evaluation metrics for controllable TTS are also summarized in this paper.
Besides, the current challenges are deeply analyzed and the promising future directions are also pointed out.
To the best of our knowledge, this is the first comprehensive survey for controllable TTS.

Writing a comprehensive survey is a challenging task and an iterative process.
Hence, we will regularly update this survey to offer researchers and practitioners a continuously evolving resource for understanding controllable speech synthesis.

</details>
<br>

在本综述中, 我们
- 首次阐述了可控 TTS 的一般流程, 并简要介绍了不可控 TTS 方法.
- 从模型架构和控制策略的角度全面回顾了现有的可控方法.
- 总结了可控 TTS 常用的数据集和评价指标.
- 深度分析了现有挑战, 并指出了未来发展方向.

据我们所知, 这是关于可控 TTS 的首篇全面综述.

撰写一篇全面综述是一个具有挑战性的任务, 也是一项迭代式的过程.
因此, 我们将不断更新本综述, 为研究者和从业者提供可持续发展的可控语音合成资源.