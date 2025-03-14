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

<table><tr><td width="50%">

Text-to-speech (TTS), also known as speech synthesis, is a prominent research area that aims to generate natural-sounding human speech from text.
Recently, with the increasing industrial demand, TTS technologies have evolved beyond synthesizing human-like speech to enabling controllable speech generation.
This includes fine-grained control over various attributes of synthesized speech such as emotion, prosody, timbre, and duration.
Besides, advancements in deep learning, such as diffusion and large language models, have significantly enhanced controllable TTS over the past several years.
In this paper, we conduct a comprehensive survey of controllable TTS, covering approaches ranging from basic control techniques to methods utilizing natural language prompts, aiming to provide a clear understanding of the current state of research.
We examine the general controllable TTS pipeline, challenges, model architectures, and control strategies, offering a comprehensive and clear taxonomy of existing methods.
Additionally, we provide a detailed summary of datasets and evaluation metrics and shed some light on the applications and future directions of controllable TTS.
To the best of our knowledge, this survey paper provides the first comprehensive review of emerging controllable TTS methods, which can serve as a beneficial resource for both academic researchers and industry practitioners.

</td><td>

文本转语音 (Text-to-Speech, TTS), 也称为语音合成 (Speech Synthesis) 是旨在从文本生成自然听感的人类语音的重要研究领域.

近年来随着工业需求的增加, TTS 技术已经从合成类似人类的语音演进为实现可控语音生成.
这包括对合成的语音的各种属性进行细粒度的控制, 例如情感, 韵律, 音色以及时长.

除此之外, 深度学习的最新进展例如扩散和大语言模型, 已经在过去几年中显著地增强了可控 TTS.

在本文中, 我们对可控 TTS 进行了全面地调查, 涵盖从基础控制技术到利用自然语言模型的方法, 旨在为研究的当前状态提供清晰的理解.

我们研究了一般可控 TTS 的流程, 挑战, 模型架构和控制策略, 提供现有方法的全面且清晰的分类.

据我们所知, 本文是新兴可控 TTS 方法的首个全面综述, 这对于学术研究者和工业实践者都可以提供有益的参考.

</td></tr></table>

## 1·Introduction: 引言

<table><tr><td width="50%">

Speech synthesis, also broadly known as text-to-speech (TTS), is a long-time developed technique that aims to synthesize human-like voices from text ([^1] [^2]), and it has extensive applications in our daily lives, such as health care ([^3] [^4]), personal assistants [^5], entertainment ([^6] [^7]), and robotics ([^8] [^9]).
Recently, TTS has gained significant attention with the rise of large language model (LLM)-powered chatbots, such as ChatGPT [^10] and [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md), due to its naturalness and convenience for human-computer interaction.
Meanwhile, the ability to achieve fine-grained control over synthesized speech attributes, such as emotion, prosody, timbre, and duration, has become a hot research topic in both academia and industry, driven by its vast potential for diverse applications.

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

Deep learning [^12] has made great progress in the past decade due to exponentially growing computational resources like GPUs [^13], leading to the explosion of numerous great works on TTS ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md); [FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md); [Deep Voice [16]](../../Models/TTS0_System/2017.02.25_DeepVoice.md); [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)).
These methods can synthesize human speech with better quality ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md)) and can achieve fine-grained control of the generated voice ([Zhang et al. [18]](../../Models/Acoustic/2018.12.11_Learning_Latent_Representations_for_Style_Control_and_Transfer_in_End-to-End_Speech_Synthesis.md); [GST [19]](../../Models/Style/2018.03.23_GST.md); [I2I [20]](../../Models/Style/2019.11.05_I2I.md); [Li et al. [21]](../../Models/Style/2020.11.17_Controllable_Emotion_Transfer_for_End-to-End_Speech_Synthesis.md); [iEmoTTS [22]](../../Models/Style/2022.06.29_iEmoTTS.md)).
Besides, some recent works synthesize speech given multi-modal input, such as face images ([FR-PSS [23]](../../Models/_Basis/2022.04.01_FR-PSS.md); [Face2Speech [24]](../../Models/_Basis/2020.10.25_Face2Speech.md)), cartoons [^7], and videos ([DiffV2S [25]](../../Models/CV/2023.08.15_DiffV2S.md)).
Moreover, with the fast development of open-source LLMs ([LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md); [Mistral [26]](../../Models/TextLM/2023.10.10_Mistral-7B.md); [Qwen [27]](../../Models/TextLM/2023.09.28_Qwen.md); [DeepSeek [28]](../../Models/TextLM/DeepSeek.md); [ChatGLM [29]](../../Models/TextLM/ChatGLM.md)), some researchers propose to synthesize fine-grained controllable speech with natural language description ([LLM+VALL-E [30]](../../Models/SpeechLM/2023.12.30_LLM&VALL-E.md); [Emo-DPO [31]](../../Modules/RLHF/2024.09.16_Emo-DPO.md); [T5-TTS [32]](../../Models/SpeechLM/2024.06.25_T5-TTS.md)), coining a new way to generate custom speech voices.
Meanwhile, powering LLMs with speech synthesis has also been a hot topic in the last few years ([LLaMA-Omni [33]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md); [SpeechGPT [34]](../../Models/SpokenDialogue/2023.05.18_SpeechGPT.md); [IntrinsicVoice [35]](../../Models/SpokenDialogue/2024.10.09_IntrinsicVoice.md)).
In recent years, a wide range of TTS methods has emerged, making it essential for researchers to gain a comprehensive understanding of current research trends, particularly in controllable TTS, to identify promising future directions in this rapidly evolving field.
Consequently, there is a pressing need for an up-to-date survey of TTS techniques.
While several existing surveys address parametric-based approaches ([Survey by Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md); [Survey by Dutoit et al. (1997) [37]](../../Surveys/1997.00.00_High-Quality_Text-To-Speech_Synthesis__An_Overview.md); [Survey by Breen et al. (1992) [38]](../../Surveys/1992.00.00_Speech_Synthesis_Models__A_Review.md); [Survey by Olive (1985) [39]](../../Surveys/1982.11.05_Text_to_Speech__An_Overview.md); [Survey by King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md); [Survey by Zen et al. (2009) [41]](../../Surveys/2009.01.14_Statistical_Parametric_Speech_Synthesis.md)) and deep learning-based TTS ([Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md); [Survey by Ning et al. (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md); [Survey by Kaur et al. (2023) [44]](../2022.11.13_Conventional_and_Contemporary_Approaches_Used_in_Text_to_Speech_Synthesis__A_Review_44P/Main.md); [Survey by Mattheyses et al. (2014) [45]](../2014.02.21_Audiovisual_Speech_Synthesis__An_Overview_of_the_SoTA.md); [Survey by Triantafyllopoulos et al. (2023) [46]](../2023.03.10_An_Overview_of_Affective_Speech_Synthesis_and_Conversion_in_the_Deep_Learning_Era/Main.md); [Survey by Mu et al. (2021) [47]](../2021.04.20_Review_of_End-to-End_Speech_Synthesis_Technology_Based_on_Deep_Learning_40P.md); [Survey by Mehrish et al. (2023) [48]](../2023.04.30_A_Review_of_DL_Techniques_for_Speech_Processing_55P/Main.md)), they largely overlook the controllability of TTS.
Additionally, these surveys do not cover the advancements in recent years, such as natural language description-based TTS methods.

[^12]: Deep Learning (2015)
[^13]: GPU Computing (2008)

This paper provides a comprehensive and in-depth survey of existing and emerging TTS technologies, with a particular focus on controllable TTS methods.
Fig.01 demonstrates the development of controllable TTS methods in recent years, showing their backbones, feature representations, and control abilities.
The remainder of this section begins with a brief comparison between this survey and previous ones, followed by an overview of the history of controllable TTS technologies, ranging from early milestones to state-of-the-art advancements.
Finally, we introduce the taxonomy and organization of this paper.

</td><td>

语音合成 (Speech Synthesis), 也称为文本转语音 (Text-to-Speech, TTS), 是一项长期发展的技术, 旨在从文本合成类似人类的声音 ([^1] [^2]), 并且在日常生活中有广泛的应用, 如医疗保健 ([^3] [^4]), 个性化助手 [^5], 娱乐 ([^6] [^7]), 以及机器人 ([^8] [^9]).
近期, 随着大语言模型驱动的聊天机器人 (如 ChatGPT [^10] 和 [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md)) 的兴起, TTS 获得了广泛关注, 因为它在人机交互中的自然性和便利性.
同时, 对合成语音的属性进行细粒度的控制, 如情感, 韵律, 音色, 以及时长, 已经成为学术界和工业界热门研究话题, 其潜在的多样化应用吸引了广泛的注意力.

深度学习 [^12] 在过去十年内由于指数增长的计算资源 (如 GPU [^13]) 取得了巨大的进步, 也促使了 TTS 领域的众多优秀工作的出现 ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md); [FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md); [Deep Voice [16]](../../Models/TTS0_System/2017.02.25_DeepVoice.md); [CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)).

这些方法可以合成更高质量的人类语音 ([NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md)) 并且可以实现对生成声音的细粒度控制 ([Zhang et al. [18]](../../Models/Acoustic/2018.12.11_Learning_Latent_Representations_for_Style_Control_and_Transfer_in_End-to-End_Speech_Synthesis.md); [GST [19]](../../Models/Style/2018.03.23_GST.md); [I2I [20]](../../Models/Style/2019.11.05_I2I.md); [Li et al. [21]](../../Models/Style/2020.11.17_Controllable_Emotion_Transfer_for_End-to-End_Speech_Synthesis.md); [iEmoTTS [22]](../../Models/Style/2022.06.29_iEmoTTS.md)).

除此之外, 一些近期工作给定多模态输入来合成语音, 例如面部图像 ([FR-PSS [23]](../../Models/_Basis/2022.04.01_FR-PSS.md); [Face2Speech [24]](../../Models/_Basis/2020.10.25_Face2Speech.md)), 卡通[^7], 以及视频 ([DiffV2S [25]](../../Models/CV/2023.08.15_DiffV2S.md)).

此外, 随着开源大语言模型的快速发展 ([LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md); [Mistral [26]](../../Models/TextLM/2023.10.10_Mistral-7B.md); [Qwen [27]](../../Models/TextLM/2023.09.28_Qwen.md); [DeepSeek [28]](../../Models/TextLM/DeepSeek.md); [ChatGLM [29]](../../Models/TextLM/ChatGLM.md)), 一些研究人员提出使用自然语言描述来合成细粒度可控语音, 创造了一种生成自定义语音声音的新方式 ([LLM+VALL-E [30]](../../Models/SpeechLM/2023.12.30_LLM&VALL-E.md); [Emo-DPO [31]](../../Modules/RLHF/2024.09.16_Emo-DPO.md); [T5-TTS [32]](../../Models/SpeechLM/2024.06.25_T5-TTS.md)).

同时, 赋予大语言模型语音合成能力也已经成为过去几年的热门话题 ([LLaMA-Omni [33]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md); [SpeechGPT [34]](../../Models/SpokenDialogue/2023.05.18_SpeechGPT.md); [IntrinsicVoice [35]](../../Models/SpokenDialogue/2024.10.09_IntrinsicVoice.md)).

近年来, 出现了很多 TTS 方法, 这使得研究人员必须全面理解当前研究趋势, 特别是在可控 TTS 方面, 以确定这一快速发展领域的有前景的未来方向.

因此, 迫切需要一份最新的 TTS 技术综述.

尽管现有的综述关注了
- 基于参数的方法
  - [Survey by Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md);
  - [Survey by Dutoit et al. (1997) [37]](../../Surveys/1997.00.00_High-Quality_Text-To-Speech_Synthesis__An_Overview.md);
  - [Survey by Breen et al. (1992) [38]](../../Surveys/1992.00.00_Speech_Synthesis_Models__A_Review.md);
  - [Survey by Olive (1985) [39]](../../Surveys/1982.11.05_Text_to_Speech__An_Overview.md);
  - [Survey by King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md);
  - [Survey by Zen et al. (2009) [41]](../../Surveys/2009.01.14_Statistical_Parametric_Speech_Synthesis.md),
- 基于深度学习的 TTS
  - [Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md);
  - [Survey by Ning et al. (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md);
  - [Survey by Kaur et al. (2023) [44]](../2022.11.13_Conventional_and_Contemporary_Approaches_Used_in_Text_to_Speech_Synthesis__A_Review_44P/Main.md);
  - [Survey by Mattheyses et al. (2014) [45]](../2014.02.21_Audiovisual_Speech_Synthesis__An_Overview_of_the_SoTA.md);
  - [Survey by Triantafyllopoulos et al. (2023) [46]](../2023.03.10_An_Overview_of_Affective_Speech_Synthesis_and_Conversion_in_the_Deep_Learning_Era/Main.md);
  - [Survey by Mu et al. (2021) [47]](../2021.04.20_Review_of_End-to-End_Speech_Synthesis_Technology_Based_on_Deep_Learning_40P.md);
  - [Survey by Mehrish et al. (2023) [48]](../2023.04.30_A_Review_of_DL_Techniques_for_Speech_Processing_55P/Main.md).

但它们大多忽略了 TTS 的可控性.
此外, 这些综述没有涵盖最近几年的进展, 例如基于自然语言描述的 TTS 方法.

本文提供了对现有和新兴 TTS 技术的全面且深入的调查, 特别关注可控 TTS 方法.
图 01 展示了近年来可控 TTS 方法的发展, 展示了它们的骨干, 特征表示和控制能力.

本节的剩余部分
- 首先简要比较了本文和之前综述,
- 随后介绍可控 TTS 技术的历史, 从早期里程碑到最先进的进展.
- 最后, 我们介绍了本文的分类体系和组织结构.


</td></tr>
<tr><td colspan="2">

![](Images/Fig.01.png)

</td></tr></table>

### A·Comparison with Existing Surveys: 与现有综述的比较

<table><tr><td width="50%">

Several survey papers have reviewed TTS technologies, spanning early approaches from previous decades ([Survey by Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md); [Survey by Dutoit et al. (1997) [37]](../../Surveys/1997.00.00_High-Quality_Text-To-Speech_Synthesis__An_Overview.md); [Survey by King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md); [Survey by Tabet et al. (2011) [49]](../2011.06.27_Speech_Synthesis_Techniques_A_Survey.md)) to more recent advancements ([Survey by Ning et al. (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md); [Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md); [Survey by Zhang et al. (2023) [50]](../2023.03.23_A_Survey_on_Audio_Diffusion_Models__TTS_Synthesis_&_Enhancement_in_Generative_AI/Main.md)).
However, to the best of our knowledge, this paper is the first to focus specifically on controllable TTS.
The key differences between this survey and prior work are summarized as follows:

</td><td>

多篇综述论文已经回顾了 TTS 技术的发展历程, 涵盖了从几十年前的早期方法到近年来的最新进展.
- 早期方法:
  - [Survey by Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md);
  - [Survey by Dutoit et al. (1997) [37]](../../Surveys/1997.00.00_High-Quality_Text-To-Speech_Synthesis__An_Overview.md);
  - [Survey by King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md);
  - [Survey by Tabet et al. (2011) [49]](../2011.06.27_Speech_Synthesis_Techniques_A_Survey.md)
- 近期进展:
  - [Survey by Ning et al. (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md);
  - [Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md);
  - [Survey by Zhang et al. (2023) [50]](../2023.03.23_A_Survey_on_Audio_Diffusion_Models__TTS_Synthesis_&_Enhancement_in_Generative_AI/Main.md)

然而据我们所知, 本文是首篇专门聚焦于可控 TTS 的论文.

本综述与之前工作的主要区别总结如下:

</td></tr></table>

#### Different Scope: 不同范围

<table><tr><td width="50%">

[Survey by Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md) provided the first comprehensive survey on formant, concatenative, and articulatory TTS methods, with a strong emphasis on text analysis.
In the early 2010s, [Survey by Tabet et al. (2011) [49]](../2011.06.27_Speech_Synthesis_Techniques_A_Survey.md) and [Survey by King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md) explored rule-based, concatenative, and HMM-based techniques.
Later, the advent of deep learning catalyzed the emergence of numerous neural-based TTS methods.
Therefore, [Survey by Ning et al. (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md) and [Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) have conducted extensive surveys on neural-based acoustic models and vocoders, while [Survey by Zhang et al. (2023) [50]](../2023.03.23_A_Survey_on_Audio_Diffusion_Models__TTS_Synthesis_&_Enhancement_in_Generative_AI/Main.md) presented the first review of diffusion model-based TTS techniques.
However, these studies offer limited discussion on the controllability of TTS systems.
To address this gap, we present the first comprehensive survey of TTS methods through the lens of controllability, providing an in-depth analysis of model architectures and strategies for controlling synthesized speech.

</td><td>

- [Klatt 等人 (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md) 的综述首次全面探讨了共振峰, 拼接式, 发音式 TTS 方法, 并特别强调了文本分析的重要性.
- 到了 2010 年代初期, [Tabet 等人 (2011) [49]](../2011.06.27_Speech_Synthesis_Techniques_A_Survey.md) 和 [King 等人 (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md) 的综述进一步研究了基于规则, 拼接式以及隐马尔可夫模型 (HMM) 的 TTS 技术.
- 随后, 深度学习的兴起催生了众多基于神经网络的 TTS 方法.
- 因此 [Ning 等人 (2019) [43]](../2019.08.01_A_Review_of_DL_Based_Speech_Synthesis_16P.md) 和 [Tan 等人 (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) 的综述对基于神经网络的声学模型和声码器进行了广泛探讨, 而 [Zhang 等人 (2023) [50]](../2023.03.23_A_Survey_on_Audio_Diffusion_Models__TTS_Synthesis_&_Enhancement_in_Generative_AI/Main.md) 的综述则首次对基于扩散模型的 TTS 技术进行了回顾.

然而, 这些研究对 TTS 系统可控性的讨论较为有限.

为了填补这一空白, 本文首次从可控性的角度对 TTS 方法进行了全面综述, 深入分析了模型架构以及控制合成语音的策略.

</td></tr></table>

#### Close to Current Demand: 接近当前需求

<table><tr><td width="50%">

With the rapid development of hardware (i.e., GPUs) and artificial intelligence techniques (i.e., transformers, LLMs, diffusion models) in the last few years, the demand for controllable TTS is becoming increasingly urgent due to its broad applications in industries such as filmmaking, gaming, robots, and personal assistants.
Despite this growing need, existing surveys pay little attention to control methods in TTS technologies.
To bridge this gap, we propose a systematic analysis of current controllable TTS methods and the associated challenges, offering a comprehensive understanding of the research state in this field.

</td><td>

近年来随着硬件 (如 GPU) 和人工智能技术 (如 Transformer, 大语言模型, 扩散模型) 的快速发展, 可控 TTS 的需求日益迫切, 因为其在影视制作, 游戏, 机器人以及个人助手等领域的应用越来越广泛.
尽管这种需求不断增长, 但现有的综述对 TTS 技术中的控制方法关注甚少.
为了填补这一空白, 本文提出对当前可控 TTS 方法及其相关挑战进行系统性分析, 旨在为该领域的研究现状提供全面深入的理解.

</td></tr></table>

#### New Insights & Directions: 新见解与方向

<table><tr><td width="50%">

This survey offers new insights through a comprehensive analysis of model architectures and control methods in controllable TTS systems.
Additionally, it provides an in-depth discussion of the challenges associated with various controllable TTS tasks.
Furthermore, we address the question: "Where are we on the path to fully controllable TTS technologies?", by examining the relationship and gap between current TTS methods and industrial requirements.
Based on these analyses, we identify promising directions for future research on TTS technologies.

Table.01 summarizes representative surveys and this paper in terms of main focus and publication year.

</td><td>

本综述
- 通过对可控 TTS 系统中模型架构和控制方法的全面分析, 提供了新的见解.
- 此外, 本文深入探讨了与各种可控 TTS 任务相关的挑战.
- 更进一步, 我们通过审视当前 TTS 方法与工业需求之间的关系与差距, 回答了这样一个问题：**"我们在实现完全可控TTS技术的道路上处于什么位置?"**
- 基于这些分析，我们为TTS技术的未来研究指明了有前景的方向.

</td></tr>
<tr><td colspan="2">

表 01 总结了代表性综述及本文的主要关注点和发表年份。

| 综述 | 主要关注点 | 发表年份 |
|---|---|:-:|
|[Klatt et al. (1987) [36]](../../Surveys/1987.05.01_Review_of_Text-To-Speech_Conversion_for_English.md)| 基于规则和拼接的 TTS | 1987 |
|[Tabet et al. (2011) [49]](../2011.06.27_Speech_Synthesis_Techniques_A_Survey.md)| 基于规则, 拼接, 参数的 TTS | 2011 |
|[King et al. (2014) [40]](../../Surveys/2014.06.30_Measuring_a_Decade_of_Progress_in_Text-To-Speech.md)|参数 TTS 与性能度量 | 2014 |
|[Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md)|神经网络, 高效, 表达性 TTS | 2021 |
|[Zhang et al. (2023) [50]](../2023.03.23_A_Survey_on_Audio_Diffusion_Models__TTS_Synthesis_&_Enhancement_in_Generative_AI/Main.md)|扩散模型 TTS, 语音增强|2023|
|本文|可控 TTS, 评估 | 2024 |

</td></tr></table>

### B·The History of Controllable TTS: 可控 TTS 的历史

<table><tr><td width="50%">

Controllable TTS aims to control various aspects of synthesized speech, such as pitch, energy, speed/duration, prosody, timbre, emotion, gender, or high-level styles.
This subsection briefly reviews the history of controllable TTS ranging from early approaches to the state-of-arts in recent years.

</td><td>

可控性 TTS 的目标是控制合成语音的各个方面, 如音高, 能量, 速度/时长, 韵律, 音色, 情感, 性别, 或高级风格.
本小节简要回顾了可控 TTS 的历史, 从早期方法到近年来工业界的最新技术.

</td></tr></table>

#### Early Approaches: 早期方法

<table><tr><td width="50%">

Before the prevalence of deep neural networks (DNNs), controllable TTS technologies were built primarily on rule-based, concatenative, and statistical methods.
These approaches enable some degree of customization and control, though they were constrained by the limitations of the underlying models and available computational resources.
1) Rule-based TTS systems ([Rbiner et al. [51]](../../Models/_Early/1967.11.01_Digital‐Formant_Synthesizer_for_Speech‐Synthesis_Studies.md); [Allen et al. [52]](../../Models/_Early/1987.04.01_From_Text_to_Speech__The_MITalk_System.md); [Purcell et al. [53]](../../Models/_Early/2006.02.10_Adaptive_Control_of_Vowel_Formant_Frequency__Evidence_from_Real-Time_Formant_Manipulation.md); [Klatt et al. [54]](../../Models/_Early/1980.03.01_Software_for_a_Cascade_or_Parallel_Formant_Synthesizer.md)), such as formant synthesis, were among the earliest methods for speech generation.
These systems use manually crafted rules to simulate the speech generation process by controlling acoustic parameters such as pitch, duration, and formant frequencies, allowing explicit manipulation of prosody and phonetic details through rule adjustments.
1) Concatenative TTS [55] [56] [57] [58] ~\cite{wouters2001control,bulut2002expressive,bulyko2001joint,Stylianou200121}, which dominated the field in the late 1990s and early 2000s, synthesize speech by concatenating pre-recorded speech segments, such as phonemes or diphones, stored in a large database~\cite{Hunt1996373} [59].
These methods can modify the prosody by manipulating the pitch, duration, and amplitude of speech segments during concatenation.
They also allow limited voice customization by selecting speech units from different speakers.
1) Parametric methods, particularly HMM-based TTS [60] [61] [62] [63]~\cite{nose2007style,ling2009integrating,nose2009hmm,yoshimura1998duration,[Yoshimura et al.(1999) [64]](../../Models/_Early/Simultaneous_Modeling_of_Spectrum_Pitch_&_Duration_in_HMM-Based_Speech_Synthesis.md); [Tokuda et al. (2000) [65]](../../Models/_Early/Speech_Parameter_Generation_Algorithms_for_HMM-Based_Speech_Synthesis.md)}, gained prominence in the late 2000s.
These systems model the relationships between linguistic features and acoustic parameters, providing more flexibility in controlling prosody, pitch, speaking rate, and timbre by adjusting statistical parameters.
Some HMM-based systems also supported speaker adaptation [66] [67]~\cite{yamagishi2009robust,yamagishi2009analysis} and voice conversion [68] [69]~\cite{masuko1997voice,wu2006voice}, enabling voice cloning to some extent.
Besides, emotion can also be limitedly controlled by some of these methods~\cite{yamagishi2003modeling,[Yamagishi et al. (2005) [71]](../../Models/_Early/Acoustic_Modeling_of_Speaking_Styles_and_Emotional_Expressions_in_HMM-Based_Speech_Synthesis.md),nose2007style, [Lorenzo et al. (2015) [72]](../../Models/_Early/Emotion_Transplantation_through_Adaptation_in_HMM-Based_Speech_Synthesis.md)} [70].
In addition, they required less storage compared to concatenative TTS and allowed smoother transitions between speech units.

</td><td>

在深度神经网络盛行之前, 可控 TTS 技术主要建立在基于规则, 拼接, 统计方法之上.
这些方法实现了一定程度上的定制和控制, 但它们受到底层模型和可用计算资源的局限性约束.
1. 基于规则的 TTS 系统 (例如共振峰合成, 几乎是最早的语音生成方法)
  - 这些系统使用手工制作的规则来模拟语音生成过程, 通过控制声学参数 (如音高, 时长, 共振峰频率), 允许通过规则调整来实现对韵律和音素细节的显式操作.
2. 拼接式 TTS (1990 年代末到 2000 年代初占据主导地位, 通过预录制的语音片段进行拼接, 如音素或双音节, 存储在大型数据库中)
   - 这些方法可以在拼接时通过操作语音片段的音高, 时长, 幅度来修改韵律.
   - 它们还允许通过选择不同发音人的语音单元来实现有限的语音自定义.
3. 参数方法, 特别是基于 HMM 的 TTS (2000 年代末突出)
   - 这些系统建模语言特征和声学参数之间的关系, 通过调整统计参数提供了在控制韵律, 音高, 说话速率, 音色方面更大的灵活性.
   - 一些基于 HMM 的系统也支持说话人适应和声音转换, 在一定程度上实现了语音克隆.
   - 除此之外, 其中一些方法还可以有限地控制情感.
   - 另外, 它们比起拼接式 TTS 所需存储更少, 且能在语音单元之间实现更平滑的过渡.

</td></tr></table>

#### Neural-Based Synthesis: 基于神经网络合成

<table><tr><td width="50%">

Neural-based TTS technologies emerged with the advent of deep learning, significantly advancing the field by enabling more flexible, natural, and expressive speech synthesis.
Unlike traditional methods, neural-based TTS leverages DNNs to model complex relationships between input text and speech, facilitating nuanced control over various speech characteristics.
Early neural TTS systems, such as [WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md) and [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md) laid the groundwork for controllability.
1) Controlling prosody features like rhythm and intonation is vital for generating expressive and contextually appropriate speech.
Neural-based TTS models achieve prosody control through explicit conditioning or learned latent representations~\cite{shen2018tacotron2,[FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md),ren2020fastspeech2,lancucki2021fastpitch,[MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md)} [75] [76] [77].
1) Speaker control has also gained significant improvement in neural-based TTS through speaker embeddings or adaptation techniques [79] [80] [81] [82]~\cite{fan2015multi,huang2022meta,chen2020multispeech,casanova2022yourtts}.
2) Besides, emotionally controllable TTS [83]~\cite{lei2022msemotts,[Emo-DPO [31]](../../Modules/RLHF/2024.09.16_Emo-DPO.md); [I2I [20]](../../Models/Style/2019.11.05_I2I.md); [iEmoTTS [22]](../../Models/Style/2022.06.29_iEmoTTS.md); [T5-TTS [32]](../../Models/SpeechLM/2024.06.25_T5-TTS.md)} has become a hot topic due to the strong modeling capability of DNNs, enabling the synthesis of speech with specific emotional tones such as happiness, sadness, anger, or neutrality.
These systems go beyond producing intelligible and natural-sounding speech, focusing on generating expressive output that aligns with the intended emotional context.
1) Neural-based TTS can also manipulate timbre (vocal quality)~\cite{[MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md); elias2021paralleltacotron,wang2023neural,[NaturalSpeech [14]](../../Models/E2E/2022.05.09_NaturalSpeech.md),shen2023naturalspeech2,[NaturalSpeech3 [87]](../../Models/Diffusion/2024.03.05_NaturalSpeech3.md)} [84] [85] [86] and style (speech mannerisms)~\cite{li2022styletts,li2024styletts2,huang2022generspeech} [88] [89] [90], allowing for creative and personalized applications.
These techniques lead to one of the most popular research topics, i.e., zero-shot TTS (particularly voice cloning)~\cite{casanova2022yourtts,[MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md),jiang2023megavoic,cooper2020zero} [91] [92].
1) Fine-grained content and linguistic control also become more powerful~\cite{peng2024voicecraft,tan2021editspeech,tae2021editts,seshadri2021emphasis} [93] [94] [95] [96].
These methods can emphasize or de-emphasize specific words or adjust the pronunciation of phonemes through speech editing or generation techniques.

Neural-based TTS technologies represent a significant leap in the flexibility and quality of speech synthesis.
From prosody and emotion to speaker identity and style, these systems empower diverse applications in fields such as entertainment, accessibility, and human-computer interaction.

</td><td>

</td></tr></table>

#### LLM-Based Synthesis: 基于大语言模型合成

<table><tr><td width="50%">

Here we pay special attention to LLM-based synthesis methods due to their superior context modeling capabilities compared to other neural-based TTS methods.
LLMs, such as GPT ([GPT-3 [97]](../../Models/TextLM/2020.05.28_GPT-3.md); )~\cite{achiam2023gpt4}, T5~\cite{raffel2020t5} [98] [99], and PaLM [100]~\cite{chowdhery2023palm}, have revolutionized various natural language processing (NLP) tasks with their ability to generate coherent, context-aware text.
Recently, their utility has expanded into controllable TTS technologies ~\cite{[PromptTTS [101]](../../Models/Acoustic/2022.11.22_PromptTTS.md),leng2023prompttts2,[VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md);shimizu2024promptttspp,[CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md)} [102] [104].
For example, users can synthesize the target speech by describing its characteristics, such as: "A young girl says `I really like it, thank you!' with a happy voice", making speech generation significantly more intuitive and user-friendly.
Specifically, an LLM can detect emotional intent in sentences (e.g., "I’m thrilled" → happiness, "This is unfortunate" → sadness).
The detected emotion is encoded as an auxiliary input to the TTS model, enabling modulation of acoustic features like prosody, pitch, and energy to align with the expressed sentiment.
By leveraging LLMs' capabilities in understanding and generating rich contextual information, these systems can achieve enhanced and fine-grained control over various speech attributes such as prosody, emotion, style, and speaker characteristics ([InstructTTS [105]](../../Models/Acoustic/2023.01.31_InstructTTS.md),[Emo-DPO [31]](../../Modules/RLHF/2024.09.16_Emo-DPO.md); [ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md)).
Integrating LLMs into TTS systems represents a significant step forward, enabling more dynamic and expressive speech synthesis.

</td><td>

</td></tr></table>

### C·Organization of This Survey: 本文结构

<table><tr><td width="50%">

This paper first presents a comprehensive and systematic review of controllable TTS technologies, with a particular focus on model architectures, control methodologies, and feature representations.
To establish a foundational understanding, this survey begins with an introduction to the TTS pipeline in [Section 2](Sec.02_Pipeline.md).
While our focus remains on controllable TTS, [Section 3](Sec.03_UnControllableTTS.md) examines seminal works in uncontrollable TTS that have significantly influenced the field's development.
[Section 4](Sec.04_ControllableTTS.md) provides a thorough investigation into controllable TTS methods, analyzing both their model architectures and control strategies.
[Section 5](Sec.05_Datasets&Evaluation.md) presents a comprehensive review of datasets and evaluation metrics.
[Section 6](Sec.06_Challenges.md) provides an in-depth analysis of the challenges encountered in achieving controllable TTS systems and discusses future directions.
[Section 7](#Sec.07) explores the broader impacts of controllable TTS technologies and identifies promising future research directions, followed by the conclusion in [Section 8](#Sec.08).

</td><td>

本文首次展示了一份全面且系统化的可控 TTS 技术综述, 着重分析模型架构, 控制策略, 和特征表示.
- [第二节](Sec.02_Pipeline.md) 介绍 TTS 流程, 以建立起基础的理解.
- [第三节](Sec.03_UnControllableTTS.md) 仔细研究了不可控 TTS, 其对整个研究领域的发展有极大影响.
- [第四节](Sec.04_ControllableTTS.md) 详细调查了可控 TTS 方法, 分析了其模型架构和控制策略.
- [第五节](Sec.05_Datasets&Evaluation.md) 展示了数据集和评价指标的全面综述.
- [第六节](Sec.06_Challenges.md) 深入分析了实现可控 TTS 系统的挑战, 并提出了未来研究方向.
- [第七节](#Sec.07) 探讨了可控 TTS 技术的广泛影响, 并提出了有前途的研究方向.
- [第八节](#Sec.08) 总结了本文.

</td></tr></table>

## 2·[TTS Pipeline: TTS 流程](Sec.02_Pipeline.md)

## 3·[Uncontrollable TTS: 非可控 TTS](Sec.03_UnControllableTTS.md)

## 4·[Controllable TTS: 可控 TTS](Sec.04_ControllableTTS.md)

## 5·[Datasets & Evaluation: 数据集和评估](Sec.05_Datasets&Evaluation.md)

## 6·[Challenges & Future Directions: 挑战与未来方向](Sec.06_Challenges.md)

## 7·Discussion: 讨论

<a id="Sec.07"></a>

### A·Impacts of Controllable TTS: 可控 TTS 的影响

#### Applications: 应用

<table><tr><td width="50%">

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

</td><td>

</td></tr></table>

#### Deepfakes: 深度伪造

<table><tr><td width="50%">

A deepfake is a type of synthetic media in which a person in an existing image, video, or audio recording is replaced with someone else’s likeness or voice.
This technology uses deep learning, particularly GANs~\cite{zhang2022deepfake}, to create highly realistic, but fabricated, content.
While deepfakes are most commonly associated with video manipulation, such as face swapping~\cite{nirkin2019fsgan}, they can also apply to audio, enabling the creation of synthetic speech that mimics a specific person’s voice, which is well known as voice cloning.

Voice cloning, especially few-shot~\cite{arik2018neural} and zero-shot TTS~\cite{wang2023neural,[MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md)}, poses a significant threat to systems that rely on voice authentication, such as banking, customer service, and other identity verification processes.
With a convincing clone of someone’s voice, attackers could potentially impersonate individuals to gain unauthorized access to sensitive information or accounts.

To address these concerns, it’s essential to establish robust security protocols, consent-based regulations, and public awareness around voice cloning.
Furthermore, advancements in detecting voice clones are equally important to help distinguish genuine voices from synthesized ones, protecting both individuals and organizations from potential misuse.

</td><td>

</td></tr></table>

### B·Limitation of this Survey: 本综述的局限性

<table><tr><td width="50%">

Although we conduct a comprehensive survey of controllable TTS in this paper, there are some limitations we want to address in the future:
1) A unified benchmark method will be provided to evaluate controllable TTS methods.
2) Detailed analysis and control strategies of each specific speech attribute will be provided in an updated version of this paper.
3) The methodologies for feature disentanglement are crucial for controllable TTS but are not adequately discussed.

</td><td>

</td></tr></table>

## 8·Conclusions: 结论

<a id="Sec.08"></a>

<table><tr><td width="50%">

In this survey paper, we first elaborate on the general pipeline for controllable TTS, followed by a glimpse of uncontrollable TTS methods.
Then we comprehensively review existing controllable methods from the perspectives of model architectures and control strategies.
Popular datasets and commonly used evaluation metrics for controllable TTS are also summarized in this paper.
Besides, the current challenges are deeply analyzed and the promising future directions are also pointed out.
To the best of our knowledge, this is the first comprehensive survey for controllable TTS.

Writing a comprehensive survey is a challenging task and an iterative process.
Hence, we will regularly update this survey to offer researchers and practitioners a continuously evolving resource for understanding controllable speech synthesis.

</td><td>

在本综述中, 我们
- 首次阐述了可控 TTS 的一般流程, 并简要介绍了不可控 TTS 方法.
- 从模型架构和控制策略的角度全面回顾了现有的可控方法.
- 总结了可控 TTS 常用的数据集和评价指标.
- 深度分析了现有挑战, 并指出了未来发展方向.

据我们所知, 这是关于可控 TTS 的首篇全面综述.

撰写一篇全面综述是一个具有挑战性的任务, 也是一项迭代式的过程.
因此, 我们将不断更新本综述, 为研究者和从业者提供可持续发展的可控语音合成资源.

</td></tr></table>
