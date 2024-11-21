# WavChat

<details>
<summary>基本信息</summary>

- 标题: "WavChat: A Survey of Spoken Dialogue Models"
- 作者:
  - 01 Shengpeng Ji (浙江大学, shengpengji@zju.edu.cn)
  - 02 Yifu Chen (浙江大学)
  - 03 Minghui Fang (浙江大学)
  - 04 Jialong Zuo (浙江大学)
  - 05 Jingyu Lu (浙江大学)
  - 06 Hanting Wang (浙江大学)
  - 07 Ziyue Jiang (浙江大学)
  - 08 Long Zhou (微软)
  - 09 Shujie Liu (微软)
  - 10 Xize Cheng (浙江大学)
  - 11 Xiaoda Yang (浙江大学)
  - 12 Zehan Wang (浙江大学)
  - 13 Qian Yang (浙江大学)
  - 14 Jian Li (腾讯优图实验室)
  - 15 Yidi Jiang (阿里巴巴)
  - 16 Jingzhen He (阿里巴巴)
  - 17 Yunfei Chu (阿里巴巴)
  - 18 Jin Xu (阿里巴巴)
  - 19 Zhou Zhao (浙江大学, zhaozhou@zju.edu.cn)
- 链接:
  - [ArXiv](https://arxiv.org/abs/2411.13577)
  - [Publication]
  - [Github](https://github.com/jishengpeng/WavChat)0
  - [Demo]
- 文件:
  - [ArXiv](2411.13577v1__Survey__WavChat__A_Survey_of_Spoken_Dialogue_Models.pdf)
  - [Publication] #TODO

</details>

## Abstract: 摘要

Recent advancements in spoken dialogue models, exemplified by systems like GPT-4o, have captured significant attention in the speech domain.
In the broader context of multimodal models, the speech modality offers a direct interface for human-computer interaction, enabling direct communication between AI and users.
Compared to traditional three-tier cascaded spoken dialogue models that comprise speech recognition (ASR), large language models (LLMs), and text-to-speech (TTS), modern spoken dialogue models exhibit greater intelligence.
These advanced spoken dialogue models not only comprehend audio, music, and other speech-related features, but also capture stylistic and timbral characteristics in speech.
Moreover, they generate high-quality, multi-turn speech responses with low latency, enabling real-time interaction through simultaneous listening and speaking capability.
Despite the progress in spoken dialogue systems, there is a lack of comprehensive surveys that systematically organize and analyze these systems and the underlying technologies.
To address this, **we have first compiled existing spoken dialogue systems in the chronological order and categorized them into the cascaded and end-to-end paradigms**.
We then provide an in-depth overview of the core technologies in spoken dialogue models, covering aspects such as **speech representation, training paradigm, streaming, duplex, and interaction capabilities**.
Each section discusses the limitations of these technologies and outlines considerations for future research.
Additionally, we present a thorough review of **relevant datasets, evaluation metrics, and benchmarks** from the perspectives of training and evaluating spoken dialogue systems.
We hope this survey will contribute to advancing both academic research and industrial applications in the field of spoken dialogue systems.
The related material is available at [Github](https://github.com/jishengpeng/WavChat).

## 1·Introduction: 引言

Spoken dialogue models \cite{defossez2024moshi,zhang2023speechgpt,xie2024miniomni2opensourcegpt4ovision} represent one of the most direct methods of human-computer interaction, evolving from traditional voice assistants such as Alexa ([Website](https://www.alexa.com/)), Siri ([Website](https://www.apple.com/siri/)), and Google Assistant ([Website](https://assistant.google.com/)) to the latest intelligent dialogue systems, such as GPT-4o ([Website](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/)).
The fundamental definition of a spoken dialogue model refers to a dialogue system capable of generating intelligent verbal responses based on the input speech.
On the one hand, the **speech modality** serves as both the input and output interface for the human-computer interaction in the spoken dialogue models.
On the other hand, the **dialogue system** \cite{dubey2024llama} requires the model to possess a certain level of textual intelligence, including the ability to comprehend the knowledge of human society and generating professional and intelligent responses.
Recently, intelligent spoken dialogue systems, exemplified by GPT-4o and Moshi \cite{defossez2024moshi}, have garnered significant attention for their ability to extend speech intelligence capabilities beyond traditional text-based dialogue models~\cite{huang2024audiogpt}.
These dialogue models can not only generate natural, human-like speech responses \cite{defossez2024moshi,speechteam2024funaudiollm} but also demonstrate an advanced understanding and generation of acoustic features beyond text, such as timbre, emotion, and style \cite{lin2024advancing,lin2024paralinguistics,xue2023chat}.
Additionally, they exhibit strong performance in processing other speech-related representations, including music and audio events~\cite{chu2024qwen2,chu2023qwen,gong2023joint,tang2023salmonn}.
Their realistic conversational interactivity \cite{fu2024vita,xie2024miniomni2opensourcegpt4ovision} and low-latency dialogue experiences~\cite{defossez2024moshi} further distinguish them among the traditional spoken dialogue models.

The history of spoken dialogue models can be traced back to early systems like dGSLM~\cite{nguyen2023generative} and AudioGPT~\cite{huang2024audiogpt}, leading up to more recent advancements such as GPT-4o and Moshi~\cite{defossez2024moshi}.
During this period, many notable spoken dialogue models have emerged.
As shown in Figure \ref{fig:img1}, we have organized these models in chronological order.
Broadly, they can be categorized into two types: cascaded spoken dialogue models~\cite{chu2024qwen2,chu2023qwen} and end-to-end~\cite{mentzer2023finite,xie2024mini,zhang2024omniflatten,zhang2024intrinsicvoice} spoken dialogue models.
Given that most current spoken dialogue models rely on alignment with the text modality, the distinction between cascaded and end-to-end models is crucial.
As illustrated in Figure \ref{fig:img2}, we classify all spoken dialogue models based on whether **the core language model can directly understand and generate speech representations**, dividing them into cascaded and end-to-end categories.
Traditional cascaded spoken dialogue systems such as AudioGPT~\cite{huang2024audiogpt} are structured around text as the central intermediary, typically comprising three cascaded modules.
First, the input audio is transcribed into text by an automatic speech recognition (ASR) module~\cite{radford2023robust}.
The transcribed text is then fed into a large language model (LLM) such as ChatGPT to generate a textual response.
Finally, this textual response is converted back into audio through a text-to-speech (TTS) module~\cite{kong2023vits2,ren2020fastspeech}.
While this cascaded architecture leverages the strong in-context capabilities of large language models, it introduces several challenges, including high latency, limited interactivity, and the inability to process non-textual information.
To address these issues, recent research has taken two primary directions.
Some approaches~\cite{chu2023qwen,tang2023salmonn} focus on optimizing the understanding and generation components within the cascaded system to mitigate the aforementioned limitations.
Some other approach~\cite{xie2024mini,xie2024miniomni2opensourcegpt4ovision,zhang2024speechgpt,zhang2024intrinsicvoice} seek to directly solve these problems by adopting end-to-end architectures for spoken dialogue systems.
Although end-to-end spoken dialogue models exhibit various differences in terms of representations and model architectures, they share a common feature: they do not rely on text as the central intermediary.
Instead, these models aim to directly comprehend and generate speech representations.
We define such systems as end-to-end spoken dialogue models.

When constructing spoken dialogue systems, we identify four core technologies closely related to spoken dialogue models, based on the different levels of intelligence involved.
The first is the design of speech representations (i.e., tokenizers and detokenizers).
The second concerns the paradigm for training, inference, and generation, specifically how to align the speech modality with the text modality while preserving or enhancing the intelligence of existing text-based dialogue models.
This part also involves selecting different model architectures, generation strategies, and multi-stage training approaches.
The third challenge involves the design of interactive, duplex, streaming for spoken dialogue systems.
Lastly, the fourth challenge relates to data—specifically, how to construct training datasets for spoken dialogue systems and evaluate their performance.

Given these considerations, in the following sections of this paper, we address these four key technologies in the order outlined above.
- In Section 2, we provide an overview of spoken dialogue systems, including typical spoken dialogue scenarios (i.e., how to define a spoken dialogue model) and recent developments in the cascaded and end-to-end spoken dialogue models.
- Section 3 focuses on the speech representations used in spoken dialogue systems.
- In Section 4, we systematically discuss the training paradigms, with particular emphasis on how to align the speech modality with the text modality, as well as multi-stage training strategies, model architectures, and generation strategies.
- Section 5 highlights the unique characteristics of spoken dialogue systems, particularly their duplex, streaming nature, which distinguishes them from text-based dialogue systems.
- In Section 6, we examine the construction of training datasets and the evaluation methodologies specific to spoken dialogue models.
At the end of each section, we include a summary and discussion to reflect on the key insights.
- Finally, in Section 7, we conclude the survey by summarizing the major findings and discussing open issues for future research.

Given the complexity of the technical points, we provide an overview of the structure of this survey in Figure \ref{fig:img4}.

## [2·Overall: 整体视角](Sec.02.md)

## [3·Representations of Spoken Dialogue Models: 口语对话模型的表示](Sec.03.md)

## [4·Training Paradigm of Spoken Dialogue Model: 口语对话模型的训练范式](Sec.04.md)

## [5·Streaming, Duplex, and Interaction: 流式, 双工, 和交互](Sec.05.md)

## [6·Training Resources and Evaluation: 训练资源和评估](Sec.06.md)

## 7·Conclusions: 结论

In this work, we systematically review the research related to spoken dialogue models, categorizing it according to two paradigms: cascaded spoken dialogue models and end-to-end spoken dialogue models.
Additionally, we provide a detailed overview of the core technologies behind spoken dialogue models, including speech representation, training paradigms, streaming duplex systems, and interaction mechanisms.
In the speech representation module, we classify and explain the representations from both the input and output perspectives, focusing on different types of semantic and acoustic representations.
In the training paradigm module, we thoroughly discuss five modalities of alignment for spoken dialogue models, multi-stage training strategies, model architectures, and generation paradigms.
Following this, we provide an in-depth analysis of streaming input and output for spoken dialogue models, as well as the related duplex interaction technologies.
Finally, we compile key training resources, evaluation metrics, and benchmarks relevant to spoken dialogue models.
We specifically address the evaluation of different levels of intelligence in spoken dialogue models across various scenarios.
It is important to note that, given that spoken dialogue models are a relatively new and emerging technology, many aspects such as semantic and acoustic representations, still lack well-established paradigms.
Therefore, at the end of each section, we include a dedicated discussion module to explore these open issues.
We hope that this survey will contribute to the further development of the field of spoken dialogue systems.