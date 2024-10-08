# Recent Advances in Speech Language Models: A Survey

<details>
<summary>基本信息</summary>

- 标题: "Recent Advances in Speech Language Models: A Survey"
- 作者:
  - 01 Wenqian Cui - 香港中文大学 CUHK
  - 02 Dianzhi Yu - 香港中文大学 CUHK
  - 03 Xiaoqi Jiao - 腾讯 LightSpeed Studios
  - 04 Ziqiao Meng - 香港中文大学 CUHK
  - 05 Guangyan Zhang - 腾讯 LightSpeed Studios
  - 06 Qichao Wang - 腾讯 AI Lab
  - 07 Yiwen Guo - 独立研究者
  - 08 Irwin King - 香港中文大学 CUHK
- 链接:
  - [ArXiv](https://arxiv.org/abs/2410.03751)
  - [Publication]
  - [Github]
  - [Demo]
- 文件:
  - [ArXiv](2410.03751v1__Survey__Recent_Advances_in_Speech_Language_Models__A_Survey.pdf)
  - [Publication] #TODO

</details>

## Abstract: 摘要

<details>
<summary>展开原文</summary>

Large Language Models (LLMs) have recently garnered significant attention, primarily for their capabilities in text-based interactions.
However, natural human interaction often relies on speech, necessitating a shift towards voice-based models.
A straightforward approach to achieve this involves a pipeline of "Automatic Speech Recognition (ASR) + LLM + Text-to-Speech (TTS)", where input speech is transcribed to text, processed by an LLM, and then converted back to speech.
Despite being straightforward, this method suffers from inherent limitations, such as information loss during modality conversion and error accumulation across the three stages.
To address these issues, Speech Language Models (SpeechLMs)---end-to-end models that generate speech without converting from text---have emerged as a promising alternative.
This survey paper provides the first comprehensive overview of recent methodologies for constructing SpeechLMs, detailing the key components of their architecture and the various training recipes integral to their development.
Additionally, we systematically survey the various capabilities of SpeechLMs, categorize the evaluation metrics for SpeechLMs, and discuss the challenges and future research directions in this rapidly evolving field.

</details>
<br>

大语言模型 (Large Language Models, LLMs) 最近获得了越来越多的关注, 主要用于文本交互的能力.
然而, 自然的人类交互往往依赖于语音, 因此需要转移到基于声音的模型.
实现这一目标的一个直接的方法是使用 "自动语音识别 (Automatic Speech Recognition, ASR) + 大语言模型 LLM + 文本转语音 (Text-to-Speech, TTS)" 构成的流程, 其中输入的语音被转录为文本, 然后被 LLM 处理, 并转换回语音.
尽管这种方法很直接, 但它也存在固有的限制, 如在模态转换过程中的信息丢失和三个阶段的错误累积.
为了解决这些问题, 出现了**语音语言模型 (Speech Language Models, SpeechLMs)**, 即无需从文本转换而直接生成语音的模型, 成为一个有前途的替代方案.
本综述首次全面概述了构造语音语言模型的近期方法, 详细介绍了其架构的关键组件和用于开发的各种训练方法.
此外, 我们系统地调查了语音语言模型的各种能力, 分类了评估指标, 并讨论了该领域的挑战和未来研究方向.

## 1.Introduction: 引言

## 2.Related Works: 相关工作

## 3.Methodology: 方法

## 4.Experiments: 实验

## 5.Results: 结果

## 6.Conclusions: 结论

<details>
<summary>展开原文</summary>

This survey provides a comprehensive overview of recent advancements in Speech Language Models (SpeechLMs).
We begin by addressing the limitations of the naive framework that combines Automatic Speech Recognition (ASR), Large Language Models (LLMs), and Text-to-Speech (TTS) systems for voice interactions.
Next, we highlight the key advantages offered by SpeechLMs.
Following this, we explore the architectures of SpeechLMs, detailing the components involved and their training recipes.
We also discuss their capabilities in various downstream applications as well as their various evaluation methods.
Finally, we identify the major challenges in developing SpeechLMs and outline potential directions for future research.
We hope this survey will illuminate the field and assist the research community in creating more powerful Speech Language Models.

</details>
<br>

本综述提供了关于语音语言模型 (Speech Language Models, SpeechLMs) 近期进展的全面概览.
我们首先讨论了将自动语音识别 (Automatic Speech Recognition, ASR), 大语言模型 (Large Language Models, LLMs), 以及文本转语音 (Text-to-Speech, TTS) 系统结合起来用于语音交互的简单框架的局限性.
然后, 我们着重介绍了语音语言模型 (Speech Language Models, SpeechLMs) 提供的关键优势.
接下来, 我们探索了语音语言模型 (Speech Language Models, SpeechLMs) 的架构, 详细介绍了所涉及的组件及其训练方法.
我们还讨论了它们在不同下游应用中的能力, 以及它们的各种评估方法.
最后, 我们确定了开发语音语言模型 (Speech Language Models, SpeechLMs) 的主要挑战, 并概述了未来研究的潜在方向.
希望本综述能够照亮该领域, 并为研究社区提供更强大的语音语言模型.
