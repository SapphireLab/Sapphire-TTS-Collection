# Direct Speech-to-Speech Neural Machine Translation: A Survey

<details>
<summary>基本信息</summary>

- 标题: "Direct Speech-to-Speech Neural Machine Translation: A Survey"
- 作者:
  - 01 Mahendra Gupta (Dept of CSE NITTTR Chandigarh, India)
  - 02 Maitreyee Dutta (Dept of CSE NITTTR Chandigarh, India)
  - 03 Chandresh Kumar Maurya (Dept of CSE IIT Indore, India)
- 链接:
  - [ArXiv](https://arxiv.org/abs/2411.14453)
  - [Publication]()
  - [Github]()
  - [Demo]()
- 文件:
  - [ArXiv](2411.14453v1__Survey__Direct_Speech-to-Speech_Neural_Machine_Translation__A_Survey.pdf)
  - [Publication] #TODO

</details>

## Abstract: 摘要

<details>
<summary>展开原文</summary>

Speech-to-Speech Translation (S2ST) models transform speech from one language to another target language with the same linguistic information.
S2ST is important for bridging the communication gap among communities and has diverse applications.
In recent years, researchers have introduced direct S2ST models, which have the potential to translate speech without relying on intermediate text generation, have better decoding latency, and the ability to preserve paralinguistic and non-linguistic features.
However, direct S2ST has yet to achieve quality performance for seamless communication and still lags behind the cascade models in terms of performance, especially in real-world translation.
To the best of our knowledge, no comprehensive survey is available on the direct S2ST system, which beginners and advanced researchers can look upon for a quick survey.
The present work provides a comprehensive review of direct S2ST models, data and application issues, and performance metrics.
We critically analyze the models' performance over the benchmark datasets and provide research challenges and future directions.

</details>
<br>

**语音到语音翻译 (Speech-to-Speech Translation, S2ST)** 模型将一种语言的语音转换为另一种目标语言的语音, 同时保留相同的语言信息.
S2ST 对于弥合社区之间的沟通障碍至关重要, 并具有多样化的应用.
近年来, 研究人员引入了直接 S2ST 模型, 这些模型有可能在不依赖中间文本生成的情况下进行语音翻译, 具有更好的解码延迟, 并能够保留语言学和非语言特征.
然而, 直接 S2ST 尚未实现无缝沟通的质量性能, 并且在性能方面仍然落后于级联模型, 尤其是在实际翻译过程中.
据我们所知, 目前还没有关于直接 S2ST 系统的全面综述, 以便初学者和高级研究人员可以快速浏览.

本工作提供了对直接 S2ST 模型, 数据, 应用问题和性能指标的全面回归.
我们批判性地分析模型在基准数据集上的性能, 提供了研究挑战和未来方向.

## 1·Introduction: 引言

The S2ST task is the process of transforming speech in a source language to speech in a target language.
It finds applications in live audio/video translation,  international conferences and meeting translations, educational video translation, and movie dubbing, to name a few.
The rising computational power of computing devices, the expanding bandwidth of communication channels, and the progress in deep learning (DL) and machine learning (ML) techniques have enabled seamless communication.

Traditionally, the S2ST problem is solved by the cascade approach in that automatic speech recognition (ASR), machine translation (MT), and text-to-speech synthesis (TTS) models are glued together \cite{Laive_1997, Nakamura2006}.
Another approach to building a cascade S2ST model is by chaining speech-to-text translation (ST) and TTS models.
Nonetheless, the cascade models have been the de-facto choice for S2ST for a long time due to the easier availability of the pre-trained component models (ASR, MT, and TTS).
However, they are being challenged by direct S2ST models in recent years owing to error propagation, higher training time, and memory cost of cascade models (refer to \S \ref{cascadevse2e} for more).

Direct S2ST models translate source speech into target speech without using intermediate text representation.
They are being popularised by the possibility of learning paralinguistic and non-linguistic features, including speaking style, emotions, energy, prosody, etc.
\cite{Jia2019, Jia2021, Lee2022}.
Languages without a writing system, often called unwritten languages, constitute 40\% of all languages \cite{lee-etal-2022-textless}.
Text-based model training is not feasible for these languages.
The direct S2ST models can potentially address the challenge posed by unwritten languages \cite{Tjandra_2019_untranscribe, Zhang2021_UWSpeech, lee-etal-2022-textless}.
Direct models follow an end-to-end (E2E) training approach, reducing error propagation and offering lower latency than cascade models \cite{Jia2019, Jia2021, Lee2022}.
However, despite their advantages, these models encounter notable challenges such as: (a) getting sufficient parallel speech corpus in two different languages is extremely hard, thus hampering model development, (b) training with the speech of unwritten languages and their evaluation, (c) the potential threat of voice cloning, (d) absence of metrics directly taking generated and reference speech as input and returning a quality score, and (e) segmentation issue especially in simultaneous S2ST, and so on.
Further advancements are still required for S2ST systems to attain the level of quality necessary for hassle-free communication.

To provide an overview of research done in the S2ST field and a bucket of open problems,  the present work comprehensively reviews the direct S2ST literature, whose broad taxonomy is shown in Fig.\ref{taxonomy}.
Overall, the manuscript is organized as follows: \S\ref{Task_definition} defines the S2ST task; \S\ref{cascadevse2e} presents the cascade vs E2E models; \S\ref{datapaucity} discusses the data scarcity.
The performance metrics are presented in \S\ref{metrics} while \S\ref{repr} elucidates the segmentation and representation issues.
Models and training strategies are discussed in \S\ref{sec:direct_s2st} and \S\ref{trainstr}, respectively.
Application issues are discussed in \S\ref{application},  experiments in \S\ref{experiment}, challenges in \S\ref{challenges}, and finally concluded in \S\ref{challenges}.

## 2·Task Definition: 任务定义

Given a parallel speech corpus, denoted as $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$.
In this context, $x=\{f^s_1,\ldots, f^s_k\}$ and $y=\{f^t_1,\ldots, f^t_l\}$ represents the source and target speech utterances, respectively.
Here, $f^s$ and $f^t$ refer to the source and target speech frames, and $k$ and $l$ represent utterance length in frames, respectively.
The objective of direct S2ST models is to maximize the conditional probability $\mathcal{P}(y|x;\theta)$ as in \eqref{condobj} or minimize the negative log-likelihood loss \eqref{nll}.

$$
  \mathcal{P}(y|x;\theta)=\overset{k}{\underset{T=1}{\prod}}\mathcal{P}(f^t_T|f^t_{<T},x;\theta)
$$

$$
  \mathcal{L}_{(x,y)\in D}=-\overset{n}{\underset{i=1}{\sum}} \log \mathcal{P}(y_i|x_i;\theta)
$$

where $\mathcal{L}_{(x,y)\in D}$ is the cumulative loss on the dataset $D$ and $\theta$ is a model parameter.
Note that the problem formulation given in \eqref{condobj} is for \emph{Autoregressive} (AR) models \footnote{Non-autoregressive (NAR) models are an alternative modeling approach that have been proposed in the past few years.
Only a sparse number of works exist in the literature on S2ST where they use NAR in decoders of encoder-decoder frameworks.
We discuss NAR briefly in \S \ref{decoderarch}}

On the other hand, cascade models have access to source transcripts denoted as a sequence of tokens (words) $w^s= \{w^s_1, \ldots, w^s_p\}$,  and the target transcript as a sequence of tokens, represented as  $w^t = \{w^t_1, \ldots, w^t_q\}$.
In the 3-stage cascade model, the optimization objectives are as follows:

$$
  \mathcal{L}_{asr}=-\overset{n}{\underset{i=1}{\sum}} \log \mathcal{P}(w^s_i|x_i;\theta^{asr})
$$

$$
  \mathcal{L}_{mt}=-\overset{n}{\underset{i=1}{\sum}} \log \mathcal{P}(w^t_i|w^s_i;\theta^{mt})
$$

$$
  \mathcal{L}_{tts}=-\overset{n}{\underset{i=1}{\sum}} \log \mathcal{P}(f^t_i|w^t_i;\theta^{tts})
$$

$$
  \mathcal{L}_{st}=-\overset{n}{\underset{i=1}{\sum}} \log \mathcal{P}(w^t_i|x_i;\theta^{st})
$$

A 3-stage cascade model is built by independently minimizing the losses in \eqref{eq:asr_loss}, \eqref{eq:mt_loss}, and \eqref{eq:tts_loss}.
A direct E2E ST model \cite{Berard2018_Audiobook, Kano2020} optimizes the loss in \eqref{eq:st_loss} (may also use losses in \eqref{eq:asr_loss}, \eqref{eq:mt_loss} in a multitask setup).
This ST model is combined with the TTS model to form a 2-stage cascade S2ST model.

## 3·Cascade vs End-to-End S2ST models: 级联模型与端到端模型

The traditional S2ST systems follow a cascade architecture \cite{Laive_1997, Ney_1999_ST, Nakamura2006, Wahlster2000_Verbmobli}.
They are designed either by chaining ASR, MT, and TTS or ST followed by TTS as illustrated in Figure \ref{fig:S2ST_Cascade_Vs_Direct} (i) and (ii) respectively.
Either way, the cascade system relies on intermediate text.
As such, they face several issues in modeling S2ST effectively.
Firstly, they face challenges when dealing with low-resource languages lacking annotated corpora or unwritten languages \cite{Chen2022}.
Secondly, paralinguistic features such as prosody, intonation, emotions, etc.
are lost when representing speech via intermediate text, quintessential for building a real-world S2ST system.
Thirdly, error propagation from one module to another \cite{Jia2019}, higher training time \cite{Huang2022_TranSpeech}, and memory footprint (due to 3 models vs 1 in direct S2ST) prohibit their application to low-powered devices.

The above issues with cascade systems catapulted the development of direct S2ST systems bypassing the intermediate text generation, reducing training time and memory cost.
There has been a lot of work developing direct S2ST models \cite{Jia2019, Jia2021, jia-etal-2022-cvss, Huang2022_TranSpeech, diwan2024textless}, etc.
Therefore, it is imperative to compare cascade and direct models on quantitative and qualitative metrics (discussed in \S \ref{metrics}).
Our literature survey reveals that there was a performance gap between cascade and direct models (both in terms of BLEU and mean opinion score (MOS)) \cite{Jia2019, Jia2021, lee-etal-2022-textless, zhu-etal-2023-diffs2ut, Huang2022_TranSpeech} which is now almost closed \cite{Chen2022, peng2024mslms2st}.
These studies, however, are done on limited language pairs and may not generalize.
Therefore, it remains to see an exhaustive comparison over multiple and distant language pairs involving large-scale datasets to truly establish the claim that the performance gap is indeed closed.

## 4·Experiments: 实验

## 5·Results: 结果

## 6·Conclusions: 结论