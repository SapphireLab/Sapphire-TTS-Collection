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

## 4·Strategies for Addressing Data Scarcity: 处理数据不足的策略

Parallel speech corpora are crucial for training direct S2ST models, and their scarcity can significantly affect their training, performance, and robustness.
Creating parallel speech corpora requires substantial resource investment \cite{Chen2023_DataAugment_survey}, resulting in limited available datasets, as shown in Table \ref{Dataset_Stats} in the appendix.
While this scarcity of supervised S2ST datasets prevents the application of direct S2ST systems to low-resource languages, techniques like data augmentation, pre-training, self-training, back-translation, knowledge distillation, and speech mining are employed to address data scarcity and enable S2ST for low-resource languages.
These methods are further elaborated in the following sections.

### Data Augmentation: 数据增强

Data augmentation (DA) refers to techniques that boost the number of data points to enhance diversity without requiring the explicit collection of real supervised data points \cite{Feng2021_DataAug_survey}.
The DA techniques suitable for various NLP tasks may not be directly applicable to S2ST since S2ST involves parallel speech, posing a challenge for augmenting existing data.
Speech data can be augmented in various ways.
For example, by adding noise, speed and pitch perturbation, time and frequency masking, etc.
In \cite{Jia2021}, data augmentation technique \emph{ConcatAug} is employed to preserve the speaker's voice on their turn.
In particular,  training data is enhanced by randomly selecting pairs of training examples and concatenating their source speech, target speech, and target phoneme sequences to create new training examples.
These new examples incorporate the voices of two different speakers in both the source and target speech, allowing the model to learn from instances that include speaker turns.
Many studies utilize synthesized speech from the target text in ST data \cite{Jia2019, Jia2021, Lee2022, nachmani_2023_translatotron3}, while others make use of MT and TTS models to label ASR corpora through pseudo-labeling \cite{Dong2022_Psudo_Labeled, Jia2022_Leveraging, Popuri2022_Enahancing_SelfSupervised,huang-etal-2023-xiaomi}.

### Pre-training: 预训练

Pre-training \cite{Popuri2022_Enahancing_SelfSupervised, Wei_2013_joint_pre-train_s2st} aims to improve the performance of the task in one domain with the learned knowledge from the tasks in another domain \cite{Pan2010_TransferLearning, Wang2022_Pre-Training_Application}.
Models learn salient features from an extensive training corpus during pre-training.
Subsequently, the pre-trained model is applied to another (target) task, employing smaller datasets for fine-tuning~\cite{Van_dem_ordo_RepresentationLearning,tang-etal-2022-unified}.
In recent works, self-supervised strategies are used for pre-training, to leverage the availability of large unlabeled datasets.
Several pre-trained DL models, including BERT \cite{Devlin2019_BERT}, BART \cite{lewis-etal-2020-BART} (using unit mBART version), Wav2Vec 2.0 \cite{baevski2020_wav2vec2.0}, HuBERT \cite{Hsu_2021_HuBERT}, w2v-BERT \cite{Chung2021_w2V-BERT}, and VQ-VAE \cite{Van_Oord_2017_VQ-VAE}, which are trained on extensive unlabeled data, serve as the foundation for various NLP and speech tasks.
The pre-trained models are used as encoders/decoders in direct S2ST models \cite{bansal2018pre, tsiamas-etal-2022-pretrained, Lee2022_Representation_Learning_for_Speech} (refer to \S \ref{sec:direct_s2st})

### Self-Training & Back-Translation: 自训练与回译

Self-training and Back-translation (BT) algorithms leverage monolingual data to train models requiring supervised data but lacking a supervised parallel corpus.
These algorithms use the model's own confident predictions for monolingual data, which are initially trained on available parallel data \cite{edunov-etal-2018-understanding_BT, yang2022survey_semi_super}.
Self-training utilizes source monolingual data, while BT utilizes target monolingual data to generate augmented data.
Let us consider a utterance-wise parallel speech corpus denoted as $\mathcal{D}=\{(x_i, y_i)\}_{i=1}^n$, with $(x_i,y_i)$ representing the source and  target utterance, respectively.
Additionally, we have source monolingual corpus $\mathcal{M}_{src} = \{u_j\}_{j = 1}^m$, and target monolingual corpus $\mathcal{M}_{tgt} = \{v_k\}_{k = 1}^q$, where \(m,q \gg n\).
The forward translation model $f_{x\rightarrow y}$ is optimized for the parameter  \(\theta_{x\rightarrow y} = \text{argmin}_\theta \sum_{(x_i, y_i) \in \mathcal{D}} \mathcal{L}(f^\theta_{x\rightarrow y}(x_i), y_i)\), where $\mathcal{L}$ is the loss function.
Similarly, another model in backward direction, $f_{y\rightarrow x}$, is optimized for the parameter \(\theta_{y\rightarrow x} = \text{argmin}_\theta \sum_{(x_i, y_i) \in \mathcal{D}} \mathcal{L}(f^\theta_{y\rightarrow x}(y_i), x_i)\).
Using the already trained model \(f_{x \rightarrow y}\) and \(\mathcal{M}_{src}\), pseudo-labeled utterances \(\hat{y}\) are generated to create a new auxiliary parallel corpus $\mathcal{A}^s = \{(u_j, \hat{y}_j)\}_{j=1}^{m'}$ by selecting confident predictions.
In self-training, as shown in Figure \ref{fig:BT_KD_ST}(a), the model \(f_{x \rightarrow y}\) is retrained on the augmented data \(\mathcal{D} \cup \mathcal{A}^s\) \cite{He2020_revisiting_SelfTraining}.
Similarly, an auxiliary parallel corpus \(\mathcal{A}^t = \{(\hat{x}_k, v_k)\}_{k=1}^{q'} \) is generated using the backward-trained model \(f_{y \rightarrow x}\).
When employing back-translation (BT) the model \(f_{x \rightarrow y}\) is trained in the forward direction on the newly augmented data \(\mathcal{D} \cup \mathcal{A}^t\) as depicted in Figure \ref{fig:BT_KD_ST}(b) \cite{sennrich-etal-2016-improving}.
Various studies utilize a denoising version of BT, where noise is added to the input \cite{fu_2023_improving_BackTranslation}.
Several studies demonstrate that self-training \cite{Pino2020_ST_4ST} and BT \cite{Pino2020_ST_4ST} are highly beneficial, particularly for low-resource languages compared to high-resource ones.
These algorithms also benefit S2ST directly or indirectly, with the indirect method proving more effective \cite{Popuri2022_Enahancing_SelfSupervised,nachmani_2023_translatotron3}.

### Knowledge Distillation: 知识蒸馏

Knowledge Distillation (KD) transfers learned knowledge from a large ensemble model to a smaller model as shown in Figure \ref{fig:BT_KD_ST}(c) \cite{Hinton2015_KnowledgeDistilation, Treviso2023_NLP_Methods_Survey}.
The KD process is based on the teacher-student learning paradigm: the larger model serves as the teacher, while the smaller model acts as the student.

KD proves to be valuable for tasks such as  ASR \cite{Hinton2015_KnowledgeDistilation}, ST \cite{Inaguma2021, Liu2019d_KnowledgeDistillation}, and S2ST \cite{Huang2022_TranSpeech} in scenarios with limited resources.
In particular, AV-TranSpeech  \citet{Huang2022_TranSpeech} applies cross-modal distillation from audio-only pre-trained S2ST \cite{Popuri2022_Enahancing_SelfSupervised} to AV-TranSpeech.
Doing so initializes the audio encoder and unit decoder and alleviates the low-resource problem of audio-visual data.
Nonetheless, this approach remains relatively under-explored in the context of direct S2ST models.

### Speech Mining: 语音挖掘

Speech mining is an extension of \textbf{bitext mining} \cite{resnik-1998-parallel}, designed to discover parallel speech corpora from monolingual speech corpora.
Bitext refers to text-to-text parallel data where the source and target sentences are in different languages.
Bitext mining has been done in a supervised and unsupervised way (no cross-lingual resources like parallel text or bilingual lexicons are used) \cite{keung-etal-2020-unsupervised}.
Multilingual fixed-length sentence embedding
\cite{heffernan-etal-2022-bitext} using KD, contrastive learning \cite{tan2022bitext} are used for mining bitext in low-resource settings.
Sentence embedding serves to cluster similar sentences closely together within the latent space \cite{schwenk-2018-filtering,artetxe-schwenk-2019-massively}.
Similarly, for speech mining, multilingual fixed-length speech embedding is employed to represent the variable-length speech utterances.
Speech mining has been used by a few works such as \citet{Duquenne_2021_Speech_Mining} where they utilize a teacher-student approach to train the multilingual speech embedding, which produces speech representations that are compatible with the output of text embedding layer.

## 5·Performance Parameters & Metrics: 性能参数与指标

Offline S2ST systems are evaluated using Various text-based metrics such as BLEU \cite{papineni-etal-2002-bleu}, ScareBLEU \cite{Post2018_scareBLEU}, BLEURT \cite{Sellam2020_BLEURT}, COMET \cite{Rei2020_COMET, Rei2022_COMET-22}, and chrF \cite{Popovic_2015_chrF} are employed to measure the quality of translated speech.
These metrics are used after transcribing the speech through ASR due to the absence of direct evaluation metrics.
These metrics depend on the performance of ASR models, and they also pose challenges for low-resource and unwritten languages.
This is primarily due to either the unavailability of ASR systems or the existence of poor-quality models \cite{Salesky2021}.
Recently, \citet{Chen2023_BLASER} introduced a metric called \textbf{BLASER}\footnote{Balanced Accuracy for Speech Evaluation by Recognition}, which is a text-free S2ST quality metric.
BLASER takes source, reference, and generated speeches, embeds them to a common representation space via a shared encoder, and finally computes a score via a neural regressor.
Formally, \emph{unsupervised} BLASER score is given below.

$$
  \text{BLASER}_U = \frac{cos(h_{src}, h_{mt})+cos(h_{mt},h_{ref})}{2}
$$

where cos(·, ·) is the cosine similarity function, $h_{src}, h_{mt}$, and $h_{ref}$ are the source, generated, and reference speech representations, respectively.

There is also a supervised BLASER score proposed \cite{Chen2023_BLASER}.

### Naturalness

Naturalness measures how closely synthesized speech resembles natural speech.
This is measured by a human-based subjective metric known as \textbf{MOS} (Mean Opinion Score).
It is a numerical measure that represents the average subjective rating given by human listeners to the quality of a speech signal.
MOS provides a simple and intuitive measure of speech quality that correlates well with human perception.
Though MOS is widely used for speech quality, it has a few limitations such as subjective evaluation, bias, limited sample size,  time, and cost.
Therefore, it is recommended also to report some \emph{objective} metrics along with MOS, especially those that capture the perception of speech quality.
For example, the Perceptual Evaluation of Speech Quality (PESQ) \cite{rix2001perceptual}, its extension Perceptual Objective Listening Quality Analysis  (POLQA) \cite{beerends2013perceptual}, Short-Time Objective Intelligibility (STOI) \cite{taal2011algorithm} are a few objective algorithms designed to predict the perceived quality of speech as heard by human listeners.

### Voice-preservation

Voice-preservation measures the extent to which predicted speech is similar to a particular speaker's voice.
It can be calculated using metrics for evaluating voice cloning.
Measuring voice cloning involves a combination of subjective and objective methods to evaluate various aspects of the cloned voice, such as its similarity to the original voice (via speaker embedding similarity as done in \cite{Dong2023_PolyVoice}), naturalness, intelligibility, and acoustic properties.

Simultaneous S2ST (Simul-S2ST) systems are evaluated using quality-based metrics for offline S2ST models along with \textbf{latency} metrics.
Calculating the latency of the offline S2ST is straightforward: It is equal to the time elapsed in starting to produce the output from the decoder.
For Simul-S2ST, calculating the average latency poses a significant challenge since it sees only the partial input.
Due to the absence of such metrics, simultaneous text-to-text (Simul-T2T) latency metrics such as AP (average proportion) \cite{cho2016neural_AP} and AL (Average Lagging) \cite{ma-etal-2019-stacl_AL} are used as proxy.
AL is further adapted for the Simul-S2T task \cite{Ren2020_SimulSpeech}, which is also used for the Simul-S2ST task with the help of ASR due to the non-availability of direct metrics.

### Discussion

Current practice in S2ST works is that authors report only text-based quality metrics ignoring objective and subjective metrics as described in this section.
It is warranted that S2ST models are evaluated holistically using objective and subjective metrics as well.
Further, the development of more effective textless S2ST quality metrics
is recommended.

## 6·Segmentation & Representation Learning: 分割与表示学习

S2ST models essentially take speech and optionally take text as inputs.
Before we can train E2E S2ST models, we need to segment the speech and text followed by how to learn their representations.

Handling long speech and text sequences is a challenging task \cite{kim2017joint, Tsiamas2022SHASAO}.
The following section discusses segmentation and representation learning related to speech and text.

### 6.1·Segmentation Learning: 分割学习

Breaking down text into segments is a simpler task--it involves splitting based on robust punctuation marks used by current MT models.
E2E S2ST models necessitate intricate speech segmentation, primarily because of the significant role played by the \emph{out-of-order} word relationships between input and output, alongside the absence of linguistic features.
Traditionally, manual segmentation has been the norm for speech processing.
However, due to its labor-intensive nature, there is a growing need for segmentation learning.
Speech segmentation typically relies on either fixed-length splits which split the speech at fixed lengths, sometimes randomly \cite{Huang2022_TranSpeech} or \emph{pause} which splits the speech on Voice Activity Detection (VAD), as outlined by  \citet{Sohn1999ASM}.
A third method, the \emph{hybrid} approach, integrates both length and linguistic cues for segmentation, as discussed by  \cite{Potapczyk2020SRPOLsSF, Gaido2021BeyondVA, Tsiamas2022SHASAO}.
Notably, the hybrid approach demonstrates superior performance compared to length and pause-based methods \citep{Gaido2021BeyondVA}.
However, there is still a gap in the hybrid and manual approaches to segmentation, and future work may consider paying attention to this.

### 6.2·Representation Learning: 表示学习

Representation learning is an important issue in S2ST because speech in two different languages are two different modalities that may reside in different representation spaces.
Hence, we not only need better representation learning methods for speech and text but also their joint representation learning.

Existing works leveraging text data for S2ST modeling utilize LSTM \cite{Kano2021_Transformer}, Transformers \cite{Lee2022}, etc.
for representation learning.
As the current trend is towards building \emph{textless} models, more recent works focus on learning efficient speech representations.
Among them, the popular choices are \emph{raw waveform, spectrogram-based, unsupervised ASR}, and \emph{discrete-units based}.
The raw waveform is the use of raw speech signal directly fed to the Sequence-to-Sequence (Seq2Seq) model and has been utilized by \cite{wang2023speechtospeech, KimLCL24}, inter alia.
Mel-Filter Cepstral Coefficient (MFCC) feature, a spectrogram-based method, has been one of the most used speech representation methods \cite{Jia2019, jia2019leveraging_S2T, Tjandra_2019_untranscribe, Kano2021_Transformer, Lee2022, Huang2022_TranSpeech, Chen2023}, etc.
where 80-dimension mel-spectrogram is computed.

Obtaining a substantial volume of labeled speech data for supervised feature representation learning poses significant challenges.
Consequently, recent studies have turned to leveraging speech features acquired through \emph{unsupervised} and \emph{self-supervised}  methods.
These approaches involve mapping continuous speech signals into discrete units-- akin to words and sub-words in the text domain.
Such representations enable the integration of NLP tools into the speech domain.
Among them, the popular choices are Wav2Vec \citep{schneider2019wav2vec} and its variants such as w2v-BERT \citep{Chung2021_w2V-BERT, Jia2022_Leveraging} and Wav2Vec 2.0 \citep{baevski2020_wav2vec2.0, Chen2022, song_2023_styles2st} and Hidden-Units BERT (HuBERT) \citep{hsu2021hubert}.
What makes these representation methods take over the MFCC is that they can extract \emph{semantic-units}.
Hence recent S2ST models invariably exploit HuBERT \cite{Huang2022_TranSpeech, wang2023speechtospeech, diwan2024textless, peng2024mslms2st, kaur2024direct} for semantic-unit discrete representation.
HuBERT utilizes a self-supervised approach to representation learning through masked prediction and employs $k$-means clustering to convert speech into discrete units \cite{Hsu_2021_HuBERT}.
Vector Quantized Variational Autoencoder (VQ-VAE) is another popular discrete unit representation model employing unsupervised speech representational learning \cite{Van_Oord_2017_VQ-VAE}.

## 7·Direct S2ST Models: 直接 S2ST 模型

Direct S2ST models can be classified broadly into three main categories: offline, simultaneous (Simul), and LLM-based S2ST\footnote{LLM-based S2ST is a very recent approach to S2ST modeling and is given special treatment due to their superior performance.}.
Offline models, as the name suggests, start decoding after having seen the entire utterance, whereas simultaneous (aka streaming models) models can start decoding only with partial utterances.
As such, Simul-S2ST is amenable to real-time translation, dubbing, etc.
In this section, we will delve deeper into these three category models.
In particular, we will discuss the architectural level changes made in vanilla Seq2Seq to devise the offline/Simul/LLM-S2ST models.

### 7.1·Offline S2ST models

As mentioned previously, offline direct S2ST models start decoding after having seen the entire utterance.
The typical architecture of such models is shown in Fig.
\ref{fig:GenericDirectS2ST} which is based on Seq2Seq with attention (not shown in Fig.).
These models have \emph{translation model} component besides speech encoder (as mentioned in \S \ref{repr}) and vocoder (for speech synthesis).
Dashed arrow components are optional and used in a few existing works.
This section will discuss various design choices used by existing works for translation model component as well as vocoder.
Style adapters and duration predictors will also be explained as they play an important role in designing offline direct S2ST models.

#### 7.1.1·Encoder Architecture

Some models use separate \emph{speech encoders} to convert raw speech into spectrograms or discrete units, which are then fed into the translation model's encoder \cite{Jia2019, diwan2024textless}.
Others integrate both into a single and directly feed raw speech into the translation model's encoder \cite{Popuri2022_Enahancing_SelfSupervised, Li2023_TextLess_Direct}.

For example, Translatotron 1 \cite{Jia2019} uses a spectrogram as input fed to the LSTM encoder.
Conformer \cite{Gulati2020_Conformer} is used as the encoder in Translatotron 2 and 3 \cite{Jia2021, nachmani_2023_translatotron3}.
Essentially, recent direct S2ST models rely on transformers and their variants for encoder \cite{lee-etal-2022-textless, Li2023_TextLess_Direct}.
To handle large speech inputs, there is often a downsampling module pre-pended before the encoder.
1-D CNN, Connectionist Temporal Classification (CTC) \cite{GravesConnectionistTC}, and adapters are commonly used for length mismatch \cite{lee-etal-2022-textless, Popuri2022_Enahancing_SelfSupervised}.
As shown in Fig.\ref{fig:GenericDirectS2ST}, there is \emph{optional} auxiliary task module attached to the encoder.
The auxiliary task again relies on transformer blocks to predict source and/or target phonemes/char/word or discrete units depending upon the text data availability \cite{Jia2021, lee-etal-2022-textless}.
Dual-encoders disentangling semantic and acoustic information have been proposed \cite{le2024transvip}.

#### 7.1.2·Decoder Architecture

Similar to the encoder, the decoder of the translation model outputs either text \cite{Wang2022SimpleAE} or discrete unit \cite{lee-etal-2022-textless, duquenne_2022_SpeechMatrix, Li2023_TextLess_Direct}.
Decoders in the existing works have been designed either in a \textbf{Autoregressive} (AR) or \textbf{Non-autoregressive} (NAR) way, as explained below.
Further, they could be \textbf{single-pass} or \textbf{double-pass} decoders.
Double-pass decoders employ a linguistic and an acoustic decoder, contrary to single-pass decoders with only an acoustic decoder.
Due to multi-level speech processing and the ability to solve complex \emph{multimodal distribution}\footnote{S2ST is considered a multimodal problem due to (a) linguistic diversity during translation, and (b) diverse acoustic conditions, e.g., pitch, duration, energy, etc.
}, double-pass decoders are shown to outperform single-pass decoders \cite{Jia2021, inaguma_2023_unity}.

In AR decoding, output tokens are predicted in a \emph{causal} manner, i.e., current output depends only on the previously generated tokens.
LSTMs and transformer-based decoders are preferred choices for autoregressive decoding.
For example, unit mBART is used as a unit decoder in \cite{Popuri2022_Enahancing_SelfSupervised, diwan2024textless} whereas Translatotron 1 and 2 use  LSTMs.
The CTC-based decoder is optionally added with the decoder as an auxiliary task \cite{lee-etal-2022-textless}.
Despite high-quality translations, two-pass autoregressive decoders suffer from \emph{high-latency} issue because of their sequential generation.

To address the high-latency decoding problem, NAR S2ST models have been proposed.
NAR models can generate target speech in parallel by assuming conditional independence among the outputs.
However, it becomes challenging to address the multimodal problem of the target speech via NAR compared to AR \cite{gu2018nonautoregressive}.
To solve the linguistic modality problem, directed acyclic transformer \cite{pmlr-v162-huang22m}  and FastSpeech 2 as two-pass decoders are used by DASpeech \cite{Fang2023_DASpeech} whereas knowledge-distillation from AR model to NAR model is used in \cite{Huang2022_TranSpeech}.

Besides AR and NAR decoding, there are other possible decoding choices.
For example, it has been proven that translating multiple sentences of varying lengths can improve performance \cite{Huang2022_TranSpeech}.
As such, \textbf{Top-K} length beam candidates are selected and decoded in parallel.
\textbf{Noisy parallel decoding} is another option that uses AR decoding as a teacher to capture the more accurate optimum of the target distribution and compute the best translation for each fertility sequence \cite{Huang2022_TranSpeech}.

#### 7.1.3·Vocoder

After the translation module is a vocoder whose job is to synthesize the speech from the decoder output which can be either text, discrete unit, or spectrogram.
Recent and advanced speech models are adapted for synthesizing speech.
For example, the majority of works leverage HiFi-GAN \cite{Kong2020_HiFiGAN} such as \cite{lee-etal-2022-textless,Popuri2022_Enahancing_SelfSupervised, Dong2022_Psudo_Labeled, diwan2024textless}, etc., as vocoder.
Other speech synthesis models include non-autoregressive Transformer stack initialized by VQ-VAE \cite{Li2023_TextLess_Direct},  Light Convolution Blocks \cite{zhang_2022_direct_botelneck}, Non-Attentive Tacotron (NAT) \cite{shen2021nonattentive} which has duration prediction module, an LSTM module, and a residual convolutional block \cite{Jia2021}.
Traditional estimation algorithms such as  Griffin  Lim \cite{GriffinLim} may also be used to convert the mel-spectrogram to waveform.
To support multilingual S2ST, multilingual vocoders have been trained with language embedding and language ID as an auxiliary loss \cite{gong2023multilingual_S2ST}.

#### 7.1.4·Style Adapter

One of the desired characteristics of S2ST systems is that the translated speech should preserve (if need be) the original speaker's paralinguistic features, such as speaking rate, tone, intonation, etc.
To this end, existing works insert \emph{style adapter} layer between the translator's encoder and decoder.
\emph{Speech normalization} \cite{lee-etal-2022-textless} is one such technique to adapt to the source speaker's voice.
The key idea is to extract discrete units from the reference speaker and then train the speech normalization module (which consists of pre-trained HuBERT  + CTC) using multi-speaker data as input and reference discrete units as targets.
For example, Translatotron 1 uses a discriminatively pre-trained encoder on speaker verification to condition the decoder for voice preservation.
Due to \textbf{voice cloning} issue, the aforementioned technique is unsuitable for production deployment.
An approach to address this issue is to train on the same speaker's speech on the source and target side so that the model can transfer prosodic information at inference time \cite{Jia2021}.
A side benefit that comes for free is that the S2ST model does not require speaker segmentation for translating multi-speaker utterances.
Interestingly, recent S2ST models implicitly model the para-/non-linguistic characteristics during decoding  \cite{Dong2023_PolyVoice, peng2024mslms2st}.

#### 7.1.5·Duration Predictor

Duration predictor is a module often applied (but not always) along with the speech synthesizer.
Its job is to predict the duration of each output element-- be it a phoneme or discrete unit, which is later converted to a waveform.
For example, Translatotron 2 uses NAT TTS \cite{shen2021nonattentive}, which predicts the duration of each element followed by upsampling.
However, unlike NAT, it optimizes $\mathcal{L}^2$ loss on the total duration predicted instead of \emph{per-phoneme} level which is costly to obtain.
 TTS models like Fastspeech 2 \cite{Ren2020_FastSpeech2} have an in-built duration predictor, which has been used by \cite{Lee2022}.
Works employing unit-to-speech conversion \cite{lee-etal-2022-textless, inaguma_2023_unity, Fang2023_DASpeech, zhu-etal-2023-diffs2ut, Shi_2023_Multiple_TTS} via HiFi-GAN TTS enhance the latter via duration prediction module from \cite{Ren2020_FastSpeech2}.

### 7.2·Simultaneous (Simul-) S2ST Models

The models following the offline setting consider all the input speech to be available before commencing the translation process.
In contrast, simultaneous models initiate translation upon receiving partial input speech \cite{agrawal-etal-IWSLT-2023-findings}.
Simul-S2ST models may borrow encoders, decoders, and vocoders used by offline models (optionally style adapter and duration predictor as well) as mentioned in the previous section.
However, they differ in how they process and segment the speech.

Simul-S2ST is an important problem; however, simultaneously enhancing \textbf{ translation quality} while reducing \textbf{ latency} presents a formidable challenge.
The Simul-S2ST problem faces several issues in practical implementation; {\bf reordering,  acoustic ambiguity, variable speech rate, and long inputs} being prominent among them.
One of the most important decisions in Simul-S2ST is to decide when to start the translation while balancing latency and quality.
As such, there are \textbf{fixed} (like wait-$k$ \cite{ma-etal-2020-simulmt}) and \textbf{adaptive} policies ( like MILk, MMA \cite{Arivazhagan2019_MonotonicLookback, ma2022directMonotonicMultiHead}, etc.
For more details on these policies, refer to \cite{Ma2019MonotonicMA}) proposed in Simul-MT and Simul-ST literature may be borrowed while designing effective and practical Simul-S2ST models.

Traditional Simul-S2ST models (cascaded Simul-S2ST) have studied latency-quality trade-off under various policies \cite{zheng-etal-2020-fluent, dugan2023_when_to_speak} and find that no-single best-policy exist for all languages.
Hence, it is recommended to tune the policy per language basis.
The research on designing direct Simul-S2ST is severely limited.
For example, a simultaneous policy, V-MMA (Variational Monotonic MultiHead Attention \cite{ma2022directMonotonicMultiHead}) considers every attention head as independent monotonic attention and models the alignment between the source and target sequences using latent variables instead of recurrent estimation which leads to an inferior estimate of the alignment matrix.
 Direct Simul-S2ST using V-MMA policy, which adopts a discrete unit-based direct approach, reduces the average latency but compromises translation accuracy \cite{ma2022directMonotonicMultiHead}.

### 7.3·Large Language Models (LLM)-S2ST

Our third category of S2ST models is LLM-based.
The recent success of  Generative Pre-Trained Transformers (GPT) \cite{ Openai_2018_GPT, Brown2020_GPT3,ouyang2022training_GPT} and BERT models \cite{Devlin2019_BERT} over various NLP tasks gives rise to what we know as LLMs.
These models exhibit in-context learning (ICL) when trained on vast datasets.
Extensive training unlocks their latent \emph{emergent abilities} \citep{Wei2022EmergentAO}, enabling them to perform few-shot and zero-shot learning through prompting.
It alludes to using LLMs for S2ST tasks as well.
On a high level, LLM-based S2ST works leverage speech LM, which is prompted for speech generation in a target language.
One issue is that LLMs operate on discrete tokens, whereas speech involves continuous values and cannot be directly applied to these models.
Several research \cite{Wu2023_SpeechGen, zhang2023speak_forign_Languages, Dong2023_PolyVoice} utilize discrete-unit speech representation of the source/target speech.

The effectiveness of LLM-based S2ST lies in several strategies, such as (a) what prompt to use? (b) how to do prompt-tuning? (c) which LM to use?  Works such as \cite{Wu2023_SpeechGen} use task ID as a prompt while others \cite{ zhang2023speak_forign_Languages, peng2024mslms2st, Dong2023_PolyVoice,  gong2024seamlessexpressivelm} use source and target semantic units and source acoustic units for prompting.
Deep prompt tuning \cite{li-liang-2021-prefix}, chain-of-thought prompting \cite{gong2024seamlessexpressivelm}
have been explored in recent works.
For LM, mBART \cite{Wu2023_SpeechGen}, VALL-E and its extension VALL-EX
\cite{zhang2023speak_forign_Languages} have been the preferred choice.
Expressive S2ST \cite{gong2024seamlessexpressivelm} claims to preserve the speaker style without relying on aligned
speech-text data or speaker-aligned speech.
 There are still some questions that need to be answered.
For example, what is the best strategy for prompt design, and how to pre-train/fine-tune them parameter-efficiently for S2ST tasks? Further, the use of LLMs for Simul-MT has been recently proposed \citep{Agostinelli2023SimulLLMAF} and it remains to see how to adapt Simul-MT to Simul-S2ST.


## 8·Training Strategies

Training of E2E S2ST, in general, follows the training of DL models \cite{DBLP:journals/corr/abs-1206-5533}.
Pre-training, self-supervised, unsupervised, and weakly supervised training approaches are primarily used to solve data scarcity as mentioned in \S \ref{datapaucity}.
Therefore, we split our discussion on training based on the availability of \emph{external data}.

### 8.1·Training with External Data

Training of direct S2ST models optimizes the negative conditional log-likelihood as given in \eqref{nll}.
However, there are cases when external sources and/or target transcripts are available along with target speech.
Therefore, a natural question arises on how to leverage such external data.

As depicted in Fig.\ref{fig:GenericDirectS2ST}, the architecture includes both an encoder and a decoder auxiliary task and is supervised by available labeled transcripts.
Different sub-tasks are optimized simultaneously employing the E2E training approach \cite{Prabhavalkar2023_ASR_Survey, Tampuu2020_End-to-End_Driving}.
Training with external data often invariably uses \textbf{Multitask Learning} (MTL) and is employed for \textbf{high-resource written} languages that have abundant text and/or speech data.

Several studies such as \citet{Jia2019, Jia2021, Kano2020_Transcoding, le2024transvip} employ MTL to propose the direct S2ST system.
For example, Translatotron \cite{Jia2019} is the first direct S2ST model trained with two different setups: one with an MTL approach employing textual supervision using an auxiliary network of decoders that predict phonemes and the other without MTL.
However, Translatotron's performance is significantly poor without MTL.

There are several issues present in Translatotron \cite{Jia2019} such as: (1) Auxiliary tasks for textual supervision are underutilized, as the learnings from auxiliary attention are not fully transferred to the main module, (2) the model faces difficulty mapping lengthy source speech spectrograms to target spectrograms, (3) over generation and under generation problem due to attention mechanism \cite{Ren2019_FastSpeech, Zheng_2019_E2E_TTS, Shen_2020_RobustControlled_TTS}.
The existing bottlenecks are mitigated through architectural changes in Translatotron2 \cite{Jia2021} with four sub-modules.
The single attention mechanism, based on Transformer \cite{Vaswani2017_Attention} alleviates the issue of lengthy spectrogram sequences by aligning them with shorter-length phonemes.
Additionally, a Non-Attentive Tacotron (NAT) \cite{shen2021nonattentive} based TTS is employed to mitigate problems of over-generation and under-generation.
Several models following the above architecture also use unsupervised data using techniques such as pre-training, BT, self-training, and pseudo-labeling to improve the performance of models \cite{Dong2022_Psudo_Labeled, Jia2022_Leveraging, nachmani_2023_translatotron3}.

### 8.2·Training w/o External Data aka Textless Training

Compared to high-resource written languages, low-resource unwritten languages lack transcripts. For these languages, recent efforts in direct S2ST modeling propose \emph{textless} training. One approach is to train the model on speech spectrograms, but this method struggles to learn generalized patterns without text. Alternatively, textless training can use self-supervised (e.g., HuBERT) or unsupervised (e.g., VQ-VAE) discrete unit speech encoders.  This encoder converts continuous speech into discrete tokens (similar to text), enabling the application of textual NLP tools to speech.
For example, The textless model in \cite{Zhang2021_UWSpeech}  comprises three modules: converter, translator, and inverter. The converter transforms target speech into discrete units, the translator translates source speech into target discrete units, and the inverter inverts the predicted discrete units back into a speech waveform. Target discrete units enable the model to use the cross-entropy loss for model optimization. This architecture is beneficial for untranscribed languages \cite{Tjandra_2019_untranscribe, Zhang2021_UWSpeech,lee-etal-2022-textless, Huang2022_TranSpeech} or speech datasets without labelled transcripts \cite{Huang2022_TranSpeech, Lee2022}. The process of acquiring phonetic knowledge from languages that share syntactic similarities and have a written form might aid in learning representations for unwritten languages \cite{yallop2007_phonetics_phonology, Kuhl2008_phonetics_learning}. Nevertheless, the extent to which this assistance proves effective depends on the degree of similarity between the languages. Hence, leveraging the benefits of related languages with writing systems, XL-VAE \cite{Zhang2021_UWSpeech} is trained in a supervised manner by using phonemes from the related language as targets.

## 9·Applications issues

Offline direct S2ST models find applications in the dubbing of movies and lectures. When deploying direct S2ST models for such applications, several things to be kept in mind. Firstly, getting a clean speech from a real-world environment is a challenging task. Cross-talk, noise, and background music removal must be done. Secondly, the challenge lies in handling various accents while translating, and voice preservation should be taken care of. The third important requirement of dubbing is {\bf isochrony}, i.e. dubbed speech has to closely match the timing of speech and pauses of the original audio. Inserting pauses, handling verbosity, and prosodic alignment are a few techniques to achieve isochrony. On the other hand, Simul-S2ST is more practical for real-world dubbing. Contrary to offline S2ST, these models require latency-quality handled in a better way along with isochrony.

## 10·Experimental Results and Discussion

Table \ref{tab:performance_table} compares the performance of cascade and direct S2ST models from various studies. Results show a significant performance gap in BLEU scores between textless direct models without external data and cascade models (ID:1-4). The discrete unit-based Translatotron 2 \cite{Li2023_TextLess_Direct} outperforms other direct and cascade models in the Fisher Es$\rightarrow$En group (ID:5-10). \emph{UnitY} \cite{inaguma_2023_unity}, with dual decoders, surpasses all counterparts in its datasets and language directions and demonstrates performance comparable to cascade models (ID:14, 21, 26, 33). Utilizing text data through MTL and external data via pre-training benefits models \cite{Jia2019, Popuri2022_Enahancing_SelfSupervised, Lee2022}. Models with dual decoders outperform single-decoder models \cite{Jia2021, inaguma_2023_unity, Li2023_TextLess_Direct}. The MOS of direct models is slightly lower than cascade models perhaps due to training with synthesized target speech (ID:5-7,11,12), limiting their scores. MOS also depends on vocoder performance; lower translation quality negatively impacts MOS ratings \cite{Jia2019}. Using natural speech in the target can surpass the MOS of cascade models (ID:18, 24). Table \ref{tab:performance_Simul_LLM} shows results for Simultaneous and LLM-based models, though limited literature and no common comparison ground exist.

A standard dataset and language pair are necessary to compare the models fairly. Therefore, we implemented some existing models on the Es$\rightarrow$En language pair of the CVSS-C dataset, as shown in Table \ref{Experiment_result}, acknowledging that only some models are implemented due to code reproducibility issues and limited computing resources. The Fairseq library \cite{ott-etal-2019-fairseq} is used on a single NVIDIA Quadro GV100 GPU machine. The models (ID: 42-48) are trained from scratch with the provided hyperparameter settings. The results indicate that the performance of the direct model is deficient without MTL (ID: 45). However, when textual supervision is introduced through MTL, the performance of the discrete unit-based model \cite{Lee2022} shows significant improvement (ID: 46), while the spectrogram-based model \cite{Jia2019} still struggles with the provided amount of data (ID: 46). The model by \citet{Popuri2022_Enahancing_SelfSupervised}, which uses a pre-trained encoder and decoder and fine-tunes on CVSS-C dataset, outperforms all models trained without pre-training (ID: 49-52) and also outperforms the cascade 2-stage models, but falls short compared to the cascade 3-stage model.

## 11·Research Challenges and Future Directions

This section highlights challenges that may be explored by researchers working on S2ST problems.

### 11.1·Cascade vs End-to-End S2ST Models

As discussed in \S \ref{cascadevse2e}, there has been limited empirical study comparing the cascade and E2E S2ST models. Furthermore, to our knowledge, no thorough assessment has been done for low-resource languages using E2E and cascade models. It may be interesting to compare E2E and cascade S2ST models on various S2ST datasets, particularly for low-resource and unwritten languages.

### 11.2·S2ST on Code-Mix data

Our literature review reveals a gap in research regarding the S2ST model applied to code-mix data. Code-mix data presents challenges like diverse lexicons, syntax variations, and a lack of labeled data. Hence, exploring the following questions would be intriguing: (a) Developing Code-mix S2ST datasets encompassing additional languages, (b) Evaluating the performance of current S2ST models on Code-mix S2ST data, and (c) Investigating whether pre-training across multiple languages aids in addressing code-mixing challenges.

### 11.3·Discrepancy between Automatic and Human Evaluation

Current S2ST systems are evaluated by conducting ASR on the generated waveform and comparing it to the ground truth target text. Errors in ASR can impede evaluation via BLEU. As emphasized in \citep{marie-etal-2021-scientific}, the BLEU score is reported by more than 99\% of MT papers without considering statistical significance testing or human evaluation. Our examination of S2ST papers indicates a similar trend. The development of metrics that directly compare and evaluate source and target speech utterances without resorting to textual analysis remains an open challenge.

### 11.4·Multiple Speakers and Noise Handling

In real-world scenarios, audio or video often feature multiple speakers in a noisy environment, each potentially with their accent, dialect, pitch, and tone. Conducting \emph{speech separation} beforehand could prove beneficial before inputting the data into the S2ST. Similarly, factors such as ambient noise, background music, cross-talk, and non-verbal sounds can pose challenges for S2ST model training. Distinguishing between meaningful speech and ambient noise presents a non-trivial task for the model.

### 11.5·Multilingual and Simultaneous S2ST

The recent surge in interest in multilingual S2ST is driven by its significance in real-world applications. For instance, a single speech may need to be delivered to multilingual communities, such as during a conference attended by a diverse audience. Multilingual S2ST can encompass various scenarios, including one-to-many, many-to-one, and many-to-many language translations. However, our literature review indicates a scarcity of research in this area. Moreover, there is an opportunity to explore simultaneous multilingual S2ST\footnote{In May 2024, Microsoft launched such a service in their Azure product.}, representing the most practical setting.

### 11.6·Low-resource S2ST Datasets and Models

Most current research efforts have concentrated on constructing S2ST models and datasets for languages with ample resources. However, given that the effectiveness of S2ST models hinges on parallel speech-to-speech corpora, there is a need for greater emphasis on developing datasets for low-resource languages. Thus, there is merit in constructing models that can transfer learning from language pairs with abundant resources to those with limited resources.

### 11.7·Voice Cloning

The true challenge for direct models lies in attaining both lower latency and higher translation quality to fulfill real-time usage requirements. Simultaneously preserving the authenticity of the voice presents a challenge, as it is essential to guard against the misuse of voice cloning.
A hybrid model design combining discrete units and spectrograms can acquire linguistic and para-linguistic features, thereby enhancing translation quality, naturalness, and voice preservation.

### 11.8·Faster Token Generation

Many direct models primarily rely on autoregressive models as their sub-modules, in which the current output depends on previous inputs, resulting in a high degree of input dependency. Decreasing the number of modules or layers is essential to reduce the average latency of direct S2ST models. Moreover, shifting the focus from autoregressive models to non-autoregressive models is advisable. This shift in focus is crucial for enabling real-time usage scenarios.