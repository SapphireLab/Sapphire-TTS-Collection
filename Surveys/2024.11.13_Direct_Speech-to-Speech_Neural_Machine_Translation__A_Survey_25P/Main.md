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
