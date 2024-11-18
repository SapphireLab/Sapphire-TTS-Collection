# 6·Advanced Transfer Learning Techniques for Speech Processing

## 6.1·Domain Adaptation

### 6.1.1·Task Description

Domain adaptation is a field that deals with adapting a model trained on a labeled dataset from a source domain to a target domain, where the source domain differs from the target domain.
The goal of domain adaptation is to reduce the performance gap between the source and target domains by minimizing the difference between their distributions.
In speech processing, domain adaptation has various applications such as speech recognition \cite{bousquet2019robustness,nidadavolu2019cycle,lee2019coral+,chowdhury2022domain,hu2022domain}, speaker verification \cite{xia2019cross,chen2021self,wang2019vae,zhang2023meta,himawan2019deep}, and speech synthesis \cite{xin2020cross,yue2022exploring}.
This section explores the use of domain adaptation in these tasks by reviewing recent literature on the subject.
Specifically, we discuss the techniques used in domain adaptation, their effectiveness, and the challenges that arise when applying them to speech processing.

### 6.1.2·Models

Various techniques have been proposed to adapt a deep learning model for speech processing tasks.
An example of a technique is reconstruction-based domain adaptation, which leverages an additional reconstruction task to generate a communal representation for all the domains.
The Deep Reconstruction Classification Network (DRCN) \cite{ghifary2016deep} is an illustration of such an approach, as it endeavors to address both tasks concurrently: (i) classification of the source data and (ii) reconstruction of the input data.
Another technique used in domain adaptation is the domain-adversarial neural network architecture, which aims to learn domain-invariant features using a gradient reversal layer \cite{9688269,9892596,8461423}.

Different domain adaptation techniques are successfully applied to different speech processing tasks, such as speaker recognition \cite{li2022coral++,hu2022domain,bousquet2019robustness,nidadavolu2019cycle} and verification \cite{chen2021self,chen2020adversarial,zhang2023meta,li2022editnet,zhu2022multi}, where the goal is to verify the identity of a speaker using their voice.
One approach for domain adaptation in speaker verification is to use adversarial domain training to learn speaker-independent features insensitive to variations in the recording environment \cite{chen2020adversarial}.

Domain adaptation has also been applied to speech recognition \cite{mani2020asr,hwang2022large,yue2022exploring,sukhadia2023domain} to improve speech recognition accuracy in a target domain.
One recent approach for domain adaptation in ASR is prompt-tuning \cite{dingliwal2021prompt}, which involves fine-tuning the ASR system on a small amount of data from the new domain.
Another approach is to use adapter modules for transducer-based speech recognition systems \cite{majumdar2023damage,sathyendra2022contextual}, which can balance the recognition accuracy of general speech and improve recognition on adaptation domains.
The Machine Speech Chain integrates both end-to-end (E2E) ASR and neural text-to-speech (TTS) into one circle \cite{yue2022exploring}.
This integration can be used for domain adaptation by fine-tuning the E2E ASR on a small amount of data from the new domain and then using the TTS to generate synthetic speech in the new domain for further training.

In addition to domain adaptation techniques used in speech recognition, there has been growing interest in adapting text-to-speech (TTS) models to specific speakers or domains.
This research direction is critical, especially in low-resource settings where collecting sufficient training data can be challenging.
Several recent works have proposed different approaches for speaker and domain adaptation in TTS, such as AdaSpeech \cite{chen2021adaspeech, yan2021adaspeech, wu2022adaspeech}.

## 6.2·Meta Learning

### 6.2.1·Task Description

Meta-learning is a branch of machine learning that focuses on improving the learning algorithms used for tasks such as parameter initialization, optimization strategies, network architecture, and distance metrics.
This approach has been demonstrated to facilitate faster fine-tuning, better performance convergence, and the ability to train models from scratch, which is especially advantageous for speech-processing tasks.
Meta-learning techniques have been employed in various speech-processing tasks, such as low-resource ASR \cite{hsu2020meta,indurthi2020end}, SV \cite{zhang2021meta}, TTS \cite{huang2022meta} and domain generalization for speaker recognition \cite{kang2020domain}.

Meta-learning has the potential to improve speech processing tasks by learning better learning algorithms that can adapt to new tasks and data more efficiently.
Meta-learning can also reduce the cost of model training and fine-tuning, which is particularly useful for low-resource speech processing tasks.
Further investigation is required to delve into the full potential of meta-learning in speech processing and to develop more effective meta-learning algorithms for different speech-processing tasks.

### 6.2.2·Models

In low-resource ASR, meta-learning is used to quickly adapt unseen target languages by formulating ASR for different languages as different tasks and meta-learning the initialization parameters from many pretraining languages \cite{hsu2020meta,singh2022improved}.
The proposed approach, MetaASR \cite{hsu2020meta}, significantly outperforms the state-of-the-art multitask pretraining approach on all target languages with different combinations of pretraining languages.
In speaker verification, meta-learning is used to improve the meta-learning training for SV by introducing two methods to improve the backbone embedding network \cite{chen2021improved}.
The proposed methods can obtain consistent improvements over the existing meta-learning training framework \cite{kye2020meta}.

Meta-learning has proven to be a promising approach in various speech-related tasks, including low-resource ASR and speaker verification.
In addition to these tasks, meta-learning has also been applied to few-shot speaker adaptive TTS and language-agnostic TTS, demonstrating its potential to improve performance across different speech technologies.
Meta-TTS \cite{huang2022meta} is an example of a meta-learning model used for a few-shot speaker adaptive TTS.
It can synthesize high-speaker-similarity speech from a few enrolment samples with fewer adaptation steps.
Similarly, a language-agnostic meta-learning approach is proposed in \cite{lux2022language} for low-resource TTS.

## 6.3·Parameter-Efficient Transfer Learning

Transfer learning has played a significant role in the recent progress of speech processing.
Fine-tuning pre-trained large models, such as those trained on  LibriSpeech ~\cite{panayotov2015librispeech} or Common Voice ~\cite{ardila2019common}, has been widely used for transfer learning in speech processing.
However, fine-tuning all parameters for each downstream task can be computationally expensive.
To overcome this challenge, researchers have been exploring parameter-efficient transfer learning techniques that optimize only a fraction of the model parameters, aiming to improve training efficiency.
This article investigates these parameter-efficient transfer learning techniques in speech processing, evaluates their effectiveness in improving training efficiency without sacrificing performance, and discusses the challenges and opportunities associated with these techniques, highlighting their potential to advance the field of speech processing.

### 6.3.1·Adapters

In recent years, retrofitting adapter modules with a few parameters to pre-trained models has emerged as an effective approach in speech processing.
This involves optimizing the adapter modules while keeping the pre-trained parameters frozen for downstream tasks.
Recent studies (Li et al., 2023; Liu et al., 2021) \cite{li2023evaluating,DBLP:journals/corr/abs-2105-01051} have shown that adapters often outperform fine-tuning while using only a fraction of the total parameters.
Different adapter architectures are available, such as bottleneck adapters (Houlsby et al., 2019)\cite{pmlr-v97-houlsby19a}, tiny attention adapters (Zhao et al., 2022)\cite{zhao2022tiny}, prefix-tuning adapters (Li and Liang, 2021)\cite{li-liang-2021-prefix}, and LoRA adapters (Hu et al., 2022)\cite{hu2022lora}, among others
Next, we will review the different approaches for parameter-efficient transfer learning.
The different approaches are illustrated in  \Cref{fig:adaptarchi} and \Cref{fig:convadapt}

#### Adapter Tuning

Adapters are a type of neural module that can be retrofitted onto a pre-trained language model, with significantly fewer parameters than the original model.
One such type is the bottleneck or standard adapter (Houlsby et al., 2019; Pfeiffer et al., 2020) \cite{houlsby2019parameter, pfeiffer2020adapterfusion}.
The adapter takes an input vector $h \in \mathbf{R}^{d}$ and down-projects it to a lower-dimensional space with dimensionality $m$ (where $m<d$), applies a non-linear function $g(\cdot)$, and then up-projects the result back to the original $d$-dimensional space.
Finally, the output is obtained by adding a residual connection.

$$
    \bm{h} \leftarrow \bm{h} + g(\bm{h} \bm{W}_{\text{down}}) \bm{W}_{\text{up}}
$$

where matrices $\bm{W}_{\text{down}}$ and $\bm{W}_{\text{up}}$ are used as down and up projection matrices, respectively, with $\bm{W}{\text{down}}$ having dimensions $\mathbb{R}^{d \times m}$ and $\bm{W}_{\text{up}}$ having dimensions $\mathbb{R}^{m \times d}$.
Previous studies have empirically shown that a two-layer feedforward neural network with a bottleneck is effective.
In this work, we follow the experimental settings outlined in \cite{pfeiffer2020adapterfusion} for the adapter, which is inserted after the feedforward layer of every transformer module, as depicted in \Cref{fig:adaptarchi}.

#### Prefix tuning

Recent studies have suggested modifying the attention module of the Transformer model to improve its performance in natural language processing tasks.
This approach involves adding learnable vectors to the pre-trained multi-head attention keys and values at every layer, as depicted in Figure \ref{fig:adaptarchi}.
Specifically, two sets of learnable prefix vectors, $\bm{P_K}$ and $\bm{P_V}$, are concatenated with the original key and value matrices $\bm{K}$ and $\bm{V}$, while the query matrix $\bm{Q}$ remains unchanged.
The resulting matrices are then used for multi-head attention, where each head of the attention mechanism is computed as follows:

$$
\text{head}_{i} = \text{Attn}(\bm{Q}\bm{W}_{Q}^{(i)},[\bm{P}_{K}^{(i)},\bm{K}\bm{W}_{Q}^{(i)}],[\bm{P}_{V}^{(i)},\bm{V}\bm{W}_{Q}^{(i)}])
$$

where Attn($\cdot$) is scaled dot-product attention given by:

$$
\text{Attn}(\bm{Q},\bm{K},\bm{V}) = \text{softmax} (\frac{\bm{Q}\bm{K}^{T}}{\sqrt{d_{k}}})\bm{V}
$$

The attention heads in each layer are modified by prefix tuning, with only the prefix vectors $\bm{P}{K}$ and $\bm{P}{V}$ being updated during training.
This approach provides greater control over the transmission of acoustic information between layers and effectively activates the pre-trained model's knowledge.

#### LoRA

LoRA is a novel approach proposed by Hu et al.
(2021) \cite{hu2021lora}, which aims to approximate weight updates in the Transformer by injecting trainable low-rank matrices into its layers.
In this method, a pre-trained weight matrix $W \in \mathbb{R}^{d \times k}$ is updated by a low-rank decomposition $\bm{W} + \Delta \bm{W} = \bm{W} + \bm{W}{\text{down}} \bm{W}{\text{up}}$, where $\bm{W}{\text{down}} \in \mathbb{R}^{d \times r}$, $\bm{W}{\text{up}} \in \mathbb{R}^{r \times k}$ are tunable parameters and $r$ represents the rank of the decomposition matrices, with $r<d$.
Specifically, for a given input $\bm{x}$ to the linear projection in the multi-headed attention layer, LoRA modifies the projection output $\bm{h}$ as follows:

$$
    \bm{h} \leftarrow \bm{h} + s\cdot \bm{x} \bm{W}_{\text{down}}\bm{W}_{\text{up}}
$$

In this work,  LoRA is integrated into four locations of the multi-head attention layer, as illustrated in \Cref{fig:adaptarchi}.
Thanks to its lightweight nature, the pre-trained model can accommodate many small modules for different tasks, allowing for efficient task switching by replacing the modules.
Additionally, LoRA incurs no inference latency and achieves a convergence rate that is comparable to that of training the original model, unlike fully fine-tuned models \cite{hu2021lora}.

#### Convolutional Adapter

CNNs have become increasingly popular in the field of speech processing due to their ability to learn task-specific information and combine channel-wise information within local receptive fields.
To further improve the efficiency of CNNs for speech processing tasks, Li et al.
(2023) \cite{li2023evaluating} proposed a lightweight adapter, called the ConvAdapter, which uses three 1D convolutional layers, layer normalization, and a squeeze-and-excite module (Zhang et al., 2017) ~\cite{https://doi.org/10.48550/arxiv.1709.01507}, as shown in \cref{fig:convadapt}.
By utilizing depth-wise convolution, which requires fewer parameters and is more computationally efficient, the authors were able to achieve better performance while using fewer resources.
In this approach, the ConvAdapter is added to the same location as the Bottleneck Adapter (\Cref{fig:adaptarchi}).

#### Summary

\Cref{sure:Task1}, \Cref{sure:Task2}, and \Cref{sure:TTS} present the results of various speech processing tasks in the SURE benchmark.
The findings demonstrate that the adapter-based methods perform comparably well in fine-tuning.
However, there is no significant advantage of any particular adapter type over others for these benchmark tasks and datasets.

### 6.3.2·Knowledge Distillation (KD)

Knowledge distillation involves training a smaller model to mimic the behavior of a larger and more complex model.
This can be done by training the smaller model to predict the outputs of the larger model or, by using, the larger model's hidden representations as input to the smaller model.
Knowledge distillation is effective in reducing the computational cost of training and inference.

\citet{Cho20a} conducted knowledge distillation (KD) by directly applying it to the downstream task.
One way to improve this approach is to use KD as pre-training for various downstream tasks, thus allowing for knowledge reuse.
A noteworthy result achieved by \citet{Denisov20a} was using KD in pretraining.
However, they achieved this by initializing an utterance encoder with a trained ASR model's backbone, followed by a trained NLU backbone.
Knowledge distillation can be applied directly into a wav2vec 2.0 encoder without ASR training and a trained NLU module to enhance this method.
\citet{Kim21a} implemented a more complex architecture, utilizing KD in both the pretraining and fine-tuning stages.

### 6.3.3·Model Compression

Researchers have also explored various architectural modifications to existing models to make them more parameter-efficient.
One such approach is \emph{pruning}~\cite{Frantar2023SparseGPTML,DBLP:journals/corr/abs-1910-04732}, where motivated by lottery-ticket hypothesis (LTH)~\cite{DBLP:journals/corr/abs-1803-03635}, the task-irrelevant parameters are masked based on some threshold defined by importance score, such as some parameter norm.
Another form of compression could be \emph{low-rank factorization}~\cite{Hsu2022LanguageMC}, where the parameter matrices are factorized into lower-rank matrices with much fewer parameters.
Finally, \emph{quantization} is a popular approach to reduce the model size and improve energy efficiency with a minimal performance penalty.
It involves transforming 32-bit floating point model weights into integers with fewer bit-counts~\cite{DBLP:journals/corr/abs-2011-10680}---8-bit, 4-bit, 2-bit, and even 1-bit---through scaling and shifting.
At the same time, the quantization of the activation is also handled based on the input.

\citet{DBLP:journals/corr/abs-2106-05933} iteratively prune and subsequently fine-tune wav2vec2.0 on downstream tasks to obtained improved results over fine-tuned wav2vec2.0.
\citet{9053878} employ low-rank transformers to excise the model size by half and increase the inference speed by 1.35 times.
\citet{peng-etal-2021-shrinking} employ KD and quantization to make wav2vec2.0 twice as fast, twice as energy efficient, and 4.8 times smaller at the cost of a 7\% increase in WER.
Without the KD step, the model is 3.6 times smaller with mere 0.1\% WER degradation.
