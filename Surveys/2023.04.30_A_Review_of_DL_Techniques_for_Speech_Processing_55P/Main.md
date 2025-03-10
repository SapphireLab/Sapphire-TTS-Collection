# A Review of Deep Learning Techniques for Speech Processing

<details>
<summary>基本信息</summary>

- 标题: "A Review of Deep Learning Techniques for Speech Processing"
- 作者:
  - 01 Ambuj Mehrish,
  - 02 Navonil Majumder,
  - 03 Rishabh Bhardwaj,
  - 04 Rada Mihalcea,
  - 05 Soujanya Poria
- 链接:
  - [ArXiv](https://arxiv.org/abs/2305.00359)
  - [Publication](https://doi.org/10.1016/j.inffus.2023.101869)
  - [Github]
  - [Demo]
- 文件:
  - [ArXiv](2305.00359v3__Survey__A_Review_of_Deep_Learning_Techniques_for_Speech_Processing.pdf)
  - [Publication](2305.00359p0__Survey__InfFus2023.pdf)

</details>

## Abstract: 摘要

The field of speech processing has undergone a transformative shift with the advent of deep learning.
The use of multiple processing layers has enabled the creation of models capable of extracting intricate features from speech data.
This development has paved the way for unparalleled advancements in speech recognition, text-to-speech synthesis, automatic speech recognition, and emotion recognition, propelling the performance of these tasks to unprecedented heights.
The power of deep learning techniques has opened up new avenues for research and innovation in the field of speech processing, with far-reaching implications for a range of industries and applications.
This review paper provides a comprehensive overview of the key deep learning models and their applications in speech-processing tasks.
We begin by tracing the evolution of speech processing research, from early approaches, such as MFCC and HMM, to more recent advances in deep learning architectures, such as CNNs, RNNs, transformers, conformers, and diffusion models.
We categorize the approaches and compare their strengths and weaknesses for solving speech-processing tasks.
Furthermore, we extensively cover various speech-processing tasks, datasets, and benchmarks used in the literature and describe how different deep-learning networks have been utilized to tackle these tasks.
Additionally, we discuss the challenges and future directions of deep learning in speech processing, including the need for more parameter-efficient, interpretable models and the potential of deep learning for multimodal speech processing.
By examining the field's evolution, comparing and contrasting different approaches, and highlighting future directions and challenges, we hope to inspire further research in this exciting and rapidly advancing field.

## Content

- 1·Introduction
- 2·Background
  - 2.1·Speech Signals
  - 2.2·Speech Features
  - 2.3·Traditional models for speech processing
- 3·Deep Learning Architectures and Their Applications in Speech Processing Tasks
  - 3.1·Recurrent Neural Networks (RNNs)
  - 3.2·Convolutional Neural Networks
  - 3.3·Transformers
  - 3.4·Conformer
  - 3.5·Sequence to Sequence Models
  - 3.6·Reinforcement Learning
  - 3.7·Graph Neural Network
  - 3.8·Diffusion Probabilistic Model
- 4·Speech Representation Learning
  - 4.1·Supervised Learning
  - 4.2·Unsupervised learning
  - 4.3·Semi-Supervised Learning
  - 4.4·Self-Supervised Representation Learning (SSRL)
- 5·Speech Processing Tasks
  - 5.1·Automatic Speech Recognition (ASR) & Conversational Multi-Speaker AST
  - 5.2·Neural Speech Synthesis
  - 5.3·Speaker Recognition
  - 5.4·Speaker Diarization
  - 5.5·Speech-to-Speech Translation
  - 5.6·Speech Enhancement
  - 5.7·Audio Super Resolution
  - 5.8·Voice Activity Detection (VAD)
  - 5.9·Speech Quality Assessment
  - 5.10·Speech Separation
  - 5.11·Spoken Language Understanding
  - 5.12·Audio/Visual Multimodal Speech Processing
- 6·Advanced Transfer Learning Techniques for Speech Processing
  - 6.1·Domain Adaptation
  - 6.2·Meta Learning
  - 6.3·Parameter-Efficient Transfer Learning
- 7·Conclusion and Future Research Directions

## 7·Conclusion and Future Research Directions

The rapid advancements in deep learning techniques have revolutionized speech processing tasks, enabling significant progress in speech recognition, speaker recognition, and speech synthesis.
This paper provides a comprehensive review of the latest developments in deep learning techniques for speech-processing tasks.
We begin by examining the early developments in speech processing, including representation learning and HMM-based modeling, before presenting a concise summary of fundamental deep learning techniques and their applications in speech processing.
Furthermore, we discuss key speech-processing tasks, highlight the datasets used in these tasks, and present the latest and most relevant research works utilizing deep learning techniques.


We envisage several lines of development in speech processing:

- **Large Speech Models**: In addition to the advancements made with wav2vec2.0, further progress in the field of ASR and TTS models involves the development of larger and more comprehensive models, along with the utilization of larger datasets.
By leveraging these resources, it becomes possible to create TTS models that exhibit enhanced naturalness and human-like prosody.
One promising approach to achieve this is through the application of adversarial training, where a discriminator is employed to distinguish between machine-generated speech and reference speech.
This adversarial framework facilitates the generation of TTS models that closely resemble human speech, providing a significant step forward in achieving more realistic and high-quality synthesized speech.
By exploring these avenues, researchers aim to push the boundaries of speech synthesis technology, ultimately enhancing the overall performance and realism of TTS systems.
- **Multilingual Models**: Self-supervised learning has emerged as a transformative approach in the field of speech recognition, particularly for low-resource languages characterized by scarce or unavailable labeled datasets.
The recent development of the XLS-R model, a state-of-the-art self-supervised speech recognition model, represents a significant milestone in this domain.
With a remarkable scale of over 2 billion parameters, the XLS-R model has been trained on a diverse dataset spanning 128 languages, surpassing its predecessor in terms of language coverage.
The notable advantage of scaling up larger multilingual models like XLS-R lies in the substantial performance improvements they offer.
As a result, these models are poised to outperform single-language models and hold immense promise for the future of speech recognition.
By harnessing the power of self-supervised learning and leveraging multilingual datasets, the XLS-R model showcases the potential for addressing the challenges posed by low-resource languages and advancing the field of speech recognition to new heights.
- **Multimodal Speech Models**: Traditional speech and text models have typically operated within a single modality, focusing solely on either speech or text inputs and outputs.
However, as the scale of generative models continues to grow exponentially, the integration of multiple modalities becomes a natural progression.
This trend is evident in the latest developments, such as the unveiling of groundbreaking language models like GPT-4~\cite{OpenAI2023GPT4TR} and Kosmos-I~\cite{Huang2023LanguageIN}, which demonstrate the ability to process both images and text jointly.
These pioneering multimodal models pave the way for the emergence of large-scale architectures that can seamlessly handle speech and other modalities in a unified manner.
The convergence of multiple modalities within a single model opens up new avenues for comprehensive understanding and generation of multimodal content, and it is highly anticipated that we will witness the rapid development of large multimodal models tailored for speech and beyond in the near future.
- **In-Context Learning**: Utilizing mixed-modality models opens up possibilities for the development of in-context learning approaches for a wide range of speech-related tasks.
This paradigm allows the tasks to be explicitly defined within the input, along with accompanying examples.
Remarkable progress has already been demonstrated in large language models (LLMs), including notable works such as InstructGPT~\cite{Ouyang2022TrainingLM}, FLAN-T5~\cite{Chung2022ScalingIL}, and LLaMA~\cite{Touvron2023LLaMAOA}.
These models showcase the efficacy of in-context learning, where the integration of context-driven information empowers the models to excel in various speech tasks.
By leveraging mixed-modality models and incorporating contextual cues, researchers are advancing the boundaries of speech processing capabilities, paving the way for more versatile and context-aware speech systems.
- **Controllable Speech Generation**:An intriguing application stemming from the aforementioned concept is controllable text-to-speech (TTS), which allows for fine-grained control over various attributes of the synthesized speech.
Attributes such as tone, accent, age, gender, and more can be precisely controlled through in-context text guidance.
This controllability in TTS opens up exciting possibilities for personalization and customization, enabling users to tailor the synthesized speech to their specific requirements.
By leveraging advanced models and techniques, researchers are making significant strides in developing controllable TTS systems that provide users with a powerful and flexible speech synthesis experience.
- **Parameter-efficient Learning**: With the increasing scale of LLMs and speech models, it becomes imperative to adapt these models with minimal parameter updates.
This necessitates the development of specialized adapters that can efficiently update these emerging mixed-modality large models.
Additionally, model compression techniques have proven to be practical solutions in addressing the challenges posed by these large models.
Recent research~\cite{DBLP:journals/corr/abs-2106-05933, 9053878, peng-etal-2021-shrinking} has demonstrated the effectiveness of model compression, highlighting the sparsity that exists within these models, particularly for specific tasks.
By employing model compression techniques, researchers can reduce the computational requirements and memory footprint of these models while preserving their performance, making them more practical and accessible for real-world applications.
- **Explainability**: Explainability remains elusive to these large networks as they grow.
Researchers are steadfast in explaining these networks' functioning and learning dynamics.
Recently, much work has been done to learn the fine-tuning and in-context learning dynamics of these large models for text under the neural-tangent-kernel (NTK) asymptotic framework~\cite{Malladi2022AKV}.
Such exploration is yet to be done in the speech domain.
More yet, explainability could be built-in as inductive bias in architecture.
To this end, brain-inspired architectures~\cite{millet2022toward} are being developed, which may shed more light on this aspect of large models.
- **Neuroscience-inspired Architectures**:In recent years, there has been significant research exploring the parallels between speech-processing architectures and the intricate workings of the human brain~\cite{millet2022toward}.
These studies have unveiled compelling evidence of a strong correlation between the layers of speech models and the functional hierarchy observed in the human brain.
This intriguing finding has served as a catalyst for the development of neuroscience-inspired speech models that demonstrate comparable performance to state-of-the-art (SOTA) models~\cite{millet2022toward}.
By drawing inspiration from the underlying principles of neural processing in the human brain, these innovative speech models aim to enhance our understanding of speech perception and production while pushing the boundaries of performance in the field of speech processing.
- **Text-to-Audio Models for Text-to-Speech**: Lately, transformer and diffusion-based text-to-audio (TTA) model development is turning into an exciting area of research.
Until recently, most of these models~\cite {Liu2023AudioLDMTG,Kreuk2022AudioGenTG,yang2022diffsound,ghosal2023texttoaudio,wang2023audit} overlooked speech in favour of general audio.
In the future, however, the models will likely strive to be equally performant in both audio and speech.
To that end, current TTS methods will likely be an integral part of those models.
Recently, \citet{bark} have aimed at striking a good balance between general audio and speech, although their implementation is not public, nor have they provided any detailed paper.
