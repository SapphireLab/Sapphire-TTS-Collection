# Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey

<details>
<summary>基本信息</summary>

- 标题: "Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey"
- 作者:
  - 01 Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, Yichi Zhang, Ruoyu Wu, Qingxiu Dong, Ge Zhang, Jian Yang, Lingwei Meng, Shujie Hu, Yulong Chen, Junyang Lin, Shuai Bai, Andreas Vlachos, Xu Tan, Minjia Zhang, Wen Xiao, Aaron Yee, Tianyu Liu, Baobao Chang
- 链接:
  - [ArXiv](https://arxiv.org/abs/2412.18619)
  - [Publication]()
  - [Github](https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction)
  - [Demo]()
- 文件:
  - [ArXiv](2412.18619v2__Next_Token_Prediction_Towards_Multimodal_Intelligence__A_Comprehensive_Survey.pdf)
  - [Publication] #TODO

</details>

## Abstract: 摘要

Building on the foundations of language modeling in natural language processing, Next Token Prediction (NTP) has evolved into a versatile training objective for machine learning tasks across various modalities, achieving considerable success.
As Large Language Models (LLMs) have advanced to unify understanding and generation tasks within the textual modality, recent research has shown that tasks from different modalities can also be effectively encapsulated within the NTP framework, transforming the multimodal information into tokens and predict the next one given the context.
This survey introduces a comprehensive taxonomy that unifies both understanding and generation within multimodal learning through the lens of NTP.
The proposed taxonomy covers five key aspects: Multimodal tokenization, MMNTP model architectures, unified task representation, datasets \& evaluation, and open challenges.
This new taxonomy aims to aid researchers in their exploration of multimodal intelligence.
An associated GitHub repository collecting the latest papers and repos is available at [this https URL](https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction).

## Introduction: 引言

Humans' engagement with the universe is a tapestry, interwoven with the threads of various modalities.
Humans can see and sketch paintings, read and write epics, listen and compose music, touch and sculpture heroes, ponder and make movements.
These modalities -- specific information types such as vision, sound, and language -- are the channels through which humans interpret and respond to the world.
This multifaceted interaction highlights the intertwined nature of perception and response in human experience.
As a specialized area within Artificial Intelligence (AI) research, multimodal Learning focuses on creating systems capable of understanding and generating various multimodal information~\citep{10.1109/TPAMI.2018.2798607}.

A paradigm shift has emerged in the field of AI across multiple modalities, transitioning from specialized unimodal models trained for a single task to versatile multimodal ones dealing with a diverse array of tasks~\citep{hao2022languagemodelsgeneralpurposeinterfaces}.
This shift is largely attributed to the advancement of Large Language Models (LLMs) in the Natural Language Processing (NLP) field such as GPT-3~\citep{gpt3}, ChatGPT~\citep{chatgpt} and LLaMA~\citep{touvron2023llama}, which unify multiple natural language understanding and generation tasks with a single Next Token Prediction (NTP) objective.
The original task of NTP is to predict the next token (which can be a word, subword, or character) in a given sequence of text based on the context provided by preceding tokens.
The NTP paradigm has been proven to be scalable given abundant data and computational resources in the lens of scaling law research~\cite{kaplan2020scalinglaw,zhai2022scaling}.

Simultaneously, researchers have explored the incorporation of non-textual input and output modalities into large language models, sparking interest within the community to develop powerful Large Multimodal Models (LMMs) featuring capabilities to conduct tasks across different modalities~\citep{cui2023surveymultimodallargelanguage,yin2024surveymultimodallargelanguage}.
For a better understanding of the historical development of LMMs based on NTP, we demonstrate a timeline in Figure~\ref{fig:history}, categorized by models' understanding or generation ability and different modalities.

In \autoref{fig:methodology}, we use the image modality as an example to illustrate the workflow of \textbf{Multimodal Learning with NTP (MMNTP)}.
The process can be divided into three key components: Tokenization, Modeling, and Training Objectives, which will be explained and discussed in details in the rest of the survey.
For vision modality, image and video understanding capabilities have been demonstrated in large vision-language models such as GPT4-V~\citep{gpt4v}, QwenVL~\citep{QwenVL}, LLaVA~\citep{liu2023llava}, Kosmos~\citep{peng2023kosmos2groundingmultimodallarge,lv2024kosmos25multimodalliteratemodel,huang2023languageneedaligningperception}, Phi-Vision~\citep{phi3_5} and Gemini~\citep{gemini1}, while Emu~\citep{sun2024emu} and Chameleon~\citep{chameleonteam2024chameleon} show visual generation could be achieved in NTP manner.
Similarly, end-to-end audio understanding and generation have been achieved in NTP-based models such as GPT4-o and Moshi \citep{gpt4o,défossez2024moshispeechtextfoundationmodel}.

To equip LLMs with visual understanding capabilities, pioneering research such as Flamingo~\cite{alayrac2022flamingo}, BLIP2~\cite{li2023blip2}, GPT4V~\cite{gpt4v}, MiniGPT4~\cite{zhu2023minigpt4} and LLaVA~\cite{liu2023llava} has demonstrated that LLMs can be easily adapted to process multimodal inputs such as images and videos, by converting multimodal information into tokens with a straightforward tokenization module, such as a visual encoder like CLIP~\cite{radford2021clip} or a simple linear projection~\cite{fuyu}.
Subsequently, these models perform multimodal instruction tuning based on image-query-answer triples using the same NTP objective.

As Large Language Models bridge understanding and generation tasks in natural language processing, there is considerable interest in extending their capabilities to generate multimodal outputs.
Recent advances in this direction include GPT-4o~\cite{gpt4o}, which can understand and generate text, audio, and images using a unified multimodal LLM.
We have also witnessed tremendous improvements from the open source community.
For visual modality, Chameleon~\cite{chameleonteam2024chameleon} and Emu3~\citep{wang2024emu3nexttokenpredictionneed} are two distinctive multimodal that unify understanding and generation in both language and image modalities.
For audio, Moshi~\citep{défossez2024moshispeechtextfoundationmodel} can conduct tasks such as automatic speech recognition (ASR) and speech generation in an NTP manner based a pretrained LLM.
As a general and fundamental approach, NTP also has promising implications for diverse fields like AI-for-Science such as designing proteins in biology~\citep{benegas2024genomiclanguagemodelsopportunities} and composing molecule structure in chemistry~\citep{flam2022language}.

To generate multimodal content using the NTP approach, it is crucial to recognize that unlike language, which is structured from discrete symbols, multimodal data like images and sounds inherently exist in a continuous space.
A common technique to address this challenge is quantization.
Vector Quantization (VQ) is a classical method that allows for the modeling of probability density functions for continuous multimodal data through discrete vector distributions~\cite{Pags2015IntroductionTV,vq}.
This technique aligns well with NTP modeling.
With the rise of deep learning, neural VQ methods such as VQVAE~\cite{Oord2017NeuralDR} and VQGAN~\cite{Esser2020TamingTF} have been developed, establishing a foundation for linking visual and audio generation with NTP.
Significant work has emerged leveraging these VQ methodologies and the language modeling task.
Examples include innovative systems such as DALL-E~\cite{ramesh2021zeroshot}, CogView~\cite{CogView}, CM3Leon~\cite{cm3leon}, Parti~\cite{text2image2}, Muse~\cite{chang2023muse}, VideoPoet~\cite{kondratyuk2023videopoet}, LVM~\cite{bai2023sequential}, Chameleon~\cite{chameleonteam2024chameleon} and Infinity~\citep{han2024infinityscalingbitwiseautoregressive}.
These methods often rely on external models, like VQGAN decoders, for image generation, making their approach a form of indirect multimodal generation.
Parallel explorations have been conducted utilizing the NTP objective to directly generate images in continuous spaces, such as VAE's latent space~\cite{tschannen2024givt}, or by simulating a diffusion process~\cite{li2024autoregressive-without,Transfusion}.
Unlike the indirect methods, only a few initiatives like ImageGPT~\citep{imagegpt} perform direct multimodal generation by predicting pixels from scratch.
Additionally, NTP models can be augmented with various external models to facilitate multimodal generation.
Notable examples include Emu~\cite{sun2023emu1}, MiniGPT5~\cite{zheng2023minigpt5}, and CoDi2~\cite{tang2023codi2}.
These approaches utilize the NTP framework to incorporate external diffusion models for image generation, showcasing another form of indirect multimodal generation.

We have covered powerful models that can understand or generate information across different modalities within the NTP paradigm.
However, developing a single model that can both comprehend and produce information across multiple modalities, similar to human abilities, remains an intriguing goal in the pursuit of Artificial General Intelligence (AGI).
Recently, a new research trend has emerged, focusing on the development of LMMs that unifies multimodal understanding and generation in the NTP paradigm.
Notable examples include Unified-IO~\citep{lu2023unifiedio,lu2023unifiedio2}, Chameleon~\citep{chameleonteam2024chameleon}, Transfusion~\citep{Transfusion}, Show-o~\citep{Show-o}, Moshi~\citep{Moshi}, and Emu3~\citep{Emu3}.
Unifying understanding and generation presents unique challenges, including the diversity of modalities and resolving conflicts between them.
We will discuss these issues further in Section~\ref{sec:challenges}.

### Overall Structure of the Survey

The structure of the survey is shown in Figure~\ref{taxonomy}.
Section~\ref{sec:tokenization} focuses on Multimodal Tokenization, and highlights the importance of tokenization as the bridge between raw multimodal data and their representations, distinguishing between discrete tokens that use Vector Quantization and continuous tokens.
Section~\ref{sec:model} delves into the Multimodal Backbone Model for NTP, indicating that an auto-regressive model, often resembling a large language model, is employed to capture multimodal tokens, utilizing distinct attention masks for different modalities to account for their specific features.
Section~\ref{sec:training} covers Training with Unified Multimodal Task Representation, explaining the training objectives varying from discrete to continuous token prediction, enabling multimodal output through VQ decoders or directly generating conditions for models like diffusion or VAE.
The section also covers prompt engineering techniques such as In-Context Learning and Chain-of-Thought reasoning of MMNTP models adopted from LLM research.
Section~\ref{sec:datasets} introduces datasets and evaluation metrics, noting the superior performance of NTP models over non-NTP models in both understanding and generation tasks.
Lastly, Section~\ref{sec:challenges} outlines unsolved challenges in MMNTP research, such as scaling up MMNTP, emergent abilities, modality-specific biases, modalities interference, and MMNTP as universal interfaces, and discusses approaches to mitigate these challenges.
Table~\ref{table:key_tables} outlines key tables and figures in our survey.

### Related Work

Several recent works have reviewed Large Multimodal Models (LMMs) in multimodal learning.
For instance, ~\citet{yin2024surveymultimodallargelanguage} delve into the understanding capabilities of early vision-language models.
Similarly, \citet{awais2023foundationalmodelsdefiningnew}, ~\citet{bordes2024introductionvisionlanguagemodeling}, ~\citet{ghosh2024exploringfrontiervisionlanguagemodels}, ~\citet{caffagni2024revolutionmultimodallargelanguage}, and ~\citet{zhang2024mmllmsrecentadvancesmultimodal} take a step forward and explore recent progress in multimodal learning with a focus on model architecture, training strategies, datasets, evaluation metrics and more.

In addition, several surveys have reviewed multimodal learning in vision-language tasks, including pre-training~\citep{caffagni2024revolutionmultimodallargelanguage}, transfer learning~\citep{zhang2024visionlanguagemodelsvisiontasks}, reasoning~\citep{wang2024exploringreasoningabilitiesmultimodal}, and reinforcement learning from human feedback (RLHF)~\citep{zhang2024visionlanguagemodelsvisiontasks}.
Beyond the discussions on the general revolution of the LMMs, specialized surveys have investigated the application of LMMs in domains such as multimodal agents~\citep{xie2024largemultimodalagentssurvey,li2023multimodalfoundationmodelsspecialists} and autonomous driving~\citep{cui2023surveymultimodallargelanguage}.

Recent surveys have also tackled key issues in multimodal learning, such as hallucinations in LMMs~\citep{liu2024surveyhallucinationlargevisionlanguage,rohleder2024variationalapproachhotspots} and efficiency of LMMs~\citep{jin2024efficientmultimodallargelanguage,xu2024surveyresourceefficientllmmultimodal}.

Diverging from prior work that primarily focused on the understanding abilities of multimodal LLMs, our survey adopts a systematic perspective by integrating both understanding and generation in multimodal learning through the paradigm of next-token prediction.
To the best of our knowledge, this is the first survey that reviews LMMs from the perspective of next token prediction, aiming to aid researchers in their exploration of multimodal intelligence.

In summary, in this survey, we aim to provide a holistic review on current multimodal models that rely on next token prediction.
An associated GitHub link collecting the latest papers is at [Github](https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction).
