# Mini-Omni

<details>
<summary>基本信息</summary>

- 标题: Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming
- 作者:
  - 01 [Zhifei Xie](../../../Authors/Zhifei_Xie.md)
  - 02 [Changqiao Wu](../../../Authors/Changqiao_Wu.md)
- 机构:
  - [清华大学](../../../Institutions/CHN-THU_清华大学.md)
- 时间:
  - 预印时间: 2024.08.29 ArXiv v1
  - 更新笔记: 2024.08.31
- 发表:
  - 期刊/会议 
- 链接:
  - [ArXiv](https://arxiv.org/abs/2408.16725)
  - [DOI]()
  - [Github](https://github.com/gpt-omni/mini-omni)
  - [Demo]()
  - [Scholar](https://scholar.google.com/scholar?cluster=)
- 标签:
  - ?
- 页数: 10
- 引用: ?
- 被引: ?
- 数据:
  - ? 
- 对比:
  - ?
- 复现:
  - ?

</details>

## Abstract: 摘要

> Recent advances in language models have achieved significant progress. GPT-4o, as a new milestone, has enabled real-time conversations with humans, demonstrating near-human natural fluency. Such human-computer interaction necessitates models with the capability to perform reasoning directly with the audio modality and generate output in streaming. However, this remains beyond the reach of current academic models, as they typically depend on extra TTS systems for speech synthesis, resulting in undesirable latency. This paper introduces the ***Mini-Omni***, an audio-based end-to-end conversational model, capable of real-time speech interaction. To achieve this capability, we propose a text-instructed speech generation method, along with batch-parallel strategies during inference to further boost the performance. Our method also helps to retain the original model's language capabilities with minimal degradation, enabling other works to establish real-time interaction capabilities. We call this training method **"Any Model Can Talk"**. 
> We also introduce the **VoiceAssistant-400K dataset** to fine-tune models optimized for speech output. To our best knowledge, ***Mini-Omni*** is the first fully end-to-end, open-source model for real-time speech interaction, offering valuable potential for future research.

## 1.Introduction: 引言

> Recent developments in large language models have progressed rapidly, with models becoming increasingly powerful, such as off-the-shelf Llama 3.1 \citep{llama3.1}, Mixtral \citep{mixtral}, Qwen-2 \citep{qwen2}, and the well-known GPT-4. As an extension of their capabilities, language models are beginning to master understanding other modalities, exemplified by LLaVA \citep{llava}, Qwen2-Audio \citep{qwen2audio} and Video-llama \citep{videollama}. Despite their strength in specific tasks, a significant gap remains that hinders further integration of large language models into daily application: real-time voice interaction capability. GPT-4o \citep{gpt4}, introduced by OpenAI, is the first model to feature real-time multimodal speech interaction capabilities. It can understand and engage with vision, audio, and text while enabling real-time speech conversations, although it remains closed-source. Other models typically adopt two approaches to incorporate speech capabilities. The first is a cascade method, where the language model generates text, followed by a text-to-speech (TTS) model for audio synthesis. This approach introduces significant latency due to the time required for text generation, severely impacting user experience. The second, an end-to-end method like SpeechGPT \citep{speechgpt}, generates text before continuing to generate audio. However, this still requires waiting for text generation. Large language models need real end-to-end speech output capabilities to provide real-time feedback.
>
> Enhancing models with speech output capabilities is a challenging task, primarily due to four factors: (1) \textbf{Complexity of Audio Reasoning}: Our experiments indicate that direct training for audio modality reasoning is highly challenging, often resulting in incoherent outputs from the model. (2) \textbf{Model Complexity}: Incorporating additional modules for speech input and output increases the overall complexity. (3) \textbf{Difficulty in Modality Alignment}: The reasoning abilities developed for text are difficult to transfer to the audio domain. (4) \textbf{Resource Demands}: Adapting a model's text capabilities to the speech modality requires converting all data labels into audio and retraining, significantly increasing resource consumption.
>
> In this paper, we propose \textbf{Mini-Omni}, the first open-source multi-model large language model with real-time conversational capabilities, featuring fully end-to-end speech input and output abilities. It also includes various other audio-to-text functionalities such as Automatic Speech Recognition (ASR). We adapt currently available off-the-shelf methods for discretizing speech tokens and employ the simplest model architecture, making it easy for our model and approach to be adapted by other researchers. 
> Direct audio reasoning poses significant challenges; however, our approach successfully addresses this using only a 0.5B model and a limited amount of synthesized audio data. Importantly, our training framework achieves this without heavy reliance on extensive model capabilities or large volumes of data.
>
> To leverage and preserve the original capabilities of the language model, we propose a parallel generation paradigm in which the transformer simultaneously produces audio and text tokens. Subsequently, we observed a minimal impact of the audio modality on text capabilities and further introduced \textbf{batch-based parallel generation}, which significantly enhances the model’s reasoning ability during streaming audio output. As a poinerr, we opted not to sacrifice audio quality for a simpler and lower bitrate audio encoder, in order to reduce the complexity of audio inference in the model. However, to ensure audio quality, we selected SNAC \citep{snac}, a music-grade encoder features 8 layers of codebooks and processes hundreds of tokens per second. Innovatively, we applied \textbf{text-instructed delayed parallel generation} to address the issue of long SNAC codebook sequences. Experiments show that the audio output quality is on par with common TTS systems.
>
> We also propose a method that requires minimal training and modification of the original model, enabling other works to rapidly develop their own speech capabilities. We refer to this approach as \textbf{"Any Model Can Talk"}, designed to achieve speech output using a limited amount of additional data. The approach extend speech capabilities through additional adapters and pre-trained models, fine-tuning with a small amount of synthesized data. This is combined with the aforementioned parallel modeling approach to enable streaming output in the new modality while preserving the original model’s reasoning capabilities. 
>
> To evaluate the capabilities of \textbf{Mini-Omni}, we first assessed its performance on traditional text-to-speech multi-modal tasks, including text-based question answering (textQA), automatic speech recognition (ASR), text-to-speech response, and speech-based question answering (speechQA). The model demonstrated strong proficiency in these fundamental tasks. Additionally, we conduct a series of experiments to investigate the impact on the original model's capabilities and assess the effectiveness and variations of our inference method. Preliminary experiments demonstrate that batch parallel inference preserves the model’s original capabilities. We will conduct further experiments and provide additional details in due course.
>
> Lastly, we observed that most open-source QA datasets contain mixed code or overly lengthy text, rendering them unsuitable for speech model. To overcome this limitation, we introduce the \textbf{VoiceAssistant-400K} dataset, comprising over 400,000 entries specifically generated by GPT-4o for speech assistant supervised fine-tuning (SFT).
>
> In summary, we make the following contributions:
>
> - We introduce \textbf{Mini-Omni}, the first open-source end-to-end multimodal large model with audio input and audio streaming output capabilities. We propose a unique text-instruct parallel generation method that enables speech inference outputs aligned with textual capabilities, achieved with minimal data. We further enhance this with delayed parallelism, accelerating audio inference speed. 
> - We introduce "\textbf{Any Model Can Talk}", an innovative approach that enhances performance without altering the architecture of large models by focusing on training and inference. Our method employs a three-phase training process for speech-to-text and text-to-speech adapters, including annealing and SFT. Our method involves minimal training and modification of the original model, aiming to provide a reference for incorporating interaction capabilities into other models.
> - We identified shortcomings in existing open-source QA datasets when training audio assistants and proposed a dedicated dataset for speech model outputs, called\textbf{ VoiceAssistant-400K}. This dataset, synthesized using GPT-4o, can be used to fine-tune models to develop the tone of a voice assistant.

## 2.Related Works: 相关工作

### Multimodal Understanding

> Recently, researchers have been increasingly focused on advancing unified models for cross-modal understanding. These approaches typically employ a well-pretrained neural network as the encoder for relevant modalities, using a lightweight adapter to align the encoder's output with the text input of language model. Classical works such as LLaVA \citep{llava}, Flamingo \citep{flamingo} and BLIP \citep{blip} are used for visual understanding, while in the audio domain, models like Whisper \citep{whisper} and Beats \citep{beats} are commonly utilized as encoders for semantic and acoustic features. In Llama 3.1, Whisper is employed, while SpeechVerse \citep{speechverse} leverages WavLM \citep{wavllm}; SALMONN \citep{salmonn}, combine Whisper and Beats to extract features. Such works are often constrained to producing output in the text modality.

### Audio Language Modeling 

> Recently, an increasing number of studies have employed audio tokenization to bridge the gap between audio and text. Audio tokenization converts continuous audio signals into discrete audio tokens, enabling large language models to perform inference and even cross-modal interactions. As a result, a variety of speech-text tasks, such as ASR, TTS, music understanding and generation, and sound editing, can be accomplished. MegaTTS \citep{megatts} utilized audio codecs for speech synthesis, while efforts like InstructTTS \citep{instructtts}, SpearTTS \citep{spearTTS}, and Voicebox \citep{voicebox} have further explored optimizations in decoding methods and conditioning techniques, employing Diffusion as the converter from tokens to audio.

### Real-Time Human-Machine Interaction Models 

> Since the introduction of GPT-4o \citep{gpt4}, real-time conversational models have achieved unprecedented results, providing near-instantaneous voice feedback to user inputs, marking a significant milestone for the next generation of multi-modal large models. However, the technical implementations remain proprietary. Models with real-time interaction capabilities are currently scarce. SpeechGPT \citep{speechgpt} is an early end-to-end speech interaction model; however, it still suffers from latency due to the Audio-Text-Text-Audio(A-T-T-A) process, similar to Spectron \citep{Spectron}. LauraGPT \citep{lauragpt} also employs a similar approach but not for voice conversation scenario. VITA \citep{vita} and Qwen-audio2 \citep{qwen2audio} are two models that support voice input, but they output text and rely on external TTS systems for speech synthesis. \textbf{Mini-Omni} is a fully end-to-end speech-to-speech conversational model. Through our exploration, we have identified the biggest challenge in advancing this field: the logical inconsistency in reasoning when only the audio modality is present, which we will address in the following chapter.

## 3.Methodology: 方法

## 4.Experiments: 实验

## 5.Results: 结果

## 6.Conclusions: 结论

> In this work, we introduce \textbf{Mini-Omni}, the first multi-modal model with direct speech-to-speech capabilities. Building on previous approaches that use text-guided speech generation, we propose a parallel text and audio generation method that leverages minimal additional data and modules to rapidly transfer a language model's text capabilities to the audio modality, supporting streaming output interactions with high model and data efficiency. We explore both text-instructed streaming parallel generation and batch parallel generation, which further enhance the model's reasoning ability and efficiency. Our approach successfully addresses challenging real-time dialogue tasks using a model with only 0.5 billion parameters. We have developed the \textbf{Any Model Can Talk} method, based on a pre and post-adapter design, to facilitate rapid speech adaptation of other models with minimal additional training. Additionally, we have released the VoiceAssistant-400K dataset for fine-tuning speech output, designed to minimize the generation of code symbols and assist humans in a voice assistant-like manner. All our data, inference, and training codes will be progressively open-sourced at https://github.com/gpt-omni/mini-omni. We hope to provide guidance and support for other work focused on language model speech interaction.
