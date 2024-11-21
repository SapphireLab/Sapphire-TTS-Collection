# 4·Training Paradigm of Spoken Dialogue Model

Existing text-based large language models have demonstrated strong contextual understanding and reasoning abilities in the field of natural language processing, such as GPT-4 \cite{achiam2023gpt}, Llama 3.1 \cite{dubey2024llama}, and Qwen-2 \cite{yang2024qwen2}.
Due to their training on large-scale corpora, these models achieve exceptional accuracy when handling complex contexts.
To further expand the capabilities of large language models, some research \cite{chen2024emova,chu2024qwen2,fu2024vita,xie2024miniomni2opensourcegpt4ovision} has explored enabling them to understand other modalities, thereby building multimodal interaction abilities.
The spoken dialogue model, also known as the speech-text dialogue model, allows users to interact with LLMs naturally and straightforwardly through speech.
However, the transition from text intelligence to speech intelligence involves two inherent hurdles: one core issue is the insufficient amount of speech data compared to the massive datasets used for pre-training text-based large language models.
For instance, Llama 3.1 \cite{dubey2024llama}  uses 800 billion training tokens, and Qwen-2 \cite{yang2024qwen2} is trained on over 7 trillion tokens, whereas pure speech pre-training data often amounts to hundreds of thousands or millions of hours.
For example, Moshi's \cite{defossez2024moshi} pre-training speech data comprises 7 million hours, and the amount of labeled speech data is even smaller, making it difficult to support LLMs in achieving powerful speech intelligence comparable to text.
Another challenge is that speech information density is not as compact as text.
Text commonly uses byte-pair encoding (BPE) \cite{gage1994new,sennrich2015neural} encoding to compress it into a tight token space, whereas the speech modality includes not only semantic information but also acoustical information, which is less dense.
This undoubtedly increases the difficulty for LLMs to learn.
Understanding and generating the inherent knowledge of the speech modality more effectively is a significant challenge.

Consequently, existing spoken dialogue models aim to build upon text-based LLMs by incorporating the speech modality into these large language models.
\cite{zhang2023speechgpt,chen2024emova,xie2024mini,defossez2024moshi} support speech-in and speech-out capabilities for LLMs, forming the foundation of basic speech dialogue capabilities.
Some of the latest advanced approaches \cite{defossez2024moshi,zhang2024omniflatten,veluri2024beyond} attempt to transition from traditional turn-based spoken dialogue systems to full-duplex systems, aiming to simulate the natural spontaneity of human conversation.
While these advancements are promising, achieving low latency and natural interaction in full-duplex systems remains a significant challenge.
Moreover, enhancing LLMs to effectively handle the speech modality—mastering both speech comprehension and generation—while maintaining robust natural language text processing capabilities, is hindered by the limited size of labeled speech datasets.
These datasets are far smaller compared to the vast amounts of pure text data available, which risks diminishing the models' original text processing capabilities.
Thus, building a truly end-to-end conversational model that meets real-world requirements necessitates careful consideration of model architecture, training paradigms, and training data.
Overall, we believe that several key aspects are crucial in the training paradigm of spoken dialogue models: aligning speech-text modalities to ensure consistent understanding, designing multi-stage training strategies for gradual adaptation, and optimizing training structures and inference paradigms for efficient performance.

## 4.1·Architecture Paradigm about Modal Alignment of Speech and Text

To enable large language models (LLMs) to handle both speech input and output, a significant amount of prior work \cite{rubenstein2023audiopalm,dubey2024llama,fang2024llama,xie2024mini,defossez2024moshi} has focused on adapting text-based foundation models into robust spoken dialogue models.
Based on different architectural paradigms, these approaches can be broadly categorized into five types, as shown in Figure ~\ref{fig:archi_img1}.

### Text-Output Only Method

These systems \cite{chu2024qwen2,chu2023qwen,gong2023joint,xue2023chat,tang2023salmonn,hu2024wavllm,das2024speechverse,fu2024vita} maintain the text-based LLM’s foundational structure unchanged, \textbf{using an audio encoder and adaptor to map speech input into the LLM's pre-trained text latent space directly.} This method of direct embedding alignment, combined with a multi-task training strategy, equips the LLM with the ability to 'listen,' thus enabling it to understand and process speech modality inputs effectively and perform exceptionally well in various audio understanding tasks.
Nevertheless, the output remains text-based, which necessitates the use of an external text-to-speech (TTS) system \cite{casanova2024xtts,du2024cosyvoice} to generate speech output.
LTU-AS \cite{gong2023joint} uses Whisper \cite{radford2023robust} and the Time and Layer-Wise Transformer (TLTR) as its audio encoder, allowing it to recognize both speech and audio events.
Qwen-Audio 1 \cite{chu2023qwen} scales up audio-language pre-training to cover over 30 tasks and various audio types, facilitating universal audio understanding abilities.
It employs a unified encoder for all audio inputs, bridging the gap between audio and textual modalities, and uses the large language model Qwen-7B \cite{bai2023qwen} as its foundational component.
Qwen-Audio 2 \cite{chu2024qwen2} simplifies the pre-training process by utilizing natural language prompts for different data and tasks, with DPO \cite{rafailov2024direct} optimizing the model’s performance in terms of factuality and adherence to desired behavior.
SALMMON \cite{tang2023salmonn} employs dual auditory encoders: a speech encoder from the Whisper model and a non-speech BEATs \cite{chen2022beats} audio encoder.
The auditory features from these two encoders are complementary, making them suitable for general audio inputs that contain both speech and non-speech information.
These inputs are then connected to a well-trained LLM using Q-former style attention to generate responses.
 VITA \cite{fu2024vita} implements a duplex solution through two independent modules: one generates text responses to user queries, while the other continuously monitors environmental input to selectively provide updated interaction content, although it still requires an external TTS system.
All the aforementioned methods frequently overlook paralinguistic information, including emotion, prosody, and non-verbal elements, rendering them insufficient for scenarios that involve emotional speech dialogue.
ParalinGPT \cite{lin2024paralinguistics} utilizes an ASR model to obtain text and a speech encoder to extract emotion embeddings, thereby more accurately simulating both the linguistic content and paralinguistic attributes of spoken responses.
E-chat \cite{xue2023chat} employs a Hubert speech encoder \cite{hsu2021hubert} to extract speech and emotion features, using a connection module to map these features to the textual space within the LLM decoder.
Although these approaches have explored emotional responses within spoken dialogue systems, they require additional systems to synthesize speech from text and suffer from high latency, making real-time dialogue challenging to achieve.

### Chain-of-Modality (CoM) Method.

This method tokenizes speech into discrete tokens and extends the LLM’s vocabulary to handle both speech input and output.
To address alignment issues between speech and text modalities, Recent works \cite{zhang2023speechgpt,zhang2024speechgpt,nachmani2023spoken,chen2024emova} utilize a prompting approach called Chain-of-Modality (CoM), which first generates response text autoregressively before producing the corresponding speech.
This technique allows the text LLM's output to guide speech generation, thereby enhancing the quality of the response content.
However, it is not suitable for live interactions, as the model must complete the entire text response before beginning speech generation, leading to increased response latency.
SpeechGPT \cite{zhang2023speechgpt} and SpeechGPT-gen \cite{zhang2024speechgpt} employ the SpeechTokenizer \cite{zhang2023speechtokenizer} model as a speech token extractor, breaking down speech generation into the prediction of semantic tokens followed by acoustic tokens.
Spectron \cite{nachmani2023spoken} performs speech continuation by predicting spectrograms frame-by-frame, optimizing the LLM with a combination of cross-entropy loss for text and reconstruction loss for speech frames.
EMOVA \cite{chen2024emova}, on the other hand, utilizes the FSPIRAL \cite{huang2022spiral} architecture for its speech encoder to capture phonetic and tonal information, which is then discretized using finite scalar quantization (FSQ) \cite{mentzer2023finite}.
Its speech response procedure is divided into three primary steps:
1) transcribing user instructions into text,
2) generating textual responses based on these instructions, and
3) producing style labels and response speech units from the textual responses.

This process enables EMOVA to facilitate emotional speech dialogue.

### Interleaving Text and Speech Tokens

Some earlier models \cite{rubenstein2023audiopalm,maiti2024voxtlm} employed supervised training methods, using specific input and output sequences, and trained on mixed speech-text tasks, including text-to-speech (TTS), automatic speech recognition (ASR), and speech-to-speech translation.
Spirit-LM \cite{nguyen2024spirit} leverages the temporal alignment between speech and its transcription, continuing training on a pre-trained text-based LLM using alternating text and speech tokens.
This significantly improves the model’s performance in both speech understanding and generation.
However, it employs discrete Hubert units \cite{hsu2021hubert} as speech representations, which results in some loss of paralinguistic information.
USDM \cite{kim2024unified} continues pretraining Mistral-7B \cite{chaplot2023albert} with interleaved speech-text data to capture multimodal semantics.
For dialogue finetuning, it constructs templates using both speech and transcripts of user input as instruction data.

### Parallel Generation of Text and Speech.

PSLM \cite{mitsui2024pslm} proposes generating speech and text tokens in parallel to reduce latency; however, this approach may compromise response quality.
Additionally, this method still relies on speech recognition for input \cite{radford2023robust}, which introduces further delay.
Llama-Omni \cite{fang2024llama} introduces a novel streaming speech decoder that can simultaneously generate text responses and discrete speech unit sequences, significantly reducing latency and meeting real-time interaction needs.
Moshi \cite{defossez2024moshi} and Mini-Omni \cite{xie2024mini} adopt similar approaches, introducing dual streams that generate both speech tokens and corresponding text tokens simultaneously on the assistant side, facilitating the transfer of the pre-trained LLM’s textual capabilities to the speech modality, enabling the model to directly engage in reasoning through speech.
The key difference lies in how speech-text alignment is handled: Moshi \cite{defossez2024moshi} uses explicit alignment information to supervise the model’s learning, while Mini-Omni \cite{xie2024mini} allows the LLM to learn implicit alignment information.
On the input side, Mini-Omni feeds continuous speech embeddings from the Whisper encoder \cite{radford2023robust} into the LLM, enhancing the model's ability to understand spoken instructions without requiring text input.
However, inconsistencies between speech input and output introduce additional computational overhead, increasing latency in multi-turn dialogue scenarios.
In contrast, Moshi allows users to input speech without relying on text, and generates both text and speech tokens in parallel on the assistant side.
Moshi further extends its architecture to model several speech streams in parallel, allowing for conceptually and practically simple handling of full-duplex dialogues with arbitrary dynamics.

### Speech-to-Speech Generation

This approach aims to remove the dependency on intermediate text, thereby reducing latency and making the system closer to real-time interaction.
SyncLLM \cite{veluri2024beyond} achieves real-time full-duplex interaction through time chunking methods, integrating time information into LLMs to enable synchronous operation with the real-world clock.
IntrinsicVoice \cite{zhang2024intrinsicvoice} utilizes a specific model to generate multiple speech tokens in a single step, effectively reducing speech token sequences to lengths comparable to text sequences while producing high-quality audio.
Align-SLM \cite{lin2024alignslmtextlessspokenlanguage} utilizes a pre-trained self-supervised Hubert model \cite{hsu2021hubert} with K-means clustering \cite{hassid2024textually} to convert continuous speech representations into discrete units.
It employs LoRA adapter \cite{hu2021lora} fine-tuning on a pre-trained Twist \cite{hassid2024textually} to produce multiple speech continuations from a given prompt and uses semantic metrics to generate preference data for Direct Preference Optimization (DPO) \cite{rafailov2024direct}.
Experimental results indicate that integrating the preference optimization method significantly improves the semantic comprehension of the Spoken LLM.

## 4.2·Multi-Stage Training Strategy

This section primarily discusses the training process of the Spoken Dialogue Model, building upon previous work on spoken dialogue systems.
Generally, this process consists of four stages: text LLM pre-training, modality adaptation and alignment post-training, followed by supervised fine-tuning, and optionally, preference optimization.
The primary goal in training most spoken dialogue systems is to preserve the model's original capabilities while integrating the speech modality for voice interaction into the LLM.
The diagram of multi-stage training can be referred to in Figure ~\ref{fig:archi_img2}.

### 4.2.1·Text LLM Pre-Training

The goal is to develop a text-intelligent LLM model capable of handling complex contexts and possessing knowledge reasoning abilities, thus preparing it for integration with speech-intelligent LLMs.
Most spoken dialogue systems utilize pre-trained large language models as foundational models rather than pre-training with separate text data themselves.
A series of approaches \cite{zhang2023speechgpt,zhang2024speechgpt,nguyen2024spirit,chen2024emova,fang2024llama,veluri2024beyond} use the LLaMA model and its variants as their foundational language model.
On the other hand, \cite{du2023lauragpt,xie2024mini,xie2024miniomni2opensourcegpt4ovision,zhang2024omniflatten} employ the Qwen \cite{bai2023qwen,yang2024qwen2} family of large language models as their backbone.
Meanwhile, Moshi \cite{defossez2024moshi} employs an RQ-Transformer for hierarchical autoregressive modeling of speech, utilizing a unique structure that involves pre-training a text-only language model with datasets from the internet (e.g., Wikipedia \footnote{\url{https://dumps.wikimedia.org/}} and StackExchange \footnote{\url{https://archive.org/details/stackexchange/}}).
The collected data was filtered using a comprehensive preprocessing pipeline to ensure quality and relevance, which included deduplication to remove redundant entries, language identification to retain text in the desired language, and quality filtering to exclude low-quality or irrelevant content based on criteria such as coherence and completeness.
VITA \cite{fu2024vita} utilizes Mixtral 8x7B1 \cite{jiang2024mixtral}, a representative LLM with a sparse mixture of experts (SMoE) architecture, and performs pure-text instruction tuning for its extended Chinese vocabulary.

### 4.2.2·Modality Adaptation and Alignment Post-training

This phase explores strategies to adapt text-based large language models (LLMs) for speech modality input, focusing on aligning text and audio modalities effectively.
The primary goal is to enhance the models' ability to understand and generate speech by bridging the gap between these two modalities.
Common approaches include multimodal training techniques, leveraging unlabeled speech corpora, and employing multi-task learning frameworks.
These methods typically involve fine-tuning existing LLMs with speech-related tasks and integrating speech-specific modules, such as speech adaptors and decoders, to facilitate seamless interaction between text and speech modalities.
Different training tasks for modality adaptation and alignment are shown in Figure ~\ref{fig:archi_img3}.
Spirit-LM \cite{nguyen2024spirit} continuously pretrains on text LLM checkpoints using interleaved text and speech tokens to improve the model's performance in speech understanding and generation.
LLaMA-Omni \cite{fang2024llama} adopts a two-stage training strategy: the first stage jointly trains a speech adaptor and LLM with speech input and text responses, while the second stage uses the same dataset to train a streaming speech decoder independently.
Consequently, this LLM primarily possesses the capability for speech input understanding, with speech generation handled by a separate decoder module.
SpeechGPT \cite{zhang2023speechgpt}, Moshi \cite{defossez2024moshi}, and VITA \cite{fu2024vita} utilize unlabeled speech corpora to train models in a next-token prediction task.
In the first phase, VITA focuses on training the audio encoder and connector, while in the second phase, it optimizes both the connector and the LLM model through multimodal training.
Although capable of processing speech input, it outputs only text.
Spectron \cite{nachmani2023spoken} addresses the alignment issue between text and speech representations by jointly supervising multiple objectives.
IntrinsicVoice \cite{zhang2024intrinsicvoice} employs a two-stage training approach, constructing multiple cross-modal tasks from a single dataset to enable the model to better learn the semantic consistency between speech and text.
Mini-Omni \cite{xie2024mini}, EMOVA \cite{chen2024emova}, and OmniFlatten \cite{zhang2024omniflatten} adopt similar methodologies, commencing with supervised multi-task fine-tuning of the text LLM backbone to achieve speech-text modality alignment and develop a multimodal LLM~\cite{jin2024efficientmllm, li2024surveybenchmarksmultimodallarge} using Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) tasks.
Notably, Mini-Omni divides the training of various modules into three phases: the first phase utilizes data from speech recognition and synthesis to enhance the model’s abilities in these aspects, training only the ASR and TTS adapters.
The second phase focuses exclusively on enhancing the model’s text capabilities when given speech inputs, updating only the LLM parameters while freezing other modules.
Through these two training phases, the original language LLM’s capabilities are maximally preserved, while adapting to speech modality input and output, thereby addressing the primary modality alignment tasks.

### 4.2.3·upervised Fine-tuning or Dialogue Dataset Fine-tuning

During this stage, most models use instruction-following datasets or dialogue data for supervised fine-tuning of the LLM, enhancing natural conversational abilities.
\cite{zhang2023speechgpt,zhang2024speechgpt} propose a two-stage instruction-tuning process that includes cross-modal instruction fine-tuning and chain-of-modality instruction fine-tuning.
Ultimately, the model follows the A-T-T-A method to achieve end-to-end speech input and output.
EMOVA \cite{chen2024emova} employs a similar chain-of-modality concept to construct instruction-tuning datasets, empowering it to respond accurately to speech instructions.
Moshi \cite{defossez2024moshi}, Mini-Omni \cite{xie2024mini}, OmniFlatten \cite{zhang2024omniflatten}, and SyncLLM \cite{veluri2024beyond} utilize spoken dialogue datasets for fine-tuning, endowing the models with conversational interaction capabilities.
Remarkably, Moshi constructs a more natural and realistic dialogue dataset that incorporates elements such as noise and overlap, enabling the model to learn authentic multi-stream interactions.
OmniFlatten fine-tunes the speech-text LLM using interleaved and serialized dialogues across three stages to progressively train the model in acquiring half-duplex and full-duplex communication capabilities.
Similarly, SyncLLM employs a three-stage training procedure that predominantly uses synthetic spoken dialogue data along with a relatively small amount of real-world spoken dialogue data to develop a full-duplex voice agent.

### 4.2.4·Preference Optimization and Reinforcement Learning

The research on leveraging preference optimization to align a spoken dialogue model with human preferences is virtually absent.
Recently, \cite{anastassiou2024seed,zhang2024speechalign,chen2024enhancing} adopted preference optimization for Text-to-Speech (TTS) models to align speech synthesis quality with human preferences but not for spoken dialogue models.
Align-SLM \cite{lin2024alignslmtextlessspokenlanguage} pioneers the integration of Direct Preference Optimization (DPO) \cite{rafailov2024direct} in textless Spoken Language Models (SLMs) to enhance semantic understanding.
It transforms continuous speech into discrete units using a pre-trained Hubert model and K-means clustering.
LoRA fine-tuning on a Spoken LLM generates multiple speech continuations from prompts.
Semantic metrics create preference data offline, making DPO training efficient and stable, eliminating the need for an external reward model.
Coupled with curriculum learning \cite{bengio2009curriculum}, Align-SLM progressively refines preference data selection, optimizing semantic feedback, and improving SLM performance.

## 4.3·Training Frameworks and Generation Strategies

Recent advanced methods in spoken dialogue models employ a variety of innovative techniques to achieve more natural speech output and lower latency. In this part, we explore various approaches that exemplify these advancements:

### LLaMA-Omni

LLama-Omni \cite{fang2024llama} adds a streaming speech decoder that operates after the LLM. This decoder runs in a non-autoregressive manner, taking the output hidden states from the LLM as input and generating the discrete unit sequence corresponding to the speech response. To model the variable-length mapping between input and output, LLama-Omni employs an upsample factor, denoted as $\lambda$, along with Connectionist Temporal Classification (CTC) loss \cite{graves2006connectionist}. This ensures that the model can generate speech responses simultaneously with text responses. Additionally, a predefined chunk size is set to further enable vocoder streaming synthesis of speech waveforms, facilitating real-time interaction and reducing latency.

### Mini-Omni

Mini-Omni \cite{xie2024mini} selects SNAC \cite{siuzdak2024snac}, a music-grade encoder, to discretize one second of audio into hundreds of tokens, which significantly increases the burden on the LLM for modeling speech tokens. Delay Pattern language model decoding strategies are often applied in modeling multiple parallel streams of acoustic tokens in speech tasks like MusicGen \cite{copet2024simple}, VoiceCraft \cite{peng2024voicecraft}, and Parler-TTS \cite{lyth2024natural}. Compared with traditional sequential step decoding, this strategy can effectively reduce the time steps required for LLM decoding and generating speech tokens. Inspired by this, Mini-Omni innovatively applies text-instructed delayed parallel generation to address the issue of long SNAC codebook sequences, simultaneously producing audio and text tokens. This effectively leverages and preserves the original capabilities of the language model. Moreover, Mini-Omni proposes a Batch Parallel Decoding method. Specifically, it generates two samples in parallel for a single input: the first predicts text tokens, and the second predicts both text and speech tokens simultaneously. The text output from the first sample is embedded into the corresponding positions of the second sample, while the second sample's text output is discarded. This further enhances the model’s reasoning capabilities during dialogue, maximizing the transfer of its text-based abilities.

### IntrinsicVoice

IntrinsicVoice \cite{zhang2024intrinsicvoice} introduces a speech encoder and a streaming vocoder for the tokenization and detokenization of speech, and a GroupFormer for modeling speech and text sequences. This architecture integrates a large language model (LLM) with a GroupModel. Specifically, it uses a pre-trained HuBERT encoder \cite{hsu2021hubert} and its corresponding KMeans quantizer \cite{hassid2024textually} to process speech inputs into discrete units. These units are organized into a grouped token sequence through a group partition operation. The grouped tokens are then passed through an embedding layer and adaptor module to map these embeddings into the LLM's embedding space. The context embeddings output by the LLM are processed through a linear layer and concatenated with a specified number of learnable queries. This input is fed into a smaller non-autoregressive transformer encoder model, dubbed the "GroupModel," to predict a group of speech tokens in one step. The introduction of GroupFormer effectively improves the model's ability to handle sequences within a group, mitigates the modality gap between speech and text, accelerates inference speed, and alleviates issues associated with long-sequence modeling.

### Moshi

Moshi \cite{defossez2024moshi} introduces a mini codec model with 8 codebooks at a frame rate of 12.5 Hz for speech representation, where one second corresponds to 100 speech tokens. It adopts an RQ-Transformer consisting of a Temporal Transformer and a smaller Depth Transformer as the backbone network for the LLM, hierarchically modeling multi-codebook audio tokens. Similar architectures have appeared in prior research, such as UniAudio \cite{yang2023uniaudio} and Megabyte \cite{yu2023megabyte}. The Depth Transformer models sub-sequence tokens conditioned on temporal context predicted by the Temporal Transformer. Given the smaller size of the Depth Transformer, sub-sequence generation can almost be viewed as parallel generation. This allows the model to scale to longer sequences by extending the temporal modeling capacity of the Temporal Transformer or to achieve greater depth by enhancing the hierarchical modeling capabilities of the Depth Transformer, rather than modeling the flattened sequence with a single model.

### SyncLLM

SyncLLM \cite{veluri2024beyond} employs an auto-regressive transformer decoder for full-duplex dialogue, integrating time synchronization to align speech units with the real-world clock. It predicts interleaved speech tokens for both dialogue partners, maintaining timing with speaker tags. The model is trained on deduplicated HuBERT token sequences to enhance semantic fidelity while managing latency by anticipating user responses. Interpolation reconstructs token sequences to fit expected structures, facilitating seamless speech synthesis.


### Text-guided generation

Some end-to-end methods like \cite{zhang2023speechgpt,zhang2024speechgpt,nachmani2023spoken,chen2024emova} use chain-of-thought reasoning, which allows guiding speech generation with the output of an underlying text LLM. However, this is fundamentally incompatible with live interactions, as the model needs to produce an entire answer as text before it starts speaking. Later methods \cite{fang2024llama,xie2024mini,defossez2024moshi} can accept user speech input and simultaneously output speech and text, ensuring high-quality responses while significantly reducing latency. Lama-Omni \cite{fang2024llama} utilizes a streaming decoder to generate text and speech tokens in parallel. Mini-Omni \cite{xie2024mini} is restructured to transfer language reasoning abilities to streaming audio output through a text-audio parallel decoding approach. Moshi \cite{defossez2024moshi} details a novel feature, the Inner Monologue, which consists of joint modeling of the textual and speech modalities on the system side to improve the quality of interactions.

### W/o text-guided generation

Other methods achieve speech-to-speech generation without relying on text stream generation. IntrinsicVoice \cite{zhang2024intrinsicvoice} introduces a novel GroupModel that predicts a group of speech tokens in one step based on global context embeddings. SyncLLM \cite{veluri2024beyond} predicts interleaved chunks of token sequences at each time step, allowing the model to handle all conversational cues such as backchannels, overlaps, interruptions, etc.
