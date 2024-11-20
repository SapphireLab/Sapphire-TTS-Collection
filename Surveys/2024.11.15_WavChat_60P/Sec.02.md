# 2·Overall: 整体视角

In this section, we will provide an overall overview of spoken dialogue models.
we begin by defining what constitutes an intelligent spoken dialogue model by examining various dialogue scenarios.
We then provide a comprehensive overview of spoken dialogue models, distinguishing between cascaded spoken dialogue models and end-to-end spoken dialogue models.

## 2.1·Functions of Spoken Dialogue Systems

Based on the demos and inference interfaces of representative models such as GPT-4o, Moshi~\cite{defossez2024moshi}, Qwen2-Audio~\cite{chu2024qwen2}, and VITA~\cite{fu2024vita}, we categorize the usage scenarios of modern intelligent spoken dialogue models into the following nine representative categories: 1) Text Intelligence, 2) Speech Intelligence, 3) Audio and Music Generation, 4) Audio and Music Understanding, 5) Multilingual Capability, 6) Context Learning, 7) Interaction Capability, 8) Streaming Latency, and 9) Multimodal Capability.
For the nine distinct use cases in spoken dialogue models, we provide corresponding examples for each scenario in Figure \ref{fig:main}.
It is clear from these usage scenarios that a spoken dialogue model is not simply an extension of a text-based dialogue model to the speech modality (i.e., where the speech modality serves merely as an interface for converting speech into text).
Rather, an intelligent spoken dialogue system must be capable of comprehending and generating acoustic information embedded in speech (such as timbre, style, and emotion) and of understanding and producing a wider range of audio representations, including information related to audio events and music.
Additionally, unlike non-streaming text-based systems, spoken dialogue models need to support real-time, interactive streaming capabilities.
These usage scenarios not only highlight the intelligence inherent in spoken dialogue systems but also present significant challenges for building end-to-end spoken dialogue models.
Below, we provide a detailed examination of each of the nine usage scenarios.

### 2.1.1·Text Intelligence

As illustrated in Figure \ref{fig:main} (a), a spoken dialogue system must retain the fundamental capabilities of the original text-based dialogue models, such as ChatGPT.
We define this usage scenario as textual intelligence.
In this context, the spoken dialogue model can intelligently respond to user requests, generating appropriate responses such as travel itineraries, work plans, and scheduling.
However, due to the limitations of voice-based interaction, the textual intelligence of current spoken dialogue systems is more focused on the daily scenarios.
In certain contexts, such as complex mathematical theorem reasoning, the performance requirements for spoken dialogue models differ from those of text-based dialogue models~\cite{touvron2023llama}.
These advanced aspects of textual intelligence warrant further exploration in unified multimodal dialogue models.

### 2.1.2·Speech Intelligence

A distinguishing feature of spoken dialogue models, compared to text-based dialogue models~\cite{touvron2023llama}, is their ability to understand and generate acoustic information beyond mere textual content.
In the speech modality, not only is the textual content present, but also additional acoustic information, such as timbre (speaker identity) and style (emotion, prosody, etc.).
As illustrated in Figure \ref{fig:main} (b), an intelligent spoken dialogue system should be capable of \textbf{understanding} the timbre and style of conversational speech and, ideally, \textbf{generating} responses with specified timbre and style in a \textbf{zero-shot} manner.

This capability about speech intelligence involves several use cases.
First, on the comprehension side, the spoken dialogue system should generate responses based on the speaker's vocal style.
For example, in the E-chat~\cite{xue2023chat}, a classic example might be: if a user asks, "My phone won't turn on, what should I do?" in a cheerful tone, the system might respond, "It looks like you're excited about getting a new phone.
What type of phone are you interested in?" Conversely, if the user asks the same question in a sad tone, the system might reply, "It's unfortunate your phone isn't working.
If you're familiar with the repair policy, let's proceed with the next steps." This situation indicates that the spoken dialogue system may generate responses with different \textbf{content} based on varying acoustic information.
Furthermore, the system should comprehend various acoustic cues, such as accents or emotional states, and adjust its responses of different \textbf{acoustic} information accordingly.
For instance, if the speaker is an American, the system might reply with a native English accent, whereas if the speaker is a Shanghainese user, the system could respond using the corresponding dialect.
Similarly, if the user speaks with a sad tone, the dialogue system should be able to generate a more encouraging and empathetic response.

On the generation side, speech intelligence is more prominently reflected in its controllability, such as voice cloning and style control.
For example, the system could be instructed to mimic a specific voice or respond in a designated style (e.g., mimicking a grandmother's soft and gentle voice for a comforting interaction).
Additionally, the system could use a voice prompt provided during the conversation to fully clone the timbre from the prompt and generate speech in that same voice.
In summary, the ability to comprehend and generate acoustic information is one of the key characteristics of an intelligent spoken dialogue model.

### 2.1.3·Audio and Music Generation

In the spoken dialogue models, beyond basic spoken dialogue capabilities, an intelligent spoken dialogue system may be required to generate music and audio.
For example, a user might instruct the system to generate a one-minute piano piece or a ten-second recording of a dog barking.
Additionally, users might provide lyrics and a musical melody, asking the spoken dialogue model to create a pop song.
The system should thus inherit the generative capabilities of large-scale music~\cite{agostinelli2023musiclm,copet2024simple,lam2024efficient,ma2024foundation} and audio~\cite{huang2023make,liu2023audioldm,liu2024audioldm} models on the output side.

### 2.1.4·Audio and Music Understanding

Complementing its music and audio generation capabilities, a spoken dialogue model should also be able to understand music and audio on the input side~\cite{chu2024qwen2,tang2023salmonn}.
For instance, when given an audio clip, the intelligent system should identify both its content and acoustic characteristics, such as recognizing whether the sound is a bird chirping or a cat meowing, or whether the music is calm or energetic.
Moreover, the system could extend its understanding by creating literary works—like poetry or songs—based on the given music or audio.

### 2.1.5·Multilingual Capability

Similar to text-based dialogue models, spoken dialogue systems are expected to possess multilingual capabilities.
Specifically, these models should be able to perform multilingual content translation, such as translating a spoken segment in Japanese into French speech clips, effectively inheriting the capabilities of simultaneous interpretation.
In addition to multilingual content translation, the system should also handle multilingual acoustic information.
This means that the intelligent spoken dialogue model should be able to generate responses in various languages and accents, replying in the corresponding accent of the target language based on the different input speech.

### 2.1.6·Context Learning

In the spoken dialogue models, the ability to handle long-form and multi-turn conversations is a key benchmark for evaluating performance~\cite{defossez2024moshi}.
This requires that spoken dialogue models not only support long-duration audio inputs but also generate extended audio outputs.
Moreover, they must be capable of engaging in multi-turn conversations based on historical context.
An important aspect of multi-turn dialogue is the ability to revise previous responses based on new user instructions.
As shown in Figure \ref{fig:main} (f), an intelligent spoken dialogue model should be able to continuously modify its previous replies according to the user’s evolving requests.

### 2.1.7·Interaction Capability

A distinguishing feature of spoken dialogue systems compared to the text-based dialogue models is their duplex and interactive nature~\cite{defossez2024moshi}.
In text-based dialogue, interactions typically follow a half-duplex structure, where the response can only be provided after the question has been completed, and the user is unable to interrupt the reply in real-time.
However, in the spoken dialogue systems, full-duplex interaction is common.
This means that a conversation does not need to be fully completed before a response can be generated.
Both the system and the user can interrupt and interact in real time.
For example, if the user is unsatisfied with the system's response, they can immediately interrupt, causing the system to halt its current generation and respond to the new input.
Additionally, to emulate more natural conversational settings, the system can also interrupt the user when appropriate, such as when clarifying the user’s intent.
Beyond the ability to interrupt, interactive dialogue often includes the use of conversational fillers, such as "okay," "haha," or "oh," which signal acknowledgment or agreement.
Including these within spoken dialogue models enhances the realism and natural flow of conversations.
The underlying requirement for interaction capabilities is that the system should be able to listen and speak simultaneously, responding dynamically to the flow of the interaction.

### 2.1.8·Streaming Latency

Streaming comprehension and generation are also fundamental functionalities of spoken dialogue models~\cite{xie2024miniomni2opensourcegpt4ovision,zhang2024intrinsicvoice,fang2024llama}.
In the real-world scenarios, a model cannot wait until an entire minute-long audio segment has been processed before generating a response.
Instead, the model must operate on a chunk-based mechanism, dynamically processing and generating audio in real time, one chunk at a time.
Additionally, the streaming requirement means that the entire system must operate in a causal manner—understanding and generating audio based solely on past information, without relying on future information.
Streaming function is often closely tied to the need for low latency.
In practical conversational experiences, the latency of the first token generated by the spoken dialogue model (i.e., the wait time for the user) and the average latency of the generation process are critical factors that influence the overall responsiveness and usability of the spoken dialogue system.

### 2.1.9·Multimodal Capability

Multimodal dialogue capability~\cite{chen2024emova,fu2024vita} represents an advanced feature of spoken dialogue models.
In existing systems, this typically refers to the ability to process inputs from multiple modalities, such as video, images, and text, while generating intelligent speech responses.
A spoken dialogue model equipped with this capability achieves the ability to “hear, see, and speak” simultaneously.
Multimodal inputs significantly enhance the potential of these systems; for instance, users can employ various gestures to improve the quality of the model’s generated responses, and the system can develop a deeper understanding of the physical world.
Beyond multimodal inputs, the future of dialogue systems lies in large multimodal models that unify the comprehension and generation capabilities across all modalities, with spoken dialogue serving as the foundational modality.
