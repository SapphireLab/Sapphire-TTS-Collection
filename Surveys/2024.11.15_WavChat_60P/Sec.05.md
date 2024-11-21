# 5·Streaming, Duplex, and Interaction

Streaming, full-duplex technology, and interactions, are crucial elements for enhancing the interactive capabilities of spoken dialogue models because they directly impact the system's responsiveness, the fluidity of natural interaction, and its ability to handle complex interactions.Unlike text language models, spoken dialogue models require real-time processing of user input.
\textbf{Streaming} allows the system to instantly acquire and process speech data; \textbf{full-duplex technology} enables both the system and user to speak simultaneously, enhancing the naturalness of interaction; and \textbf{handling of interactions} provides the model with the ability to recognize and adapt to various conversational contexts, making the dialogue more intelligent and realistic.
Building on early explorations, GPT-4o's advanced spoken dialogue capabilities have ignited a surge of research interest.
With real-time voice processing and natural conversational interaction, these models offer users a seamless and efficient communication experience.
However, achieving these capabilities requires deep research into model architecture, data collection, system design, and training methods.
The model needs to be carefully designed and optimized in terms of real-time performance, stability, and response speed.
At the same time, duplex technology is an indispensable key implementation, which ensures that the voice model has both "ears" and "mouths".
Next, we will first discuss the streaming processing method in Section 5.1, then introduce the key technologies of duplex communication and explains how to handle interactation to improve user experience in Section 5.2.

## 5.1·Streaming Spoken Dialogue Models

The core of streaming speech models lies in their "real-time" and "continuous" capabilities, meaning they can process input and generate output simultaneously without waiting for complete input.
This includes two main aspects:
- **Streaming Understanding**.
The model can process audio input as the user speaks, without needing to wait for the user to finish entirely, allowing it to align more naturally with the flow of conversation.
- **Streaming Generation**.
This concept refers to the model's ability to generate output without waiting for all intermediate hidden states.
Instead, it can produce output progressively as processing occurs, which improves responsiveness and allows for smoother, more efficient interactions.

These streaming capabilities allow the model to perform more fluidly in real-time interactions, providing a seamless communication experience for users.
We will explore streaming techniques in both end-to-end and cascaded spoken dialogue models, discussing the implementation methods of streaming in each system and highlighting their similarities and differences.

### 5.1.1·Streaming End-to-End Spoken Dialogue Models

End-to-end streaming spoken dialogue models often leverage the knowledge of pre-trained text language models alongside an audio tokenizer, employing an tokenizer-detokenizer architecture to process and output audio signals.
Based on the concepts of streaming input and output discussed above, end-to-end models also require specific design considerations to enable streaming capabilities.
These designs center around the model’s input and output handling and can be distilled into three core techniques: causal convolution, causal attention mechanisms, and queue management.

#### Causal Convolution

Causal Convolution~\cite{bai2018empirical} is a specialized form of convolution widely used in time-series processing, especially suitable for streaming speech models.
The key feature of causal convolution is that the current output depends only on the current and past inputs, without being influenced by future inputs, thereby strictly respecting temporal order.
Unlike regular convolution, causal convolution achieves this by "shifting" the convolution kernel to avoid accessing future information.
In a one-dimensional time series, if the convolution kernel size is \(k\), a standard convolution would use data from \((t - k/2)\) to \((t + k/2)\) at the current time step \(t\).
Causal convolution, however, pads the input on the left with \(k-1\) zeros so that the kernel only uses data from \(t - k + 1\) to \(t\), aligning the kernel to only consider current and past inputs.
This padding ensures that each layer's output depends solely on current and prior information, maintaining causality.
To further expand the model’s receptive field while preserving causality, \textbf{dilated causal convolution} can be used.
This technique introduces gaps within the kernel by inserting zeros between weights, effectively expanding the convolution’s range.
This allows the model to capture longer dependencies in the data without increasing latency, which is particularly useful for streaming applications.
In streaming spoken dialogue models, causal convolution plays a critical role in:
- **Ensuring real-time processing**.
Causal convolution allows the model to compute outputs without accessing future frames, enabling real-time processing by generating outputs as input is received, which is essential for streaming.
- **Reducing latency**.
By not requiring future input data, causal convolution significantly lowers the latency in speech models, making it more suitable for real-time interaction applications, such as voice assistants and live translation.

#### Causal Attention.

Causal Attention is a specialized form of the attention mechanism designed to ensure that each position in a sequence can only attend to previous positions, thus preserving the temporal order crucial for streaming models.
This approach ensures that the model’s current output depends only on past and present information, preventing any “leakage” of future information, which is essential for real-time processing tasks.
In causal attention, the attention mask is typically used to achieve causality.
By applying a mask that blocks connections to future time steps, the model restricts each token’s receptive field to only the tokens before it.
Specifically, a lower triangular mask is applied to the attention matrix, setting values to negative infinity for positions corresponding to future tokens.
This masking technique ensures that the model’s predictions for each time step only consider current and past inputs, thereby adhering to a strict causal structure.
In streaming speech models, causal attention plays a significant role in enabling real-time interaction.
Unlike standard attention, which requires access to the entire sequence, causal attention can operate incrementally.
As new inputs are processed, the model can generate outputs without waiting for future context.

#### Queue Management~\cite{wu2023audiodec}.

Audio streams are typically split into frames, then processed in sequence via a queue management system that ensures real-time, orderly processing.


Some end-to-end models, such as Llama-Omni\cite{fang2024llama}, Mini-Omni\cite{xie2024mini} and Mini-Omni2\cite{xie2024miniomni2opensourcegpt4ovision}, employ non-streaming ASR model Whisper as an audio encoder components.
These models have made improvements on the output side to reduce latency.
- **Mini-Omni**.
Mini-Omni use a generation strategy delayed parallel decoding is a that layer-by-layer delays during audio token generation.
This allows the model to generate text and multiple audio tokens simultaneously at each step, accelerating streaming audio generation and ensuring low-latency real-time output.
- **Llama-Omni**.
Llama-Omni incorporates a non-autoregressive streaming speech decoder that leverages connectionist temporal classification (CTC) to directly generate a sequence of discrete audio tokens as the response.
- **Intrinsicvoice**.
~\cite{zhang2024intrinsicvoice}
Intrinsicvoice introduced GroupFormer module  to group speech tokens, reducing the length of speech sequences to match that of text sequences.
This approach accelerates inference, alleviates the challenges of long-sequence modeling, and effectively narrows the gap between speech and text modalities.We think they cannot be considered fully streaming because they are not designed to be streaming on the input side.
- **Moshi**~\cite{defossez2024moshi}.
In contrast, Moshi references the architecture of SpeechTokenizer to train a streaming codec from scratch, serving as the audio tokenizer-detokenizer.
The entire model, including the codec, transformer, and attention mechanism, is built on a causal structure.
- **OmniFlatten** ~\cite{zhang2024omniflatten}.
OmniFlatten proposes chunk-based processing of text and speech along with gradual learning techniques and data handling to reduce turn-taking delays, such as response delays when users finish speaking or interrupt the system.
These models have achieved true streaming capabilities and established a foundation for diverse, bidirectional interactions.

### 5.1.2·Streaming Cascaded Spoken Dialogue Models

Consistent with the above, ensuring streaming capability in a model relies on designing both input and output for streaming.
Due to its cascaded nature, a cascaded model typically relies on external streaming ASR and TTS components, placing the streaming responsibility on these ASR and TTS modules.

In~\cite{wang2024full}, comparative studies were conducted on the streaming ASR model \textbf{U2++ Conformer}~\cite{wu2021u2++}, streaming TTS model \textbf{XTTS-v2}~\cite{casanova2024xtts}, non-streaming ASR \textbf{Whisper}, and non-streaming TTS \textbf{VITS}~\cite{kong2023vits2}.
The combination of streaming components achieved the lowest latency and significantly contributed to interactive interruption capabilities.

## 5.2·Duplex Technology and Interaction

### 5.2.1·Duplex Technology

The term Duplex originates from the field of communications, used to describe interaction modes between two parties in data transmission.
Depending on the type of communication, duplex is divided into half-duplex and full-duplex.

With the development of audio processing and generation technology , the concept of duplex has been introduced to speech systems, especially within the context of speech language models.
Here, duplex doesn’t just refer to signal transmission but emphasizes the synchronization and natural interaction in human-computer dialogue.
Specifically, within model architecture, it means that the model must retain its ability to perceive external input even while generating a response---essentially, the ability to listen while speaking.

#### Simplex

In simplex communication, data flows in only one direction.
The speaker can send data, while the listener can only receive it.
As shown in Figure \ref{fig:simplex}, the robot continuously transmits audio, while the user has no ability to respond.
This fixed-direction, one-way communication has the limitation of lacking interactivity.

#### Half-Duplex

In half-duplex communication, data flows in both directions but not simultaneously.
The two parties must take turns speaking and listening.
As illustrated in Figure \ref{fig:half-duplex}, the user speaks first, followed by a response delay during which the robot "thinks" before replying.
The robot’s response occurs only after the user has finished speaking, and vice versa.
This turn-taking method is similar to using a walkie-talkie, where each party can only transmit after the other has finished, limiting efficiency.Half-duplex is a common mode in early voice interaction systems.
In a typical half-duplex interaction, there are noticeable pauses in the conversation; the user and the system cannot “speak”  simultaneously, making the conversation feel less smooth, much like communication through a walkie-talkie.
For example, voice assistants like Siri use wake words or button presses to trigger the dialogue and require the speaker to finish a complete sentence before responding.
These systems typically adopt an ASR-LM-TTS cascaded structure and are often constrained by cascade delays and the turn-based nature of text language models.
Although this interaction method is simple and easy to implement, it can feel rigid and disjointed in natural conversational settings, with notable latency.
It is designed more for command execution rather than interactive communication.

#### Full-Duplex

Full-duplex communication allows both parties to send and receive data simultaneously~\cite{ma2024language}.
Figure \ref{fig:full-duplex} shows the user and robot engaging in overlapping, real-time interaction, where backchannels and interruptions are possible.
This mode enables a natural, two-way conversation, where both the user and robot can speak, respond, and even interrupt each other as needed, much like a phone call.In dialogue systems, full-duplex means that the system and user can speak simultaneously and interrupt each other, making it closer to natural conversation in real life.
Full-duplex large voice models allow the system not only to listen and understand the user while they speak but also to interrupt at appropriate moments or respond with backchannel cues.
Moreover, the system can detect the user’s intent to interrupt and pause itself accordingly, maintaining a smooth flow in the interaction.

#### Summary

The ultimate goal of a spoken dialogue moded is to make the user feel as though they are conversing with a real human friend.
Clearly, full-duplex technology is essential for achieving natural voice dialogue systems, enabling the system to send and receive audio signals simultaneously, thus facilitating real-time interaction.
Unlike text-based models, it doesn’t “cover its ears” while speaking.
Users and intelligent agents can interrupt each other while listening or express their attitude through non-verbal signals, such as interjections or laughter.
The challenges in realizing this lie in ensuring conversational fluidity, seamless turn-taking, and precise timing of interactions.
Developing a full-duplex system that can both generate and receive voice signals in complex interactive scenarios remains a key focus in academic and industrial research.

### 5.2.2·Interaction

Now that we understand duplex technology, we can further explore duplex spoken dialogue model.

We start with some concept.Turn-taking is the core concept in duplex dialogue.
It refers to the process in which speakers take turns speaking in an orderly manner during a conversation, forming a pattern of turn-taking.
Over the past few decades and has been extensively studied across fields such as linguistics, phonetics, and sociology.
Some research ~\cite{raux2009finite,sacks1974simplest}uses a non-deterministic finite-state machine with six states to describe the turn-taking behavior between the system and the user in a spoken dialogue system (SDS).
It outlines all possible states of turn-taking within an SDS, defining the objective of turn-taking as minimizing mutual silence or overlap between interlocutors, thereby improving communication efficiency.
Turn-taking encompasses three fundamental concepts:

- Turn-taking cues ~\cite{duncan1972some,duncan1974signalling}.
These include voice, rhythm, breathing, gaze, or gestures.
Agents can use these cues to determine whether to take a turn from the user or to relinquish the turn.

- Turn-end detection or prediction
The distinction between detection~\cite{hara2019turn,lala2017attentive} and prediction~\cite{lala2019smooth,ekstedt2020turngpt} lies in that detection determines whether the agent should take a turn at the current moment, whereas prediction decides when the turn-taking should occur in the future.

- Overlap
This mainly involves two situations.
When the user and agent’s voices overlap, if the user intends to take the turn from the agent, this behavior is defined as an \textit{interruption}~\cite{khouzaimi2016reinforcement,marge2022spoken}.
If the user has no intention of taking the turn, this behavior is considered \textit{backchannel}~\cite{hara2018prediction} or a listener response, such as "uh-huh," "right."

Through these concepts, we can better understand turn-taking behavior in duplex dialogues.
In summary, our interactions with voice dialogue systems can be categorized as \textit{interruptions}, \textit{backchannels}, and \textit{normal turn exchanges}.

The earliest full-duplex systems used a simple Voice Activity Detection (VAD) component to model whether the user intended to interrupt.
However, this approach is inadequate for handling backchannel interaction forms, leading to frequent interruptions and introducing considerable delays.

We can briefly categorize the exploration of interactions into cascaded systems and end-to-end systems based on duplex technology.
Regardless of the system type, the critical core idea is that the system must continuously track external information in real-time, analyze it, and determine the model’s operational state accordingly.
An interactive voice system must meet two requirements:
1) The ability to accept external information in real-time at any moment.
2) The ability to respond to this information accurately.

This includes:

- Detecting User Interactions
When the user tries to interject or provide new information, the system can recognize this intent and immediately stop its output to allow the user to speak.
- Backchanneling During User Speech
While the user is speaking, the system can provide brief acknowledgments like "uh-huh" or "I see" to indicate active listening, which encourages the user to continue.
- Quickly Responding After User Completion
When the user finishes speaking, the system can promptly recognize this cue and respond without unnecessary delays, maintaining a smooth conversational flow.
- Handling Pauses in User Speech
When the user briefly pauses, the system can interpret this as a moment of thought rather than an invitation to respond, thus avoiding premature interruptions and preserving the natural flow.
- Interrupting the User When Necessary
In situations where the system detects critical information, it can choose to interrupt the user to provide immediate feedback.
For example, if the user is speaking but the system needs to alert them to an error, it can intervene in real-time to ensure effective communication.

#### Cascaded Systems

To enable interactive functionality, cascaded spoken dialogue models typically require explicit modeling of dialogue turns.
As the core, the large language model needs effective context and turn management.
Next, we introduce several representative works on interaction in cascaded systems.


##### Duplex Conversation

In \cite{lin2022duplex}, three core modules are proposed to achieve smooth full-duplex dialogue: user state detection, response signal selection, and interruption detection.
The user state detection module not only focuses on traditional turn-end detection but also identifies whether the user intends to switch turns, continue speaking, or hesitates during their speech.
To achieve this, the system uses a multimodal model, taking audio and text as inputs, and incorporates features such as speech rhythm, pitch, and pauses for more accurate assessment of the user’s state, determining whether to respond immediately or wait longer.
The response signal selection module inserts small backchannel cues (such as "uh-huh" or "right") at appropriate times to simulate natural human conversation.
By analyzing a large volume of real dialogues, this module extracts and trains suitable response signals for various conversation scenarios.
Using multi-label classification, the system selects the optimal response for each dialogue context, significantly reducing user waiting time and enhancing conversation flow.
The interruption detection module flexibly responds to user interruptions.
Unlike traditional rule-based detection methods, this system builds an end-to-end detection model with multimodal input (audio and text) that not only identifies genuine user interruptions but also avoids misinterpreting background noise or unintended voice signals as interruptions.

##### Outbound Agent System

\cite{jin2021duplex} proposed a full-duplex dialogue scheme for outbound systems, focusing on the issues of conversational fluidity and timing of interaction in speech dialogue.
This scheme uses semantic analysis to determine whether the user truly intends to interrupt the system and can handle disjointed expressions when users mention named entities.
The core of this system is a full-duplex interaction finite-state machine (FSM), which retrieves text snippets from ASR results every 300 milliseconds to decide whether to interrupt.
Through continuous semantic analysis of user speech, the interruption model identifies meaningful user interruptions and avoids frequent interruptions caused by brief, meaningless responses (like "uh-huh").
The model employs a pre-trained BERT-based text classifier and utilizes streaming input, ensuring that the system can process and analyze user speech in real-time as it is received.
Additionally, the system includes a Discontinuous Expression module to handle user pauses when mentioning named entities.
Specifically, when users hesitate over entities (such as numbers, locations, or company names), VAD may erroneously detect turn-end.

The advent of Large Language Models  has significantly advanced generative AI development.
Models like ChatGPT demonstrate strong capabilities in semantic understanding and logical reasoning, offering a simplified method to integrate various dialogue components into a unified framework, which may simplify SDS construction.
GPT-4o represents a milestone for dialogue systems, showcasing a nearly human-like conversational voice model.
Its flexible interaction style and interruption mechanisms make human-computer interaction more natural and fluid.
However, as a commercial model, its training data and implementation details remain proprietary, making replication challenging.

##### Full-duplex LLM

\cite{wang2024full} proposed a full-duplex spoken dialogue models based on LLMs, enabling simultaneous reception and transmission of voice signals through a perception module, an action module, and a neural finite-state machine (FSM).
The perception module uses a streaming ASR model, capturing and processing user speech in real-time with 640-millisecond intervals per time step, converting it into token inputs for the LLM.
The action module, utilizing a streaming TTS model, instantly converts the LLM-generated text into audio output and can pause or resume playback as needed, ensuring the system can generate audio while receiving user input.
At the core is the neural FSM, allowing the LLM to switch between "speaking" and "listening" states.
Controlled by FSM signals, the system can dynamically decide to continue speaking, listen, or interrupt based on the dialogue context.
Experimental results show that Wang et al.'s full-duplex streaming system reduces response latency by threefold, achieves a response time within 500 milliseconds in over 50\% of dialogues, and handles user interruptions at a rate of 96.7\%, with an interruption accuracy of 54.7\%.

##### VITA

VITA is an open-source multimodal large language model which aimed at enhancing multimodal interaction experiences.
VITA can process multiple modalities, such as video, image, text, and audio, and achieves fluid human-computer interaction through a new duplex architecture involving two simultaneously operating models: one for generating responses to user queries, and another for continuously monitoring environmental inputs.
When a new user query is detected, the generation model pauses, and the monitoring model processes the new query and generates an updated response.
This setup enables VITA to support audio interruption, allowing users to ask new questions during system generation, with the system immediately pausing the current response to handle new input.
VITA’s perception abilities are achieved through multimodal alignment and instruction fine-tuning, enabling it to switch automatically between different inputs.
Additionally, VITA employs state tokens to distinguish user input types, such as query audio, background noise, and text input, facilitating wake-free interaction.
VITA's enhanced listening module prevents unnecessary user feedback from interrupting system responses, improving robustness.

\quad$\bullet$ \emph{CleanS2S.}\cite{CleanS2S}
This model employs a structured pipeline to enable responsive and flexible interactions in a spoken dialogue setting.
Designed to facilitate seamless turn-taking and interruption handling, the model consists of several interconnected modules working in a coordinated sequence to optimize user experience.
Starting with user input, the system uses a Voice Activity Detection (VAD) module to continuously monitor for incoming audio signals.
As soon as a user starts speaking, VAD captures the input and immediately initiates processing by sending the audio data to the Automatic Speech Recognition (ASR) module.
This quick detection and response setup allows the system to react to user input without delay.
Once ASR transcribes the audio into text, the transcription is passed to the Large Language Model (LLM), which generates a relevant response based on the user’s query.
Meanwhile, the model is designed to be interruption-aware.
During response generation, if VAD detects a new user input (indicating an interruption or a follow-up query), the system can promptly adjust its processing flow.
In this case, the LLM temporarily pauses its current task, allowing ASR to transcribe the new input, which the LLM then uses to generate an updated response.
This interruption capability is achieved through the model’s layered processing design, allowing for adaptive turn-taking that feels natural and responsive.
The Text-to-Speech (TTS) module then converts the generated text response into audio, which is transmitted to the user via WebSocket.
To further support interruption handling, TTS breaks down lengthy responses into smaller audio segments that are sent progressively.
This segmentation allows the system to stop audio output instantly if an interruption occurs, switching to the new input without delay.
Each segment is prepared and sent only after a brief VAD check, ensuring that the system is ready to pause and handle new input at any time.
This interconnected processing chain—VAD detecting input, ASR transcribing, LLM generating responses, and TTS outputting segmented audio—creates a duplex interaction framework that balances response generation and user-driven interruptions.
By seamlessly coordinating these components, the model provides a fluid, real-time dialogue experience that adapts to user interactions dynamically.

#### End-to-End Systems

In contrast, end-to-end spoken dialogue models do not require explicit modeling of dialogue turns; instead, they learn interaction modeling directly from training data.
Next, we introduce several representative works on interaction in end-to-end systems.

##### dGSLM

In end-to-end systems, the introduction of the dGSLM model marks a significant milestone in full-duplex technology development.
Within the dGSLM framework, duplex technology is effectively implemented.
This model demonstrates how to capture complex interactions within dialogues directly from raw audio data through generative spoken dialogue modeling, without relying on text.
The core innovation of dGSLM is the dual-tower Transformer architecture, called the Dialogue Transformer Language Model (DLM), which uses a cross-attention mechanism to enable the system to process two parallel audio channels simultaneously.
Through this architecture, the model not only independently generates speech for each channel but also shares information between channels using cross-attention, effectively modeling silences and interaction events.
It leverages the HuBERT encoder and HiFi-GAN decoder, combined with the dual-tower DLM, and is trained on 2,000 hours of dual-channel telephone conversation audio (Fisher dataset), where each speaker in a conversation is allocated an independent audio track.
The dGSLM model transforms the audio on both channels into discrete tokens using HuBERT, and the DLM model autoregressively predicts the next audio token and its duration.
Finally, the HiFi-GAN\cite{kong2020hifi} decoder reconstructs the audio for both channels.
This approach differs significantly from traditional text-dependent spoken dialogue models, with a particular emphasis on modeling turn-taking and backchanneling capabilities.
This capability gives dGSLM a notable advantage in duplex voice interaction, better mimicking the natural dynamics of human conversation.
Through its duplex model design, dGSLM represents an essential step forward in interactive capabilities and provides a foundation for further advancements.

##### Moshi

As a novel full-duplex architecture, Moshi incorporates a rich array of design concepts.
Unlike dGSLM, Moshi does not abandon the language model’s ability in text dialogue.
Moshi’s architecture is based on the Helium language model and Mimi neural audio codec, both trained from scratch.
Helium, as a large pre-trained text language model, provides strong reasoning capabilities, while Mimi handles audio signal encoding and decoding.
To achieve real-time interaction, Moshi is designed as a multi-stream architecture, simultaneously processing "user" and "moshi" audio streams without explicitly modeling speaker turns.
Moshi also introduces the "Inner Monologue" method within the "moshi" audio stream, a process that jointly models text and audio tokens during training and inference.
This approach allows the model to fully utilize textual knowledge while maintaining speech-to-speech system characteristics, significantly enhancing generation quality.
Mimi, a neural audio codec integrating semantic and acoustic information through residual vector quantization and knowledge distillation, captures high-quality user input audio and Moshi’s output voice efficiently.
To jointly model Moshi and user audio streams alongside Moshi’s text tokens, Depth Transformer with streaming inference capabilities is employed.
The Mimi encoder and decoder combine convolutional and Transformer layers, with causal convolutions, allowing for streaming operation.
Moshi is pre-trained on unsupervised audio data to handle speech scenarios and then fine-tuned on the Fisher dataset to address overlapping speech and interruptions.
Finally, the system is further optimized on a custom instruction-tuning dataset, ensuring robust performance across various interactive scenarios.
Experimental results show that Moshi excels in speech modeling and spoken QA tasks, especially in latency, achieving a theoretical latency of 160 milliseconds and 200 milliseconds in practice, significantly lower than the typical 230 milliseconds in natural conversation, enhancing real-time interaction and conversation flow.

##### Parrot

Parrot~\cite{meng2024sd} model incorporates multiple features specifically designed to enhance interaction in spoken dialogue.
It uses a dual-channel audio setup, where each channel represents a different speaker.
This configuration allows Parrot to manage both sides of a conversation independently, facilitating real-time turn-taking.
By distinguishing between the user’s input and the system’s response on separate channels, the model can listen and respond in parallel, creating a more natural conversational flow.
To handle simultaneous speaker inputs effectively, Parrot employs a "next-token-pair prediction" mechanism, allowing it to predict tokens for both channels in a coordinated sequence.
This approach helps the model manage conversational dynamics such as overlapping speech and smooth transitions between turns, adjusting response timing based on the user’s input.
During inference, Parrot supports streaming input, enabling continuous processing of user audio on one channel while generating responses on the other.
This streaming capability allows the model to respond to live spoken input in real-time, handling turn-taking, pauses, and interruptions dynamically.
Unlike cascaded systems that rely on intermediate text conversions, Parrot processes audio directly, reducing latency and allowing immediate responses to spoken input.
These interaction-focused design choices make Parrot highly responsive, enabling it to manage turn-taking naturally, respond to interruptions, and handle overlapping speech,

##### Mini-Omni2

Mini-Omni2 is an open-source multimodal large language model aimed at simulating the multimodal capabilities of GPT-4o in vision, hearing, and text, supporting real-time full-duplex interaction.
Mini-Omni2 combines visual and audio encoders with a language model to enable simultaneous input and output of images, audio, and text.
The model incorporates an interrupt mechanism based on instruction design for more flexible user interactions.
This system uses a delayed parallel generation algorithm, allowing the model to generate text and audio responses simultaneously, greatly improving conversational real-time capabilities and response speed.
To achieve full-duplex interaction, Mini-Omni2 introduces an interrupt mechanism based on a limited instruction approach, trained on a specially constructed dataset with specific irq (interrupt) and n-irq (non-interrupt) state markers for model optimization.
For training Mini-Omni2’s interruption functionality, the researchers used noisy speech data synthesized with specific command phrases (such as "Stop Omni") in various voices and tones to simulate scenarios where users might issue interrupt commands.
The dataset also includes background noises, such as environmental sounds, music, and other dialogues, enhancing the model’s robustness in complex environments.
During training, Mini-Omni2 controls output flow through irq and n-irq state markers, generating these markers in real-time to determine whether to continue output.
In this way, the model can immediately halt generation based on user instructions and switch to "listening" mode in real-time dialogue.
The training data consists of long audio streams from which the model extracts and encodes user commands like "Stop Omni." Researchers inserted interrupt commands at various time points, marking data after the insertion point as irq (interrupt) and data before as n-irq (non-interrupt).
This labeling method ensures that the model learns to accurately identify interrupt commands in complex audio inputs and respond appropriately.

##### SyncLLM

SyncLLM achieves full-duplex dialogue and interruption capabilities through multi-stream interleaving and chunk processing.
SyncLLM divides the conversation's audio stream into fixed-sized chunks, each corresponding to a specific time interval.
The model alternates between generating user and system speech segments within each time step (chunk), ensuring real-time system responses while processing user speech input.
To maintain temporal synchronization with the user, SyncLLM predicts the user’s speech at each time step before generating each system chunk, using it as context to infer the system’s next response.
This mechanism enables the system to keep pace with the conversation even with network latency.
The chunk method allows SyncLLM to handle both user and system audio streams simultaneously, supporting complex dialogue features like speech overlap, interruption, and real-time feedback.
Additionally, by using de-duplicated speech token sequences and periodic synchronization markers, the model efficiently performs chunk-level real-time inference, making conversation more fluid and natural.

##### OmniFlatten

Similar to SyncLLM, the OmniFlatten model achieves full-duplex and interruption functionality primarily through multi-stream data processing and progressive training.
To enable full-duplex dialogue, the model adopts a multi-stream architecture that interleaves the user’s speech stream with the assistant’s speech and text streams into a single sequence for training, simplifying multimodal modeling and enhancing real-time capability.
The model first aligns the text language model with modality through multitask supervised fine-tuning, enabling it to understand and generate both speech and text, ensuring basic capability for handling speech and text simultaneously.
Through a progressive training process, OmniFlatten attains full-duplex capability in three stages: initial training for half-duplex dialogue, then removing the user’s text stream to support real-time prediction with multi-stream data, and finally removing the assistant’s text stream to enable pure speech stream generation.
These steps reduce reliance on text and decrease latency, allowing the system to generate voice responses while receiving user speech input.
By using a block-by-block generation strategy, OmniFlatten divides the input and output speech sequences into fixed-size blocks, processing each segment in turn.
This effectively implements streaming processing, ensuring low latency and high responsiveness in full-duplex dialogue, thereby providing a more natural response to user interruptions.

##### Freeze-Omni

To support duplex dialogue, Freeze-Omni~\cite{xiong2024freeze} uses a chunk-level state prediction mechanism for natural turn-taking.
When the user begins speaking, a voice activity detection module identifies the audio input, prompting the model to process the audio chunk by chunk.
After processing each chunk, the model's classification layer predicts the conversation state to determine the next action.
There are three possible states: State 0, where the model continues listening for more input, assuming the user hasn’t completed their turn; State 1, where the model interrupts to provide an immediate response if a quick acknowledgment or feedback is needed; and State 2, where the model has completed processing the current user input and is ready to generate and output a response, thus transitioning smoothly into the response phase without further listening.
This chunk-wise state prediction enables the model to decide effectively when to respond and when to continue listening, enhancing its ability to handle natural conversational cues and support interactive dialogue.

### 5.2.3·Discussions about streaming and interaction

Significant progress has been made in dialogues models, particularly in real-time interaction and semantic understanding, with notable achievements in streaming processing and full-duplex interaction.
Current systems exhibit strong technical capabilities in reducing response latency, enhancing interruption handling, and improving the naturalness of conversation.
However, existing spoken dialogues models still lack a unified system that can handle all forms of interaction seamlessly.
Future research could explore new frameworks to better manage both user interruptions and the system’s ability to interrupt users, making interactions more natural.
Additionally, standardized benchmarks for evaluating interaction capabilities remain underdeveloped.
A unified evaluation benchmark would provide a consistent method for assessing and comparing the performance of different models, thereby advancing the development of more intelligent and responsive interaction systems.
