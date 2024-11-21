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

The core of streaming speech models lies in their "real-time" and "continuous" capabilities, meaning they can process input and generate output simultaneously without waiting for complete input. This includes two main aspects:
- **Streaming Understanding**. The model can process audio input as the user speaks, without needing to wait for the user to finish entirely, allowing it to align more naturally with the flow of conversation.
- **Streaming Generation**. This concept refers to the model's ability to generate output without waiting for all intermediate hidden states. Instead, it can produce output progressively as processing occurs, which improves responsiveness and allows for smoother, more efficient interactions.

These streaming capabilities allow the model to perform more fluidly in real-time interactions, providing a seamless communication experience for users. We will explore streaming techniques in both end-to-end and cascaded spoken dialogue models, discussing the implementation methods of streaming in each system and highlighting their similarities and differences.

### 5.1.1·Streaming End-to-End Spoken Dialogue Models

End-to-end streaming spoken dialogue models often leverage the knowledge of pre-trained text language models alongside an audio tokenizer, employing an tokenizer-detokenizer architecture to process and output audio signals. Based on the concepts of streaming input and output discussed above, end-to-end models also require specific design considerations to enable streaming capabilities. These designs center around the model’s input and output handling and can be distilled into three core techniques: causal convolution, causal attention mechanisms, and queue management.

#### Causal Convolution

Causal Convolution~\cite{bai2018empirical} is a specialized form of convolution widely used in time-series processing, especially suitable for streaming speech models. The key feature of causal convolution is that the current output depends only on the current and past inputs, without being influenced by future inputs, thereby strictly respecting temporal order. Unlike regular convolution, causal convolution achieves this by "shifting" the convolution kernel to avoid accessing future information. In a one-dimensional time series, if the convolution kernel size is \(k\), a standard convolution would use data from \((t - k/2)\) to \((t + k/2)\) at the current time step \(t\). Causal convolution, however, pads the input on the left with \(k-1\) zeros so that the kernel only uses data from \(t - k + 1\) to \(t\), aligning the kernel to only consider current and past inputs. This padding ensures that each layer's output depends solely on current and prior information, maintaining causality. To further expand the model’s receptive field while preserving causality, \textbf{dilated causal convolution} can be used. This technique introduces gaps within the kernel by inserting zeros between weights, effectively expanding the convolution’s range. This allows the model to capture longer dependencies in the data without increasing latency, which is particularly useful for streaming applications. In streaming spoken dialogue models, causal convolution plays a critical role in:
- **Ensuring real-time processing**. Causal convolution allows the model to compute outputs without accessing future frames, enabling real-time processing by generating outputs as input is received, which is essential for streaming.
- **Reducing latency**. By not requiring future input data, causal convolution significantly lowers the latency in speech models, making it more suitable for real-time interaction applications, such as voice assistants and live translation.

#### Causal Attention.

Causal Attention is a specialized form of the attention mechanism designed to ensure that each position in a sequence can only attend to previous positions, thus preserving the temporal order crucial for streaming models. This approach ensures that the model’s current output depends only on past and present information, preventing any “leakage” of future information, which is essential for real-time processing tasks. In causal attention, the attention mask is typically used to achieve causality. By applying a mask that blocks connections to future time steps, the model restricts each token’s receptive field to only the tokens before it. Specifically, a lower triangular mask is applied to the attention matrix, setting values to negative infinity for positions corresponding to future tokens. This masking technique ensures that the model’s predictions for each time step only consider current and past inputs, thereby adhering to a strict causal structure. In streaming speech models, causal attention plays a significant role in enabling real-time interaction. Unlike standard attention, which requires access to the entire sequence, causal attention can operate incrementally. As new inputs are processed, the model can generate outputs without waiting for future context.

#### Queue Management~\cite{wu2023audiodec}.

Audio streams are typically split into frames, then processed in sequence via a queue management system that ensures real-time, orderly processing.


Some end-to-end models, such as Llama-Omni\cite{fang2024llama}, Mini-Omni\cite{xie2024mini} and Mini-Omni2\cite{xie2024miniomni2opensourcegpt4ovision}, employ non-streaming ASR model Whisper as an audio encoder components. These models have made improvements on the output side to reduce latency.
- **Mini-Omni**.
Mini-Omni use a generation strategy delayed parallel decoding is a that layer-by-layer delays during audio token generation. This allows the model to generate text and multiple audio tokens simultaneously at each step, accelerating streaming audio generation and ensuring low-latency real-time output.
- **Llama-Omni**.
Llama-Omni incorporates a non-autoregressive streaming speech decoder that leverages connectionist temporal classification (CTC) to directly generate a sequence of discrete audio tokens as the response.
- **Intrinsicvoice**. ~\cite{zhang2024intrinsicvoice}
Intrinsicvoice introduced GroupFormer module  to group speech tokens, reducing the length of speech sequences to match that of text sequences. This approach accelerates inference, alleviates the challenges of long-sequence modeling, and effectively narrows the gap between speech and text modalities.We think they cannot be considered fully streaming because they are not designed to be streaming on the input side.
- **Moshi**~\cite{defossez2024moshi}.
In contrast, Moshi references the architecture of SpeechTokenizer to train a streaming codec from scratch, serving as the audio tokenizer-detokenizer. The entire model, including the codec, transformer, and attention mechanism, is built on a causal structure.
- **OmniFlatten** ~\cite{zhang2024omniflatten}.
OmniFlatten proposes chunk-based processing of text and speech along with gradual learning techniques and data handling to reduce turn-taking delays, such as response delays when users finish speaking or interrupt the system. These models have achieved true streaming capabilities and established a foundation for diverse, bidirectional interactions.

### 5.1.2·Streaming Cascaded Spoken Dialogue Models

Consistent with the above, ensuring streaming capability in a model relies on designing both input and output for streaming. Due to its cascaded nature, a cascaded model typically relies on external streaming ASR and TTS components, placing the streaming responsibility on these ASR and TTS modules.

In~\cite{wang2024full}, comparative studies were conducted on the streaming ASR model \textbf{U2++ Conformer}~\cite{wu2021u2++}, streaming TTS model \textbf{XTTS-v2}~\cite{casanova2024xtts}, non-streaming ASR \textbf{Whisper}, and non-streaming TTS \textbf{VITS}~\cite{kong2023vits2}. The combination of streaming components achieved the lowest latency and significantly contributed to interactive interruption capabilities.
