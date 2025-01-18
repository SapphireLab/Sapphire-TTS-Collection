# Backbone Model for Multimodal Next Token Prediction

After multimodal information is tokenized into sequential tokens, we need a model capable of handling multimodal information.
In the literature, two classic MMNTP model structures are depicted in Fig.~\ref{fig:two type of MMNTP models}: 1) the Compositional Model and 2) the Unified Model.
The key distinction lies in their design: the Compositional Model relies on heavily trained external encoders and decoders (such as ~\citep{radford2021clip}), and Diffusion models~\citep{ho2020denoising}, for understanding and generation tasks respectively.
In contrast, the Unified Model features lightweight encoders and decoders, with multimodal understanding and generation tasks primarily occurring within the backbone model, typically a large transformer decoder.
A categorization of current MMNTP models is shown in Table~\ref{table:mmntp_structure_summary}.
We will introduce the general structure of MMNTP model in Section~\ref{sec: general structure}, the recent advances in compostional and unified models in Sections~\ref{sec:comp model} and~\ref{sec:unified model}, and compare them in Section~\ref{sec:comparision}.

## Basic Structure of MMNTP Model

As shown in Fig.~\ref{fig:arch}, to implement multimodal understanding and generation as next token prediction, this typically involves three steps.
\textbf{Step 1.} Encode various inputs – images, text, audio, action, boxes etc., into sequences of tokens in a shared representation space.
\textbf{Step 2.} Use a multi-modal transformer to predict next token in an auto-regressive manner.
\textbf{Step 3.} Decode the predicted tokens into the space of their respective modalities.

Fig.~\ref{fig:arch} also showcases the key modules of the NTP-based multimodal model, including tokenizers (encoders) and de-tokenizers (decoders) for each modality, as well as the multimodal Transformer.
The tokenizer (encoder) and de-tokenizer (decoder) modules often appear together and are pretrained using unimodal data through techniques such as reconstruction.
They have the capability to split the original input into tokens using the tokenizer (encoder) and restore the tokens back to their original form using the de-tokenizer (decoder).
Once all the tokenizers (encoders) and de-tokenizers (decoders) for each modality are pretrained, we can activate the required tokenizer (encoder) separately for tokenization of input containing multiple modalities, enabling us to obtain a multimodal token sequence.
Finally, these multimodal token sequences are fed into the multimodal Transformer for NTP training.

For multimodal Transformers, we can use different attention masks to control the flow of information from different modalities~\citep{Show-o,VAR}.
As shown in Fig~\ref{fig:attn-mask}, a common attention mask is the causal mask, which requires each token to only depend on preceding context for generation.
However, certain tasks require generating subsequent text conditioned on a content-rich input prefix, such as generating summaries based on a rich-text-format document.
For such tasks, we can also utilize a non-causal mask, which applies a bidirectional attention to the prefix, allowing the context within the prefix to interdepend and provide better representation, while using causal attention for autoregressive generation of the content to be generated.
In summary, we can flexibly select attention masks based on the requirements of the task.

### A Unified Structure for Vision Tasks

As illustrated in Fig.~\ref{fig:image_ntp}, various tasks in the vision modality can be encapsulated within the framework of MMNTP.
Currently, a majority of large multimodal models (LMMs), such as LLaVA~\citep{liu2023llava} and the Qwen-VL~\citep{QwenVL,Qwen2vl} series, adhere to the NTP-based visual question answering paradigm.
In this approach, images and text instructions are tokenized and sent to the transformer decoder to obtain the answer tokens.
Another line of research, focusing on auto-regressive image generation, primarily adopts the NTP-based text-to-image generation paradigm, as seen in models like LlamaGen~\citep{llamagen}, VAR~\citep{VAR}, and DnD-Transformer~\citep{dnd-transformer}.
Alternatively, the output image tokens can be generated in a non-causal order, as demonstrated by works like MaskGIT~\citep{MaskGIT} and RAR~\citep{RAR}.
Additionally, these tokens can be continuous and later sent to a diffusion-based image de-tokenizer, as seen in recent developments like MAR~\citep{MAR} and Transfusion~\citep{Transfusion}.
Some research combines the above paradigms to enable LMMs to perform both visual understanding and generation, as evidenced by models such as Show-o~\citep{Show-o}, Janus~\citep{Janus}, and Emu3~\citep{Emu3}.
Specifically, the NTP paradigm also supports various image-to-image tasks, such as image editing and semantic segmentation, as distinguished by Unified-IO2 and LVM~\citep{bai2023sequential}.

### A Unified Structure for Audio Tasks

As illustrated in Fig.\ref{fig:audio_ntp}, distinct NTP-based model architectures are required for various audio processing and generation tasks.
For audio understanding \cite{hu2024wavllm, chu2023qwen, tang2023salmonn}, large-scale data pre-trained encoders demonstrate superior performance in extracting information from speech compared to discrete tokens.
Additionally, an adapter is employed to facilitate the connection between the audio and text domains.
Meanwhile, text instructions can specify specific audio processing tasks, such as automatic speech recognition, speech translation, and speech question answering.
For audio generation, audio signals are typically transformed into discrete tokens \cite{wang2023neural, kreuk2022audiogen, lajszczak2024base} or continuous tokens \cite{meng2024autoregressive}.
These tokens can subsequently be converted back into waveform format through the use of corresponding decoders or vocoders.
The text serves as either the specific speech content to be synthesized,  or a detailed description of the audio.
Leveraging the in-context learning capabilities and the scalability potential of the NTP based model, it achieves exceptional performance in zero-shot text-to-audio synthesis where a prompt audio is provided.
Recently, the exploration of full-duplex real-time spoken dialogue \cite{defossez2024moshi, gpt4o} has been progressing at a rapid pace, which requires strong audio understanding and streaming speech generation capabilities.
In Moshi~\cite{defossez2024moshi}, to address these requirements, multiple audio streams, encompassing both user inputs and model outputs, are modeled concurrently, and a novel streaming audio tokenizer is introduced.
For these tasks, the parameters of the Transformer decoder can be effectively initialized using those derived from an LLM.

## Compositional Model

As shown in Fig.~\ref{fig:two type of MMNTP models}, the Compositional Model utilizes advanced external models to serve as the encoder and decoder for processing multimodal information.
The section introduces the two components individually.

### Connecting External Encoders for Understanding

A common architectural approach to enabling multimodal information understanding ability in LLM is using a robust external encoder to encode raw multimodal data to better representations.
Pioneering work includes MiniGPT4~\citep{zhu2023minigpt4} and LLaVA~\citep{liu2023llava}, which combine a vision encoder, an alignment layer and an LLM for general-purpose visual and language understanding.
The LLaVA-style structure~\citep{liu2023llava}, which uses CLIP~\citep{radford2021clip} as the encoder and an MLP as the alignment layer, has been utilized in numerous subsequent models.
Recent studies reveal that scaling up the visual encoder~\citep{chen2023internvl,QwenVL} and allowing for more flexible input image resolutions~\citep{llava-uhd,QwenVL} can significantly improve the model's visual perception abilities.
Similar architectural approaches are employed within the audio domain to equip LLMs with the ability to perceive and process speech signals, as exemplified by models such as SALMONN~\citep{tang2023salmonn}, Qwen-Audio~\citep{chu2023qwen}, and WavLLM~\citep{hu2024wavllm}.
For a detailed discussion on encoder design, please refer to Section~\ref{sec: Tokenize Continuous Input}.

### Connecting External Decoders for Generation

To enable the LLM to generate multimodal outputs, including images, a straightforward approach is to connect it to a powerful image generation model such as a latent diffusion model~\citep{ldm}.
In this context, it is crucial to ensure that the LLM generates continuous features beyond just language tokens, aligning the output with the input space of the diffusion models.
Typical work includes Emu~\cite{sun2023emu1}, which adds a regression head on top of the LLM's output hidden state to predict the visual embedding for the diffusion model.
For a detailed discussion on decoder design, please refer to Section~\ref{sec: De-tokenize Continuous Output}.

To enable both multimodal understanding and generation abilities of LLM in compositional manner, an external encoder and decoder can be attached to the backbone model simultaneously.
A classic structure is exemplified by Emu1 and Emu2~\citep{sun2023emu1,sun2024emu}, which adopts EVA-CLIP~\citep{eva-clip} as the encoder and SDXL as the image decoder.
For the audio domain, LLaMA-Omni~\citep{fang2024llama} utilizes Whisper-large-v3~\citep{radford2023robust} as the encoder and a Transformer based decoder.
