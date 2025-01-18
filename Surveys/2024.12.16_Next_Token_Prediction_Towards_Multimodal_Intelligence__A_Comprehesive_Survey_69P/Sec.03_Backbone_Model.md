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

## Unified Model

As shown in Fig.~\ref{fig:two type of MMNTP models}, the Unified Model leverages a light-weight encoder and decoder to process and generate multimodal information.
The backbone model takes up most of the roles in understanding and generation tasks.
This section will introduce two main structures of the unified model.

### Quantization-based Autoregression

The quantization-based method is widely applied in building a unified model for multimodal understanding and generation due to its simplicity and similarity to the causal language modeling task.
Typically, the encoder and decoder are derived from VQVAEs, trained to reconstruct the input from a discrete representation space.
Focusing on generation, research explores generating images~\citep{DALLE,llamagen,VAR,dnd-transformer} and audio~\citep{kreuk2022audiogen, yang2023uniaudio, copet2024simple,lajszczak2024base} with higher quality in an autoregressive manner and integrating advanced techniques for optimizing LLMs.
Another line of work focuses on both understanding and generating multimodal information using quantization-based methods.
Notable examples include Unified-IO~\cite{lu2022unifiedio}, Chameleon~\cite{chameleonteam2024chameleon}, Emu-3~\cite{wang2024emu3nexttokenpredictionneed} and Moshi~\cite{defossez2024moshi}, which employ a unified NTP training objective for multimodal understanding and generation tasks.

### Autoregressive Diffusion

The quantization-based method often faces criticism regarding generation quality.
It typically produces images in a raster-scan order, which contradicts the intrinsic nature of 2D images.
Additionally, the quantization process can lead to information loss.
Several works aim to integrate the diffusion process into the NTP to enhance generation quality.
Unlike compositional methods, the diffusion model is trained from scratch alongside the entire transformer model.
Distinctive works such as Transfusion~\citep{Transfusion}, MAR~\citep{MAR}, CosyVoice~\citep{du2024cosyvoice} and Fluid~\citep{FLUID} demonstrate that diffusion models can be jointly trained with language modeling tasks, offering superior image generation quality compared to quantization-based methods.

The debate between quantization-based and diffusion-based autoregressive  methods for image generation is on-going, highlighting the need for further research.
For instance, while many diffusion-based AR methods~\citep{MAR,Transfusion} claim better generation quality compared to quantization method, Emu3~\citep{wang2024emu3nexttokenpredictionneed} significantly outperforms diffusion baselines like SDXL using a quantization-based AR approach.
DnD-Transformer~\citep{dnd-transformer} showcased that quantization-based AR generation has superior performance in generating rich-text images than diffusion models.
In summary, it is not concluded yet which modeling method has superior performance than another currently.

## Comparison Between Compositional and Unified Models

This subsection delves into a detailed comparison between compositional and unified models, evaluating their respective strengths and weaknesses in terms of general multimodal intelligence, training and deployment efficiency, and their potential to scale with increasing computational resources.

### General Multimodal Intelligence

Unified models handle multimodal understanding and reasoning within a single backbone model, whereas compositional models assign different tasks to specialized external models.
Although NTP has transformed language intelligence, its impact on multimodal intelligence remains uncertain.
Given this context, unified models are closer to a multimodal foundation model~\citep{li2023multimodalfoundationmodelsspecialists, Emu3} due to its end-to-end nature and it may hold more potential than their compositional counterparts, as they rely on a single NTP training objective, making them easier to scale compared to multi-module systems.
We will discuss the scaling behavior of MMNTP models in Section~\ref{sec:challange_scaling_law}.

### Training Efficiency

Compositional models benefit from leveraging highly specialized external encoders and decoders, often resulting in reduced training time for new tasks since these components are pretrained separately.
This modular approach allows for targeted updates, reusing existing powerful models without the need for extensive retraining of the entire system.
In contrast, unified models leave most of the understanding and generation responsibility to one backbone model, leading to sub-optimal performance given the same amount of computation~\citep{Show-o}.
This integrated training can be more resource-intensive, but it potentially facilitates a more coherent feature space across modalities within the LLM backbone, potentially enhancing overall performance on diverse multimodal tasks.

### Deployment Efficiency

The unified model, particularly when using quantization-based methods, demonstrates significantly superior deployment efficiency compared to the compositional approach.
A single unified transformer decoder backbone can effectively leverage the advanced techniques developed by the LLM community for accelerating both training and inference, such as Flash-Attention~\citep{flash-attention} and vLLM~\citep{vllm}.
This capability is frequently cited as a key advantage of unified models, as highlighted by works like~\citep{Emu3,llamagen}.