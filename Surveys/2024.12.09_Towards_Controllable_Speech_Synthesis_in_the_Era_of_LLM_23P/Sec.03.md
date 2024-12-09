# 3·Uncontrollable TTS: 非可控文本转语音

The development of Uncontrollable Text-To-Speech (UC-TTS) systems represents a significant shift from traditional, linguistics-based synthesis to modern, data-driven deep learning techniques. This shift highlights the integration of both local and global information to produce speech with human-like quality and naturalness. This survey explores UC-TTS evolution, emphasizing the role of local and global information in enhancing speech fidelity and expressiveness.

In the context of UC-TTS, "uncontrollable" refers to the absence of explicit control mechanisms for speech features such as emotion, timbre, and speaker style. Despite this lack of explicit control, the goal is to achieve natural, fluid speech while minimizing issues like mispronunciations and omissions.

## A·Early Approaches: Statistical Models: 早期方法: 统计模型

Early Text-To-Speech (TTS) systems relied on statistical models such as Hidden Markov Models (HMMs)~\cite{yoshimura1999simultaneous, tokuda2000speech} and early neural network-based parametric methods~\cite{zen2013statistical, fan2014tts}. These models operated at the frame level, using acoustic models and vocoders for text-to-speech conversion. Notable contributions from Tokuda et al.~\cite{tokuda2013speech} employed HMMs for statistical parametric synthesis, focusing on local features like phonemes, accents, and prosody to improve speech naturalness.

While robust, these statistical methods were limited by their reliance on pre-segmented data, leading to oversimplified assumptions about speech dynamics. Local linguistic features were well-modeled, but the global phonetic context was often overlooked, resulting in speech that sounded monotone and lacked emotional depth, as noted by Zen et al.~\cite{zen2009statistical}.

## B·Sequence-to-Sequence Models: 序列到序列模型

The emergence of sequence-to-sequence models represents a significant breakthrough by removing the need for explicit linguistic features, thereby enabling the capture of the nuances and idiosyncrasies of human speech. Models such as Tacotron~\cite{wang2017tacotron} and Tacotron 2 ~\cite{shen2018natural} utilize recurrent neural networks (RNNs) with attention mechanisms to effectively model the complex, nonlinear nature of speech sequences. These innovations allow for precise tuning of speech parameters, enhancing prosody and rhythm by modeling entire utterances rather than isolated phonetic units.

Building on these advancements, Deep Voice 3\cite{ping2017deep} introduces a fully convolutional sequence-to-sequence architecture that significantly accelerates training speed compared to RNN-based models. This approach achieves training times an order of magnitude faster, enabling scalability to handle large datasets. Additionally, the use of a position-augmented attention mechanism in Deep Voice 3 enhances the naturalness of synthesized speech, achieving competitive mean opinion scores, especially when paired with advanced neural vocoders like WaveNet. This development not only improves training efficiency but also enhances the scalability and naturalness of text-to-speech systems.

## C·Transformer-Based Models: 基于 Transformer 的模型

Transformer-based architectures advanced the field by enabling computational parallelization and effectively capturing long-range dependencies. Models like Transformer TTS overcame RNN challenges, such as gradient vanishing, by using efficient training paradigms~\cite{li2019neural}. Self-attention mechanisms allowed simultaneous modeling of local phonetic details and global prosodic contexts, resulting in more sophisticated and human-like speech synthesis.

Although transformers improved contextual information incorporation, challenges remained in preserving local phonetic precision. To address these, techniques such as relative position encodings and localized attention were integrated~\cite{vaswani2017attention}.

## D·Advanced Architectures: Integrating Flow and Diffusion Models: 集成 Flow 和 Diffusion 模型

Recent advancements have shifted towards integrating global information within end-to-end architectures to enhance speech naturalness and coherence. Flow-based models like Glow-TTS\cite{kim2020glow} and Flow-TTS\cite{miao2020flow} exemplify this by employing invertible transformations that maintain the balance between local precision and global coherence. These architectures enable the synthesis of high-fidelity speech by modeling complex dependencies across the entire utterance, thus improving the overall fluidity and naturalness of the generated speech.

Moreover, the introduction of diffusion models in TTS, such as WaveGrad 2\cite{chen2021wavegrad2}, highlights the shift towards models that can iteratively refine speech output. These models use score matching and diffusion processes to generate speech directly from phoneme sequences, effectively capturing both local nuances and overarching global patterns. The iterative nature of these models allows for adjustments that enhance the quality of the synthesized audio, accommodating variations in speech without explicit control over specific attributes.

The integration of adversarial training and variational autoencoders (VAEs) further exemplifies the evolution towards incorporating global information. Systems like VITS\cite{kim2021conditional} leverage these techniques to enhance expressiveness and naturalness by learning complex mappings between text and speech. This approach allows the model to manage variations in prosody and rhythm inherently derived from the textual input, aligning with the objectives of UC-TTS to produce diverse and natural speech outputs.

The evolution from HMMs to advanced architectures in UC-TTS exemplifies progress toward synthesizing speech that is both expressive and precise. The interplay of local and global information is crucial for enhancing speech quality and customizability. Future UC-TTS research aims to produce high-fidelity, customizable speech by harmonizing deep contextual insights with precise local adjustments, meeting diverse user needs and communication contexts.
