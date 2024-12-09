# 2·TTS Pipeline: 文本转语音流程

In this section, we elaborate on the general pipeline that supports controllable TTS technologies, including acoustic models, speech vocoders, and feature representations.
Fig.\ref{fig:sec2_pipeline} depicts the general pipeline of controllable TTS, containing various model architectures and feature representations, but the control strategies will be discussed in Section~\ref{sec:ch4_controllable}.
Readers can jump to Section~\ref{sec:ch3_uncontrollable} if familiar with TTS pipelines.

## A·Overview: 总览

A TTS pipeline generally contains three key components, \ie linguistic analyzer, acoustic model, speech vocoder, and with a conditional input, \eg prompts, for controllable speech synthesis.
Besides, some end-to-end methods use a single model to encode the input and decode the speech waveforms without generating intermediate features like mel-spectrograms~\cite{wiki2024spectrogram}.
\emph{Linguistic analyzer} aims to extract linguistic features, \eg phoneme duration and position, syllable stress, and utterance level, from the input text, which is a necessary step in HHM-based methods~\cite{yoshimura1999simultaneous,tokuda2000speech} and a few neural-based methods~\cite{zen2013statistical,fan2014tts}, but is time-consuming and error-prone.
\emph{Acoustic model} is a parametric or neural model that predicts the acoustic features from the input texts.
Modern neural-based acoustic models like Tacotron~\cite{wang2017tacotron} and later works~\cite{ren2019fastspeech,ren2020fastspeech2,jeong2021difftts} directly take character~\cite{chen2015joint} or word embeddings~\cite{almeida2019word} as the input, which is much more efficient than previous methods.
\emph{Speech vocoder} is the last component that converts the intermediate acoustic features into a waveform that can be played back.
This step bridges the gap between the acoustic features and the actual sounds produced, helping to generate high-quality, natural-sounding speech~\cite{van2016wavenet,kong2020hifigan}.
Tan \etal\cite{tan2021survey} have presented a comprehensive and detailed review of acoustic models and vocoders.
Therefore, the following subsections will briefly introduce some representative acoustic models and speech vocoders, followed by a discussion of acoustic feature representations.

## B·Acoustic Models: 声学模型

Acoustic modeling is a crucial step in TTS because it ensures the generated acoustic features capture the subtleties of human speech.
By accurately modeling acoustic features, modern TTS systems can help generate high-quality and expressive audio that sounds close to human speech.

### Parametric Models: 参数模型

Early acoustic models rely on parametric approaches, where predefined rules and mathematical functions are utilized to model speech generation.
These models often utilize HMMs to capture acoustic features from linguistic input and generate acoustic features by parameterizing the vocal tract and its physiological properties such as pitch and prosody~\cite{Tokuda20001315,Zen2007294,nose2013intuitive,nishigaki2015prosody,lorenzo2015emotion,yamagishi2005acoustic}.
These methods have relatively low computational costs and can produce a range of voices by adjusting model parameters.
However, the speech quality of these methods is robotic and lacks natural intonation, and the expressiveness is also limited~\cite{nishigaki2015prosody,lorenzo2015emotion}.

### RNN-Based Models: 基于 RNN 的模型

Recurrent Neural Networks (RNNs) proved particularly effective in early neural-based TTS due to their ability to model sequential data and long-range dependencies, which helps in capturing the sequential nature of speech, such as the duration and natural flow of phonemes.
Typically, these models have an encoder-decoder architecture, where an encoder encodes input linguistic features, such as phonemes or text, into a fixed-dimensional representation, and the decoder sequentially decodes this representation into acoustic features (\eg mel-spectrogram frames) that capture the frequency and amplitude of sound over time.
Tacotron 2~\cite{shen2018tacotron2} is one of the pioneering TTS models that uses RNNs with an attention mechanism, which helps align the text sequence with the generated acoustic features.
It takes raw characters as input and produces mel-spectrogram frames, which are subsequently converted to waveforms.
Another example is MelNet~\cite{vasquez2019melnet}, which leverages autoregressive modeling to generate high-quality mel-spectrograms, demonstrating versatility in generating both speech and music, achieving high fidelity and coherence across temporal scales.

### CNN-Based Models: 基于 CNN 的模型

Unlike RNNs, which process sequential data frame by frame, CNNs process the entire sequence at once by applying filters across the input texts.
This parallel approach enables faster training and inference, making CNN-based TTS particularly appealing for real-time and low-latency applications.
Furthermore, by stacking multiple convolutional layers with varying kernel sizes or dilation rates, CNNs can capture both short-range and long-range dependencies, which are essential for natural-sounding speech synthesis.
Deep Voice~\cite{arik2017deepvoice} is one of the first prominent CNN-based TTS models by Baidu, designed to generate mel-spectrograms directly from phoneme or character input.
ParaNet~\cite{peng2020paranet} also utilizes a RNN model to achieve sequence-to-sequence mel-spectrogram generation.
It uses a non-autoregressive architecture, which enables significantly faster inference by predicting multiple time steps simultaneously.

### Transformer-Based Models: 基于 Transformer 的模型

Transformer model~\cite{vaswani2017attention} uses self-attention layers to capture relationships within the input sequence, making them well-suited for tasks requiring an understanding of global contexts, such as prosody and rhythm in TTS.
Transformer-based TTS models often employ an encoder-decoder architecture, where the encoder processes linguistic information (\eg phonemes or text) and captures contextual relationships, and the decoder generates acoustic features (like mel-spectrograms) from these encoded representations, later converted to waveforms by a vocoder.
TransformerTTS~\cite{li2019transformertts} is one of the first TTS models that apply transformers to synthesize speech from text.
It utilizes a standard encoder-decoder transformer architecture and relies on multi-head self-attention mechanisms to model long-term dependencies, which helps maintain consistency and natural flow in speech over long utterances.
FastSpeech~\cite{ren2019fastspeech} is a non-autoregressive model designed to overcome the limitations of autoregressive transformers in TTS, achieving faster synthesis than previous methods.
It introduces a length regulator to align text with output frames, enabling the control of phoneme duration.
FastSpeech 2~\cite{ren2020fastspeech2} extends FastSpeech by adding pitch, duration, and energy predictors, resulting in more expressive and natural-sounding speech.

### LLM-Based Models: 基于大语言模型的模型

LLMs~\cite{devlin2018bert,brown2020gpt3,touvron2023llama,jiang2023mistral}, known for their large-scale pre-training on text data, have shown remarkable capabilities in natural language understanding and generation.
LLM-based TTS models generally use a text description to guide the mel-spectrogram generation, where the acoustic model processes the input text to generate acoustic tokens that capture linguistic and contextual information, such as tone, sentiment, and prosody.
For example, PromptTTS~\cite{guo2023prompttts} uses a textual prompt encoded by BERT~\cite{devlin2018bert} to guide the acoustic model on the timbre, tone, emotion, and prosody desired in the speech output.
PromptTTS first generates mel-spectrograms with token embeddings and then converts them to audio using a vocoder.
InstructTTS~\cite{yang2024instructtts} generates expressive and controllable speech using natural language style prompts.
It leverages discrete latent representations of speech and integrates natural language descriptions to guide the synthesis process, which bridges the gap between TTS systems and natural language interfaces, enabling fine-grained style control through intuitive prompts.

### Other Acoustic Models: 其他声学模型

In TTS, GANs~\cite{lee2021multi,ma2018neural,guo2019new}, VAEs~\cite{hsu2018hierarchical,zhang2019learning}, and diffusion models~\cite{jeong2021difftts,popov2021gradtts} can also be used as acoustic models.
Flow-based methods~\cite{miao2020flow,kim2020glow} are also popular in waveform generation.
Refer to the survey paper from Tan \etal\cite{tan2021survey} for more details.

The choice of an acoustic model depends on the specific requirements and is a trade-off between synthesis quality, computational efficiency, and flexibility. For real-time applications, CNN-based or lightweight transformer-based models are preferable, while for high-fidelity, expressive speech synthesis, transformer-based and LLM-based models are better suited.

## C·Speech Vocoders: 语音声码器

Vocoders are essential for converting acoustic features such as mel-spectrograms into intelligible audio waveforms and are vital in determining the naturalness and quality of synthesized speech.
We broadly categorize existing vocoders according to their model architectures, \ie RNN-, CNN-, GAN-, and diffusion-based vocoders.

### RNN-Based Vocoders: 基于 RNN 的声码器

Unlike traditional vocoders~\cite{hideki2006straight,morise2016world} that depend on manually designed signal processing pipelines, RNN-based vocoders~\cite{mehri2016samplernn,kalchbrenner2018wavernn,valin2019lpcnet,yu2020durian} leverage the temporal modeling capabilities of RNNs to directly learn the complex patterns in speech signals, enabling the synthesis of natural-sounding waveforms with improved prosody and temporal coherence.
For instance, WaveRNN~\cite{kalchbrenner2018wavernn} generates speech waveforms sample-by-sample using a single-layer recurrent neural network, typically with Gated Recurrent Units (GRU). It improves upon earlier neural vocoders like WaveNet~\cite{van2016wavenet} by significantly reducing the computational requirements without sacrificing audio quality.
MB-WaveRNN~\cite{yu2020durian} extends WaveRNN by incorporating a multi-band decomposition strategy, where the speech waveform is divided into multiple sub-bands, with each sub-band synthesized at a lower sampling rate. These sub-bands are then combined to reconstruct the full-band waveform, thereby accelerating the synthesis process while preserving audio quality.

### CNN-Based Vocoders: 基于 CNN 的声码器

By leveraging the parallel nature of convolutional operations, CNN-based vocoders~\cite{van2016wavenet,oord2018parallelwavenet,jin2018fftnet} can generate high-quality speech more efficiently, making them ideal for real-time applications.
A key strength of CNN-based vocoders is their ability to balance synthesis quality and efficiency. However, they often require extensive training data and careful hyperparameter tuning to achieve optimal performance.
WaveNet~\cite{van2016wavenet} is a probabilistic autoregressive model that generates waveforms sample by sample conditioned on all preceding samples and auxiliary inputs, such as linguistic features and mel-spectrograms. It employs stacks of dilated causal convolutions, enabling long-range dependence modeling in speech signals without relying on recurrent connections.
Parallel WaveNet~\cite{oord2018parallelwavenet} addresses WaveNet's inference speed limitations while maintaining comparable synthesis quality. It introduces a non-autoregressive mechanism based on a teacher-student framework, where the original WaveNet (teacher) distills knowledge into a student model. The student generates samples in parallel, enabling real-time synthesis without waveform quality degradation.

### GAN-Based Vocoders: 基于 GAN 的声码器

GANs have been widely adopted in vocoders for high-quality speech generation~\cite{donahue2018wavegan,binkowski2019gantts,kong2020hifigan,yamamoto2020parallelwavegan,kumar2019melgan}, leveraging adversarial losses to improve realism.
GAN-based vocoders typically consist of a generator that produces waveforms conditioned on acoustic features, such as mel-spectrograms, and a discriminator that distinguishes between real and synthesized waveforms. Models like Parallel WaveGAN~\cite{yamamoto2020parallelwavegan} and HiFi-GAN~\cite{kong2020hifigan} have demonstrated the effectiveness of GANs in vocoding by introducing tailored loss functions, such as multi-scale and multi-resolution spectrogram losses, to ensure naturalness in both time and frequency domains. These models can efficiently handle the complex, non-linear relationships inherent in speech signals, resulting in high-quality synthesis.
A key advantage of GAN-based vocoders is their parallel inference capability, enabling real-time synthesis with lower computational costs compared to autoregressive models. However, training GANs can be challenging due to instability and mode collapse. Despite these challenges, GAN-based vocoders continue to advance the state-of-the-art in neural vocoding, offering a compelling combination of speed and audio quality.

### Diffusion-Based Vocoders: 基于扩散的声码器

Inspired by diffusion probabilistic models~\cite{ho2020denoising} that have shown success in visual generation tasks, diffusion-based vocoders \cite{huang2022fastdiff,jeong2021difftts,kong2020diffwave,chen2020wavegrad,lee2021priorgrad} present a novel approach to natural-sounding speech synthesis.
The core mechanism of diffusion-based vocoders involves two stages: a forward process and a reverse process. In the forward process, clean speech waveforms are progressively corrupted by adding noise in a controlled manner, creating a sequence of intermediate noisy representations. During training, the model learns to reverse this process, progressively denoising the corrupted signal to reconstruct the original waveform.
Diffusion-based vocoders, such as WaveGrad~\cite{chen2020wavegrad} and DiffWave~\cite{kong2020diffwave}, have demonstrated remarkable performance in generating high-fidelity waveforms while maintaining temporal coherence and natural prosody. They offer advantages over previous vocoders, including robustness to over-smoothing~\cite{ren2022revisiting} and the ability to model complex data distributions. However, their iterative sampling process can be computationally intensive, posing challenges for real-time applications.

### Other Vocoders: 其他声码器

There are also many other types of vocoders such as flow-based~\cite{kim2024pflow,guo2024voiceflow,lee2024periodwave,ping2020waveflow,kim2018flowavenet} and VAE-based vocoders~\cite{peng2020non,guo2023msmc,kim2021conditional}.
These methods provide unique strengths for speech synthesis such as efficiency and greater flexibility in modeling complex speech variations. Readers can refer to the survey paper from Tan \etal\cite{tan2021survey} for more details.

The choice of vocoder depends on various factors.
While high-quality models like GAN-based and diffusion-based vocoders excel in naturalness, they may not be suitable for real-time scenarios.
On the other hand, models like Parallel WaveNet~\cite{oord2018parallelwavenet} balance quality and efficiency for practical use cases. The best choice will ultimately depend on the specific use case, available resources, and the importance of factors such as model size, training data, and inference speed.

## D·Fully End-to-end TTS models: 完全端到端 TTS 模型

Fully end-to-end TTS methods~\cite{sotelo2017char2wav,ping2018clarinet,ren2020fastspeech2,kim2021conditional,donahue2020end} directly generate speech waveforms from textual input, simplifying the ``acoustic model → vocoder'' pipeline and achieving efficient speech generation.
Char2Wav~\cite{sotelo2017char2wav} is an early neural text-to-speech (TTS) system that directly synthesizes speech waveforms from character-level text input. It integrates two components and jointly trains them: a recurrent sequence-to-sequence model with attention, which predicts acoustic features (e.g., mel-spectrograms) from text, and a SampleRNN-based neural vocoder~\cite{mehri2016samplernn} that generates waveforms from these features.
Similarly, FastSpeech 2s~\cite{ren2020fastspeech2} directly synthesizes speech waveforms from texts by extending FastSpeech 2~\cite{ren2020fastspeech2} with a waveform decoder, achieving high-quality and low-latency synthesis. ~
VITS~\cite{kim2021conditional} is another fully end-to-end TTS framework. It integrates a variational autoencoder (VAE) with normalizing flows~\cite{rezende2015variational} and adversarial training, enabling the model to learn latent representations that capture the intricate variations in speech, such as prosody and style. VITS combines non-autoregressive synthesis with stochastic latent variable modeling, achieving real-time waveform generation without compromising naturalness.
There are more end-to-end TTS models such as Tacotron~\cite{wang2017tacotron}, ClariNet~\cite{ping2018clarinet}, and EATS~\cite{donahue2020end}, refer to another survey~\cite{tan2021survey} for more details.
End-to-end controllable methods that emerged in recent years will be discussed in Section~\ref{sec:ch4_controllable}.

## E·Acoustic Feature Representations: 声学特征表示

In TTS, the choice of acoustic feature representations impacts the model's flexibility, quality, expressiveness, and controllability.
This subsection investigates continuous representations and discrete tokens as shown in Fig.\ref{fig:sec2_pipeline}, along with their pros and cons for TTS applications.

### Continuous Representations: 连续表示

Continuous representations (\eg mel-spectrograms) of intermediate acoustic features use a continuous feature space to represent speech signals. These representations often involve acoustic features that capture frequency, pitch, and other characteristics without discretizing the signal.
The advantages of continuous features are: 1) Continuous representations retain fine-grained detail, enabling more expressive and natural-sounding speech synthesis. 2) Since continuous features inherently capture variations in tone, pitch, and emphasis, they are well-suited for prosody control and emotional TTS. 3) Continuous representations are more robust to information loss and can avoid quantization artifacts, allowing smoother, less distorted audio.
GAN-based~\cite{kong2020hifigan,yamamoto2020parallelwavegan,kumar2019melgan} and diffusion-based methods~\cite{kong2020diffwave,huang2022fastdiff} often utilize continuous feature representations, \ie mel-spectrograms.
However, continuous representations are typically more computationally demanding and require larger models and memory, especially in high-resolution audio synthesis.

### Discrete Tokens: 离散 Tokens

In discrete token-based TTS, the intermediate acoustic features (\eg quantized units or phoneme-like tokens) are discrete values, similar to words or phonemes in languages. These are often produced using quantization techniques or learned embeddings, such as HuBERT~\cite{hsu2021hubert} and SoundStream~\cite{zeghidour2021soundstream}.
The advantages of discrete tokens are: 1) Discrete tokens can encode phonemes or sub-word units, making them concise and less computationally demanding to handle. 2) Discrete tokens often allow TTS systems to require fewer samples to learn and generalize, as the representations are compact and simplified. 3) Using discrete tokens simplifies cross-modal TTS applications like voice cloning or translation-based TTS, as they map well to text-like representations such as LLM tokens.
LLM-based~\cite{wang2024maskgct,zhou2024voxinstruct,ji2024controlspeech,yang2024instructtts} and zero-shot TTS methods~\cite{du2024cosyvoice,wang2024maskgct,ju2024naturalspeech3} often adopt discrete tokens as their acoustic features.
However, discrete representation learning may result in information loss or lack the nuanced details that can be captured in continuous representations.

Table~\ref{tab:sec5_controllable_methods_ar} and~\ref{tab:sec5_controllable_methods_nar} summarize the types of acoustic features of representative methods. Table \ref{tab:sec2_quantization} summarizes popular open-source speech quantization methods.