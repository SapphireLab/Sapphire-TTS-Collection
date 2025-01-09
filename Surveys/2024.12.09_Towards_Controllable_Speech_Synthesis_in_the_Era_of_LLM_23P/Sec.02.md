# 2·TTS Pipeline: 文本转语音流程

<details>
<summary>展开原文</summary>

In this section, we elaborate on the general pipeline that supports controllable TTS technologies, including acoustic models, speech vocoders, and feature representations.
Fig.02 depicts the general pipeline of controllable TTS, containing various model architectures and feature representations, but the control strategies will be discussed in [Section 4](Sec.04.md).
Readers can jump to [Section 3](Sec.03.md) if familiar with TTS pipelines.

</details>
<br>

在本节中, 我们详细介绍支持可控 TTS 技术的一般流程, 包括声学模型, 语音声码器和特征表示.

![](Images/Fig.02.png)

图 02 展示了可控 TTS 的一般流程, 包含各种模型架构和特征表示, 但控制策略将在 [第 4 节](Sec.04.md) 中讨论.

如果读者熟悉 TTS 流程, 可以直接跳转到 [第 3 节](Sec.03.md).

## A·Overview: 总览

<details>
<summary>展开原文</summary>

A TTS pipeline generally contains three key components, i.e., linguistic analyzer, acoustic model, speech vocoder, and with a conditional input, e.g., prompts, for controllable speech synthesis.
Besides, some end-to-end methods use a single model to encode the input and decode the speech waveforms without generating intermediate features like mel-spectrograms ([Spectrogram [110] [URL]](https://en.wikipedia.org/wiki/Spectrogram)).
- **Linguistic analyzer** aims to extract linguistic features, e.g., phoneme duration and position, syllable stress, and utterance level, from the input text, which is a necessary step in HMM-based methods ([Yoshimura et al.(1999) [64]](../../Models/HMM/Simultaneous_Modeling_of_Spectrum_Pitch_&_Duration_in_HMM-Based_Speech_Synthesis.md); [Tokuda et al. (2000) [65]](../../Models/HMM/Speech_Parameter_Generation_Algorithms_for_HMM-Based_Speech_Synthesis.md)) and a few neural-based methods ([Statistical Parametric Speech Synthesis Using DNNs [111]](../../Models/SPSS/2013.05.26_Statistical_Parametric_Speech_Synthesis_Using_DNNs.md); [DBLSTM-RNN [112]](../../Models/SPSS/DBLSTM-RNN.md)), but is time-consuming and error-prone.
- **Acoustic model** is a parametric or neural model that predicts the acoustic features from the input texts.
Modern neural-based acoustic models like [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md) and later works ([FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md); [FastSpeech2 [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md); [Diff-TTS [113]](../../Models/Acoustic/2021.04.03_Diff-TTS.md)) directly take character ([CWE[114]](../../Models/SpeechRepresentation/CWE.md)) or word embeddings ([Survey by Almeida et al. (2019) [115]](../2019.01.25__Survey__Word_Embeddings.md)) as the input, which is much more efficient than previous methods.
- **Speech vocoder** is the last component that converts the intermediate acoustic features into a waveform that can be played back.
This step bridges the gap between the acoustic features and the actual sounds produced, helping to generate high-quality, natural-sounding speech ([WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md); [HiFi-GAN [116]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md)).
[Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) have presented a comprehensive and detailed review of acoustic models and vocoders.
Therefore, the following subsections will briefly introduce some representative acoustic models and speech vocoders, followed by a discussion of acoustic feature representations.

</details>
<br>

文本转语音流程通常包含三个关键组件, 即语言分析器, 声学模型, 语音声码器, 以及条件化输入 (如用于可控语音合成的提示).
除此之外, 一些端到端的模型使用单个模型来编码输入并解码出语音波形, 而无需生成中间特征 (如梅尔频谱图 [Spectrogram [110] [URL]](https://en.wikipedia.org/wiki/Spectrogram)).
- **语言分析器 (Linguistic Analyzer)** 的目的是从输入文本中提取语言学特征 (如音素时长和位置, 音节重音, 句子级别等), 这是基于 HMM 的方法 ([Yoshimura et al.(1999) [64]](../../Models/HMM/Simultaneous_Modeling_of_Spectrum_Pitch_&_Duration_in_HMM-Based_Speech_Synthesis.md); [Tokuda et al. (2000) [65]](../../Models/HMM/Speech_Parameter_Generation_Algorithms_for_HMM-Based_Speech_Synthesis.md)) 和一些神经网络方法 ([Statistical Parametric Speech Synthesis Using DNNs [111]](../../Models/SPSS/2013.05.26_Statistical_Parametric_Speech_Synthesis_Using_DNNs.md); [DBLSTM-RNN [112]](../../Models/SPSS/DBLSTM-RNN.md)) 的必要步骤, 但耗时且容易出错.
- **声学模型 (Acoustic Model)** 是参数化或神经模型, 从输入文本预测声学特征.
现代基于神经网络的声学模型如 [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md) 及后续工作 ([FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md); [FastSpeech2 [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md); [Diff-TTS [113]](../../Models/Acoustic/2021.04.03_Diff-TTS.md)) 直接采用字符 ([CWE[114]](../../Models/SpeechRepresentation/CWE.md)) 或词嵌入 ([Survey by Almeida et al. (2019) [115]](../2019.01.25__Survey__Word_Embeddings.md)) 作为输入, 这比以前的方法要更高效.
- **语音声码器 (Speech Vocoder)** 是将中间声学特征转换为可以播放的波形的最后一个组件.
这一步弥合声学特征和实际发出的声音之间的差距, 以生成高质量, 听感自然的语音 ([WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md); [HiFi-GAN [116]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md)).

[Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) 展示了关于声学模型和声码器的全面且详细的综述.
因此, 下面的章节将简要介绍一些代表性的声学模型和语音声码器, 并对声学特征表示进行讨论.

## B·Acoustic Models: 声学模型

Acoustic modeling is a crucial step in TTS because it ensures the generated acoustic features capture the subtleties of human speech.
By accurately modeling acoustic features, modern TTS systems can help generate high-quality and expressive audio that sounds close to human speech.

### Parametric Models: 参数模型

Early acoustic models rely on parametric approaches, where predefined rules and mathematical functions are utilized to model speech generation.
These models often utilize HMMs to capture acoustic features from linguistic input and generate acoustic features by parameterizing the vocal tract and its physiological properties such as pitch and prosody~\cite{Tokuda20001315,Zen2007294,nose2013intuitive,nishigaki2015prosody,lorenzo2015emotion,yamagishi2005acoustic}. [71] [72] [117] [118] [119] [120]
These methods have relatively low computational costs and can produce a range of voices by adjusting model parameters.
However, the speech quality of these methods is robotic and lacks natural intonation, and the expressiveness is also limited~\cite{nishigaki2015prosody,lorenzo2015emotion} [72] [120].

### RNN-Based Models: 基于 RNN 的模型

Recurrent Neural Networks (RNNs) proved particularly effective in early neural-based TTS due to their ability to model sequential data and long-range dependencies, which helps in capturing the sequential nature of speech, such as the duration and natural flow of phonemes.
Typically, these models have an encoder-decoder architecture, where an encoder encodes input linguistic features, such as phonemes or text, into a fixed-dimensional representation, and the decoder sequentially decodes this representation into acoustic features (e.g., mel-spectrogram frames) that capture the frequency and amplitude of sound over time.
[Tacotron2 [75]](../../Models/Acoustic/2017.12.16_Tacotron2.md) is one of the pioneering TTS models that uses RNNs with an attention mechanism, which helps align the text sequence with the generated acoustic features.
It takes raw characters as input and produces mel-spectrogram frames, which are subsequently converted to waveforms.
Another example is [MelNet [121]](../../Models/Acoustic/2019.06.04_MelNet.md), which leverages autoregressive modeling to generate high-quality mel-spectrograms, demonstrating versatility in generating both speech and music, achieving high fidelity and coherence across temporal scales.

### CNN-Based Models: 基于 CNN 的模型

Unlike RNNs, which process sequential data frame by frame, CNNs process the entire sequence at once by applying filters across the input texts.
This parallel approach enables faster training and inference, making CNN-based TTS particularly appealing for real-time and low-latency applications.
Furthermore, by stacking multiple convolutional layers with varying kernel sizes or dilation rates, CNNs can capture both short-range and long-range dependencies, which are essential for natural-sounding speech synthesis.
[Deep Voice [16] [122]](../../Models/TTS0_System/2017.02.25_DeepVoice.md) is one of the first prominent CNN-based TTS models by Baidu, designed to generate mel-spectrograms directly from phoneme or character input.
[ParaNet [123]](../../Models/Acoustic/2019.05.21_ParaNet.md) also utilizes a RNN model to achieve sequence-to-sequence mel-spectrogram generation.
It uses a non-autoregressive architecture, which enables significantly faster inference by predicting multiple time steps simultaneously.

### Transformer-Based Models: 基于 Transformer 的模型

[Transformer model [124]](../../Models/_Transformer/2017.06.12_Transformer.md) uses self-attention layers to capture relationships within the input sequence, making them well-suited for tasks requiring an understanding of global contexts, such as prosody and rhythm in TTS.
Transformer-based TTS models often employ an encoder-decoder architecture, where the encoder processes linguistic information (e.g., phonemes or text) and captures contextual relationships, and the decoder generates acoustic features (like mel-spectrograms) from these encoded representations, later converted to waveforms by a vocoder.
[TransformerTTS [125]](../../Models/Acoustic/2018.09.19_TransformerTTS.md) is one of the first TTS models that apply transformers to synthesize speech from text.
It utilizes a standard encoder-decoder transformer architecture and relies on multi-head self-attention mechanisms to model long-term dependencies, which helps maintain consistency and natural flow in speech over long utterances.
[FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md) is a non-autoregressive model designed to overcome the limitations of autoregressive transformers in TTS, achieving faster synthesis than previous methods.
It introduces a length regulator to align text with output frames, enabling the control of phoneme duration.
[FastSpeech2 [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md) extends FastSpeech by adding pitch, duration, and energy predictors, resulting in more expressive and natural-sounding speech.

### LLM-Based Models: 基于大语言模型的模型

LLMs ([BERT [126]](../../Models/TextLM/2018.10.11_BERT.md); [GPT-3 [97]](../../Models/TextLM/2020.05.28_GPT-3.md); [LLaMA [11]](../../Models/TextLM/2023.02.27_LLaMA.md); [Mistral [26]](../../Models/TextLM/2023.10.10_Mistral-7B.md)), known for their large-scale pre-training on text data, have shown remarkable capabilities in natural language understanding and generation.
LLM-based TTS models generally use a text description to guide the mel-spectrogram generation, where the acoustic model processes the input text to generate acoustic tokens that capture linguistic and contextual information, such as tone, sentiment, and prosody.
For example, PromptTTS~\cite{guo2023prompttts} [101] uses a textual prompt encoded by [BERT [126]](../../Models/TextLM/2018.10.11_BERT.md) to guide the acoustic model on the timbre, tone, emotion, and prosody desired in the speech output.
PromptTTS first generates mel-spectrograms with token embeddings and then converts them to audio using a vocoder.
InstructTTS [105]~\cite{yang2024instructtts} generates expressive and controllable speech using natural language style prompts.
It leverages discrete latent representations of speech and integrates natural language descriptions to guide the synthesis process, which bridges the gap between TTS systems and natural language interfaces, enabling fine-grained style control through intuitive prompts.

### Other Acoustic Models: 其他声学模型

In TTS, GANs [127] [128] [129] ~\cite{lee2021multi,ma2018neural,guo2019new}, VAEs~\cite{hsu2018hierarchical,[Zhang et al. [18]](../../Models/_Full/2018.12.11_Learning_Latent_Representations_for_Style_Control_and_Transfer_in_End-to-End_Speech_Synthesis.md)} [130], and diffusion models ([Diff-TTS [113]](../../Models/Acoustic/2021.04.03_Diff-TTS.md); [131])~\cite{popov2021gradtts} can also be used as acoustic models.
Flow-based methods~\cite{miao2020flow,kim2020glow} [132] [133] are also popular in waveform generation.
Refer to the survey paper from [Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) for more details.

The choice of an acoustic model depends on the specific requirements and is a trade-off between synthesis quality, computational efficiency, and flexibility.
For real-time applications, CNN-based or lightweight transformer-based models are preferable, while for high-fidelity, expressive speech synthesis, transformer-based and LLM-based models are better suited.

## C·Speech Vocoders: 语音声码器

Vocoders are essential for converting acoustic features such as mel-spectrograms into intelligible audio waveforms and are vital in determining the naturalness and quality of synthesized speech.
We broadly categorize existing vocoders according to their model architectures, i.e., RNN-, CNN-, GAN-, and diffusion-based vocoders.

### RNN-Based Vocoders: 基于 RNN 的声码器

Unlike traditional vocoders ([134]; [WORLD [135]](../../Models/Vocoder/2015.11.11_WORLD.md)) ~\cite{hideki2006straight} that depend on manually designed signal processing pipelines, RNN-based vocoders ([SampleRNN [136]](../../Models/Vocoder/2016.12.22_SampleRNN.md); [WaveRNN [137]](../../Models/Vocoder/2018.02.23_WaveRNN.md); [LPCNet [138]](../../Models/Vocoder/2018.10.28_LPCNet.md); [Multi-Band WaveRNN [139]](../../Models/Vocoder/2019.09.04_Multi-Band_WaveRNN.md)) leverage the temporal modeling capabilities of RNNs to directly learn the complex patterns in speech signals, enabling the synthesis of natural-sounding waveforms with improved prosody and temporal coherence.
For instance, [WaveRNN [137]](../../Models/Vocoder/2018.02.23_WaveRNN.md) generates speech waveforms sample-by-sample using a single-layer recurrent neural network, typically with Gated Recurrent Units (GRU).
It improves upon earlier neural vocoders like [WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md) by significantly reducing the computational requirements without sacrificing audio quality.
[MB-WaveRNN [139]](../../Models/Vocoder/2019.09.04_Multi-Band_WaveRNN.md) extends WaveRNN by incorporating a multi-band decomposition strategy, where the speech waveform is divided into multiple sub-bands, with each sub-band synthesized at a lower sampling rate.
These sub-bands are then combined to reconstruct the full-band waveform, thereby accelerating the synthesis process while preserving audio quality.

### CNN-Based Vocoders: 基于 CNN 的声码器

By leveraging the parallel nature of convolutional operations, CNN-based vocoders ([WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md); [Parallel WaveNet [140]](../../Models/Vocoder/2017.11.28_Parallel_WaveNet.md); [FFTNet [141]](../../Models/Vocoder/2018.04.15_FFTNet.md)) can generate high-quality speech more efficiently, making them ideal for real-time applications.
A key strength of CNN-based vocoders is their ability to balance synthesis quality and efficiency.
However, they often require extensive training data and careful hyperparameter tuning to achieve optimal performance.
[WaveNet [73]](../../Models/Vocoder/2016.09.12_WaveNet.md) is a probabilistic autoregressive model that generates waveforms sample by sample conditioned on all preceding samples and auxiliary inputs, such as linguistic features and mel-spectrograms.
It employs stacks of dilated causal convolutions, enabling long-range dependence modeling in speech signals without relying on recurrent connections.
[Parallel WaveNet [140]](../../Models/Vocoder/2017.11.28_Parallel_WaveNet.md) addresses WaveNet's inference speed limitations while maintaining comparable synthesis quality.
It introduces a non-autoregressive mechanism based on a teacher-student framework, where the original WaveNet (teacher) distills knowledge into a student model.
The student generates samples in parallel, enabling real-time synthesis without waveform quality degradation.

### GAN-Based Vocoders: 基于 GAN 的声码器

GANs have been widely adopted in vocoders for high-quality speech generation ([WaveGAN [142]](../../Models/Vocoder/2018.02.12_WaveGAN.md); [GAN-TTS [143]](../../Models/Vocoder/2019.09.25_GAN-TTS.md); [HiFi-GAN [116]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md); [Parallel WaveGAN [144]](../../Models/Vocoder/2019.10.25_Parallel_WaveGAN.md); [MelGAN [145]](../../Models/Vocoder/2019.10.08_MelGAN.md)), leveraging adversarial losses to improve realism.
GAN-based vocoders typically consist of a generator that produces waveforms conditioned on acoustic features, such as mel-spectrograms, and a discriminator that distinguishes between real and synthesized waveforms.
Models like [Parallel WaveGAN [144]](../../Models/Vocoder/2019.10.25_Parallel_WaveGAN.md) and [HiFi-GAN [116]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md) have demonstrated the effectiveness of GANs in vocoding by introducing tailored loss functions, such as multi-scale and multi-resolution spectrogram losses, to ensure naturalness in both time and frequency domains.
These models can efficiently handle the complex, non-linear relationships inherent in speech signals, resulting in high-quality synthesis.
A key advantage of GAN-based vocoders is their parallel inference capability, enabling real-time synthesis with lower computational costs compared to autoregressive models.
However, training GANs can be challenging due to instability and mode collapse.
Despite these challenges, GAN-based vocoders continue to advance the state-of-the-art in neural vocoding, offering a compelling combination of speed and audio quality.

### Diffusion-Based Vocoders: 基于扩散的声码器

Inspired by [diffusion probabilistic models [146]](../../Models/Diffusion/2020.06.19_DDPM.md) that have shown success in visual generation tasks, diffusion-based vocoders ([FastDiff [147]](../../Models/Vocoder/2022.04.21_FastDiff.md); [DiffWave [148]](../../Models/Vocoder/2020.09.21_DiffWave.md); [WaveGrad [149]](../../Models/Vocoder/2020.09.02_WaveGrad.md); [PriorGrad [150]](../../Models/Vocoder/2021.06.11_PriorGrad.md)) present a novel approach to natural-sounding speech synthesis.
The core mechanism of diffusion-based vocoders involves two stages: a forward process and a reverse process.
In the forward process, clean speech waveforms are progressively corrupted by adding noise in a controlled manner, creating a sequence of intermediate noisy representations.
During training, the model learns to reverse this process, progressively denoising the corrupted signal to reconstruct the original waveform.
Diffusion-based vocoders, such as [WaveGrad [149]](../../Models/Vocoder/2020.09.02_WaveGrad.md) and [DiffWave [148]](../../Models/Vocoder/2020.09.21_DiffWave.md), have demonstrated remarkable performance in generating high-fidelity waveforms while maintaining temporal coherence and natural prosody.
They offer advantages over previous vocoders, including robustness to over-smoothing~\cite{ren2022revisiting} and the ability to model complex data distributions.
However, their iterative sampling process can be computationally intensive, posing challenges for real-time applications.

### Other Vocoders: 其他声码器

There are also many other types of vocoders such as flow-based ([VoiceFlow](../../Models/Diffusion/2023.09.10_VoiceFlow.md); )~\cite{kim2024pflow,guo2024voiceflow,lee2024periodwave,ping2020waveflow,kim2018flowavenet} and VAE-based vocoders~\cite{peng2020non,guo2023msmc,[VITS [159]](../../Models/E2E/2021.06.11_VITS.md)}.
These methods provide unique strengths for speech synthesis such as efficiency and greater flexibility in modeling complex speech variations.
Readers can refer to the survey paper from [Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md) for more details.

The choice of vocoder depends on various factors.
While high-quality models like GAN-based and diffusion-based vocoders excel in naturalness, they may not be suitable for real-time scenarios.
On the other hand, models like [Parallel WaveNet [140]](../../Models/Vocoder/2017.11.28_Parallel_WaveNet.md) balance quality and efficiency for practical use cases.
The best choice will ultimately depend on the specific use case, available resources, and the importance of factors such as model size, training data, and inference speed.

## D·Fully End-to-end TTS models: 完全端到端 TTS 模型

Fully end-to-end TTS methods ([FastSpeech2s [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md); [VITS [159]](../../Models/E2E/2021.06.11_VITS.md); [Char2Wav [160]](../../Models/E2E/2017.02.18_Char2Wav.md); [ClariNet [161]](../../Models/E2E/2018.07.19_ClariNet.md); [EATS [162]](../../Models/E2E/2020.06.05_EATS.md)) directly generate speech waveforms from textual input, simplifying the ``acoustic model → vocoder'' pipeline and achieving efficient speech generation.
[Char2Wav [160]](../../Models/E2E/2017.02.18_Char2Wav.md) is an early neural text-to-speech (TTS) system that directly synthesizes speech waveforms from character-level text input.
It integrates two components and jointly trains them: a recurrent sequence-to-sequence model with attention, which predicts acoustic features (e.g., mel-spectrograms) from text, and a [SampleRNN-based neural vocoder [136]](../../Models/Vocoder/2016.12.22_SampleRNN.md) that generates waveforms from these features.
Similarly, [FastSpeech2s [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md) directly synthesizes speech waveforms from texts by extending [FastSpeech2 [76]](../../Models/Acoustic/2020.06.08_FastSpeech2.md) with a waveform decoder, achieving high-quality and low-latency synthesis. ~
[VITS [159]](../../Models/E2E/2021.06.11_VITS.md) is another fully end-to-end TTS framework.
It integrates a variational autoencoder (VAE) with normalizing flows~\cite{rezende2015variational} and adversarial training, enabling the model to learn latent representations that capture the intricate variations in speech, such as prosody and style.
VITS combines non-autoregressive synthesis with stochastic latent variable modeling, achieving real-time waveform generation without compromising naturalness.
There are more end-to-end TTS models such as [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md), [ClariNet [161]](../../Models/E2E/2018.07.19_ClariNet.md), and [EATS [162]](../../Models/E2E/2020.06.05_EATS.md), refer to another survey ([Survey by Tan et al. (2021) [42]](../2021.06.29_A_Survey_on_Neural_Speech_Synthesis_63P/Main.md)) for more details.
End-to-end controllable methods that emerged in recent years will be discussed in [Section 4](Sec.04.md).

## E·Acoustic Feature Representations: 声学特征表示

In TTS, the choice of acoustic feature representations impacts the model's flexibility, quality, expressiveness, and controllability.
This subsection investigates continuous representations and discrete tokens as shown in Fig.02, along with their pros and cons for TTS applications.

### Continuous Representations: 连续表示

Continuous representations (e.g., mel-spectrograms) of intermediate acoustic features use a continuous feature space to represent speech signals.
These representations often involve acoustic features that capture frequency, pitch, and other characteristics without discretizing the signal.
The advantages of continuous features are:
1) Continuous representations retain fine-grained detail, enabling more expressive and natural-sounding speech synthesis.
2) Since continuous features inherently capture variations in tone, pitch, and emphasis, they are well-suited for prosody control and emotional TTS.
3) Continuous representations are more robust to information loss and can avoid quantization artifacts, allowing smoother, less distorted audio.

GAN-based ([HiFi-GAN [116]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md); [Parallel WaveGAN [144]](../../Models/Vocoder/2019.10.25_Parallel_WaveGAN.md); [MelGAN [145]](../../Models/Vocoder/2019.10.08_MelGAN.md)) and diffusion-based methods ([FastDiff [147]](../../Models/Vocoder/2022.04.21_FastDiff.md); [DiffWave [148]](../../Models/Vocoder/2020.09.21_DiffWave.md)) often utilize continuous feature representations, i.e., mel-spectrograms.
However, continuous representations are typically more computationally demanding and require larger models and memory, especially in high-resolution audio synthesis.

### Discrete Tokens: 离散 Tokens

In discrete token-based TTS, the intermediate acoustic features (e.g., quantized units or phoneme-like tokens) are discrete values, similar to words or phonemes in languages.
These are often produced using quantization techniques or learned embeddings, such as HuBERT~\cite{hsu2021hubert} and SoundStream~\cite{zeghidour2021soundstream}.
The advantages of discrete tokens are:
1) Discrete tokens can encode phonemes or sub-word units, making them concise and less computationally demanding to handle.
2) Discrete tokens often allow TTS systems to require fewer samples to learn and generalize, as the representations are compact and simplified.
3) Using discrete tokens simplifies cross-modal TTS applications like voice cloning or translation-based TTS, as they map well to text-like representations such as LLM tokens.

LLM-based~\cite{wang2024maskgct,zhou2024voxinstruct,ji2024controlspeech,yang2024instructtts} and zero-shot TTS methods~\cite{[CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md); [MaskGCT]wang2024maskgct,ju2024naturalspeech3} often adopt discrete tokens as their acoustic features.
However, discrete representation learning may result in information loss or lack the nuanced details that can be captured in continuous representations.

Table~\ref{tab:sec5_controllable_methods_ar} and~\ref{tab:sec5_controllable_methods_nar} summarize the types of acoustic features of representative methods.
Table \ref{tab:sec2_quantization} summarizes popular open-source speech quantization methods.