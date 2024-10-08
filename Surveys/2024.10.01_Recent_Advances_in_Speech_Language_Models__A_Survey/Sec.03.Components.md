# 3.Components in SpeechLM: 语音语言模型的组件

There are three main components within a SpeechLM, namely speech tokenizer, language model, and token-to-speech synthesizer (vocoder), as illustrated in Figure \ref{fig:SLM architecture}.
The fundamental reason for such a three-staged design pattern is to use the language modeling architecture (e.g., decoder-only transformer) to model speech autoregressively in the format of audio waveforms.
Since both the input and output of a language model are discrete tokens, additional modules need to be attached to the language model to handle the I/O format.
Specifically, the speech tokenizer first transforms continuous audio waveforms into discrete tokens to serve as input to the language model, then the language model performs the next-token prediction based on the input speech tokens.
Finally, the vocoder transforms the discrete tokens outputted by the language model back into audio waveforms.
We note that our focus here is on how the three components are grouped together to form a SpeechLM rather than a comprehensive overview of each component.
Therefore, for speech tokenizer and vocoder, we mainly summarize the methods used in existing SpeechLMs.
Table \ref{tab:components_summary} summarizes the popular choices of the three components in various SpeechLM papers.

## 3.1.Speech Tokenizer: 语音分词器

Speech tokenizer is the first component in SpeechLMs, which encodes continuous audio signals (waveforms) into latent representations and then converts the latent representations into discrete tokens (or sometimes called speech units).
This conversion allows the audio input to be effectively processed by a language model for tasks such as speech recognition or synthesis.
Speech tokenizer aims to capture essential features of the audio while reducing its dimensionality, facilitating the subsequent modeling and analysis of speech patterns.
In this section, we categorize speech tokenizers based on their focus on modeling different aspects of the raw audio.

### Semantic Understanding Objective: 语义理解目标

Speech tokenizers designed with a semantic understanding objective aim to convert speech waveforms into tokens that accurately capture the content and meaning of the speech.
These tokenizers focus on extracting semantic features from the waveforms, which enhances tasks like ASR.

A semantic understanding speech tokenizer typically comprises a speech encoder and a quantizer, where the speech encoder encodes the essential information from the waveform and the quantizer discretizes continuous representations into discrete tokens.
Let $f_E(\cdot)$ denote the speech encoder parameterized by $\theta_{f_E}$, we have $\textbf{v} = f_E(\textbf{a}; \theta_{f_E})$, where $\textbf{v} = (v_1, v_2, \ldots, v_P)$ represents the encoded representations.
Since $\textbf{v}$ is still continuous, a quantizer $d(\cdot)$ is utilized to discretize the representation.
Depending on different design choices, the discrete speech tokens \(\textbf{s} = (s_1, s_2, \ldots, s_P)\) can either be derived from $\textbf{a}$ or $\textbf{v}$.
Therefore, we have $\textbf{s} = d(\textbf{v}; \theta_d)$ or $\textbf{s} = d(\textbf{a}; \theta_d)$.
After that, $\textbf{s}$ can be used to train the speech tokenizer as a target label (such as masking $\textbf{a}_\text{mask} \subset \textbf{a}$ and reconstructing its corresponding label $\textbf{s}_\text{mask} \subset \textbf{s}$ [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md)) or to train the following language model.

The key design choices lie in how to effectively encode and quantize the speech into discrete tokens.
[Wav2Vec 2.0 (2020)](../../Models/Speech_Representaion/2020.06.20_Wav2Vec2.0.md) uses a convolutional encoder followed by a [product quantization module](../../Modules/VQ/PQ.md) to discretize the continuous waveform.
Then, a portion of the quantized representations is masked and modeled using a contrastive loss.
[W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) is built upon wav2vec 2.0 and proposes to use Masked Language Modeling (MLM) loss ([BERT (2018)](../../Models/LLM/2018.10.11_BERT.md)) in addition to contrastive loss.
Similarly, [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) uses the k-means algorithm to cluster the speech utterances into a certain number of hidden units, and then perform MLM to predict the target hidden units from the masked speech utterances.
To better align the representation of text and speech modalities, [Google USM (2023)](../../Models/Speech_LLM/2023.03.02_USM.md) utilizes text-injection loss ([Maestro (2022)](../../Models/Speech_Representaion/2022.04.07_Maestro.md)) at the second pre-training stage to improve the performance and robustness of the downstream tasks.
[WavLM (2021)](../../Models/Speech_Representaion/2021.10.26_WavLM.md) adds the speech denoising objective during pre-training.
While the majority of speech tokenizer studies focus on semantic-related tasks such as ASR and TTS, WavLM shows that speech denoising can boost the performance of non-semantic tasks such as speaker verification and speech separation.
A full list of downstream tasks is listed in [section 5](Sec.05.Applications.md).

### Acoustic Generation Objective: 声学生成目标

Speech tokenizers with an acoustic generation objective focus on capturing the acoustic features necessary for generating high-quality speech waveforms.
These tokenizers prioritize the preservation of essential acoustic characteristics over semantic content, making them suitable for speech synthesis tasks.

To generate high-quality speech waveforms, acoustic generation speech tokenizers employ a speech synthesis or speech reconstruction objective.
To achieve this, the architecture typically includes an encoder, a quantizer, and a decoder.
Same as before, the encoder $f_E(\cdot)$ and quantizer $d(\cdot)$ transform the original waveform into discrete tokens.
After that, the decoder $f_D(\cdot)$ reconstructs these tokens back into speech waveforms.
This process is represented by $\hat{\textbf{a}} = f_D(\textbf{s}; \theta_{f_E})$, where $\hat{\textbf{a}}$ is the generated or reconstructed waveform.

Neural audio codecs are very suitable for and are primarily employed as acoustic generation speech tokenizers.
These codecs utilize the advanced modeling capabilities of deep neural networks to compress audio signals into a compact representation, typically in the form of quantized tokens.
For example, both [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md) and [EnCodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) use convolution blocks as the encoder and use Residual Vector Quantization (RVQ) ([SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)) as the quantizer.
This mechanism allows for codecs to efficiently transmit or store audio with minimal loss in quality.
Since the output of codecs is in discrete format, they can also be leveraged by SpeechLM to autoregressively generate speech.

### Mixed Objective: 混合目标

Speech tokenizers with a mixed objective aim to balance both semantic understanding and acoustic generation.
Currently, the development of these tokenizers is in its early stages.
Most existing mixed speech tokenizers primarily adopt the architecture of acoustic generation speech tokenizers and focus on distilling information from semantic tokenizers into the acoustic tokenizer.
[SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md) utilizes the RVQ-GAN ([EnCodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md); [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md)) architecture, distilling semantic information from [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) to the first layer of RVQ.
Building on SpeechTokenizer, Mimi ([Moshi (2024)](../../Models/Speech_LLM/2024.09.17_Moshi.md)) employs a single vector quantizer (VQ) to extract information from [WavLM (2021)](../../Models/Speech_Representaion/2021.10.26_WavLM.md) and incorporates another RVQ module to learn the acoustic information.

## 3.2.Language Model: 语言模型

Due to the success of TextLMs ([GPT-4 (2023)](../../Models/LLM/2023.03.15_GPT-4.md); [Gemini (2023)](../../Models/LLM/2023.12.19_Gemini.md); [LLaMA 3 (2024)](../../Models/LLM/2024.07.31_LLaMA3.md)), most SpeechLMs follow their architectures.
They primarily employ [Transformers (2017)](../../Models/_Transformer/2017.06.12_Transformer.md) or decoder-only architectures (such as [OPT (2022)](../../Models/LLM/2022.05.02_OPT.md), [LLaMA (2023)](../../Models/LLM/2023.02.27_LLaMA.md)) to generate speech in an autoregressive manner.
To formally define it, given $|V_t|$ as the vocabulary size and $h$ as the hidden dimension, a typical text-based decoder-only transformer language model consists of an embedding matrix $E_t \in \mathbb{R}^{|V_t| \times h}$, a sequence of \( L \) transformer decoder blocks \(\textbf{De} = \{ De_1, De_2, \ldots, De_L \} \), and an output embedding matrix $E'_t \in \mathbb{R}^{h \times |V_t|}$.
Therefore, the language model (LM) can be represented as

$$
    \textbf{t}^{\text{out}} \sim \text{LM}(\textbf{t}^{\text{in}}, (E_t, \textbf{De}, E'_t)).
$$

To adapt the language model to generate speech, the original text tokenizer is changed to the speech tokenizers illustrated in section \ref{sec:speechencoder}.
$E_t \in \mathbb{R}^{|V_t| \times h}$ is thus changed to a speech embedding matrix $E_s \in \mathbb{R}^{|V_s| \times h}$, where $|V_s|$ represents the vocabulary size of the speech tokenizer.
The output embedding matrix is also changed from $E'_t \in \mathbb{R}^{h \times |V_t|}$ to $E'_s \in \mathbb{R}^{h \times |V_s|}$.
As a result, the language model in an SpeechLM is represented as

$$
\textbf{s}^{\text{out}} \sim \text{LM}(\textbf{s}^{\text{in}}, (E_s, \textbf{De}, E'_s)).
$$

Because the language model architecture of SpeechLMs is borrowed from TextLMs, it is natural that the resulting model can jointly model both text and speech modalities ([SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md); [SpeechGPT (2023)](../../Models/Speech_LLM/2023.05.18_SpeechGPT.md)).
To achieve this, a naive and most adopted approach is to expand the vocabulary of the original TextLM to incorporate both text and speech tokens.
Specifically, the speech embedding matrix is usually appended to the end of the text embedding matrix, resulting in a larger embedding matrix $E_m \in \mathbb{R}^{(|V_t|+|V_s|) \times h}$.
Let $\textbf{m}$ be a token sequence containing both speech and text tokens, the resulting language model becomes

$$
\textbf{m}^{\text{out}} \sim \text{LM}(\textbf{m}^{\text{in}}, (E_j, \textbf{De}, E'_j)).
$$

By doing so, the model can generate both text and speech in a single sequence, enabling much more diverse applications (see \cref{sec:downstreamApps}).

## 3.3.Token-to-Speech Synthesizer (Vocoder): 音素转语音合成器

After the tokens have been autoregressively generated by the language model component, a token-to-speech module, often known as vocoder, is utilized to synthesize all the speech tokens back into speech waveforms.
This process involves converting the linguistic and paralinguistic information represented by the generated speech tokens into audio waveforms that can be heard.
This can be seen as a reverse process to the speech tokenizer and therefore can be represented as

$$
\textbf{a} = V(\textbf{s};\theta_V),
$$

where $V$ is the vocoder model parameterized by $\theta_V$.

The pipeline of the SpeechLM vocoder can vary depending on the underlying vocoder model.
There are two main pipelines: **Direct synthesis** and **input-enhanced synthesis**.

**Direct synthesis** is the pipeline where the vocoder directly converts speech tokens generated by the language model into audio waveforms.
For example, [Polyak et al. (2021)](../../Models/TTS3_Vocoder/2021.04.01_Speech_Resynthesis_from_Discrete_Disentangled_Self-Supervised_Representations.md) adapts the [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) architecture and takes discrete tokens as inputs.

In contrast, **input-enhanced synthesis** employs an additional module to transform the tokens into a continuous latent representation before they are fed into the vocoder ([Seed-TTS (2024)](../../Models/Speech_LLM/2024.06.04_Seed-TTS.md); [TorToise-TTS (2023)](../../Models/Diffusion/2023.05.12_TorToise-TTS.md)).
The main reason for using this pipeline is that vocoders typically require intermediate audio representations, such as mel-spectrograms ([MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md); [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md); [BigVGAN (2022)](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md)), as input.
When comparing the two pipelines, direct synthesis is generally simpler and faster than Input-Enhanced Synthesis.
However, the choice of pipeline depends on the type of tokens used as input.
Tokens from acoustic generation tokenizers contain sufficient acoustic information, making them suitable for direct synthesis.
Conversely, tokens from semantic understanding tokenizers provide rich semantic information but lack fine acoustic details, particularly in higher frequencies.
Therefore, these tokens are better enhanced into an acoustic-rich representation, such as mel-spectrograms, before synthesizing the final speech.

Vocoders can be categorized by their architectural choice.
In the following sections, we summarize vocoders that are mostly adopted in the development of SpeechLMs.

### GAN-based Vocoder

Generative Adversarial Network (GAN) is the most adopted architecture of the vocoders ([MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md); [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md);[Polyak et al. (2021)](../../Models/TTS3_Vocoder/2021.04.01_Speech_Resynthesis_from_Discrete_Disentangled_Self-Supervised_Representations.md); [Fre-GAN (2021)](../../Models/TTS3_Vocoder/2021.06.04_Fre-GAN.md); [BigVGAN (2022)](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md)).
It is well known for its fast and high-fidelity generation in speech synthesis tasks.
The architecture of GAN includes a generator and a discriminator.
Specifically, the generator creates realistic audio waveforms from random noise or input features, while the discriminator evaluates the authenticity of the generated audio against real audio samples.

To utilize GAN to synthesize high-fidelity speech, various training objectives are designed, focusing on different aspects.
First, \textbf{GAN loss} is utilized as the fundamental objective for the operation of the generator and the discriminator.
Specifically, the typical choice GAN loss for the generator ($G$) and discriminator ($D$) is to use the least squares loss function.
The GAN loss for the generator ($\mathcal{L}_{\text{GAN}}(G; D)$) and the discriminator ($\mathcal{L}_{\text{GAN}}(D; G)$) are

$$
    \mathcal{L}_{\text{GAN}}(G; D) = \mathbb{E}_{ms} \left[ \left( D(G(ms)) - 1 \right)^2 \right]
$$

and

$$
    \mathcal{L}_{\text{GAN}}(D; G) = \mathbb{E}_{(x, ms)} \left[ \left( D(x) - 1 \right)^2 + \left( D(G(ms)) \right)^2 \right],
$$

respectively.

In these loss functions, $x$ represents the ground truth audio and $ms$ represents its mel-spectrogram.
Second, most GAN-based vocoders synthesize speech waveform from mel-spectrograms, so \textbf{mel-spectrogram loss} is proposed to align the mel-spectrogram synthesized by the generator and the mel-spectrogram transformed from the ground-truth waveform, in order to improve the fidelity of the generated speech.
Mel-spectrogram loss ($\mathcal{L}_{\text{Mel}}(G)$) works by minimizing the L1 distance between the two versions of mel-spectrograms mentioned above.
Its formula is shown below:

$$
    \mathcal{L}_{\text{Mel}}(G) = \mathbb{E}_{(x, ms)} \left[ \left\| \phi(x) - \phi(G(ms)) \right\|_1 \right],
$$
where $\phi(\cdot)$ is the function to transform a waveform into the corresponding mel-spectrogram.

Third, to further enhance the generation fidelity, \textbf{feature matching loss} ($\mathcal{L}_{FM}(G;D)$) is proposed to align the discriminator-encoded features of the ground truth sample and the generated sample with L1 distance, which has the following formula:

$$
    \mathcal{L}_{FM}(G;D) = \mathbb{E}_{(x,ms)} \left[ \sum_{i=1}^{T} \frac{1}{N_i} \left\lVert D^i(x) - D^i(G(ms)) \right\rVert_1 \right],
$$

where $D^i(\cdot)$ and $N_i$ denote the features and the number of features in the $i$-th layer of the discriminator, respectively.

For architectural choices, GAN-based vocoders focus on injecting inductive biases to generate audio waveforms.
[MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md) adds residual blocks with dilations in the generator to model the long-range correlation among the audio time steps and proposes a multi-scale architecture for the discriminator to model the different frequency ranges of the audio.
Based on the idea of the multi-scale discriminator, [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) proposes a multi-period discriminator to model the diverse periodic patterns within the audio waveforms.
To preserve high-frequency content, [Fre-GAN (2021)](../../Models/TTS3_Vocoder/2021.06.04_Fre-GAN.md) employs the Discrete Wavelet Transform (DWT) to downsample and learn spectral distributions across multiple frequency bands.
Unlike traditional approaches like Average Pooling (AP), DWT efficiently decomposes the signal into low-frequency and high-frequency sub-bands.
[BigVGAN (2022)](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md) introduces a periodic activation function called snake function along with an anti-aliased representation to reduce the high-frequency artifacts in the synthesized audio.

### GAN-based Neural Audio Codec

Given that many neural audio codecs employ a GAN architecture, they can be effectively discussed within the context of GAN-based vocoders.
Similar to its role as a tokenizer, although the primary objective of neural audio codecs is for audio compression, the encoded compact token sequences capture the essential information buried in the audio waveforms and therefore can be leveraged as a vocoder in SpeechLMs.
[EnCodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) uses a GAN architecture and proposes a novel generator including an encoder, a quantizer, and a decoder.
The compressed audio representations are outputted by the quantizer by using Residual Vector Quantization (RVQ).
[Polyak et al. (2021)](../../Models/TTS3_Vocoder/2021.04.01_Speech_Resynthesis_from_Discrete_Disentangled_Self-Supervised_Representations.md) utilizes [HiFi-GAN (2020)](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) as the vocoder backbone and proposes to disentangle the input features of a vocoder into distinct properties, which include semantic tokens, pitch tokens, and speaker embeddings.
Such a design choice enables the codec to better perform on pitch and speaker-related tasks such as voice conversion and $F_0$ manipulation.

### Other Types of Vocoder

The variety of vocoders is not restricted to the ones mentioned earlier, as those are the ones commonly employed in SpeechLMs.
This section briefly outlines other potential vocoder types that are seldom explored as a component in SpeechLMs.

#### Pure Signal Processing Vocoder.

Pure signal processing vocoders are traditional methods that rely on deterministic algorithms rather than deep learning models to synthesize speech ([Griffin-Lim (1984)](../../Models/TTS3_Vocoder/1984.04.00_Griffin-Lim.md); [WORLD (2016)](../../Models/TTS3_Vocoder/2016.07.01_WORLD.md)).
However, this kind of vocoders introduces noticeable artifacts in the synthesized audio and is thus rarely used.

#### Autoregressive Vocoder.

Autoregressive vocoders generate audio waveforms one sample at a time, with each sample conditioned on the previously generated samples ([WaveNet (2016)](../../Models/TTS3_Vocoder/2016.09.12_WaveNet.md)).
This approach allows for high-quality audio synthesis due to its sequential nature and the ability to capture intricate temporal dependencies within the audio signal.
However, the sequential generation process can be computationally expensive and time-consuming, making autoregressive models less efficient compared to parallelized methods like GAN-based vocoders.

#### Flow-based Vocoder.

Flow-based vocoder aims to establish a series of invertible transformations that map a simple distribution, such as a Gaussian, to the complex distribution of audio samples.
This mechanism allows for efficient sampling and density evaluation, enabling the model to synthesize audio in parallel rather than sequentially, which significantly enhances both speed and quality ([WaveGlow (2018)](../../Models/TTS3_Vocoder/2018.10.31_WaveGlow.md)).
Compared to GAN-based vocoders, Flow-based vocoders typically need more parameters and memory to train the model, which hinders them from being effectively utilized ([MelGAN (2019)](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md)).

#### VAE-based Vocoders.

Variational Autoencoders (VAEs) are powerful generative models that learn to encode input data into a compressed latent space while allowing for the reconstruction of the original data ([VQ-VAE (2017)](../../Modules/VQ/2017.11.02_VQ-VAE.md); [VAE-WNV-VC (2018)](../../Models/Voice_Conversion/2018.11.27_VAE-WNV-VC.md)).
However, VAE is seldom explored as the underlying architecture of vocoders.

#### Diffusion-based Vocoder.

Diffusion models have emerged in recent years as a powerful class of generative models that can be used for high-fidelity speech synthesis.
They work by gradually adding noise to the input data (e.g., audio waveforms) to create a sequence of increasingly noisy representations, then learning to reverse this process to generate new samples ([DiffWave (2020)](../../Models/TTS3_Vocoder/2020.09.21_DiffWave.md); [WaveGrad (2020)](../../Models/TTS3_Vocoder/2020.09.02_WaveGrad.md); [PriorGrad (2021)](../../Models/TTS3_Vocoder/2021.06.11_PriorGrad.md)).
For instance, [DiffWave (2020)](../../Models/TTS3_Vocoder/2020.09.21_DiffWave.md) uses Denoising Diffusion Probabilistic Models (DDPM) to synthesize audio.
Additionally, [CosyVoice (2024)](../../Models/Speech_LLM/2024.07.07_CosyVoice.md) introduces a Conditional Flow-Matching (CFM) model that serves as a vocoder in TTS systems.
