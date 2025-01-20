# 4·Controllable TTS: 可控文本转语音

<details>
<summary>展开原文</summary>

In this section, we first review recent TTS work from the perspective of model architecture, followed by a detailed discussion of control modes in controllable TTS.

</details>
<br>

在本节中, 我们首先从模型架构的角度回顾近期的 TTS 工作, 然后详细讨论可控 TTS 中的控制模式.

## A·Model Architecture: 模型架构

<details>
<summary>展开原文</summary>

Current model architectures can be broadly classified into two main categories: the first is the non-autoregressive (NAR) generative models, which are based on HMMs, neural networks, VAEs, diffusion models, flow matching, and other NAR techniques.
The second category relies on autoregressive (AR) codec language models, which typically quantize speech into discrete tokens and use decoder-only models to autoregressively generate these tokens.
We summarize the NAR-based and AR-based controllable TTS methods in Table.03 and Table.04, respectively.

</details>
<br>

现有模型架构可以大致分类为两个主要类别:
- 非自回归生成式模型: 基于 HMM, 神经网络, 变分自编码器, 扩散模型, 流匹配, 和其他.
- 自回归编解码器语言模型: 将语音量化为离散 Token, 并使用仅解码器的模型来自回归生成这些 Token.

我们在表格 03 和表格 04 中分别总结了非自回归和自回归可控 TTS 方法.

### Fully Non-Autoregressive (NAR) Architectures: 完全非自回归架构

| NAR Method | Zero-Shot TTS | Pitch | Energy | Speed | Prosody | Timbre | Emotion | Environment | Description | Acoustic Model | Vocoder | Acoustic Feature | Release Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|FastSpeech\cite{ren2019fastspeech} |  |  |  | √ | √ |  |  |  |  | Transformer | WaveGlow | MelS | 2019.05 |
|DWAPI~\cite{vekkot2020emotional} |  | √ |  |  | √ |  | √ |  |  | DNN | Straight | MelS + F0 + Intensity | 2020.04 |
|FastSpeech 2\cite{ren2020fastspeech} |  | √ | √ | √ | √ |  |  |  |  | Transformer | Parallel WaveGAN | MelS | 2020.06 |
|FastPitch\cite{lancucki2021fastpitch} |  | √ |  |  | √ |  |  |  |  | Transformer | WaveGlow | MelS | 2020.06 |
|Parallel Tacotron\cite{elias2021parallel} |  |  |  |  | √ |  |  |  |  | Transformer + CNN | WaveRNN | MelS | 2020.10 |
|StyleTagging-TTS~\cite{kim2021expressive} | √ |  |  |  |  | √ | √ |  |  | Transformer + CNN | HiFi-GAN | MelS | 2021.04 |
|SC-GlowTTS~\cite{casanova2021sc} | √ |  |  |  |  | √ |  |  |  | Transformer + Conv | HiFi-GAN | MelS | 2021.06 |
|Meta-StyleSpeech~\cite{min2021meta} | √ |  |  |  |  | √ |  |  |  | Transformer | MelGAN | MelS | 2021.06 |
|DelightfulTTS\cite{liu2021delightfultts} |  | √ |  | √ | √ |  |  |  |  | Transformer + CNN | HiFiNet | MelS | 2021.11 |
|YourTTS~\cite{casanova2022yourtts} | √ |  |  |  |  | √ |  |  |  | Transformer | HiFi-GAN | LinS | 2021.12 |
|DiffGAN-TTS\cite{liu2022diffgan} |  | √ |  | √ | √ |  |  |  |  | Diffusion + GAN | HiFi-GAN | MelS | 2022.01 |
|StyleTTS~\cite{li2022styletts} | √ |  |  |  |  | √ |  |  |  | CNN + RNN + GAN | HiFi-GAN | MelS | 2022.05 |
|GenerSpeech~\cite{huang2022generspeech} | √ |  |  |  |  | √ |  |  |  | Transformer + Flow-based | HiFi-GAN | MelS | 2022.05 |
|NaturalSpeech 2~\cite{shen2023naturalspeech2} | √ |  |  |  |  | √ |  |  |  | Diffusion | Codec Decoder | Token | 2022.05 |
|Cauliflow\cite{abbas2022expressive} |  |  |  | √ | √ |  |  |  |  | BERT + Flow | UP WaveNet | MelS | 2022.06 |
|CLONE\cite{liu2022controllable} |  | √ |  | √ | √ |  |  |  |  | Transformer + CNN | WaveNet | MelS + LinS | 2022.07 |
|PromptTTS~\cite{guo2023prompttts} |  | √ | √ | √ |  | √ | √ |  | √ | Transformer | HiFi-GAN | MelS | 2022.11 |
|Grad-StyleSpeech~\cite{kang2023grad} | √ |  |  |  |  | √ |  |  |  | Score-based Diffusion | HiFi-GAN | MelS | 2022.11 |
|PromptStyle~\cite{liu2023promptstyle} | √ | √ |  |  | √ | √ | √ |  | √ | VITS | HiFi-GAN | MelS | 2023.05 |
|StyleTTS 2~\cite{li2024styletts2} | √ |  |  |  | √ | √ | √ |  |  | Diffusion + GAN | HifiGAN / iSTFTNet | MelS | 2023.06 |
|VoiceBox~\cite{le2024voicebox} | √ |  |  |  |  | √ |  |  |  | Flow Matching Diffusion | HiFi-GAN | MelS | 2023.06 |
|MegaTTS 2~\cite{jiang2024mega} | √ |  |  |  | √ | √ |  |  |  | Diffusion + GAN | HiFi-GAN | MelS | 2023.07 |
|PromptTTS 2~\cite{leng2023prompttts2} |  | √ | √ | √ |  | √ |  |  | √ | Diffusion | Codec Decoder | Token | 2023.09 |
|VoiceLDM~\cite{lee2024voiceldm} |  | √ |  |  | √ | √ | √ | √ | √ | Diffusion | HiFi-GAN | MelS | 2023.09 |
|DuIAN-E\cite{gu2023durian} |  | √ |  | √ | √ |  |  |  |  | CNN + RNN | HiFi-GAN | MelS | 2023.09 |
|PromptTTS++~\cite{shimizu2024prompttts++} |  | √ |  | √ | √ | √ | √ |  | √ | Transformer + Diffusion | BigVGAN | MelS | 2023.09 |
|SpeechFlow~\cite{liu2023generative} | √ |  |  |  |  | √ |  |  |  | Flow Matching Diffusion | HiFi-GAN | MelS | 2023.10 |
|P-Flow~\cite{kim2024p} | √ |  |  |  |  | √ |  |  |  | Flow Matching | HiFi-GAN | MelS | 2023.10 |
|E3 TTS~\cite{gao2023e3} | √ |  |  |  |  | √ |  |  |  | Diffusion | / | Waveform→Unet | 2023.11 |
|HierSpeech++~\cite{lee2023hierspeech++} | √ |  |  |  |  | √ |  |  |  | Hierarchical Conditional VAE | BigVGAN | MelS | 2023.11 |
|Audiobox~\cite{vyas2023audiobox} | √ | √ |  | √ | √ | √ |  | √ | √ | Flow Matching | EnCodec | MelS | 2023.12 |
|FlashSpeech~\cite{ye2024flashspeech} | √ |  |  |  |  | √ |  |  |  | Latent Consistency Model | EnCodec | Token | 2024.04 |
|NaturalSpeech 3~\cite{ju2024naturalspeech3} | √ |  |  | √ | √ | √ |  |  |  | Diffusion | EnCodec | Token | 2024.04 |
|InstructTTS~\cite{yang2024instructtts} |  | √ |  | √ | √ | √ | √ |  | √ | Transformer + Diffusion | HiFi-GAN | Token | 2024.05 |
|ControlSpeech~\cite{ji2024controlspeech} | √ | √ | √ | √ | √ | √ | √ | | √ | Transformer + Diffusion | FACodec Decoder | Token | 2024.06 |
|AST-LDM~\cite{kim2024speak} |  |  |  |  |  | √ |  | √ | √ | Diffusion | HiFi-GAN | MelS | 2024.06 |
|SimpleSpeech~\cite{yang2024simplespeech} | √ |  |  |  |  | √ |  |  |  | Transformer Diffusion | SQ Decoder | Token | 2024.06 |
|DiTTo-TTS~\cite{lee2024ditto} | √ |  |  | √ |  | √ |  |  |  | DiT | BigVGAN | Token | 2024.06 |
|E2 TTS~\cite{eskimez2024e2} | √ |  |  |  |  | √ |  |  |  | Flow Matching Transformer | BigVGAN | MelS | 2024.06 |
|MobileSpeech~\cite{ji2024mobilespeech} | √ |  |  |  |  | √ |  |  |  | ConFormer Decoder | Vocos | Token | 2024.06 |
|DEX-TTS~\cite{park2024dex} | √ |  |  |  |  | √ |  |  |  | Diffusion | HiFi-GAN | MelS | 2024.06 |
|ArtSpeech~\cite{wang2024artspeech} | √ |  |  |  |  | √ |  |  |  | RNN + CNN | HiFI-GAN | MelS+Energy+F0+TV | 2024.07 |
|CCSP~\cite{xiao2024contrastive} | √ |  |  |  |  | √ |  |  |  | Diffusion | Codec Decoder | Token | 2024.07 |
|SimpleSpeech 2~\cite{yang2024simplespeech2} | √ |  |  | √ |  | √ |  |  |  | Flow-based Transformer Diffusion | SQ Decoder | Token | 2024.08 |
|E1 TTS~\cite{liu2024e1} | √ |  |  |  |  | √ |  |  |  | DiT | BigVGAN | Continuous Token | 2024.09 |
|VoiceGuider~\cite{yeom2024voiceguider} | √ |  |  |  |  | √ |  |  |  | Diffusion | BigVGAN | MelS | 2024.09 |
|StyleTTS-ZS~\cite{li2024stylettszs} | √ |  |  |  |  | √ |  |  |  | Diffusion + GAN | HifiGAN / iSTFTNet | Token | 2024.09 |
|NansyTTS~\cite{yamamoto2024description} | √ | √ |  | √ | √ | √ |  |  | √ | Transformer | NANSY++ | MelS | 2024.09 |
|NanoVoice~\cite{park2024nanovoice} | √ |  |  |  |  | √ |  |  |  | Diffusion | BigVGAN | MelS | 2024.09 |
|MS$^{2}$KU-VTTS~\cite{he2024multi} |  |  |  |  |  |  |  | √ | √ | Diffusion | BigvGAN | MelS | 2024.10 |
|MaskGCT~\cite{wang2024maskgct} | √ |  |  | √ |  | √ |  |  |  | Masked Generative Transformers | DAC + Vocos | Token | 2024.10 |

#### HMM-Based Approaches: 基于 HMM 的方法

In the realm of Controllable Text-To-Speech (CTTS), advancements in Hidden Markov Model (HMM) architectures have significantly enhanced the manipulation of speech elements such as emotion and prosody.
Yamagishi et al.~\cite{yamagishi2003modeling} pioneered this field by introducing style-dependent and style-mixed modeling, which allowed precise emulation of human-like emotional nuances and versatile synthesis across various styles by incorporating style as a contextual variable.
Building on this foundation, Qin et al.\cite{qin2006hmm} developed the "average emotion model," which utilized MLLR-based adaptation to modulate emotions like happiness and sadness even with limited data, thus advancing the emotional intelligence of synthetic speech systems.

Furthering expressive variability, [Nose et al. (2012) [119]](../../Models/_Early/An_Intuitive_Style_Control_Technique_in_HMM-Based_Expressive_Speech_Synthesis_Using_Subjective_Style_Intensity_and_Multiple-Regression_Global_Variance_Model.md) integrated subjective style intensities and a multiple-regression global variance model into HMM frameworks, addressing over-smoothing and enabling nuanced emotional expressions.
[Lorenzo-Trueba et al. (2015) [72]](../../Models/_Early/Emotion_Transplantation_through_Adaptation_in_HMM-Based_Speech_Synthesis.md) expanded on these capabilities with CSMAPLR adaptation, introducing "emotion transplantation" to transfer emotional states between speakers while preserving voice distinctiveness, enhancing personalized human-computer interaction.
These innovations in HMM architectures have broadened the expressiveness and individuality in synthetic speech, augmenting technological interfaces and paving the way for future developments in adaptive, lifelike speech solutions.

#### Transformer-Based Approaches: 基于 Transformer 的方法

Advancements in Controllable Text-to-Speech (TTS) technology highlight the integration of deep learning with audio processing, driven by Transformer-based architectures.
Ren et al. introduced [FastSpeech [15]](../../Models/Acoustic/2019.05.22_FastSpeech.md), a feed-forward non-autoregressive Transformer model that significantly enhances TTS efficiency by reducing inference time and improving the stability issues found in autoregressive models like Tacotron 2.
This model provides precise control over prosodic features through duration prediction, effectively tackling the one-to-many mapping challenge.
[FastSpeech2](../../Models/Acoustic/2020.06.08_FastSpeech2.md) builds on this by integrating pitch and energy control, eliminating the need for the complex teacher-student distillation process, thus enhancing training efficiency and improving voice quality.
[Parallel Tacotron]\cite{elias2021paralleltacotron} further advances TTS by employing a variational autoencoder-based residual encoder, capturing intricate prosodic nuances.
This approach, combined with iterative spectrogram loss, significantly enhances the naturalness and quality of synthesized speech.
Additionally, FastPitch\cite{lancucki2021fastpitch} incorporates direct pitch prediction into its architecture, enabling fully parallelized synthesis and precise pitch manipulation.
This capability enhances expressiveness and retains the efficiency benefits established by FastSpeech.
These innovations significantly contribute to the development of more interactive and natural AI-driven communication systems, underscoring the potential of integrating AI with human-centric disciplines to craft a future where technology and humanity coexist harmoniously.

#### VAE-Based Approaches: 基于 VAE 的方法

Recent advancements in Controllable Text-To-Speech (TTS) systems are largely driven by the integration of Variational Autoencoder (VAE) architectures, which enhance the flexibility and precision of speech modulation.
[Zhang et al. [18]](../../Models/Acoustic/2018.12.11_Learning_Latent_Representations_for_Style_Control_and_Transfer_in_End-to-End_Speech_Synthesis.md) pioneered the use of VAEs in end-to-end speech synthesis, creating disentangled latent representations that allow effective style control and transfer, especially in prosody and emotion management, outperforming the Global Style Token model in style transfer tasks.
Building on this, Hsu et al.\cite{hsu2018hierarchical} developed a hierarchical generative model with a conditional VAE framework and a Gaussian mixture model, enabling precise control over complex speech attributes such as environment and style, thus improving expressive speech synthesis through refined noise and speaker characteristic management.
Liu et al.\cite{liu2022controllable} further advanced the field with the CLONE model, a single-stage TTS system that resolves the one-to-many mapping issue and enhances high-frequency information reconstruction.
By employing a conditional VAE with normalizing flows and a dual path adversarial training mechanism with multi-band discriminators, CLONE achieves nuanced control over prosody and energy, demonstrating superior performance in both speech quality and prosody control compared to state-of-the-art models.
These collective innovations highlight the adaptability of VAEs in managing complex speech generation tasks, marking significant progress toward more dynamic and versatile TTS technologies, with ongoing research promising even greater advancements.

#### Diffusion-Based Approaches: 基于扩散模型的方法

The core concept of diffusion-based models is to generate target data by progressively removing noise.
During the forward diffusion phase, noise is incrementally added to the original data to form a noise distribution.
In the generation phase, a reverse denoising process is employed to gradually recover high-quality speech from the noise.
Grad-StyleSpeech~\cite{kang2023grad} introduces a hierarchical transformer encoder to create a representative noise prior distribution for speaker-adaptive settings using score-based diffusion models.
NaturalSpeech 2~\cite{shen2023naturalspeech2} uses a neural audio codec with residual vector quantizers to obtain quantized latent vectors, which are then generated using a diffusion model conditioned on text input.
[NaturalSpeech3 [87]](../../Models/Diffusion/2024.03.05_NaturalSpeech3.md) decomposes speech into distinct subspaces that represent different attributes and generates each subspace independently.
DEX-TTS~\cite{park2024dex} improves DiT-based diffusion networks by applying overlapping patchify and convolution-frequency patch embedding strategies.
E3 TTS~\cite{gao2023e3} models the temporal structure of the waveform through the diffusion process, eliminating the need for any intermediate representations such as spectrogram features or alignment information.

Applying diffusion models to TTS requires a complex pipeline due to the need for precise temporal alignment between text and speech and the high fidelity required for audio data.
This includes domain-specific modeling, such as phoneme and duration~\cite{shen2023naturalspeech2}.
To address the issue of reduced naturalness caused by the addition of duration models, DiTTo-TTS~\cite{lee2024ditto} leverages the off-the-shelf pre-trained text and speech encoders without relying on speech domain-specific modeling by incorporating cross-attention mechanisms with the prediction of the total length of speech representations.
Similarly, SimpleSpeech~\cite{yang2024simplespeech} proposes a speech codec model (SQ-Codec) based on scalar quantization and uses the sentence duration to control the generated speech length.

#### Flow-Based Approaches: 基于流模型的方法

Flow-based methods leverage invertible flow transformations to learn mappings from target speech features to simple distributions, typically standard Gaussian distributions.
Due to their invertibility, this mechanism can directly sample from the simple distribution and generate high-fidelity speech in the reverse direction.
Audiobox~\cite{vyas2023audiobox} and P-flow~\cite{kim2024p} employ non-autoregressive flow-matching models for efficient and stable speech synthesis.
VoiceBox~\cite{le2024voicebox} also employs a flow-matching to generate speech, effectively casting the TTS task into a speech infilling task.
SpeechFlow~\cite{liu2023generative} is trained on 60k hours of untranscribed speech with flow matching and mask conditions and can be fine-tuned with task-specific data to match or surpass existing expert models.
This highlights the potential of generative models as foundation models for speech applications.
HierSpeech++~\cite{lee2023hierspeech++} proposes a hierarchical variational inference method.
FlashSpeech~\cite{ye2024flashspeech} is built on a latent consistency model and applies a novel adversarial consistency training approach that can train from scratch without the need for a pre-trained diffusion model as the teacher, achieving speech generation in one or two steps.

Recently, E2 TTS~\cite{eskimez2024e2} converts text input into a character sequence with filler tokens and trains a mel spectrogram generator based on audio infilling task, achieving human-level naturalness.
Inspired by E2 TTS, F5-TTS~\cite{chen2024f5} refines the text representation with ConvNext v2~\cite{woo2023convnext}, facilitating easier alignment with speech.
E1 TTS~\cite{liu2024e1} further distills a diffusion-based TTS model into a one-step generator with distribution matching distillation~\cite{luo2024diff,yin2024improved}, reducing the number of network evaluations in sampling from diffusion models.
SimpleSpeech 2~\cite{yang2024simplespeech2} introduces a flow-based scalar transformer diffusion model.
The work also provides a theoretical analysis, showing that the inclusion of a small number of noisy labels in a large-scale dataset is equivalent to introducing classifier-free guidance during model optimization.

#### Other NAR Approaches: 其他方法

Other works leverage GAN-based or Masked Generative-based methods for TTS generation.
StyleTTS 2~\cite{li2024styletts2} employs large pre-trained speech language models (SLMs) such as Wav2Vec 2.0~\cite{baevski2020wav2vec}, [HuBERT [166]](../../Models/SpeechRepresentation/2021.06.14_HuBERT.md), and WavLM~\cite{chen2022wavlm} as discriminators, in combination with a novel differentiable duration modeling approach.
This setup uses SLM representations to enhance the naturalness of the synthesized speech.
[MaskGCT [78]](../../Models/SpeechLM/2024.09.01_MaskGCT.md) proposes masked generative transformers without requiring text-speech alignment supervision and phone-level duration.
The model employs a two-stage system, both trained using a mask-and-predict learning paradigm.

### Autoregressive (AR) Architectures: 自回归架构

| AR Method | Zero-Shot TTS | Pitch | Energy | Speed | Prosody | Timbre | Emotion | Environment | Description | Acoustic Model | Vocoder | Acoustic Feature | Release Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Prosody-Tacotron\cite{skerry2018towards} |  | √ |  |  | √ |  |  |  |  | RNN | WaveNet | MelS | 2018.03 |
|GST-Tacotron\cite{stanton2018predicting} |  | √ |  |  | √ |  |  |  |  | CNN + RNN | Griffin-Lim | LinS | 2018.03 |
|GMVAE-Tacotron\cite{hsu2018hierarchical} |  | √ |  | √ | √ |  |  | √ |  | CNN + RNN | WaveRNN | MelS | 2018.12 |
|VAE-Tacotron\cite{zhang2019learning} |  | √ |  | √ | √ |  |  |  |  | CNN + RNN | WaveNet | MelS | 2019.02 |
|DurIAN\cite{yu2019durian} |  | √ |  | √ | √ |  |  |  |  | CNN + RNN | MB-WaveRNN | MelS | 2019.09 |
|Flowtron\cite{valle2020flowtron} |  | √ |  | √ | √ |  |  |  |  | CNN + RNN | WaveGlow | MelS | 2020.07 |
|MsEmoTTS\cite{lei2022msemotts} |  | √ |  |  | √ |  | √ |  |  | CNN + RNN | WaveRNN | MelS | 2022.01 |
|VALL-E~\cite{wang2023neural} | √ |  |  |  |  | √ |  |  |  | LLM | EnCodec | Token | 2023.01 |
|SpearTTS~\cite{kharitonov2023speak} | √ |  |  |  |  | √ |  |  |  | LLM | SoundStream | Token | 2023.02 |
|VALL-E X~\cite{zhang2023speak} | √ |  |  |  |  | √ |  |  |  | LLM | EnCodec | Token | 2023.03 |
|Make-a-voice~\cite{huang2023make} | √ |  |  |  |  | √ |  |  |  | LLM | BigVGAN | Token | 2023.05 |
|TorToise~\cite{betker2023better} |  |  |  |  |  | √ |  |  |  | Transformer + DDPM | Univnet | MelS | 2023.05 |
|MegaTTS~\cite{jiang2023megavoic} | √ |  |  |  |  | √ |  |  |  | LLM + GAN | HiFi-GAN | MelS | 2023.06 |
|SC VALL-E~\cite{kim2023sc} | √ | √ |  | √ | √ | √ | √ |  |  | LLM | EnCodec | Token | 2023.07 |
|Salle~\cite{ji2024textrolspeech} |  | √ | √ | √ | √ | √ | √ |  | √ | LLM | Codec Decoder | Token | 2023.08 |
|UniAudio~\cite{yang2023uniaudio} | √ | √ |  | √ | √ | √ |  |  | √ | LLM | EnCodec | Token | 2023.10 |
|ELLA-V~\cite{song2024ella} | √ |  |  |  |  | √ |  |  |  | LLM | EnCodec | Token | 2024.01 |
|BaseTTS~\cite{lajszczak2024base} | √ |  |  |  |  | √ |  |  |  | LLM | UnivNet | Token | 2024.02 |
|ClaM-TTS~\cite{kim2024clam} | √ |  |  |  |  | √ |  |  |  | LLM | BigVGAN | MelS+Token | 2024.04 |
|RALL-E~\cite{xin2024rall} | √ |  |  |  |  | √ |  |  |  | LLM | SoundStream | Token | 2024.05 |
|ARDiT~\cite{liu2024autoregressive} | √ |  |  | √ |  | √ |  |  |  | Decoder-only Diffusion Transformer | BigVGAN | MelS | 2024.06 |
|VALL-E R~\cite{han2024vall} | √ |  |  |  |  | √ |  |  |  | LLM | Vocos | Token | 2024.06 |
|VALL-E 2~\cite{chen2024vall} | √ |  |  |  |  | √ |  |  |  | LLM | Vocos | Token | 2024.06 |
|Seed-TTS~\cite{anastassiou2024seed} | √ |  |  |  |  | √ | √ |  |  | LLM + Diffusion Transformer | / | Token | 2024.06 |
|VoiceCraft~\cite{peng2024voicecraft} | √ |  |  |  |  | √ |  |  |  | LLM | HiFi-GAN | Token | 2024.06 |
|XTTS~\cite{casanova2024xtts} | √ |  |  |  |  | √ |  |  |  | LLM + GAN | HiFi-GAN | MelS+Token | 2024.06 |
|CosyVoice~\cite{du2024cosyvoice} | √ | √ |  | √ | √ | √ | √ |  | √ | LLM + Conditional Flow Matching | HiFi-GAN | Token | 2024.07 |
|MELLE~\cite{meng2024autoregressive} | √ |  |  |  |  | √ |  |  |  | LLM | HiFi-GAN | MelS | 2024.07 |
|Bailing TTS~\cite{di2024bailing} | √ |  |  |  |  | √ |  |  |  | LLM + Diffusion Transformer | / | Token | 2024.08 |
|VoxInstruct~\cite{zhou2024voxinstruct} | √ | √ | √ | √ | √ | √ | √ |  | √ | LLM | Vocos | Token | 2024.08 |
|Emo-DPO~\cite{gao2024emo} |  |  |  |  |  |  | √ |  | √ | LLM | HiFi-GAN | Token | 2024.09 |
|FireRedTTS~\cite{guo2024fireredtts} | √ |  |  |  | √ | √ |  |  |  | LLM + Conditional Flow Matching | BigVGAN-v2 | Token | 2024.09 |
|CoFi-Speech~\cite{guo2024speaking} | √ |  |  |  |  | √ |  |  |  | LLM | BigVGAN | Token | 2024.09 |
|Takin~\cite{chen2024takin} | √ | √ |  | √ | √ | √ | √ |  | √ | LLM | HiFi-Codec | Token | 2024.09 |
|HALL-E~\cite{nishimura2024hall} | √ |  |  |  |  | √ |  |  |  | LLM | EnCodec | Token | 2024.10 |
|
#### RNN-Based Approaches: 基于 RNN 的方法

Controllable Text-To-Speech (TTS) technology has seen significant advancements through innovations in neural network architectures, facilitating speech that is both natural-sounding and adaptable in terms of emotion, prosody, and pitch.
A key breakthrough was the introduction of [Tacotron [74]](../../Models/Acoustic/2017.03.29_Tacotron.md), a sequence-to-sequence architecture that effectively integrates prosodic variations, laying the groundwork for precise control over speech attributes.
[Tacotron 2 [175]](../../Models/Acoustic/2017.12.16_Tacotron2.md) further enhanced this capability by better managing prosodic variability, though it averaged these variations, indicating a need for more sophisticated control methods.
To address these constraints, Wang et al. introduced [Global Style Tokens (GSTs) [19]](../../Models/Style/2018.03.23_GST.md), using an unsupervised approach to encapsulate diverse speech styles into fixed tokens, thus enabling versatile style transfer within the Tacotron framework.
Skerry-Ryan et al.\cite{skerry2018towards} further advanced this by incorporating prosodic embeddings, providing detailed control over timing and intonation, and significantly improving the replication of emotional expressions in synthetic speech.

Building on these innovations, emotion-controllable models developed by [Li et al. [21]](../../Models/Style/2020.11.17_Controllable_Emotion_Transfer_for_End-to-End_Speech_Synthesis.md) focus on calibrating emotional nuances using emotion embedding networks and style loss alignment, allowing detailed modulation of emotional strength.
Hierarchical models like MsEmoTTS\cite{lei2022msemotts} refine this approach by segmenting synthesis into global, utterance-level, and local emotional strengths, offering enhanced emotional expressiveness and intuitive control.
These advancements have expanded the scope to produce nuanced TTS outputs, enabling precise control over emotion, prosody, and pitch, with applications ranging from virtual assistants to interactive narratives.
As researchers continue to explore the potential of neural networks in TTS, the technology promises even richer, more engaging digital experiences, moving towards speech synthesis that is indistinguishable from natural human interaction.

#### LLM-Based Approaches: 基于大语言模型的方法

Inspired by the success of large language models (LLMs) in natural language processing (NLP), recent studies have explored leveraging in-context learning for zero-shot TTS generation.

VALL-E~\cite{wang2023neural} is a pioneering work in this area, formulating TTS as a conditional language modeling problem.
It utilizes EnCodec~\cite{defossez2022high} to discretize waveforms into tokens as intermediate representations and employs a two-stage modeling pipeline: an autoregressive model first generates coarse audio tokens, followed by a non-autoregressive model that iteratively predicts additional codebook codes for refinement.
This hierarchical modeling of semantic and acoustic tokens has set the foundation for many subsequent LLM-based TTS approaches~\cite{kharitonov2023speak,huang2023make,kim2023sc,chen2024takin}.

Building on VALL-E, various improvements have been proposed.
VALL-E X~\cite{zhang2023speak} extends VALL-E to multilingual scenarios, supporting zero-shot cross-lingual speech synthesis and speech-to-speech translation.
ELLA-V~\cite{song2024ella} introduces a sequence order rearrangement step, enhancing local alignment between phoneme and acoustic modalities.
RALL-E~\cite{xin2024rall} incorporates prosody tokens as chain-of-thought prompting~\cite{wei2022chain} to stabilize the generation of speech tokens.
VALL-E R~\cite{han2024vall} improves phoneme-to-acoustic alignment and adopts codec-merging to boost decoding efficiency and reduce computational overhead.
VALL-E 2~\cite{chen2024vall} introduces repetition aware sampling and grouped code modeling for greater stability and faster inference.
HALL-E~\cite{nishimura2024hall} adopts a hierarchical post-training framework, effectively managing the trade-off between reducing frame rate and producing high-quality speech.

Beyond the foundational improvements introduced by VALL-E and its immediate extensions, further advancements have focused on enhancing speech alignment, quality, and robustness.
SpearTTS~\cite{kharitonov2023speak} and Make-a-voice~\cite{huang2023make} use semantic tokens to bridge the gap between text and acoustic features.
FireRedTTS~\cite{guo2024fireredtts} further optimizes the tokenizer architecture to enhance speech quality.
CoFi-Speech~\cite{guo2024speaking} generates speech in a coarse-to-fine manner via a multi-scale speech coding and generation approach, producing natural and intelligible speech.
[CosyVoice [17]](../../Models/SpeechLM/2024.07.07_CosyVoice.md) employs supervised semantic tokens to enhance content consistency and speaker similarity in zero-shot voice cloning.
Similarly, BASE TTS~\cite{lajszczak2024base} introduces discrete speech representations based on the WavLM~\cite{chen2022wavlm} self-supervised model, focusing on phonemic and prosodic information.
SeedTTS~\cite{anastassiou2024seed} also proposes a self-distillation method for speech decomposition and a reinforcement learning approach to enhance the robustness, speaker similarity and controllability of generated speech.
Based on this framework, Bailing-TTS~\cite{di2024bailing} enhances the alignment of text and speech tokens using a continual semi-supervised learning strategy, enabling high-quality synthesis of Chinese dialect speech.

Although models using discrete tokens as intermediate representations have achieved notable success in zero-shot TTS, they still face fidelity issues compared to the continuous representation like Mel spectrograms~\cite{meng2024autoregressive,liu2024autoregressive}.
MELLE~\cite{meng2024autoregressive} optimizes the training objectives and sampling strategy, marking the first exploration of using continuous-valued tokens instead of discrete-valued tokens within the paradigm of autoregressive speech synthesis models.
Similar to MELLE, ARDiT~\cite{liu2024autoregressive} encodes audio as vector sequence in continuous space and autoregressively generates these sequences by a decoder-only transformer.

## B·Control Strategies: 控制策略

The control strategies in existing controllable TTS can be broadly classified into four categories: style tagging using discrete labels, speech reference prompt for customizing a new speaker's voice with just a few seconds of voice input, controlling speech style using natural language descriptions, and the instruction-guided mode.
We illustrate taxonomies of controllable TTS from the perspective of control strategies in Fig.03.

### Style Tagging: 风格标记

This paradigm typically employs target control attributes, primarily emotion-related controls, as categorical label inputs to enable controllable speech synthesis.
StyleTagging-TTS~\cite{kim2021expressive} utilizes a short phrase or word to represent the style of an utterance and learns the relationship between linguistic embedding and style embedding space by a pre-trained language model.

However, these methods are limited in expressive diversity, as they can only model a small set of pre-defined styles.

### Reference Speech Prompt: 参考语音提示

This paradigm aims to customize a new speaker’s voice with just a few seconds of voice prompt.
The architecture can be abstracted into two main components: a speaker encoder that processes the reference speech and outputs a speaker embedding, and a conditional TTS decoder that takes both text and speaker embedding as input to generate speech that matches the style of the reference prompt.
MetaStyleSpeech~\cite{min2021meta} and StyleTTS~\cite{li2022styletts} use adaptive normalization as a style conditioning method, enabling robust zero-shot performance.
GenerSpeech~\cite{huang2022generspeech} introduces a multi-level style adapter to improve zero-shot style transfer for out-of-domain custom voices.
SC VALL-E~\cite{kim2023sc} facilitates control over synthesized speech’s emotions, speaking styles, and various acoustic features by incorporating style tokens and scale factors.
ArtSpeech~\cite{wang2024artspeech} revisits the sound production system by integrating articulatory representations into the TTS framework, improving the physical interpretability of articulation movements.

To enhance the learning of contextual information and address the challenge of limited voice data from the target speaker, CCSP~\cite{xiao2024contrastive} proposes a contrastive context-speech pretraining framework that learns cross-modal representations combining both contextual text and speech expressions.
DEX-TTS~\cite{park2024dex} separates styles into time-invariant and time-variant components, enabling the extraction of diverse styles from expressive reference speech.
StyleTTS-ZS~\cite{li2024stylettszs} leverages distilled time-varying style diffusion to capture diverse speaker identities and prosodies.

Some works also decouple timbre and style information from the reference speech, allowing more flexible control over the speaking style~\cite{[NaturalSpeech3 [87]](../../Models/Diffusion/2024.03.05_NaturalSpeech3.md),jiang2024mega,[ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md)}.
MegaTTS 2~\cite{jiang2024mega} introduces an acoustic autoencoder that separately encodes prosody and timbre into the latent space, enabling the transfer of various speaking styles to the desired timbre.
[ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md) uses bidirectional attention and mask-based parallel decoding to capture codec representations in a discrete decoupling codec space, allowing independent control of timbre, style, and content in a zero-shot manner.

### Natural Language Descriptions: 自然语言描述

Recent studies explore controlling speech style using natural language descriptions that include attributes such as pitch, gender, and emotion, making the process more user-friendly and interpretable.
In this paradigm, several speech datasets with natural language descriptions~\cite{[PromptTTS [101]](../../Models/Acoustic/2022.11.22_PromptTTS.md),ji2024textrolspeech,[ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md)} and associated prompt generation pipelines~\cite{ji2024textrolspeech,leng2023prompttts2,lyth2024natural} have been proposed.
Detailed information about these datasets will be discussed in [Section 5](Sec.05.md).
[PromptTTS [101]](../../Models/Acoustic/2022.11.22_PromptTTS.md) uses manually annotated text prompts to describe five speech attributes, including gender, pitch, speaking speed, volume, and emotion.
[InstructTTS [105]](../../Models/Acoustic/2023.01.31_InstructTTS.md) introduces a three-stage training procedure to capture semantic information from natural language style prompts and adds further annotation to the NLSpeech dataset’s speech styles.
PromptStyle~\cite{liu2023promptstyle} constructs a shared space for stylistic and semantic representations through a two-stage training process.
TextrolSpeech~\cite{ji2024textrolspeech} proposes an efficient prompt programming methodology and a multi-stage discrete style token-guided control framework, demonstrating strong in-context capabilities.
NansyTTS~\cite{yamamoto2024description} combines a TTS trained on the target language with a description control model trained on another language, which shares the same timbre and style representations to enable cross-lingual controllability.

Considering not all details about voice variability can be described in the text prompt, PromptTTS++~\cite{shimizu2024prompttts++} and PromptSpeaker~\cite{zhang2023promptspeaker} tries to construct text prompts with more details.
PromptTTS 2~\cite{leng2023prompttts2} designs a variation network to capture voice variability not conveyed by text prompts.
[ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md) proposes the Style Mixture Semantic Density (SMSD) module, incorporating a noise perturbation mechanism to tackle the many-to-many problem in style control and enhance style diversity.

Other works also focus on improving controllability in additional aspects, such as the surrounding environment.
Audiobox~\cite{vyas2023audiobox} introduces both description-based and example-based prompting, integrating speech and sound generation paradigms to independently control transcript, vocal, and other audio styles during speech generation.
VoiceLDM~\cite{lee2024voiceldm} and AST-LDM~\cite{kim2024speak} extend AudioLDM~\cite{liu2023audioldm} to incorporate environmental context in TTS generation by adding a content prompt as a conditional input.
Building on VoiceLDM, MS$^{2}$KU-VTTS~\cite{he2024multi} further expands the dimensions of environmental perception, enhancing the generation of immersive spatial speech.

### Instruction-Guided Control: 指令引导控制

The natural language description-based TTS methods discussed above require splitting inputs into content and description prompts, which limits fine-grained control over speech and does not align with other AIGC models.
[VoxInstruct [103]](../../Models/SpeechLM/2024.08.28_VoxInstruct.md) proposes a new paradigm that extends traditional text-to-speech tasks into a general human instruction-to-speech task.
Here, human instructions are freely written in natural language, encompassing both the spoken content and descriptive information about the speech.
To enable automatic extraction of the synthesized speech content from raw text instructions, VoxInstruct uses speech semantic tokens as an intermediate representation, bridging the gap in current research by allowing the simultaneous use of both text description prompts and speech prompts for speech generation.