# 4·Controllable TTS: 可控文本转语音

In this section, we first review recent TTS work from the perspective of model architecture, followed by a detailed discussion of control modes in controllable TTS.

## A·Model Architecture: 模型架构

Current model architectures can be broadly classified into two main categories: the first is the non-autoregressive (NAR) generative models, which are based on HMMs, neural networks, VAEs, diffusion models, flow matching, and other NAR techniques.
The second category relies on autoregressive (AR) codec language models, which typically quantize speech into discrete tokens and use decoder-only models to autoregressively generate these tokens.
We summarize the NAR-based and AR-based controllable TTS methods in Table \ref{tab:sec5_controllable_methods_nar} and Table \ref{tab:sec5_controllable_methods_ar}, respectively.

### Fully Non-Autoregressive (NAR) Architectures: 完全非自回归架构

#### HMM-Based Approaches: 基于 HMM 的方法

In the realm of Controllable Text-To-Speech (CTTS), advancements in Hidden Markov Model (HMM) architectures have significantly enhanced the manipulation of speech elements such as emotion and prosody.
Yamagishi et al.~\cite{yamagishi2003modeling} pioneered this field by introducing style-dependent and style-mixed modeling, which allowed precise emulation of human-like emotional nuances and versatile synthesis across various styles by incorporating style as a contextual variable.
Building on this foundation, Qin et al.\cite{qin2006hmm} developed the "average emotion model," which utilized MLLR-based adaptation to modulate emotions like happiness and sadness even with limited data, thus advancing the emotional intelligence of synthetic speech systems.

Furthering expressive variability, Nose et al.\cite{nose2013intuitive} integrated subjective style intensities and a multiple-regression global variance model into HMM frameworks, addressing over-smoothing and enabling nuanced emotional expressions.
Lorenzo-Trueba et al.\cite{lorenzo2015emotion} expanded on these capabilities with CSMAPLR adaptation, introducing "emotion transplantation" to transfer emotional states between speakers while preserving voice distinctiveness, enhancing personalized human-computer interaction.
These innovations in HMM architectures have broadened the expressiveness and individuality in synthetic speech, augmenting technological interfaces and paving the way for future developments in adaptive, lifelike speech solutions.

#### Transformer-Based Approaches: 基于 Transformer 的方法

Advancements in Controllable Text-to-Speech (TTS) technology highlight the integration of deep learning with audio processing, driven by Transformer-based architectures.
Ren et al.\cite{ren2019fastspeech} introduced FastSpeech, a feed-forward non-autoregressive Transformer model that significantly enhances TTS efficiency by reducing inference time and improving the stability issues found in autoregressive models like Tacotron 2.
This model provides precise control over prosodic features through duration prediction, effectively tackling the one-to-many mapping challenge.
FastSpeech 2\cite{ren2020fastspeech} builds on this by integrating pitch and energy control, eliminating the need for the complex teacher-student distillation process, thus enhancing training efficiency and improving voice quality.
Parallel Tacotron\cite{elias2021paralleltacotron} further advances TTS by employing a variational autoencoder-based residual encoder, capturing intricate prosodic nuances.
This approach, combined with iterative spectrogram loss, significantly enhances the naturalness and quality of synthesized speech.
Additionally, FastPitch\cite{lancucki2021fastpitch} incorporates direct pitch prediction into its architecture, enabling fully parallelized synthesis and precise pitch manipulation.
This capability enhances expressiveness and retains the efficiency benefits established by FastSpeech.
These innovations significantly contribute to the development of more interactive and natural AI-driven communication systems, underscoring the potential of integrating AI with human-centric disciplines to craft a future where technology and humanity coexist harmoniously.

#### VAE-Based Approaches: 基于 VAE 的方法

Recent advancements in Controllable Text-To-Speech (TTS) systems are largely driven by the integration of Variational Autoencoder (VAE) architectures, which enhance the flexibility and precision of speech modulation.
Zhang et al.\cite{zhang2019learning} pioneered the use of VAEs in end-to-end speech synthesis, creating disentangled latent representations that allow effective style control and transfer, especially in prosody and emotion management, outperforming the Global Style Token model in style transfer tasks.
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
NaturalSpeech 3~\cite{ju2024naturalspeech3} decomposes speech into distinct subspaces that represent different attributes and generates each subspace independently.
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
StyleTTS 2~\cite{li2024styletts2} employs large pre-trained speech language models (SLMs) such as Wav2Vec 2.0~\cite{baevski2020wav2vec}, HuBERT~\cite{hsu2021hubert}, and WavLM~\cite{chen2022wavlm} as discriminators, in combination with a novel differentiable duration modeling approach.
This setup uses SLM representations to enhance the naturalness of the synthesized speech.
MaskGCT~\cite{wang2024maskgct} proposes masked generative transformers without requiring text-speech alignment supervision and phone-level duration.
The model employs a two-stage system, both trained using a mask-and-predict learning paradigm.

### Autoregressive (AR) Architectures: 自回归架构

#### RNN-Based Approaches: 基于 RNN 的方法

Controllable Text-To-Speech (TTS) technology has seen significant advancements through innovations in neural network architectures, facilitating speech that is both natural-sounding and adaptable in terms of emotion, prosody, and pitch.
A key breakthrough was the introduction of Tacotron\cite{wang2017tacotron}, a sequence-to-sequence architecture that effectively integrates prosodic variations, laying the groundwork for precise control over speech attributes.
Tacotron 2\cite{shen2018natural} further enhanced this capability by better managing prosodic variability, though it averaged these variations, indicating a need for more sophisticated control methods.
To address these constraints, Wang et al.\cite{wang2018style} introduced Global Style Tokens (GSTs), using an unsupervised approach to encapsulate diverse speech styles into fixed tokens, thus enabling versatile style transfer within the Tacotron framework.
Skerry-Ryan et al.\cite{skerry2018towards} further advanced this by incorporating prosodic embeddings, providing detailed control over timing and intonation, and significantly improving the replication of emotional expressions in synthetic speech.

Building on these innovations, emotion-controllable models developed by Li et al.\cite{li2021controllable} focus on calibrating emotional nuances using emotion embedding networks and style loss alignment, allowing detailed modulation of emotional strength.
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
CosyVoice~\cite{du2024cosyvoice} employs supervised semantic tokens to enhance content consistency and speaker similarity in zero-shot voice cloning.
Similarly, BASE TTS~\cite{lajszczak2024base} introduces discrete speech representations based on the WavLM~\cite{chen2022wavlm} self-supervised model, focusing on phonemic and prosodic information.
SeedTTS~\cite{anastassiou2024seed} also proposes a self-distillation method for speech decomposition and a reinforcement learning approach to enhance the robustness, speaker similarity and controllability of generated speech.
Based on this framework, Bailing-TTS~\cite{di2024bailing} enhances the alignment of text and speech tokens using a continual semi-supervised learning strategy, enabling high-quality synthesis of Chinese dialect speech.

Although models using discrete tokens as intermediate representations have achieved notable success in zero-shot TTS, they still face fidelity issues compared to the continuous representation like Mel spectrograms~\cite{meng2024autoregressive,liu2024autoregressive}.
MELLE~\cite{meng2024autoregressive} optimizes the training objectives and sampling strategy, marking the first exploration of using continuous-valued tokens instead of discrete-valued tokens within the paradigm of autoregressive speech synthesis models.
Similar to MELLE, ARDiT~\cite{liu2024autoregressive} encodes audio as vector sequence in continuous space and autoregressively generates these sequences by a decoder-only transformer.

## B·Control Strategies: 控制策略

The control strategies in existing controllable TTS can be broadly classified into four categories: style tagging using discrete labels, speech reference prompt for customizing a new speaker's voice with just a few seconds of voice input, controlling speech style using natural language descriptions, and the instruction-guided mode.
We illustrate taxonomies of controllable TTS from the perspective of control strategies in Fig.~\ref{fig:controllablemode}.

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

Some works also decouple timbre and style information from the reference speech, allowing more flexible control over the speaking style~\cite{ju2024naturalspeech3,jiang2024mega,ji2024controlspeech}.
MegaTTS 2~\cite{jiang2024mega} introduces an acoustic autoencoder that separately encodes prosody and timbre into the latent space, enabling the transfer of various speaking styles to the desired timbre.
ControlSpeech~\cite{ji2024controlspeech} uses bidirectional attention and mask-based parallel decoding to capture codec representations in a discrete decoupling codec space, allowing independent control of timbre, style, and content in a zero-shot manner.

### Natural Language Descriptions: 自然语言描述

Recent studies explore controlling speech style using natural language descriptions that include attributes such as pitch, gender, and emotion, making the process more user-friendly and interpretable.
In this paradigm, several speech datasets with natural language descriptions~\cite{guo2023prompttts,ji2024textrolspeech,ji2024controlspeech} and associated prompt generation pipelines~\cite{ji2024textrolspeech,leng2023prompttts2,lyth2024natural} have been proposed.
Detailed information about these datasets will be discussed in Section \ref{sec:ch5_datasets_eval}.
PromptTTS~\cite{guo2023prompttts} uses manually annotated text prompts to describe five speech attributes, including gender, pitch, speaking speed, volume, and emotion.
InstructTTS~\cite{yang2024instructtts} introduces a three-stage training procedure to capture semantic information from natural language style prompts and adds further annotation to the NLSpeech dataset’s speech styles.
PromptStyle~\cite{liu2023promptstyle} constructs a shared space for stylistic and semantic representations through a two-stage training process.
TextrolSpeech~\cite{ji2024textrolspeech} proposes an efficient prompt programming methodology and a multi-stage discrete style token-guided control framework, demonstrating strong in-context capabilities.
NansyTTS~\cite{yamamoto2024description} combines a TTS trained on the target language with a description control model trained on another language, which shares the same timbre and style representations to enable cross-lingual controllability.

Considering not all details about voice variability can be described in the text prompt, PromptTTS++~\cite{shimizu2024prompttts++} and PromptSpeaker~\cite{zhang2023promptspeaker} tries to construct text prompts with more details.
PromptTTS 2~\cite{leng2023prompttts2} designs a variation network to capture voice variability not conveyed by text prompts.
ControlSpeech~\cite{ji2024controlspeech} proposes the Style Mixture Semantic Density (SMSD) module, incorporating a noise perturbation mechanism to tackle the many-to-many problem in style control and enhance style diversity.

Other works also focus on improving controllability in additional aspects, such as the surrounding environment.
Audiobox~\cite{vyas2023audiobox} introduces both description-based and example-based prompting, integrating speech and sound generation paradigms to independently control transcript, vocal, and other audio styles during speech generation.
VoiceLDM~\cite{lee2024voiceldm} and AST-LDM~\cite{kim2024speak} extend AudioLDM~\cite{liu2023audioldm} to incorporate environmental context in TTS generation by adding a content prompt as a conditional input.
Building on VoiceLDM, MS$^{2}$KU-VTTS~\cite{he2024multi} further expands the dimensions of environmental perception, enhancing the generation of immersive spatial speech.

### Instruction-Guided Control: 指令引导控制

The natural language description-based TTS methods discussed above require splitting inputs into content and description prompts, which limits fine-grained control over speech and does not align with other AIGC models.
VoxInstruct~\cite{zhou2024voxinstruct} proposes a new paradigm that extends traditional text-to-speech tasks into a general human instruction-to-speech task.
Here, human instructions are freely written in natural language, encompassing both the spoken content and descriptive information about the speech.
To enable automatic extraction of the synthesized speech content from raw text instructions, VoxInstruct uses speech semantic tokens as an intermediate representation, bridging the gap in current research by allowing the simultaneous use of both text description prompts and speech prompts for speech generation.