# Towards Audio Language Modeling: An Overview

作者列表

- Haibin Wu, 
- Xuanjun Chen, 
- Yi-Cheng Lin, 
- [Kai-wei Chang](../../Authors/张凯为_Kai-wei_Chang.md), 
- Ho-Lam Chung, 
- Alexander H. Liu, 
- [Hung-yi Lee](../../Authors/李宏毅_Hung-Yi_Lee.md)

## Abstract

> Neural audio codecs are initially introduced to compress audio data into compact codes to reduce transmission latency.
> Researchers recently discovered the potential of codecs as suitable tokenizers for converting continuous audio into discrete codes, which can be employed to develop audio language models (LMs).
> Numerous high-performance neural audio codecs and codec-based LMs have been developed.
> The paper aims to provide a thorough and systematic overview of the neural audio codec models and codec-based LMs.

## 1.Introduction

> Neural audio codec models were first introduced to compress audio for efficient data transmission. 
> The encoder converts the audio into codec codes, which are then transmitted.
> The receiver then uses the codec decoder to reconstruct the audio using the received codes.

> Language modeling has proven to be highly successful in the field of Natural Language Processing (NLP). 
> Audio data encompasses not only textual content but also rich information about speaker timbre, emotion, and general audio, offering deeper possibilities for language model applications.
> Researchers, especially those in large companies with significant computational resources, recently leverage the potential of neural codecs [1]–[8] as suitable tokenizers for converting continuous audio into discrete codes, which can be employed to develop audio language models (LMs) [9]–[20].
> The current codec-based language models and codec models are summarized in Figure 1. 
> These findings promptly garnered the community’s attention, sparking a fervor for developing codecs tailored to audio language modeling. 
> Numerous high performance neural audio codec models and audio LMs have been developed.

> An ideal codec should maintain content while preserving paralinguistic and speaker-related information. 
> Similarly, a universal audio language model should be able to generalize across various audio types, such as speech, music, and general audio, covering a wide range of applications. 
> The arms race in developing codecs and audio LMs is still ongoing.

> Given the significant advancements in codecs and audio language models over the past three years as shown in Figure , there has yet to be a comprehensive review comparing them and providing inspiration to the community.
> In this study, we aim to fill this research gap by thoroughly reviewing and comparing various existing neural codec models and audio codec-based language models.
> Firstly, we specifically conduct an in-depth analysis of six representative open-source neural codec models to cover their training methodologies, implementation settings, and training data. 
> Secondly, we expand our analysis to include eleven diverse codec-based language models, examining how they utilize the codecs and the tasks to which they can be applied. 
> Through this comprehensive review, we aim to offer the community insights into the diverse methodologies and potential directions in the field of neural codecs and codec-based language modeling.

## 2.Comprehensive Comparison for Neural Audio Codec Models

> Codec models aim to compress and decompress speech signals efficiently.
> Traditional codecs are developed based on psycho-acoustics and speech synthesis [21], [22].
> Recently, the neural codec models demonstrated highly effective for compression and signal reconstruction, outperforming traditional codecs.
> Considering the broad spectrum of codec models within the research community, each trained with its distinct configurations and training techniques, there is a clear need for a thorough examination that covers the training methodologies, implementation settings, and training data employed across these codec models.
> The six codec models have distinct training details, resulting in a collection of fifteen different codec models, as summarized in Tab.01.

## 2.1.Brief Method Overview for Codecs

> SoundStream [2] stands as one of the pioneering implementations of neural codec models, embodying a classic neural codec architecture comprising encoder, quantizer, and decoder modules.
> It utilizes the streaming SEANets [23] as its encoder and decoder.
> The quantizer incorporates a speech enhancement system with a Residual Vector Quantization (RVQ) [2], [24] bottleneck to obtain parallel token streams.
> During training, the model parameters are optimized using a combination of reconstruction and adversarial loss.
> SoundStorm [3] is an improved version of SoundStream to achieve both efficiency and high-quality audio generation.
> It accomplishes this by employing an architecture specifically tailored to the hierarchical structure of audio tokens.
> Moreover, it pioneers a parallel, non-autoregressive decoding scheme, which relies on confidence-based strategies for residual vector-quantized token sequences.

> Encodec [1] builds upon a framework similar to SoundStream.
> Nonetheless, it further augments its capabilities by integrating supplementary LSTM [25] layers and harnessing a Transformer-based language model [26] to model the RVQ codes, thereby amplifying its sequence modeling performance.
> Then, there is a stream of work aimed at making codec models more general and powerful.
> AudioDec [4] represents an enhanced version of Encodec, implementing a group convolution mechanism to facilitate the real-time operation of the streamable network while also harnessing the capabilities of HiFi-GAN [27] to effectively generate high-fidelity audio at a high sampling rate of 48 kHz. 

> In the AcademiCodec model introduced by [5], a novel technique known as group-residual vector quantization is presented.
> It employs multiple parallel RVQ groups.
> This technique is specifically tailored for generation tasks.
> It aims to enhance the reconstruction performance while using a limited number of codebooks, consequently achieving an impressively low bit rate per second (BPS).
> This low BPS is of utmost significance as it effectively addresses the challenge of lengthy speech tokens in speech language modeling, resulting in reduced sequence lengths.

> SpeechTokenizer [7] is a unified speech tokenizer designed for speech language models.
> It implements an Encoder-Decoder architecture enhanced with RVQ.
> By integrating both semantic and acoustic tokens, SpeechTokenizer hierarchically separates various aspects of speech information across different RVQ layers.
> Specifically, SpeechTokenizer is designed to regularize the first RVQ layer to highlight semantic information by learning the Hubert tokens [28].
> Using such techniques can enhance the disentanglement of information across different RVQ layers.

> Descript-Audio-Codec (DAC) [8], a universal neural codec model, distinguishes itself through its exceptional ability to maintain high-fidelity audio quality across a wide spectrum of data types, encompassing general audio, music, and speech.
> It accomplishes this feature by employing a number of training techniques, such as periodic activation functions [29], enhanced residual vector quantization using factorized and L2-normalized codes, random quantizer dropout to preserve audio reconstruction quality, as well as refining adversarial and reconstruction loss during the training process.
> The authors highlight the crucial importance of the periodic activation function among the employed techniques.

> Unlike most models focusing on the time domain, FunCodec [6] proposes a frequency-domain codec.
> The authors claim they can achieve comparable performance with fewer parameters and lower computation complexity.
> Meanwhile, it also finds that incorporating semantic information in the codec tokens improves speech quality at low bit rates.

### 2.2.Comparison from Methodology Angles

> We compare several techniques proposed by these codecs in Tab.02.
> The abbreviation “A-F” represents different codec models.
> Please refer to Tab.01 for the corresponding model full name.
> The design of discriminators constitutes a pivotal element within codec models.
> Encodec initially introduces the Multi-scale-STFT Discriminator (MS-STFTD).
> In contrast to the multi-scale discriminator (MSD) proposed in MelGAN [24], which captures long-term dependencies, the multi-period discriminator (MPD) proposed in HiFi-GAN [30] exhibits a capacity to discern more nuanced periodic details.
> Consequently, AudioDec replaces the conventionally employed STFTD with a HiFi-GAN-based MPD, observing an enhancement in audio quality within their model.
> AcademiCodec integrates prior research efforts by incorporating the MS-STFTD from Encodec and both HiFi-GAN-based MPD and MSD.
> Both SpeechTokenizer and FunCodec adopt identical discriminators to AcademiCodec, with Funcodec offering a unified interface adaptable to any combination of these three discriminator types.
> DAC identifies that employing MSD and MPD alone generates audio displaying blurriness and artifacts.
> To address this, they propose the application of a multi-scale, multi-band STFT discriminator (MS-MB-STFTD) to improve phase modeling and mitigate aliasing artifacts.

> SpeechTokenizer utilizes semantic tokens from Hubert L9 as a teacher for the RVQ process.
> This guidance enables the disentanglement of content information into the first layer of the tokenizer, while paralinguistic information is retained in subsequent layers.
> FunCodec seeks to integrate semantic information by combining, adding, or residualizing the audio codec with semantic tokens.
> The study reveals that including semantic tokens enhances audio quality, particularly with the residual inclusion method.
> Additionally, SpeechTokenizer and FunCodec utilize K-means to cluster samples in the first mini-batch for initializing the VQ codebook, leading to improved code utilization.
> DAC follows the approach of BigVGAN [31], employing snake activation [29] for trainable control over the frequency of periodic signals.
> AcademiCodec employs multiple RVQ codebooks (multiple residual groups) to represent intermediate features.
> They demonstrate that using multiple residual groups achieves good reconstruction performance while employing only a few codebooks.
> Encodec trains an additional small transformer model for entropy coding over the quantized units, which reduces bandwidth and accelerates encoding and decoding.

### 2.3.Implementation Details

> We compare the codebook number, training data, sampling rate, and bit rate per second in Tab.01.
> From the training data perspective, SpeechTokenizer [7], AudioDec [4], and FunCodec [6] utilize only English speech dataset.
> AcademiCodec [5] incorporates bilingual speech datasets, including AISHELL for Chinese and LibriTTS and VCTK for English.
> Both DAC [8], and Encodec [1] encompass diverse modality data, including speech, music, and audio, in the training data.

## 3.Current Codec-Based Speech Language Models

> As shown in Figure 2, the process of neural codec-based audio language modeling begins by converting context information, such as text and MIDI, into context codes, while simultaneously encoding the audio into codec codes.
> These context and codec codes are then employed in the language modeling phase to generate the desired target codec code sequence.
> Subsequently, the target codec code sequence is passed to the codec decoder to produce the audio output.
> The entire pipeline embodies an audio-to-audio modeling approach.

### 3.1.Overview of Codec-Based Language Models

> AudioLM [9] is the pioneering model in introducing codec codes for language modeling, utilizing a hierarchical approach that encompasses two distinct stages.
> The first stage generates semantic tokens using a self-supervised w2v-BERT model [32].
> These tokens are then leveraged in the second stage as conditioning elements to create acoustic tokens using a SoundStream neural codec [2].

> VALL-E [12], VALL-E X [13], and SpeechX [17], all originate from Microsoft and are neural codec language models trained to generate discrete codes derived from EnCodec [1], based on textual or acoustic inputs.
> VALL-E can generate high-quality personalized speech with only a 3-second enrollment recording from an unseen speaker.
> Furthermore, VALL-E X can produce high-quality speech in the target language with just a single speech utterance in the source language as a prompt.
> Additionally, SpeechX introduces a unified framework to address not only zero-shot TTS but also various types of speech transformation tasks, including speech enhancement and speech editing.

> What sets ViaLA [14], AudioPaLM [10], and LauraGPT [16] apart is their dual capability to generate both text and audio.
> VioLA tries to tackle the question “Is one decoder-only generative model all you need for speech recognition, synthesis, and translation?” by employing language modeling that integrates both text tokens and audio tokens (extracted by EnCodec [1]), along with the use of task IDs and language IDs.
> AudioPaLM constructs a unified vocabulary comprising both text and audio tokens.
> It is a decoder-only, autoregressive model capable of processing and generating both text and speech.
> Additionally, AudioPaLM’s initialization stems from PaLM-2 [33], a text-only language model.
> AudioPaLM’s approach to audio tokenization resembles that of AudioLM.
> Moreover, AudioPaLM adopts and extends the SoundStream model to SoundStorm [3].
> LauraGPT [16] is a versatile language model built on a decoder-only text-based language model, Qwen-2B [34].
> LauraGPT has the capability to process both audio and text inputs, generating outputs in either modality.
> LauraGPT encodes input audio into continuous representations using a Conformer encoder and decodes output audio using FunCodec [6] discrete codes.
> The authors claim this specific audio features design for inputs and outputs will result in improved performance for speech generation using some preliminary experimental results.

> UniAudio [15] utilizes language modeling to generate a wide range of audio types, including speech, sounds, music, and singing, using textual or acoustic tokens as inputs.
> UniAudio stands out for its ability to enhance autoregressive prediction speed by introducing a multi-scale Transformer model [35], which employs a large global transformer to predict the first-layer codec codes and a small local transformer to predict the codec codes for the subsequent codec layers.
> The codec model in UniAudio is revised from EnCodec.

> Additionally, there are other codec-based language models designed for sound modeling.
> AudioGen [20] trained a SoundStream model to get audio tokens and subsequently trained a language model to utilize textual features as conditions for generating audio tokens.
> MusicLM [11] follows a training strategy similar to AudioLM but extends its scope to encompass music features.
> It approaches the task of conditional music generation through a hierarchical sequence-to-sequence modeling approach.
> Initially, it utilizes music tokens from Mulan [36] to generate semantic tokens from the w2v-BERT model.
> Subsequently, it employs both music tokens and semantic tokens to generate acoustic features through SoundStream.
> MusicGen [18] is a music language model designed to work with EnCodec discrete tokens.
> It accepts textual descriptions or melodic features as input conditions to generate tokens, which can be reconstructed to high-fidelity music.

> Another branch of speech language modeling aims to utilize discrete units obtained by quantizing self-supervised speech representations.
> While these discrete units contain rich acoustic and linguistic information [37], they lack speaker and paralinguistic information [38].
> This research direction focuses on modeling the semantics of speech, with the optional use of encoders to learn about speaker characteristics and prosody.
> Pioneering work is speech-resynthesis [38], which utilizes these discrete units in conjunction with prosody and speaker encoders to encode speech into low-bitrate codes.
> These codes can then be resynthesized into a speech signal with a decoder to achieve low-bitrate transmission.
> Additionally, these discrete units can be regarded as “pseudo-text,” serving as a foundation for training textless speech language models.
> Notable examples include GSLM [39], pGSLM [40], dGSLM [41], and TWIST [42].
> By engaging in the pre-trained task of next-token prediction, these speech LMs perform spoken language modeling and can conduct the task of speech continuation.
> In the field of speech translation, recent advancements have been made possible through these discrete units. [43] pre-trained a Unit mBART combined with a wav2vec 2.0 [44] encoder to directly predict the translated discrete units.
> UnitY [45] further incorporates text modality to enhance speech translation.
> The Seamless models [46], [47] integrate the UnitY framework to perform expressive and streaming speech-to-text and speech-to-speech translation.
> With the development of these powerful speech LMs, researchers have begun to explore the use of prompting on speech LMs for various speech processing tasks, including prompt tuning [48]–[50], in-context learning [51], and instruction tuning [52], [53].

### 3.2.Comparison for Codec-Based Audio Language Models

> In Tab.03, we compare the inputs, outputs, and downstream tasks of different codec-based language models.
> We also summarize that the downstream tasks conducted by differ-ent codec-based language models: Speech Continuation (SC), Piano Continuation (PC), Audio Continuation (AC), Text-to-Speech (TTS), Music Generation (MG), Stereophonic Generation (SG), Speech to Speech Translation (S2ST), Automatic Speech Recognition (ASR), Spoken Language Understanding (SLU), Automated Audio Captioning (AAC), Speech to Text Translation (S2TT), Machine Translation (MT), Speech Enhancement (SE), Speech Removal (SR), Target Speaker Extraction (TSE), Speech Editing (SPED), Voice Conversion (VC), Singing Voice Synthesis (SVS), Text-to-Sound (TTSO), Text-to-Music (TTM), Audio Editing (AUED), Speech Dereverb (SD), Instructed TTS (ITTS).
> Finally, we show the codec models adopted by different LMs.

## 4.Conclusions

> The paper fills the research blank to review the neural codec models and LMs built upon them.
> We hope the comprehensive review and comparisons can inspire future research works to boost the development of neural codec models and codec-based LMs.