# 4.Training Recipes: 训练方法

In this section, we categorize and summarize the commonly used training recipes found in recent SpeechLM papers.
This includes an overview of the types of features modeled in SpeechLMs, the various training stages along with the techniques employed in each stage, and the different paradigms for generating speech.

## 4.1.Features Modeled

The features modeled refer to the types of features outputted by the speech tokenizer and modeled by the language model component within a SpeechLM.
These features play a crucial role in determining the capabilities and performance of SpeechLMs.
Different features model the speech waveforms from different aspects.
Based on recent developments, we can categorize the features modeled by SpeechLMs into two main types, including discrete features and continuous features.

### Discrete Features

Discrete features refer to quantized representations of speech signals that can be represented as distinct, countable units or tokens.
These features are typically derived from speech signals through various encoding and quantization processes, resulting in a finite set of possible values.
Discrete features are the most used features by SpeechLMs as they can be represented as tokens and be modeled exactly the same as the text tokens within a TextLM.
The majority of speech tokenizers produce discrete tokens that better model the **semantic information** within a speech waveform (semantic understanding speech tokenizers, \cref{sec:semanticUnderstandingTokenizer}), such as [W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md); [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md).
This is because they primarily use understanding objectives such as MLM to model the contextual information of the waveforms when training the tokenizer.
We refer to them as **semantic tokens** here.

Most SpeechLMs only employ **semantic tokens** to represent speech.
[GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md), the first-ever SpeechLM, compares three tokenizers, which include [Contrastive Predictive Coding (CPC) (2018)](../../Models/Speech_Representaion/2018.07.10_CPC.md), [Wav2Vec 2.0 (2020)](../../Models/Speech_Representaion/2020.06.20_Wav2Vec2.0.md), and [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md).
It concludes that HuBERT performs the best on various tasks such as speech resynthesis and speech generation.
A large number of works follow this setting and use HuBERT as the speech tokenizer ([TWIST (2023)](../../Models/Speech_LLM/2023.05.22_TWIST.md); [SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md); [SpeechGPT (2023)](../../Models/Speech_LLM/2023.05.18_SpeechGPT.md)).
[AudioPaLM (2023)](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md) experiments the choice between [W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md) , USM-v1 ([Google USM (2023)](../../Models/Speech_LLM/2023.03.02_USM.md)), and USM-v2 ([AudioPaLM (2023)](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md)) (which is a modified version of USM-v1), and it concludes that USM-v2 is the best-performing speech tokenizer on ASR and Speech Translation (ST) tasks.

Although semantic tokens excel at generating semantically meaningful speech because of the modeling of the contextual information within speech waveforms, researchers find out that the speech generated solely upon semantic tokens lacks expressive information such as prosody and different pitches or timbres ([Expresso (2023)](../../Datasets/2023.08.10_Expresso.md); [SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md)).
To conquer this limitation, **paralinguistic tokens** can be integrated into the modeling process to capture expressive information with speeches.
Specifically, [pGSLM (2021)](../../Models/Speech_LLM/2021.09.07_pGSLM.md) proposes to use the fundamental frequency (F0) and unit duration as prosody features to complement the HuBERT semantic tokens, and trains a multi-stream transformer language model to predict the semantic tokens, pitch (F0), and unit duration separately.
Similarly, [SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md) complements the HuBERT semantic tokens with pitch and style tokens ([Sonar Expressive (2023)](../../Models/S2ST/Sonar_Expressive.md)).
This incorporation of extra acoustic tokens allows SpeechLMs to more effectively capture expressive elements without significantly compromising semantic understanding ([SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md)).

Another type is **acoustic tokens**, which are tokens aiming to capture the essential acoustic features to reconstruct high-fidelity speech, primarily obtained from neural audio codec models (see \cref{sec:acousticGenerationTokenizer}).
Codec models aim to learn the compressed representation of audio, so it is anticipated that both the semantic and acoustic information present in a speech waveform can be encoded in the representation.
Some studies attempt to directly model the codec tokens in an autoregressive manner.
[VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md) utilizes codec tokens to achieve zero-shot TTS.
It encodes a 3-second audio clip using [EnCodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md) as a prompt, enabling the TTS system to synthesize speech that matches the timbre information of the prompt.
[ViolLA (2023)](../../Models/Speech_LLM/2023.05.25_VioLA.md) uses codec tokens in a SpeechLM capable of performing ASR, TTS, and Machine Translation (in text).

#### Discussion

Different types of tokens influence the speech quality of SpeechLMs in different ways, often resulting in trade-offs ([AudioLM (2022)](../../Models/Speech_LLM/2022.09.07_AudioLM.md)).
For example, while semantic tokens align well with text and excel in producing semantically coherent speech, the generated speech often lacks acoustic details, such as high-frequency information.
Recovering and enhancing these details typically requires postprocessing, like a diffusion model, which significantly increases the model's latency.
Conversely, acoustic tokens can facilitate the generation of high-fidelity audio but often struggle with inaccuracies in content generation ([SpeechTokenizer (2023)](../../Models/Speech_Neural_Codec/2023.08.31_SpeechTokenizer.md)).
Researchers have tried two ways to balance these trade-offs.
The first involves combining semantic and acoustic tokens into a single sequence.
[AudioLM (2022)](../../Models/Speech_LLM/2022.09.07_AudioLM.md) proposes a hierarchical modeling scheme that first models semantic tokens from [W2V-BERT (2021)](../../Models/Speech_Representaion/2021.08.07_W2V-BERT.md)  and then uses these tokens to predict acoustic tokens from [SoundStream (2021)](../../Models/Speech_Neural_Codec/2021.07.07_SoundStream.md), which ultimately generates speech.
However, this kind of approach increases sequence length, which increases modeling complexity.
The second strategy leverages **mixed tokens** (see \cref{sec:mixedTokenizer}) to jointly model semantic and acoustic information, showing promising results in [Moshi (2024)](../../Models/Speech_LLM/2024.09.17_Moshi.md) and [SpeechGPT-Gen (2024)](../../Models/Speech_LLM/2024.01.24_SpeechGPT-Gen.md).

### Continuous Features

Continuous features, in contrast to discrete features, are unquantized, real-valued representations of speech signals that exist on a continuous scale.
These features capture fine-grained, nuanced aspects of speech that may be lost in discretization processes.
Continuous features can include spectral representations like mel-spectrograms or latent representations extracted from neural networks.
The exploration of leveraging continuous features to condition SpeechLMs is still in its infancy.
[Spectron (2023)](../../Models/Speech_LLM/2023.05.24_Spectron.md) performs speech continuation by predicting the spectrograms frame-by-frame.
However, the generation of speech spectrograms still needs to be conditioned on text transcripts, which is not an end-to-end speech generation approach.
[Mini-Omni (2024)](../../Models/MultiModal/2024.08.27_Mini-Omni.md) extracts intermediate representations from a frozen Whisper encoder as input for the SpeechLM, whereas [LauraGPT (2023)](../../Models/Speech_LLM/2023.10.07_LauraGPT.md) employs an audio encoder trained alongside the SpeechLM to derive latent representations from input speech.

## 4.2.Training Stages

Training a SpeechLM involves training the three main components: speech tokenizer, language model, and vocoder.
Similar to TextLMs, the key to training SpeechLMs lies in effectively modeling speech continuation, which is primarily the responsibility of the language model.
The speech tokenizer and vocoder usually rely on established methods and are trained using distinct training datasets specific to each SpeechLM approach.
Therefore, This section reviews the main techniques used to train the language model component.

Following TextLMs, the training process for SpeechLMs can be divided into three stages: pre-training, instruction-tuning, and alignment.
However, to our knowledge, there is currently no research specifically focused on the alignment process following instruction tuning.
Therefore, we only discuss the works related to the pre-training and instruction-tuning stages of SpeechLMs.

### Language Model Pre-Training

The pre-training of the language model in SpeechLMs is a critical phase that significantly influences the model's ability to generate coherent and contextually relevant speech.
This phase typically involves training the language model to autoregressively predict the next token on a large corpus of speech tokens.
The primary objective during this stage is to learn the statistical patterns and dependencies inherent in the speech data, enabling the model to predict the next token in a sequence based on the preceding context.
Table.02 includes popular datasets used in pre-training stage of SpeechLMs.

![](Images/Tab.02.png)

#### Training data.

SpeechLMs pre-training mainly leverages large-scale open-sourced speech data.
Commonly used datasets include those for ASR ([LibriSpeech (2015)](../../Datasets/2015.04.19_LibriSpeech.md); [Libri-Light (2019)](../../Datasets/2019.12.17_Libri-Light.md); [The People's Speech (2021)](../../Datasets/2021.11.17_The_People's_Speech.md); [VoxPopuli (2021)](../../Datasets/2021.01.02_VoxPopuli.md)), TTS ([LibriTTS (2019)](../../Datasets/2019.04.05_LibriTTS.md)), ST ([CVSS (2022)](../../Datasets/2022.01.11_CVSS.md); [VoxPopuli (2021)](../../Datasets/2021.01.02_VoxPopuli.md)), podcasts ([Spotify Podcast Dataset (2020)](../../Datasets/2020.04.08_Spotify_Podcast_Dataset.md)), and dialogues ([Fisher Corpus (2004)](../../Datasets/Fisher_Corpus.md)).
Some datasets consist solely of speech data, while others include both speech and corresponding text transcripts.
The inclusion of text transcripts can enhance the model's representation by allowing it to learn the relationship between spoken language and its written form, which will be discussed later.

#### Cold Initialization.

Some SpeechLMs use cold initialization during the pre-training phase, where model parameters are initialized randomly.
The pioneering SpeechLM---[GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md)---trained a [Transformer (2017)](../../Models/_Transformer/2017.06.12_Transformer.md) from scratch to serve as the language model.
This study demonstrated the effectiveness of the SpeechLM pipeline and compared performance across various speech tokenizer options.
They found that [HuBERT (2021)](../../Models/Speech_Representaion/2021.06.14_HuBERT.md) outperformed [CPC (2018)](../../Models/Speech_Representaion/2018.07.10_CPC.md) and [Wav2vec 2.0 (2020)](../../Models/Speech_Representaion/2020.06.20_Wav2Vec2.0.md) in understanding speech content and generating natural speech.
[SUTLM (2023)](../../Models/Speech_LLM/2023.10.12_SUTLM.md) also uses a transformer as the language model.
They studied the critical problem of jointly modeling speech and text tokens by comparing four different modeling methods: speech-only, text-only, concatenated speech-text, and alternating (interleaving) speech-text.
They showed that the setting of alternating speech-text performs the best in cross-modal evaluations.
Table.03 illustrates the four modeling methods.

![](Images/Tab.03.png)

Some works leverage a different architecture from the standard transformer.
Since there are no existing checkpoints for those self-proposed architectures, it is necessary to train them from scratch.
For example, [pGSLM (2021)](../../Models/Speech_LLM/2021.09.07_pGSLM.md) proposes a multi-stream transformer language model (MS-TLM) that takes multiple streams of input and predicts multiple streams of output to generate speech units, duration, and pitch embeddings simultaneously.
[dGSLM (2022)](../../Models/Speech_LLM/2022.03.30_dGSLM.md) introduced a dialogue transformer language model (DLM) to jointly model the dialogue speech data from the two speakers.
To enable the listening ability of SpeechLMs while speaking, [LSLM (2024)](../../Models/Speech_LLM/2024.08.05_LSLM.md) proposes to attach a streaming self-supervised learning (SSL) Encoder to an autoregressive token-based TTS Model.
[ViolLA (2023)](../../Models/Speech_LLM/2023.05.25_VioLA.md) introduced a multi-task auto-regressive codec language model to autoregressively generate codec tokens instead of speech unit tokens.

#### Continued Pre-Training.

In contrast to cold initialization, continued Pre-Training involves initializing the language model with pre-trained weights from a TextLM and then adapting it to handle speech tokens.
This approach leverages the linguistic knowledge embedded in TextLMs, allowing for more efficient and effective training of SpeechLMs.
Research by [TWIST (2023)](../../Models/Speech_LLM/2023.05.22_TWIST.md) found that starting with a textually pre-trained language model ([OPT (2022)](../../Models/LLM/2022.05.02_OPT.md) and [LLaMA (2023)](../../Models/LLM/2023.02.27_LLaMA.md)) can enhance the model's convergence rate and significantly improve its speech understanding capabilities.
They also demonstrated that while training from text-pretrained checkpoints outperforms cold initialization, training from image-pretrained checkpoints yields poorer results compared to cold initialization.
This indicates that not all pre-trained checkpoints are equally effective.
Additionally, [AudioPaLM (2023)](../../Models/Speech_LLM/2023.06.22_AudioPaLM.md) trained the SpeechLM using [PaLM (2022)](../../Models/LLM/2022.04.05_PaLM.md) and [PaLM-2 (2023)](../../Models/LLM/2023.05.17_PaLM2.md), showing that the SpeechLM benefits from both an increased size of the pre-trained checkpoint and a larger training dataset.

The performance of SpeechLMs can be further enhanced by **aligning** the text and speech modality representations.
[SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md) found that continually pretraining on TextLM checkpoints using interleaving text and speech tokens can significantly boost the model's performance on speech understanding and generation.
Additionally, their visualizations demonstrate that the similarity between text and speech features is notably higher in models trained with interleaved token sequences compared to those trained without this approach.
[AudioChatLLaMA (2023)](../../Models/Speech_LLM/2023.11.12_AudioChatLLaMA.md) aims to ensure that the model produces consistent outputs regardless of whether the input is text or speech.
They address this challenge by treating text data in ASR datasets as prompts, allowing LLaMA to generate the corresponding responses.
Consequently, both text and speech versions of the prompt can be utilized to train the model to provide the appropriate response.
[Spectron (2023)](../../Models/Speech_LLM/2023.05.24_Spectron.md) solves the text-speech representation alignment problem by jointly supervising multiple objectives.
Specifically, the input speech prompt is first transcribed into its text tokens, and then the model predicts the text token response.
Finally, the text response is synthesized to output speech.

### Language Model Instruction-Tuning

Instruction-tuning refers to the process of fine-tuning SpeechLMs to follow specific instructions to perform a wide range of tasks.
This phase is crucial for enhancing the pre-trained model's generalization capabilities and making it more adaptable to diverse applications.
Therefore, the key focus is on creating effective instruction-following datasets.

Several approaches have been proposed to construct instruction-following datasets for SpeechLMs.
[SpeechGPT (2023)](../../Models/Speech_LLM/2023.05.18_SpeechGPT.md) and [SpeechGPT-Gen (2024)](../../Models/Speech_LLM/2024.01.24_SpeechGPT-Gen.md) propose a two-stage instruction-tuning, including cross-modal instruction fine-tuning and chain-of-modality instruction fine-tuning.
In the first stage, instruction data are generated based on ASR datasets by appending the instruction to paired ASR data, asking the model to convert speech into text.
Similarly, paired data is also used to create instruction data for performing TTS.
In the second stage, they construct a speech-in-speech-out dataset by transforming a text-based instruction-following dataset using TTS.
[LLaMA-Omni (2024)](../../Models/MultiModal/2024.09.10_LLaMA-Omni.md) also creates instruction-following data by synthesizing text-based datasets, adhering to specific constraints.
First, they transform the input text prompt into a format that mimics natural speech patterns.
Next, they discard the original text response and employ a TextLM to generate answers to the converted prompts, ensuring these responses also follow natural speech patterns.
Finally, they synthesize the prompt/response pairs using TTS.
[COSMIC (2023)](../../Models/Speech_LLM/2023.11.03_COSMIC.md) constructed speech QA data by asking GPT-3.5 to generate question-answer pairs based on the transcriptions of English TED talk speeches.
They showed the model trained on their proposed speech QA dataset can generalize to unseen tasks such as speech-to-text translation using in-context learning.

## 4.3.Speech Generation Paradigm

In the previous sections, we discuss the typical generation paradigm for SpeechLMs, which involves taking a predefined input sequence and generating a complete response.
However, this approach does not reflect the natural flow of voice interactions.
For instance, during a conversation, one person may interrupt another, switching from listening to speaking.
Additionally, a person might choose not to respond if the other is engaged in a conversation with someone else.
Based on these observations, we identify two key aspects of advanced voice interaction skills for SpeechLMs: real-time interaction and silence mode.

### Real-time Interaction

Real-time Interaction refers to the capability of SpeechLMs to engage with users instantaneously.
This interaction consists of two key components:

- User Interruption: SpeechLMs should be able to be interrupted by users and should respond appropriately to new instructions provided during the conversation.
- Simultaneous Response: SpeechLMs should be capable of generating responses while the user is still speaking.

Both of these abilities require the model to effectively perform speech understanding (processing input) and speech generation (producing output) simultaneously.
The study by [dGSLM (2022)](../../Models/Speech_LLM/2022.03.30_dGSLM.md) introduces a dual-transformer architecture to model two-speaker dialogues, using one transformer to handle speech from each speaker.
A cross-attention transformer layer is included to capture the interactions between the speakers' content.
In contrast, [LSLM (2024)](../../Models/Speech_LLM/2024.08.05_LSLM.md) proposes a different approach, utilizing a single decoder-only Transformer to model one speaker's speech in the dialogue.
This model incorporates a streaming SSL encoder that continuously processes input from the listening channel and fuses its embeddings with those from the speaking channel.

### Silence Mode

Silence Mode refers to the state in which the SpeechLMs remain inactive or silent during periods of non-interaction.
This mode is essential for creating a natural conversational flow, allowing the model to avoid unnecessary interruptions.
It is crucial for situations where a small group of users is having a discussion, as the SpeechLM needs to discern when to join in and when to stay silent.
Additionally, it is important for the model to learn when to disregard instructions when users are not speaking at it.
[VITA (2024)](../../Models/MultiModal/2024.08.09_VITA.md) is currently the only work that integrates silence mode.
This method involves training the model on both query speech and non-query audio, which may include environmental sounds or non-query speech.
As a result, the model learns to output the **end-of-sequence** token to terminate its response when non-query audio is detected.
