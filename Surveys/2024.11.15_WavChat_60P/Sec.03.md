# 3·Representations of Spoken Dialogue Models

Representations play a critical role in spoken dialogue systems as they determine how the spoken dialogue system comprehends, processes, and generates speech signals.
Additionally, they serve as a bridge between speech and other modalities, thereby directly influencing the system’s performance, functionality, and range of applications.
Compared to text and visual representations, speech representations possess a unique complexity.
Text representations primarily rely on a well-defined symbolic system, conveying meaning through structured elements like vocabulary and syntax.
Visual representations, on the other hand, focus on capturing spatial relationships and visual features in images.
In contrast, speech signals contain both dynamic acoustic features (such as timbre, prosody and emotion) and rich semantic content, requiring representations that not only capture temporal variations but also preserve an understanding of the underlying meaning.

The unique nature of speech has led to the development of two types of representation models.
The representations obtained by these two modeling approaches are often classified as semantic tokens and acoustic tokens.
**One category (semantic) is prediction-based modeling**, these models are trained for representation learning by predicting future frames in an autoregressive manner \cite{chung2020vector, shain2020acquiring} or by using surrounding frames to predict masked frames \cite{chi2021audio, hsu2021hubert, liu2020mockingjay}.
This approach tends to prioritize capturing linguistic information within speech, making it particularly useful for recognition and understanding tasks.
**The other category (acoustic) focuses on speech compression and reconstruction** \cite{ji2024wavtokenizer, defossez2022high, kumar2024high, zeghidour2021soundstream}.
These models quantify speech features (which are downsampled from raw waveforms by one encoder) into a series of discrete tokens, then use one decoder to upsample these discrete tokens into the speech, calculating the reconstruction loss against the original signal.
By this approach, we can get discrete acoustic tokens with impressive compression rates and high-fidelity acoustic information, making it more suitable for tasks such as speech synthesis and emotion analysis.

In the spoken dialogue systems, as illustrated in Figure \ref{fig:img2}, different spoken dialogue models employ various approaches for representation selection.
In the following part, we will enumerate the commonly used speech representations in spoken dialogue models from both the input and output perspectives.
At the end of this section, we will thoroughly discuss the advantages and limitations of these representations, as well as the future trends in the development of representations used in spoken dialogue models.

## 3.1·Speech Representations at the Inputs

### Semantic

To enhance language models' ability to understand speech representations and align multimodal data at input, using pretrained models such as Wav2Vec \cite{schneider2019wav2vec}, HuBERT \cite{hsu2021hubert}, Whisper \cite{radford2023robust}, and WavLM \cite{chen2022wavlm} to extract high-level semantic features from speech has become a core strategy for many spoken dialogue systems.

#### Wav2Vec

Wav2Vec \cite{schneider2019wav2vec} is a foundational work in the field of speech representation learning, pioneering the extraction of self-supervised speech representations from unlabeled speech data.
This approach has driven technological advancements in tasks such as speech recognition, speaker identification, and other speech processing applications.
Wav2Vec employs a multi-layer, one-dimensional convolutional neural network directly on raw speech waveforms to progressively extract temporal speech features.
Training is accomplished through contrastive learning: the model selects a "correct" target (from the current speech frame) alongside several "incorrect" targets (negative samples).
By learning to distinguish positive samples from negatives, the model effectively learns to represent speech features in latent space.
As an improved version of Wav2Vec, Wav2Vec 2.0 \cite{baevski2020wav2vec} introduces the Transformer architecture and masked modeling.
Wav2Vec 2.0 quantizes the latent speech representations extracted by the CNN and then uses a Transformer to model semantic information, similar to BERT \cite{devlin2018bert}.
It also employs a contrastive learning objective, requiring the model to distinguish the correct quantized representations from multiple candidate representations.
ParalinGPT \cite{lin2024paralinguistics} aims to incorporate emotional expression in conversational interactions, choosing Wav2Vec 2.0 for its proven capability to encode rich prosodic information, beneficial for speech emotion recognition \cite{li2023exploration}.
Specifically, ParalinGPT uses Wav2Vec 2.0’s intermediate layer (the 12th layer) for frame-by-frame feature extraction, as this layer has shown optimal results in linear probing tasks for emotion analysis.
Additionally, ParalinGPT applies mean pooling and a linear feature projector to extract utterance embeddings.

#### XLS-R

XLS-R \cite{babu2021xls} is a multilingual self-supervised speech representation model based on the Wav2Vec 2.0 architecture.
It extends and optimizes Wav2Vec 2.0 to support a broader range of languages, particularly low-resource languages.
During cross-lingual training, XLS-R employs multilingual data augmentation and denoising techniques, enhancing the model's adaptability when processing speech in various languages.
USDM \cite{kim2024unified} uses XLS-R to obtain continuous intermediate representations at 50Hz, followed by a quantizer \cite{barrault2023seamless} with $K$=10000 to generate speech tokens.

#### HuBERT

HuBERT \cite{hsu2021hubert} is a commonly used unsupervised learning model that performs K-Means clustering on the MFCC \cite{zheng2001comparison} features of speech to assign pseudo-labels to each frame.
It uses a convolutional encoder to generate a sequence of features at a 20ms frame rate from 16kHz sampled speech.
Finally, it randomly masks a portion of features from consecutive frames as input to the Transformer \cite{vaswani2017attention}.
HuBERT generates masked content based on surrounding context, enabling it to capture temporal and semantic information within speech and gain a deeper understanding of contextual details.
Spoken dialogue systems, such as E-Chat \cite{xue2023chat}, SpeechGPT \cite{zhang2023speechgpt}, PSLM \cite{mitsui2024pslm}, IntrinsicVoice \cite{zhang2024intrinsicvoice}, widely use HuBERT as their speech encoder.
E-Chat extracts the weighted sum of the 24 layers from the HuBERT to serve as speech embeddings, and incorporates an additional set of weighted parameters to extract emotion embeddings, thereby enabling emotion-aware capabilities.
SpeechGPT applies K-Means clustering to quantize the continuous features extracted from HuBERT, converting them into discrete unit sequences.
These discrete units are then integrated into the vocabulary of the large language model, enabling direct alignment between the text and speech modalities.
To more effectively integrate the language model with speech streams, PSLM adds an additional embedding layer after extracting features with HuBERT.
IntrinsicVoice uses HuBERT as the speech tokenizer, grouping speech tokens to reduce sequence length.
An embedding layer then converts these tokens into dense embeddings, which are subsequently mapped into the language model's embedding space using a trainable speech adapter.
Spirit-LM \cite{nguyen2024spirit} extracts semantic features using HuBERT, employing a K-Means model with 500 units as the basic unit.
It trains a feedforward quantizer with data augmentation techniques \cite{gat2022augmentation} to produce discrete speech tokens.
In the Align-SLM \cite{lin2024alignslmtextlessspokenlanguage}, HuBERT is used and the cluster number K is set to 500.
Notably, when continuous representations are clustered into discrete units, they primarily capture content information, which can be leveraged for modeling and understanding.
This process first extracts 25Hz frame-level continuous representations from the 11-th layer of the HuBERT model, assigns each frame to its closest cluster index, and then de-duplicates consecutive identical indices to shorten the sequence.

#### Whisper

Whisper \cite{radford2023robust}, based on the classic encoder-decoder architecture, has gained widespread attention in the field of speech recognition.
The encoder transforms input speech into high-level feature representations, while the decoder generates the corresponding text output from these representations.
Pretrained on large-scale data across various speech environments with text as the target, Whisper demonstrates strong capabilities in extracting semantic information from speech.
Qwen-Audio \cite{chu2023qwen}, Qwen-Audio 2 \cite{chu2024qwen2} use Whisper’s encoder to convert speech into continuous representations, which are then combined with text representations and fed into the large language model.
Mini-Omni \cite{xie2024mini}, Mini-Omni 2 \cite{xie2024miniomni2opensourcegpt4ovision}, and LLama-Omni \cite{fang2024llama} follow a similar approach, connecting a speech adapter after the Whisper encoder.
Their shared objective is to map speech representations into the text embedding space of the large language model, enhancing the model's ability to understand speech by forcibly aligning them through vocabulary expansion.

#### WavLM

WavLM \cite{chen2022wavlm} is a pretrained model designed for comprehensive speech processing tasks, playing a critical role in advancing speech technology.
Specifically, WavLM employs a masked speech denoising and prediction framework, where some inputs consist of simulated noise or overlapping speech with masked sections.
The goal is to predict pseudo-labels of the original speech in the masked areas.
This approach enables the model to learn ASR-related information through masked speech prediction, while also gaining knowledge relevant to non-ASR tasks through speech denoising modeling.
The masking and prediction pipeline for speech frames in WavLM is similar to that of HuBERT.
However, WavLM introduces an additional gated relative position bias to enhance the model's sensitivity to temporal information in speech.
SpeechVerse \cite{das2024speechverse} leverages the pretrained WavLM Large as its backbone speech encoder, encoding all intermediate layer features from WavLM to capture various forms of semantics and achieve better generalization performance.
To address the significant length disparity between speech features and text tokens, SpeechVerse applies a learnable convolutional module for downsampling the speech features.

#### $S^3$ Tokenizer

CosyVoice \cite{du2024cosyvoice} proposes using a supervised automatic speech recognition module to generate a supervised semantic speech($S^3$) tokenizer.
Unlike a standard ASR model, the $S^3$ tokenizer splits the encoder into two parts and introduces a vector quantization layer in between.
The first encoder converts the mel spectrogram into context-aware representations, while the second encoder transforms discrete speech units into continuous hidden states.
Finally, a Transformer-based ASR decoder predicts the posterior probabilities of text labels.
Through supervision in multilingual ASR tasks, the $S^3$ tokenizer can convert speech into semantically consistent tokens that facilitate both speech understanding and generation.
OmniFlatten \cite{zhang2024omniflatten} uses the $S^3$ tokenizer to extract discrete speech tokens, which are then directly fed into a text-speech pre-trained Transformer.

#### SPIRAL

SPIRAL \cite{huang2022spiral} aims to learn representations from speech data that are robust to noise and perturbations.
It uses a teacher-student network, where various perturbations—such as noise addition, gain adjustment, and time-frequency warping—are applied to the speech input of the student model.
The teacher model then guides the student model to produce consistent representations despite these perturbations.
EMOVA \cite{chen2024emova} utilizes the SPIRAL’s architecture as a speech encoder to process speech, and employs the finite scalar quantization \cite{mentzer2023finite} to discretize these features.
This process aligns speech with the text vocabulary, allowing for a more natural integration into the LLM.

#### Others

Some spoken dialogue systems do not use pre-trained representation models; instead, they process input features by stacking fundamental modules.
VITA \cite{fu2024vita} initially decomposes the speech signal using mel filter banks, mimicking the nonlinear perception of sound in humans.
It then processes the input features with a 4-layer CNN downsampling module followed by a 24-layer Transformer.
To align with the subsequent language model, VITA employs a simple 2-layer MLP as an adapter.
Freeze-Omni \cite{xiong2024freeze} utilizes a chunk-wise streaming speech encoder to transform input speech features into high-dimensional representations.
An adapter module then maps these high-dimensional representations into the embedding space of the main LLM, ensuring a quick, low-latency response to the input speech.
The speech encoder module consists of several downsampling convolutional layers and Transformer blocks, while the adapter includes only a few downsampling convolutional layers.
Downsampling layers are used to reduce the frame rate of speech features, increase the LLM's processing speed during the prefill phase, and minimize latency.

### Acoustic

Considering that semantic features are insufficient to capture the emotion, timbre, and style of speech, some representation models, such as Emotion2Vec \cite{ma2023emotion2vec}, attempt to extract acoustic information through self-supervised training.
Others focus on reconstruction objectives to ensure high-fidelity speech, including models like Encodec \cite{defossez2022high}, SpeechTokenizer~\cite{zhang2023speechtokenizer}, Mimi \cite{defossez2024moshi}.

#### EnCodec

EnCodec \cite{defossez2022high} is a straightforward, streaming, convolution-based encoder-decoder architecture.
Raw speech is downsampled through a series of convolutional layers, mapping it to latent feature representations.
Residual vector quantization \cite{zeghidour2021soundstream} then discretizes the encoder’s continuous latent features.
The quantization objective is to map continuous features to a predefined set of discrete tokens (known as a "codebook") for subsequent compression and transmission.
The decoder restores the discrete features to a waveform close to the original speech through a series of de-convolution layers.
LauraGPT \cite{du2023lauragpt} employs an enhanced version of EnCodec as its speech encoder with specific modifications: (1) adding a reconstruction loss in the magnitude spectral domain to improve mid-to-high frequency signal quality; (2) stacking five strided convolutional blocks with strides of (8, 5, 4, 2, 2) to address the challenges of long sequence lengths, resulting in a token rate of 25Hz per token group; and (3) using 32 quantizers with structured dropout in the Residual Vector Quantization (RVQ) module, each with a vocabulary size of 1024.
This revision increases speech quality by incorporating more quantizers while preserving most information in the shallow quantizers.
LauraGPT ultimately selects the output from the first quantizer layer as the speech token, balancing performance with sequence length efficiency.
The remaining quantizers are used only during the training of the encoder-decoder model.

#### SpeechTokenizer

SpeechTokenizer \cite{zhang2023speechtokenizer} unifies semantic and acoustic tokens, hierarchically decomposing different aspects of speech information across various RVQ layers.
It is built on the framework of RVQ-GANs, following the same pattern as SoundStream \cite{zeghidour2021soundstream} and EnCodec \cite{defossez2022high}.
Notably, SpeechTokenizer has substituted the two-layer LSTM, originally following the convolution blocks in the EnCodec encoder, with a two-layer BiLSTM to augment the semantic modeling ability.
SpeechTokenizer uses HuBERT as a semantic teacher, given HuBERT’s proven capacity to encode substantial content information \cite{mohamed2022self}.
During training, it introduces two types of distillation: continuous representation distillation and pseudo-label prediction.
For continuous representation distillation, SpeechTokenizer employs the 9th layer HuBERT representation or the average representation across all HuBERT layers as semantic teachers.
The training objective is to maximize the cosine similarity at the dimension level across all timesteps between the outputs of RVQ first layer and semantic teacher representations.
For pseudo-label prediction, SpeechTokenizer adopts HuBERT units as the target label.
In dialogue systems, SpeechGPT-Gen uses SpeechTokenizer RVQ-1 to process raw speech, primarily enhancing the large language model's ability to model the semantics of speech.

#### Mimi

Taking inspiration from previous work on SpeechTokenizer, Mimi \cite{defossez2024moshi} uses distillation to transfer non-causal, high-level semantic information into the tokens produced by a causal model, allowing for streaming encoding and decoding of semantic-acoustic tokens.
To improve the ability of Mimi to encode speech into compact representations while reconstructing high-quality speech, Transformer modules are added in the encoder and decoder.
Mimi uses WavLM to distill RVQ-1, enriching it with semantic information.
Notably, performing distillation significantly enhances the speech discrimination capability of the first quantizer; however, it can also negatively impact speech quality.
Mimi hypothesizes that this is due to distilling semantic information into the first level of a single RVQ: As higher-order quantizers operate on the residual of the first one, the latter needs to trade speech quality for phonetic discriminability.
Mimi addresses this issue by introducing a split-RVQ approach.
Instead of using a single 8-level RVQ, it extracts semantic information into a simple VQ and applies a parallel 7-level RVQ, combining their outputs at the end.
This removes the constraint that acoustic information must be preserved in the residuals of the semantic quantizer.
After careful design, Mimi serves as the speech encoder in Moshi \cite{defossez2024moshi}, this approach enhances the model's ability to capture both semantic and acoustic details.

#### Emotion2Vec

Emotion2Vec \cite{ma2023emotion2vec} is a versatile speech emotion representation model designed to extract emotional features from speech.
During the pre-training phase, Emotion2Vec conducts online distillation with a teacher network and a student network.
When a specific downstream task is performed, Emotion2Vec is frozen and a lightweight downstream model is trained.
Emotion2Vec introduces an utterance-level loss to control global emotion and employs a frame-level loss to build a frame-wise pretext task, enabling it to learn contextual emotions.
Spoken-LLM~\cite{lin2024advancing} uses features extracted by Emotion2Vec as input for the large language model, aiming to enable the model to understand and respond to emotions.

## 3.2·Speech Representations at the Outputs

### Semantic

At the output stage, Most spoken dialogue systems choose to autoregressively model semantic tokens, such as $S^3$ tokens \cite{du2024cosyvoice} and HuBERT \cite{hsu2021hubert} units.
It is worth noting that these semantic tokens lack acoustic conditioning and therefore require a vocoder \cite{kong2020hifi, polyak2021speech} or decoder, which futher takes semantic discrete units as input to synthesize speech consistent with the speakers encountered during training.

#### $S^3$ Tokenizer

OmniFlatten \cite{zhang2024omniflatten} uses the LLM to autoregressively predict $S^3$ tokens at the speech output stage.
When converting discrete tokens back into speech, it adopts the same optimal transport conditional flow matching model (OT-CFM) as used in CosyVoice \cite{du2024cosyvoice}.
OT-CFM transforms the speech token sequence into Mel spectrogram, which is then used to generate the final speech with the HiFi-GAN vocoder \cite{kong2020hifi}.

#### HuBERT

Speech tokens extracted by the pre-trained HuBERT \cite{hsu2021hubert} are widely used as generation targets for large language models in the spoken dialogue systems.
SpeechGPT \cite{zhang2023speechgpt} and Spirit-LM \cite{nguyen2024spirit} use LLaMA \cite{touvron2023llama} to autoregressively predict a sequence of units and are trained with a HuBERT unit-based HiFi-GAN \cite{kong2020hifi} to decode the speech signal from discrete representations.
PSLM \cite{mitsui2024pslm} introduces an additional speech projection layer after the Transformer layers to process the hidden states, obtaining semantic tokens via the softmax layler.
The speech decoder in LLama-Omni \cite{fang2024llama} operates in a non-autoregressive manner, taking the output hidden states of the large language model as input to generate a discrete HuBERT unit sequence corresponding to the speech response.
The discrete units can be converted into waveform with an additional unit-based vocoder \cite{polyak2021speech}.
IntrinsicVoice \cite{zhang2024intrinsicvoice} introduces Group-Former to enhance the large language model’s capability in sequence modeling.
When the large language model predicts the $<speech>$ token, the global embedding is passed through a projection layer and delivered, along with a set of learnable queries, to the group model, which then predicts units.
IntrinsicVoice uses HiFi-GAN \cite{kong2020hifi}, a non-autoregressive neural vocoder that efficiently generates high-fidelity waveforms, for speech detokenization to reduce overall latency.
Align-SLM \cite{lin2024alignslmtextlessspokenlanguage} also uses a HiFiGAN-based \cite{kong2020hifi} model to convert discrete units back into waveforms, utilizing model checkpoints from the textlesslib \cite{kharitonov2022textless} library.

#### Others

USDM \cite{kim2024unified} does not generate speech directly from input speech; instead, it first transcribes the speech, generates the response text, and then produces corresponding speech token in an end-to-end pipeline.
By inserting text-related tasks between speech input and output, the model benefits from both pre-trained LLMs and chain-of-thought \cite{wei2022chain} reasoning in the intermediate modality.
Since each stage in the pipeline processes all input and output tokens generated by the previous stage.
USDM is more robust to transcription errors and better able to produce contextually relevant spoken responses compared to a cascaded approach with separate modules.
USDM uses the Voicebox \cite{le2024voicebox} architecture to train a unit-to-speech model for reconstructing speech from units.
EMOVA \cite{chen2024emova} generates a response in the form of speech units when given an image or speech input, which is then converted into an output waveform using the U2S detokenizer.
The U2S detokenizer follows the VAE architecture: it uses a speech unit encoder to convert the predicted speech units into continuous embeddings, combines these with style embeddings predicted by the large language model to determine duration, and finally reconstructs the speech waveform through the decoder.

### Acoustic

Many spoken dialogue systems choose to directly generate tokens from acoustic representation models, such as EnCodec \cite{defossez2022high}, SpeechTokenizer \cite{zhang2023speechtokenizer}, and Mimi \cite{defossez2024moshi}.
These acoustic tokens are then upsampled into the raw waveform through the frozen codec decoder directly.

#### EnCodec

LauraGPT \cite{du2023lauragpt} uses Qwen-1.8B \cite{bai2023qwen} to predict speech tokens.
When synthesizing speech, it conditions the predictor not only on the speech tokens predicted by the LLM but also on text and speech inputs.
Such text and speech conditionings allow the model to generate high-quality speech signals by leveraging the diverse information in prompt and noisy speeches, which is lacked in the discrete tokens (output from the first quantizer of the Encodec).
The predicted speech tokens and conditioning inputs are delivered together to the codec vocoder.
An encoder-only Transformer models these inputs into dense embeddings, which are then reconstructed into speech by the codec decoder.

#### SNAC

SNAC \cite{siuzdak2024snac} encodes speech into hierarchical tokens, similar to EnCodec \cite{defossez2022high} and DAC \cite{kumar2024high}, by introducing quantization at different time resolutions to form a multi-scale discrete representation of speech.
In this approach, shallow RVQ layers have a lower sampling frequency, covering a broader time span, while deeper RVQ layers sample at higher frequencies.
SNAC introduces modest enhancements over RVQ-GAN by incorporating residual noise blocks, deep convolutions, and local window attention.
The Mini-Omni \cite{xie2024mini, xie2024miniomni2opensourcegpt4ovision} series continues the parallel generation method introduced by MusicGen\cite{copet2024simple}, utilizing SNAC \cite{siuzdak2024snac} as the speech encoder, which comprises seven complementary token layers.
In a single step, it generates eight tokens, including text, while maintaining a one-step delay between layers.
Furthermore, Mini-Omni and Mini-Omni 2 incorporates a batch approach that involves two samples: one requiring both text and speech responses and the other necessitating a text-only response.
By discarding the text token from the first sample and embedding the output from the second sample into the first, it effectively transfer the model’s text-based capabilities to speech tasks, significantly enhancing reasoning abilities with minimal resource overhead.

#### SpeechTokenizer

On the output side, SpeechGPT-Gen synthesizes speech tokens using flow matching\cite{lipman2022flow}.
Flow matching effectively models the transformation from a simple prior distribution to complex data distributions, yielding promising results in speech generation.
SpeechGPT-Gen \cite{zhang2024speechgpt} applies flow matching for perceptual modeling, generating speech tokens that align with those of SpeechTokenizer \cite{zhang2023speechtokenizer}.
Specifically, given speech $S$, semantic representation $V_1$, perceptual representation $V_{2:8}$ and the complete information representation $V_{1:8} = V_1 + V_{2:8}$ extracted by SpeechTokenizer, perceptual modeling refers to predicting the complete representation $V_{1:8}$ given the prompt speech a and the semantic representation $V_1$.
SpeechGPT-Gen synthesizes response speech by concatenating the output of SpeechGPT \cite{zhang2023speechgpt} with the prompt speech and using a flow matching model.

#### Mimi

Mimi \cite{defossez2024moshi} has eight codebooks at a frame rate of 12.5Hz, which requires 100 autoregressive steps to generate one second speech.
This results in high computational costs and incompatibility with streaming inference.
To address these issues, Moshi \cite{defossez2024moshi} proposes the RQ-Transformer, comprising a temporal Transformer and a deep Transformer.
The RQ-Transformer breaks down a flattened sequence of length $K \cdot S$ into $S$ timesteps for a large temporal Transformer which produces a context embedding used to condition a smaller depth Transformer over $K$ steps.
This allows scaling to longer sequences by increasing $S$ or to a higher depth by increasing $K$ than modeling the flattened sequence with a single model.

#### TiCodec

Ti-Codec~\cite{ren2024fewer} is a decoupled codec model which can separate the time-varying and time-invariant information in speech and quantize them separately.
Inspired by VALL-E \cite{wang2023neural}, Freeze-Omni \cite{xiong2024freeze} uses a token-based speech decoder which contains NAR prefill and AR generate stage to achieve speech output capabilities.
The speech decoder mainly consists of the NAR decoder, the AR decoder, and the frozen decoder of a codec model \cite{ren2024fewer}.
Both the NAR decoder and AR decoder are built upon transformer blocks.
The NAR decoder is used to model the semantic features from the output of LLM, and then the AR decoder generates speech tokens based on the output of the NAR decoder.
Finally, the decoder of the codec model converts the speech tokens into a speech stream.
