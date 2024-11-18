# 5·Speech Processing Tasks

In recent times, the field of speech processing has gained significant attention due to its rapid evolution and its crucial role in modern technological applications.
This field involves the use of diverse techniques and algorithms to analyse and understand spoken language, ranging from basic speech recognition to more complex tasks such as spoken language understanding and speaker identification.
Since speech is one of the most natural forms of communication, speech processing has become a critical component of many applications such as virtual assistants, call centres, and speech-to-text transcription.
In this section, we provide a comprehensive overview of the various speech-processing tasks and the techniques used to achieve them, while also discussing the current challenges and limitations faced in this field and its potential for future development.

The assessment of speech-processing models depends greatly on the calibre of datasets employed.
By utilizing standardized datasets, researchers are enabled to objectively gauge the efficacy of varying approaches and identify scopes for advancement.
The selection of evaluation metrics plays a critical role in this process, hinging on the task at hand and the desired outcome.
Therefore, it is essential that researchers conduct a meticulous appraisal of different metrics to make informed decisions.
This paper offers a thorough summary of frequently utilized datasets and metrics across diverse downstream tasks, as presented in \Cref{dataset} and, \Cref{objective}.

## 5.1·Automatic Speech Recognition (ASR) \& Conversational Multi-Speaker AST

### 5.1.1·Task Description

Automatic speech recognition (ASR) technology enables machines to convert spoken language into text or commands, serving as a cornerstone of human-machine communication and facilitating a wide range of applications such as speech-to-speech translation and information retrieval \cite{lu2020automatic}.
ASR involves multiple intricate steps, starting with the extraction and analysis of acoustic features, including spectral and prosodic features, which are then employed to recognize spoken words.
Next, an acoustic model matches the extracted features to phonetic units, while a language model predicts the most probable sequence of words based on the recognized phonetic units.
Ultimately, the acoustic and language model outcomes are merged to produce the transcription of spoken words.
Deep learning techniques have gained popularity in recent years, allowing for improved accuracy in ASR systems \cite{baevski2020wav2vec,radford2022robust}.
This paper provides an overview of the key components involved in ASR and highlights the role of deep learning techniques in enhancing the technology's accuracy.

Most speech recognition systems that use deep learning aim to simplify the processing pipeline by training a single model to directly map speech signals to their corresponding text transcriptions.
Unlike traditional ASR systems that require multiple components to extract and model features, such as HMMs and GMMs, end-to-end models do not rely on hand-designed components \cite{audhkhasi2019forget,li2022recent}.
Instead, end-to-end ASR systems use DNNs to learn acoustic and linguistic representations directly from the input speech signals \cite{li2022recent}.
One popular type of end-to-end model is the encoder-decoder model with attention.
This model uses an encoder network to map input audio signals to hidden representations, and a decoder network to generate text transcriptions from the hidden representations.
During the decoding process, the attention mechanism enables the decoder to selectively focus on different parts of the input signal \cite{li2022recent}.

End-to-end ASR models can be trained using various techniques such as CTC \cite{Karita2019ImprovingTE}, which is used to train models without explicit alignment between the input and output sequences, and RNNs, which are commonly used to model temporal dependencies in sequential data such as speech signals.
Transfer learning-based approaches can also improve end-to-end ASR performance by leveraging pre-trained models or features \cite{liu2023towards,deng2022improving,sertolli2021representation}.
While end-to-end ASR models have shown promising results in various applications, there is still room for improvement to achieve human-level performance \cite{liu2023towards,yoon2022hubert,kanda2022streaming,kanda2022transcribe,deng2022improving,fazel2021synthasr}.
Nonetheless, deep learning-based end-to-end ASR architecture offers a promising and efficient approach to speech recognition that can simplify the processing pipeline and improve recognition accuracy.

### 5.1.2·Datasets

The development and evaluation of ASR systems are heavily dependent on the availability of large datasets.
As a result, ASR is an active area of research, with numerous datasets used for this purpose.
In this context, several popular datasets have gained prominence for use in ASR systems.

- Common Voice: Mozilla's Common Voice project \cite{ardila2019common} is dedicated to producing an accessible, unrestricted collection of human speech for the purpose of training speech recognition systems.
This ever-expanding dataset features contributions from more than $9,000$ speakers spanning $60$ different languages.
- LibriSpeech: LibriSpeech \cite{panayotov2015librispeech} is a corpus of approximately 1,000 hours of read English speech created from audiobooks in the public domain.
It is widely used for speech recognition research and is notable for its high audio quality and clean transcription.
- VoxCeleb: VoxCeleb \cite{Nagrani17} is a large-scale dataset containing over 1 million short audio clips of celebrities speaking, which can be used for speech recognition and recognition research.
It includes a diverse range of speakers from different backgrounds and professions.
- TIMIT: The TIMIT corpus \cite{garofolo1993timit} is a widely used speech dataset consisting of recordings consisting of 630 speakers representing eight major dialects of American English, each reading ten phonetically rich sentences.
It has been used as a benchmark for speech recognition research since its creation in 1986.
- CHiME-5: The CHiME-5 dataset \cite{barker2018fifth} is a collection of recordings made in a domestic environment to simulate a real-world speech recognition scenario.
It includes 6.5 hours of audio from multiple microphone arrays and is designed to test the performance of ASR systems in noisy and reverberant environments.

Other notable datasets include Google's Speech Commands Dataset \cite{warden2018speech}, the Wall Street Journal dataset\footnote{https://www.ldc.upenn.edu/}, and TED-LIUM \cite{rousseau2012ted}.

### 5.1.3·Models

The use of RNN-based architecture in speech recognition has many advantages over traditional acoustic models.
One of the most significant benefits is their ability to capture long-term temporal dependencies \cite{karita2019comparative} in speech data, enabling them to model the dynamic nature of speech signals.
Additionally, RNNs can effectively process variable-length audio sequences, which is essential in speech recognition tasks where the duration of spoken words and phrases can vary widely.
RNN-based models can efficiently identify and segment phonemes, detect and transcribe spoken words, and can be trained end-to-end, eliminating the need for intermediate steps.
These features make RNN-based models particularly useful in real-time applications, such as speech recognition in mobile devices or smart homes \cite{dong2020rtmobile,he2019streaming}, where low latency and high accuracy are crucial.

In the past, RNNs were the go-to model for ASR.
However, their limited ability to handle long-range dependencies prompted the adoption of the Transformer architecture.
For example, in 2019, Google's Speech-to-Text API transitioned to a Transformer-based architecture that surpassed the previous RNN-based model, especially in noisy environments and for longer sentences, as reported in \cite{zhang2020transformer}.
Additionally, Facebook AI Research introduced wav2vec 2.0, a self-supervised learning approach that leverages a Transformer-based architecture to perform unsupervised speech recognition.
wav2vec 2.0 has significantly outperformed the previous RNN-based model and achieved state-of-the-art results on several benchmark datasets.

Transformer for the ASR task is first proposed in \cite{8462506}, where authors include CNN layers before submitting preprocessed speech features to the input.
By incorporating more CNN layers, it becomes feasible to diminish the gap between the sizes of the input and output sequences, given that the number of frames in audio exceeds the number of tokens in text.
This results in a favorable impact on the training process.
The change in the original architecture is minimal, and the model achieves a competitive word error rate (WER) of $10.9\%$ on the Wall Street Journal (WSK) speech recognition dataset (Table \ref{performance:ASR}).
Despite its numerous advantages, Transformers in its pristine state has several issues when applied to ASR.
RNN, with its overall training speed (i.e., convergence) and better WER because of effective joint training and decoding methods, is still the best option.

The authors in \cite{8462506} propose the Speech Transformer, which has the advantage of faster iteration time, but slower convergence compared to RNN-based ASR.
However, integrating the Speech Transformer with the naive language model (LM) is challenging.
To address this issue, various improvements in the Speech Transformer architecture have been proposed in recent years.
For example, \cite{Karita2019ImprovingTE} suggests incorporating the Connectionist Temporal Classification (CTC) loss into the Speech Transformer.
CTC is a popular technique used in speech recognition to align input and output sequences of varying lengths and one-to-many or many-to-one mappings.
It introduces a blank symbol representing gaps between output symbols and computes the loss function by summing probabilities across all possible paths.
The loss function encourages the model to assign high probabilities to correct output symbols and low probabilities to incorrect output symbols and the blank symbol, allowing the model to predict sequences of varying lengths.
The CTC loss is commonly used with RNNs such as LSTM and GRU, which are well-suited for sequential data.
CTC loss is a powerful tool for training neural networks to perform sequence-to-sequence tasks where the input and output sequences have varying lengths and mappings between them are not one-to-one.

Various other improvements have also been proposed to enhance the performance of Speech Transformer architecture and integrate it with the naive language model, as the use of the transformer directly for ASR has not been effective in exploiting the correlation among the speech frames.
The sequence order of speech, which the recurrent processing of input features can represent, is an important distinction.
The degradation in performance for long sentences is reported using absolute positional embedding (AED) \cite{chorowski2015attention}.
The problems associated with long sequences can become more acute for transformer \cite{zhou2019improving}.
To address this issue, a transition was made from absolute positional encoding to relative positional embeddings  \cite{zhou2019improving}.
Whereas authors in  \cite{tsunoo2019transformer} replace positional embeddings with pooling layers.
In a considerably different approach, the authors in \cite{mohamed2019transformers} propose a novel way of combining positional embedding with speech features by replacing positional encoding with trainable convolution layers.
This update further improves the stability of
optimization for large-scale learning of transformer networks.
The above works confirmed the superiority of their techniques against sinusoidal positional encoding.

In 2016, Baidu introduced a hybrid ASR model called Deep Speech 2 \cite{amodei2016deep}  that uses both RNNs and Transformers.
The model also uses CNNs to extract features from the audio signal, followed by a stack of RNNs to model the temporal dependencies and a Transformer-based decoder to generate the output sequence.
This approach achieved state-of-the-art results on several benchmark datasets such as LibriSpeech, VoxForge, WSJeval92 etc.
The transition of ASR models from RNNs to Transformers has significantly improved performance, especially for long sentences and noisy environments.

The Transformer architecture has been widely adopted by different companies and research groups for their ASR models, and it is expected that more organizations will follow this trend in the upcoming years.
One of the advanced speech models that leverage this architecture is the Universal Speech Model (USM) \cite{zhang2023google} developed by Google, which has been trained on over 12 million hours of speech and 28 billion sentences of text in more than 300 languages.
With its 2 billion parameters, USM can recognize speech in both common languages like English and Mandarin and less-common languages.
Other popular acoustic models for speech recognition include Quartznet \cite{kriman2020quartznet}, Citrinet \cite{majumdar2021citrinet}, and Conformer \cite{gulati2020conformer}.
These models can be chosen and switched based on the specific use case and performance requirements of the speech recognition pipeline.
For example, Conformer-based acoustic models are preferred for addressing robust ASR, as shown in a recent study.
Another study found that Conformer-1\footnote{https://www.assemblyai.com/blog/conformer-1/} is more effective in handling real-world data and can produce up to $43\%$ fewer errors on noisy data than other popular ASR models.
Additionally, fine-tuning pre-trained models such as BERT  \cite{devlin2018bert} and GPT \cite{radford2018improving} has been explored for ASR tasks, leading to state-of-the-art performance on benchmark datasets like LibriSpeech (refer to Table \ref{performance:ASR}).
An open-source toolkit called Vosk\footnote{https://alphacephei.com/vosk/lm} provides pre-trained models for multiple languages optimized for real-time and efficient performance, making it suitable for applications that require such performance.

The field of speech recognition has made significant progress by adopting unsupervised pre-training techniques, such as those utilized by Wav2Vec 2.0 \cite{baevski2020wav2vec}.
Another recent advancement in automatic speech recognition (ASR) is the whisper model, which has achieved human-level accuracy when transcribing the LibriSpeech dataset.
These two cutting-edge frameworks, Wav2Vec 2.0 and whisper, currently represent the state-of-the-art in ASR.
The whisper model is trained on an extensive supervised dataset, including over 680,000 hours of audio data collected from the web, which has made it more resilient to various accents, background noise, and technical jargon.
The whisper model is also capable of transcribing and translating audio in multiple languages, making it a versatile tool.
OpenAI has released inference models and code, laying the groundwork for the development of practical applications based on the whisper model.

In contrast to its predecessor, Wav2Vec 2.0 is a self-supervised learning framework that trains models on unlabeled audio data before fine-tuning them on specific datasets.
It uses a contrastive predictive coding (CPC) loss function to learn speech representations directly from raw audio data, requiring less labeled data.
The model's performance has been impressive, achieving state-of-the-art results on several ASR benchmarks.
These advances in unsupervised pre-training techniques and the development of novel ASR frameworks like Whisper and Wav2Vec 2.0 have greatly improved the field of speech recognition, paving the way for new real-world applications.
In summary, the \Cref{w2vandwhisper} highlights the varying effectiveness of wav2vec2.0 large and whisper models across different datasets.

## 5.2·Neural Speech Synthesis

### 5.2.1·Task Description

Neural speech synthesis is a technology that utilizes artificial intelligence and deep learning techniques to create speech from text or other inputs.
Its applications are widespread, including in healthcare, where it can be used to develop assistive technologies for those who are unable to communicate due to neurological impairments.
To generate speech, deep neural networks like CNNs, RNNs, transformers, and diffusion models are trained using phonemes and the mel spectrum.
The process involves several components, such as text analysis, acoustic models, and vocoders, as shown in \Cref{fig:TTS}.
Acoustic models convert linguistic features into acoustic features, which are then used by the vocoder to synthesize the final speech signal.
Various architectures, including neural vocoders based on GANs like HiFi-GAN \cite{kong2020hifi}, are used by the vocoder to generate speech.
Neural speech synthesis also enables manipulation of voice, pitch, and speed of speech signals using frameworks such as Fastspeech2  \cite{ren2020fastspeech} and NANSY/NANSY++ \cite{choi2021neural,choi2022nansy++}.
These frameworks use information bottleneck to disentangle analysis features for controllable synthesis.
The research in neural speech synthesis can be classified into two prominent approaches: autoregressive and non-autoregressive models.
Autoregressive models generate speech one element at a time, sequentially, while non-autoregressive models generate all the elements simultaneously, in parallel.
\Cref{TTS:Landscape} outlines the different architecture proposed under each category.

The evaluation of synthesized speech is of paramount importance for assessing its quality and fidelity.
It serves as a means to gauge the effectiveness of different speech synthesis techniques, algorithms, and parameterization methods.
In this regard, the application of statistical tests has emerged as a valuable approach to objectively measure the similarity between synthesized speech and natural speech \cite{franco2019application}.
These tests complement the traditional Mean Opinion Score (MOS) evaluations and provide quantitative insights into the performance of speech synthesis systems.
Additionally, widely used objective metrics such as Mel Cepstral Distortion (MCD) and Word Error Rate (WER) contribute to the comprehensive evaluation of synthesized speech, enabling researchers and practitioners to identify areas for improvement and refine the synthesis process.
By employing these objective metrics and statistical tests, the evaluation of synthesized speech becomes a rigorous and systematic process, enhancing the overall quality and fidelity of speech synthesis techniques.

### 5.2.2·Datasets

The field of neural speech synthesis is rapidly advancing and relies heavily on high-quality datasets for effective training and evaluation of models.
One of the most frequently utilized datasets in this field is the LJ Speech \cite{ljspeech17}, which features about $24$ hours of recorded speech from a single female speaker reading passages from the public domain LJ Speech Corpus.
This dataset is free and has corresponding transcripts, making it an excellent choice for text-to-speech synthesis tasks.
Moreover, it has been used as a benchmark for numerous neural speech synthesis models, including Tacotron \cite{wang2017tacotron}, WaveNet \cite{oord2016wavenet}, and DeepVoice \cite{arik2017deep,gibiansky2017deep}.

Apart from the LJ Speech dataset, several other datasets are widely used in neural speech synthesis research.
The CMU Arctic \cite{kominek2004cmu}  and L2 Arctic \cite{zhao2018l2} datasets contain recordings of English speakers with diverse accents reading passages designed to capture various phonetic and prosodic aspects of speech.
The LibriSpeech \cite{panayotov2015librispeech}, VoxCeleb \cite{Nagrani17}, TIMIT Acoustic-Phonetic Continuous Speech Corpus \cite{garofolo1993timit}, and Common Voice Dataset \cite{ardila2019common} are other valuable datasets that offer ample opportunities for training and evaluating text-to-speech synthesis models.

### 5.2.3·Models

Neural network-based text-to-speech (TTS) systems have been proposed using neural networks as the basis for speech synthesis, particularly with the emergence of deep learning.
In Statistical Parametric Speech Synthesis (SPSS), early neural models replaced HMMs for acoustic modeling.
The first modern neural TTS model, WaveNet \cite{oord2016wavenet}, generated waveforms directly from linguistic features.
Other models, such as DeepVoice 1/2 \cite{arik2017deep,gibiansky2017deep}, used neural network-based models to follow the three components of statistical parametric synthesis.
End-to-end models, including Tacotron 1 \& 2 \cite{wang2017tacotron,shen2018natural}, Deep Voice 3, and FastSpeech 1 \& 2  \cite{ren2019fastspeech,ren2020fastspeech}, simplified text analysis modules and utilized mel-spectrograms to simplify acoustic features with character/phoneme sequences as input.
Fully end-to-end TTS systems, such as ClariNet \cite{ping2018clarinet}, FastSpeech 2 \cite{ren2020fastspeech}, and EATS \cite{donahueend}, are capable of directly generating waveforms from text inputs.
Compared to concatenative synthesis~\footnote{https://en.wikipedia.org/wiki/Concatenative\_synthesis} and statistical parametric synthesis, neural network-based speech synthesis offers several advantages including superior voice quality, naturalness, intelligibility, and reduced reliance on human preprocessing and feature development.
Therefore, end-to-end TTS systems represent a promising direction for advancing the field of speech synthesis.

Transformer models have become increasingly popular for generating mel-spectrograms in TTS systems \cite{ren2020fastspeech,li2019neural}.
These models are preferred over RNN structures in end-to-end TTS systems because they improve training and inference efficiency \cite{ren2019fastspeech,li2019neural}.
In a study conducted by \citet{li2019neural}, a multi-head attention mechanism replaced both RNN structures and the vanilla attention mechanism in Tacotron 2 \cite{shen2018natural}.
This approach addressed the long-distance dependency problem and improved pluralization.
Phoneme sequences were used as input to generate the mel-spectrogram, and speech samples were synthesized using WaveNet as a vocoder.
Results showed that the transformer-based TTS approach was $4.25$ times faster than Tacotron 2 and achieved similar MOS (Mean Opinion Score) performance.

Aside from the work mentioned above, there are other studies that are based on the Tacotron architecture.
For example, \citet{skerry2018towards} and \citet{wang2018style} proposed Tacotron-based models for prosody control.
These models use a separate encoder to compute style information from reference audio that is not provided in the text.
Another noteworthy work is the Global-style-Token (GST) \cite{wang2018style} which improves on style embeddings by adding an attention layer to capture a wider range of acoustic styles.

The FastSpeech \cite{ren2019fastspeech} algorithm aims to improve the inference speed of TTS systems.
To achieve this, it utilizes a feedforward network based on 1D convolution and the self-attention mechanism in transformers to generate Mel-spectrograms in parallel.
Additionally, it solves the issue of sequence length mismatch between the Mel-spectrogram sequence and its corresponding phoneme sequence by employing a length regulator based on a duration predictor.
The FastSpeech model was evaluated on the LJSpeech dataset and demonstrated significantly faster Mel-spectrogram generation than the autoregressive transformer model while maintaining comparable performance.
FastPitch builds on FastSpeech by conditioning the TTS model on fundamental frequency or pitch contour, which improves convergence and eliminates the need for knowledge distillation of Mel-spectrogram targets in FastSpeech.

FastSpeech 2 \cite{ren2020fastspeech} represents a transformer-based Text-to-Speech (TTS) system that addresses the limitations of its predecessor, FastSpeech, while effectively handling the challenging one-to-many mapping problem in TTS.
It introduces the utilization of a broader range of speech information, including energy, pitch, and more accurate duration, as conditional inputs.
Furthermore, FastSpeech 2 trains the system directly on a ground-truth target, enhancing the quality of the synthesized speech.
Additionally, a simplified variant called FastSpeech 2s has been proposed in [61], eliminating the requirement for intermediate Mel-spectrograms and enabling the direct generation of speech from text during inference.
Experimental evaluations conducted on the LJSpeech dataset demonstrated that both FastSpeech 2 and FastSpeech 2s offer a streamlined training pipeline, resulting in fast, robust, and controllable speech synthesis compared to FastSpeech.

Furthermore, in addition to the transformer-based TTS systems like FastSpeech 2 and FastSpeech 2s, researchers have also been exploring the potential of Variational Autoencoder (VAE) based TTS models \cite{lee2021bidirectional,hsuhierarchical,guo2022multi,kim2021conditional}.
These models can learn a latent representation of speech signals from textual input and may be able to produce high-quality speech with less training data and greater control over the generated speech characteristics.
For example, authors in \cite{kim2021conditional} used a conditional variational autoencoder (CVAE) to model the acoustic features of speech and an adversarial loss to improve the naturalness of the generated speech.
This approach involved conditioning the CVAE on the linguistic features of the input text and using an adversarial loss to match the distribution of the generated speech to that of natural speech.
Results from this method have shown promise in generating speech that exhibits natural prosody and intonation.

WaveGrad \cite{chen2020wavegrad} and DiffWave \cite{kong2020diffwave} have emerged as significant contributions in the field, employing diffusion models to generate raw waveforms with exceptional performance.
In contrast, GradTTS \cite{popov2021grad} and DiffTTS \cite{jeong2021diff} utilize diffusion models to generate mel features rather than raw waveforms.
Addressing the intricate challenge of one-shot many-to-many voice conversion, DiffVC \cite{popov2021diffusion} introduces a novel solver based on stochastic differential equations.
Expanding the scope of sound generation to include singing voice synthesis, DiffSinger \cite{liu2022diffsinger} introduces a shallow diffusion mechanism.
Additionally, Diffsound \cite{yang2022diffsound} proposes a sound generation framework that incorporates text conditioning and employs a discrete diffusion model, effectively resolving concerns related to unidirectional bias and accumulated errors.

EdiTTS \cite{DBLP:journals/corr/abs-2110-02584} introduces a diffusion-based audio model that is specifically tailored for the text-to-speech task.
Its innovative approach involves the utilization of the denoising reversal process to incorporate desired edits through coarse perturbations in the prior space.
Similarly, Guided-TTS \cite{kim2022guided} and Guided-TTS2 \cite{kim2022guided2} stand as early text-to-speech models that have effectively harnessed diffusion models for sound generation.
Furthermore, Levkovitch et al.
\cite{levkovitch2022zero} have made notable contributions by combining a voice diffusion model with a spectrogram domain conditioning technique.
This combined approach facilitates text-to-speech synthesis, even with previously unseen voices during the training phase, thereby enhancing the model's versatility and capabilities.

InferGrad \cite{chen2022infergrad} enhances the diffusion-based text-to-speech model by incorporating the inference process during training, particularly when a limited number of inference steps are available.
This improvement results in faster and higher-quality sampling.
SpecGrad \cite{koizumi2022specgrad} introduces adaptations to the time-varying spectral envelope of diffusion noise based on conditioning log-mel spectrograms, drawing inspiration from signal processing techniques.
ItoTTS \cite{wu2021hat} presents a unified framework that combines text-to-speech and vocoder models, utilizing linear SDE (Stochastic Differential Equation) as its fundamental principle.
ProDiff \cite{huang2022prodiff} proposes a progressive and efficient diffusion model specifically designed for generating high-quality text-to-speech synthesis.
Unlike traditional diffusion models that require a large number of iterations, ProDiff parameterizes the model by predicting clean data and incorporates a teacher-synthesized mel-spectrogram as a target to minimize data discrepancies and improve the sharpness of predictions.
Finally, Binaural Grad \cite{leng2022binauralgrad} explores the application of diffusion models in binaural audio synthesis, aiming to generate binaural audio from monaural audio sources.
It accomplishes this through a two-stage diffusion-based framework.

### 5.2.4·Alignment

Improving the alignment of text and speech in TTS architecture has been the focus of recent research \cite{kim2020glow,popov2021grad,ju2022trinitts,miao2021efficienttts,li2022styletts,shih2021rad,9746686,9747707,chen2021speech,ren2021portaspeech,bai20223,zhang2018forward,battenberg2020location,shen2020non}.
Traditional TTS models require external aligners to provide attention alignments of phoneme-to-frame sequences, which can be complex and inefficient.
Although autoregressive TTS models use an attention mechanism to learn these alignments online, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, resulting in missing or repeating words.

In their study \cite{drexler2019explicit}, the authors presented a novel text encoder network that includes an additional objective function to explicitly align text and speech encodings.
The text encoder architecture is straightforward, consisting of an embedding layer, followed by two bidirectional LSTM layers that maintain the input's resolution.
The study utilized the same subword segmentation for the input text as for the ASR output targets.
While RNN models with soft attention mechanisms have been proven to be highly effective in various tasks, including speech synthesis, their use in online settings results in quadratic time complexity due to the pass over the entire input sequence for generating each element in the output sequence.
In \cite{raffel2017online}, the authors proposed an end-to-end differentiable method for learning monotonic alignments, enabling the computation of attention in linear time.
Several enhancements, such as those proposed in \cite{chiu2017monotonic}, have been proposed in recent years to improve alignment in TTS models.
Additionally, in \cite{badlani2022one}, the authors introduced a generic alignment learning framework that can be easily extended to various neural TTS models.

The use of normalizing flow has been introduced to address output diversity issues in parallel TTS architectures.
This technique is utilized to model the duration of speech, as evidenced by studies conducted in \cite{kim2020glow,shih2021rad,miao2021efficienttts}.
One such flow-based generative model is Glow-TTS \cite{kim2020glow}, developed specifically for parallel TTS without the need for an external aligner.
The model employs the generic Glow architecture previously used in computer vision and vocoder models to produce mel-spectrograms from text inputs, which are then converted to speech audio.
Glow-TTS has demonstrated superior synthesis speed over the autoregressive model, Tacotron 2, while maintaining comparable speech quality.

Recently, a new TTS model called EfficientTTS \cite{miao2021efficienttts} has been introduced.
This model outperforms previous models such as Tacotron 2 and Glow-TTS in terms of speech quality, training efficiency, and synthesis speed.
The EfficientTTS model uses a multi-head attention mechanism to align input text and speech encodings, enabling it to generate high-quality speech with fewer parameters and faster synthesis speed.
Overall, the introduction of normalizing flow and the development of models such as Glow-TTS and EfficientTTS have significantly improved the quality and efficiency of TTS systems.

To resolve output diversity issues in parallel TTS architectures, normalizing flow has been introduced to model the duration of speech \cite{kim2020glow,shih2021rad,miao2021efficienttts}.
Glow-TTS \cite{kim2020glow} is a flow-based generative model for parallel TTS that does not require any external aligner12345.
It is built on the generic Glow model that is previously used in computer vision and vocoder models3.
Glow-TTS is designed to produce mel-spectrograms from text input, which can then be converted to speech audio4.
It has been shown to achieve an order-of-magnitude speed-up over the autoregressive model, Tacotron 2, at synthesis with comparable speech quality.
EfficientTTS is a recent study that proposed a new TTS model, which significantly outperformed models such as Tacotron 2 \cite{shen2018natural} and Glow-TTS \cite{kim2020glow} in terms of speech quality, training efficiency, and synthesis speed.
The EfficientTTS \cite{miao2021efficienttts} model uses a multi-head attention mechanism to align the input text and speech encodings, enabling it to generate high-quality speech with fewer parameters and faster synthesis speed.

### 5.2.5·Speech Resynthesis

Speech resynthesis is the process of generating speech from a given input signal.
The input signal can be in various forms, such as a digital recording, text, or other types of data.
The aim of speech resynthesis is to create an output that closely resembles the original signal in terms of sound quality, prosody, and other acoustic characteristics.
Speech resynthesis is an important research area with various applications, including speech enhancement \cite{tan2019learning,hsu2022revise,maiti2020speaker}, and voice conversion \cite{maimon2022speaking}.
Recent advancements in speech resynthesis have revolutionized the field by incorporating self-supervised discrete representations to generate disentangled representations of speech content, prosodic information, and speaker identity.
These techniques enable the generation of speech in a controlled and precise manner, as seen in \cite{lakhotia2021generative,polyak2021speech,qian2022contentvec,sicherman2023analysing}.
The objective is to generate high-quality speech that maintains or degrades acoustic cues, such as phonotactics, syllabic rhythm, or intonation, from natural speech recordings.

Speech resynthesis is a vital research area with various applications, including speech enhancement and voice conversion, and recent advancements have revolutionized the field by incorporating self-supervised discrete representations.
These techniques enable the generation of high-quality speech that maintains or degrades acoustic cues from natural speech recordings, and they have been used in the GSLM \cite{lakhotia2021generative}  architecture for acoustic modeling, speech recognition, and synthesis, as outlined in Figure \ref{fig:GSLM}.
It comprises a discrete speech encoder, a generative language model, and a speech decoder, all trained without supervision.
GSLM is the only prior work addressing the generative aspect of speech pre-training, which builds a text-free language model using discovered units.

### 5.2.6·Voice Conversion

Modifying a speaker's voice in a provided audio sample to that of another individual is called voice conversion, preserving linguistic content information.
TTS and Voice conversion share a common objective of generating natural speech.
While models based on RNNs and CNNs have been successfully applied to voice conversion, the use of the transformer has shown promising results.
Voice Transformer Network (VTN) \cite{huang2019voice} is a seq2seq voice conversion (VC) model based on the transformer architecture with TTS pre-training.
Seq2seq VC models are attractive as they can convert prosody, and the VTN is a novel approach in this field that has been proven to be effective in converting speech from a source to a target without changing the linguistic content.

ASR and TTS-based voice conversion is a promising approach to voice conversion \cite{tian2019vocoder}.
It involves using an ASR model to transcribe the source speech into the linguistic representation and then using a TTS model to synthesize the target speech with the desired voice characteristics \cite{polyak2019tts}.
However, this approach overlooks the modeling of prosody, which plays an important role in speech naturalness and conversion similarity.
To address this issue, researchers have proposed to directly predict prosody from the linguistic representation in a target-speaker-dependent manner \cite{zhang2020voice}.
Other researchers have explored using a mix of ASR and TTS features to improve the quality of voice conversion \cite{huang2021prosody,zhao2021towards,chou2019one,zhang2019non}.

CycleGAN \cite{kaneko2019cyclegan,kaneko2020cyclegan,kaneko2021maskcyclegan}, VAE \cite{choi2021neural,9053854,kameoka2019acvae}, and VAE with the generative adversarial network \cite{hsu2017voice} are other popular VC other popular approaches for non-parallel-voice conversion.
CycleGAN-VC \cite{kaneko2019cyclegan} uses a cycle-consistent adversarial network to convert the source voice to the target voice and can generate high-quality speech without any extra data, modules, or alignment procedure.
Several improvements and modifications are also proposed in recent years \cite{kaneko2020cyclegan,kaneko2021maskcyclegan,hsu2017voice}.
VAE-based voice conversion is a promising approach that can generate high-quality speech with a small amount of training data \cite{choi2021neural,9053854,kameoka2019acvae}.

### 5.2.7·Vocoders

The field of audio synthesis has undergone significant advancements in recent years, with various approaches proposed to enhance the quality of synthesized audio.
Prior studies have concentrated on improving discriminator architectures or incorporating auxiliary training losses.
For instance, MelGAN introduced a multiscale discriminator that uses window-based discriminators at different scales and applies average pooling to downsample the raw waveform.
It enforces the correspondence between the input Mel spectrogram and the synthesized waveform using an L1 feature matching loss from the discriminator.
In contrast, GAN-TTS \cite{binkowski2019high} utilizes an ensemble of discriminators that operate on random windows of different sizes and enforce the mapping between the conditioner and the waveform adversarially using conditional discriminators.
Another approach, parallel WaveGAN \cite{yamamoto2020parallel}, extends the single short-time Fourier transform loss to multi-resolution and employs it as an auxiliary loss for GAN training.
Recently, some researchers have improved MelGAN by integrating the multi-resolution short-time Fourier transform loss.
HiFi-GAN reuses the multi-scale discriminator from MelGAN and introduces the multi-period discriminator for high-fidelity synthesis.
UnivNet employs a multi-resolution discriminator that takes multi-resolution spectrograms as input and can enhance the spectral structure of a synthesized waveform.
In contrast, CARGAN integrates partial autoregression into the generator to enhance pitch and periodicity accuracy.
The recent generative models for modeling raw audio can be categorized into the following groups.

#### Autoregressive models

Although WaveNet is renowned for its exceptional ability to generate high-quality speech, including natural-sounding intonation and prosody, other neural vocoders have emerged as potential alternatives in recent years.
For instance, LPCNet \cite{valin2019lpcnet} employs a combination of linear predictive coding (LPC) and deep neural networks (DNNs) to generate speech of similar quality while being computationally efficient and capable of producing low-bitrate speech.
Similarly, SampleRNN \cite{mehri2016samplernn}, an unconditional end-to-end model, has demonstrated potential as it leverages a hierarchical RNN architecture and is trained end-to-end to generate raw speech of high quality.

#### Generative Adversarial Network (GAN) vocoders

Numerous vocoders have been created that employ Generative Adversarial Networks (GANs) to generate speech of exceptional quality.
These GAN-based vocoders, which include MelGAN  MelGAN \cite{kumar2019melgan}and HiFIGAN \cite{kong2020hifi}, are capable of producing high-fidelity raw audio by conditioning on mel spectrograms.
Furthermore, they can synthesize audio at speeds several hundred times faster than real-time on a single GPU, as evidenced by research conducted in \cite{donahue2018adversarial,binkowskihigh,yamamoto2020parallel,kong2020hifi,kumar2019melgan}.

#### Diffusion-based models

In recent years, there have been several novel architectures proposed that are based on diffusion.
Two prominent examples of these are WaveGrad \cite{chenwavegrad} and DiffWave \cite{kong2020diffwave}.
The WaveGrad model architecture builds upon prior works from score matching and diffusion probabilistic models, while the DiffWave model uses adaptive noise spectral shaping to adapt the diffusion noise.
This adaptation, achieved through time-varying filtering, improves sound quality, particularly in high-frequency bands.
Other examples of diffusion-based vocoders include InferGrad \cite{chen2022infergrad}, SpecGrad \cite{koizumi2022specgrad}, and Priorgrad \cite{leepriorgrad}.
InfraGrad incorporates the inference process into training to reduce inference iterations while maintaining high quality.
SpecGrad adapts the diffusion noise distribution to a given acoustic feature and uses adaptive noise spectral shaping to generate high-fidelity speech waveforms.

#### Flow-based models:

Parallel WaveNet, WaveGlow, etc.\cite{luong2021flowvocoder,prenger2019waveglow,kim2018flowavenet,ping2020waveflow,lee2020nanoflow} are based on normalizing flows and are capable of generating high-fidelity speech in real-time.
While flow-based vocoders generally perform worse than autoregressive vocoders with regard to modeling the density of speech signals, recent research \cite{luong2021flowvocoder} has proposed new techniques to improve their performance.

#### Summary

Universal neural vocoding is a challenging task that has achieved limited success to date.
However, recent advances in speech synthesis have shown a promising trend toward improving zero-shot performance by scaling up model sizes.
Despite its potential, this approach has yet to be extensively explored.
Nonetheless, several approaches have been proposed to address the challenges of universal vocoding.
For example, WaveRNN has been utilized in previous studies to achieve universal vocoding (\citet{lorenzo2018towards}; \citet{paul2020speaker}).
Another approach \citet{jiao2021universal} developed involves constructing a universal vocoder using a flow-based model.
Additionally, the GAN vocoder has emerged as a promising candidate for this task, as suggested by ~\citet{you2021gan}.

### 5.2.8·Controllable Speech Synthesis

Controllable Speech Synthesis \cite{9054556,9640518,9003829,9053732,ren2019fastspeech,wang2018style,8778667} is a rapidly evolving research area that focuses on generating natural-sounding speech with the ability to control various aspects of speech, including pitch, speed, and emotion.
Controllable Speech Synthesis is positioned in the emerging field of affective computing at the intersection of three disciplines: expressive speech analysis \cite{tits2019visualization}, natural language processing, and machine learning.
This field aims to develop systems capable of recognizing, interpreting, and generating human-like emotional responses in interactions between humans and machines.

Expressive speech analysis is a critical component of this field.
It provides mathematical tools to analyse speech signals and extract various acoustic features, including pitch, loudness, and duration, that convey emotions in speech.
Natural language processing is also crucial to this field, as it helps to process the text input and extract the meaning and sentiment of the words.
Finally, machine learning techniques are used to model and control the expressive features of the synthesized speech, enabling the systems to produce more expressive and controllable speech \cite{9053678,9420276,valle2020flowtron,kulkarni2020transfer,sorin2020principal,zhao2023emotion,pamisetty2023prosody,huang2022generspeech,lee2022hierspeech}.

In the last few years, notable advancements have been achieved in this field \cite{raitio2020controllable,kenter2019chive,habibie2022motion}, and several approaches have been proposed to enhance the quality of synthesized speech.
For example, some studies propose using deep learning techniques to synthesize expressive speech and conditional generation models to control the prosodic features of speech \cite{raitio2020controllable,kenter2019chive}.
Others propose using motion matching-based algorithms to synthesize gestures from speech  \cite{habibie2022motion}.

### 5.2.9·Disentangling and Transferring

The importance of disentangled representations for neural speech synthesis cannot be overstated, as it has been widely recognized in the literature that this approach can greatly improve the interpretability and expressiveness of speech synthesis models \cite{ma2019neural, hsu2019disentangling, qian2020unsupervised}.
Disentangling multiple styles or prosody information during training is crucial to enhance the quality of expressive speech synthesis and control.
Various disentangling techniques have been developed using adversarial and collaborative games, the VAE framework, bottleneck reconstructions, and frame-level noise modeling combined with adversarial training.

For instance, \citet{ma2019neural} have employed adversarial and collaborative games to enhance the disentanglement of content and style, resulting in improved controllability.
\citet{hsu2019disentangling} have utilized the VAE framework with adversarial training to separate speaker information from noise.
\citet{qian2020unsupervised} have introduced speech flow, which can disentangle rhythm, pitch, content, and timbre through three bottleneck reconstructions.
In another work based on, adversarial training, \citet{zhang2021denoispeech} have proposed a method that disentangles noise from the speaker by modeling the noise at the frame level.

Developing high-quality speech synthesis models that can handle noisy data and generate accurate representations of speech is a challenging task.
To tackle this issue, \citet{zhang2022hifidenoise} propose a novel approach involving multi-length adversarial training.
This method allows for modeling different noise conditions and improves the accuracy of pitch prediction by incorporating discriminators on the mel-spectrogram.
By replacing the traditional pitch predictor model with this approach, the authors demonstrate significant improvements in the fidelity of synthesized speech.

### 5.2.10·Robustness

Using neural TTS models can present issues with robustness, leading to low-quality audio samples for unseen or atypical text.
In response, \citet{li2020robutrans} proposed RobuTrans \cite{li2020robutrans}, a robust transformer that converts input text to linguistic features before feeding it to the encoder.
This model also includes modifications to the attention mechanism and position embedding, resulting in improved MOS scores compared to other TTS models.
Another approach to enhancing robustness is the s-Transformer, introduced by \citet{wang2020s}, which models speech at the segment level, allowing it to capture long-term dependencies and use segment-level encoder-decoder attention.
This technique performs similarly to the standard transformer, exhibiting robustness for extra-long sentences.
Lastly, \citet{zheng2020improving} proposed an approach that combines a local recurrent neural network with the transformer to capture sequential and local information in sequences.
Evaluation of a $20$-hour Mandarin speech corpus demonstrated that this model outperforms the transformer alone in performance.

In their recent paper \cite{yang2022norespeech}, the authors proposed a novel method for extracting dynamic prosody information from audio recordings, even in noisy environments.
Their approach employs probabilistic denoising diffusion models and knowledge distillation to learn speaking style features from a teacher model, resulting in a highly accurate reproduction of prosody and timber.
This model shows great potential in applications such as speech synthesis and recognition, where noise-robust prosody information is crucial.
Other noteworthy advances in the development of robust TTS systems include the work by \cite{shih2021rad}, which focuses on a robust speech-text alignment module, as well as the use of normalizing flows for diverse speech synthesis.

### 5.2.11·Low-Resource Neural Speech Synthesis

High-quality paired text and speech data are crucial for building high-quality Text-to-Speech (TTS) systems \cite{gabrys2022voice}.
Unfortunately, most languages are not supported by popular commercialized speech services due to the lack of sufficient training data \cite{xu2020lrspeech}.
To overcome this challenge, researchers have developed TTS systems under low data resource scenarios using various techniques \cite{gabrys2022voice, xu2020lrspeech, elneima2022adversarial, tu2019end}.

Several techniques have been proposed by researchers to enhance the efficiency of low-resource/Zero-shot TTS systems.
One of these is the use of semi-supervised speech synthesis methods that utilize unpaired training data to improve data efficiency, as suggested in a study by \citet{liu2022simple}.
Another method involves cascading pre-trained models for ASR, MT, and TTS to increase data size from unlabelled speech, as proposed by \citet{nguyen2022improving}.
In addition, researchers have employed crowdsourced acoustic data collection to develop TTS systems for low-resource languages, as shown in a study by \citet{butryna2020google}.
\citet{huang2022generspeech} introduced a zero-shot style transfer approach for out-of-domain speech synthesis that generates speech samples exhibiting a new and distinctive style, such as speaker identity, emotion, and prosody.

## 5.3·Speaker Recognition

### 5.3.1·Task Description

Speech signal consists of information on various characteristics of a speaker, such as origin, identity, gender, emotion, etc.
This property of speech allows speech-based speaker profiling with a wide range of applications in forensics, recommendation systems, etc.
The research on recognizing speakers is extensive and aims to solve two major tasks: speaker identification (what is the identity?) and speaker verification (is the speaker he/she claims to be?).
Speaker recognition/verification tasks require extracting a fixed-length vector, called speaker embedding, from unconstrained utterances.
These embeddings represent the speakers and can be used for identification or verification tasks.
Recent state-of-the-art speaker-embedding-extractor models are based on DNNs and have shown superior performance on both speaker identification and verification tasks.

- Speaker Recognition (SR) relies on speaker identification as a key aspect, where an unknown speaker's speech sample is compared to speech models of known speakers to determine their identity.
The primary aim of speaker identification is to distinguish an individual's identity from a group of known speakers.
This process involves a detailed analysis of the speaker's voice characteristics such as pitch, tone, accent, and other pertinent features to establish their identity.
Recent advancements in deep learning techniques have significantly enhanced speaker identification, leading to the creation of accurate, efficient, and end-to-end models.
Various deep learning-based models such as CNNs, RNNs, and their combinations have demonstrated exceptional performance in several subtasks of speaker identification, including verification, identification, diarization, and robust recognition \cite{ravanelli2020multi,kawakami2020learning,kinoshita2020improving}.
- Speaker Verification (SV) is a process that involves confirming the identity of a speaker through their speech.
It differs from speaker identification, which aims to identify unknown speakers by comparing their voices with that of registered speakers in a database.
Speaker verification verifies whether a speaker is who they claim to be by comparing their voice with an available speaker template.
Deep learning-based speaker verification relies on Speaker Representation based on embeddings, which involves learning low-dimensional vector representations from speech signals that capture speaker characteristics, such as pitch and speaking style, and can be used to compare different speech signals and determine their similarity.

### 5.3.2·Dataset

The VoxCeleb dataset (VoxCeleb 1 \& 2) is widely used in speaker recognition research, as mentioned in \cite{Nagrani17}.
This dataset consists of speech data collected from publicly available media, employing a fully automated pipeline that incorporates computer vision techniques.
The pipeline retrieves videos from YouTube and applies active speaker verification using a two-stream synchronization CNN.
Speaker identity is further confirmed through CNN-based facial recognition.
Another commonly employed dataset is TIMIT, which comprises recordings of phonetically balanced English sentences spoken by a diverse set of speakers.
TIMIT is commonly used for evaluating speech recognition and speaker identification systems, as referenced in \cite{garofolo1993timit}.

Other noteworthy datasets in the field include the SITW database \cite{mclaren2016speakers}, which provides hand-annotated speech samples for benchmarking text-independent speaker recognition technology, and the RSR2015 database \cite{larcher2012rsr2015}, which contains speech recordings acquired in a typical office environment using multiple mobile devices.
Additionally, the RedDots project \cite{lee2015reddots} and VOICES corpus \cite{richey2018voices} offer unique collections of offline voice recordings in furnished rooms with background noise, while the CN-CELEB database \cite{fan2020cn} focuses on a specific person of interest extracted from bilibili.com using an automated pipeline followed by human verification.

The BookTubeSpeech dataset \cite{pham2020toward} was also collected using an automated pipeline from BookTube videos, and the Hi-MIA database \cite{qin2020hi} was designed specifically for far-field scenarios using multiple microphone arrays.
The FFSVC20 challenge \cite{qin2020ffsvc} and DIHARD challenge \cite{ryant2018first} are speaker verification and diarization research initiatives focusing on far-field and robustness challenges, respectively.
Finally, the LibriSpeech dataset \cite{panayotov2015librispeech}, originally intended for speech recognition, is also useful for speaker recognition tasks due to its included speaker identity labels.

### 5.3.3·Models

Speaker identification (SI) and verification (SV) are crucial research topics in the field of speech technology due to their significant importance in various applications such as security \cite{edu2020smart}, forensics \cite{koval2020practice}, biometric authentication \cite{hanifa2021review}, and speaker diarization \cite{xiao2021microsoft}.
Speaker recognition has become more popular with technological advancements, including the Internet of Things (IoT), smart devices, voice assistants, smart homes, and humanoids.
Therefore, a significant quantity of research has been conducted in this field, and many methods have been developed, making the state-of-the-art in this field quite mature and versatile.
However, it has become increasingly challenging to provide an overview of the various methods due to the high number of studies in the field.

A neural network approach for speaker verification was first attempted by \citet{6854363} in 2014, utilizing four fully connected layers for speaker classification.
Their approach has successfully verified speakers with short-duration utterances by obtaining the $d$-vector by averaging the output of the last hidden layer across frames.
Although various attempts have been made to directly learn speaker representation from raw waveforms by other researchers (\citet{ravanelli2018speaker,jung2019rawnet}), other well-designed neural networks like CNNs and RNNs have been proposed for speaker verification tasks by \citet{ye2021deep}.
Nevertheless, the field still requires more powerful deep neural networks for superior extraction of speaker features.

Speaker verification has seen notable advancements with the advent of more powerful deep neural networks.
One such model is the $x$-vector-based system proposed by \citet{snyder2018x}, which has gained widespread popularity due to its remarkable performance.
Since its introduction, the $x$-vector system has undergone significant architectural enhancements and optimized training procedures \cite{deng2019arcface}.
The widely-used ResNet \cite{he2016deep} architecture has been incorporated into the system to improve its performance further.
Adding residual connections between frame-level layers has been found to improve the embeddings \cite{garcia2020jhu,zeinali2019but}.
This technique has also aided in faster convergence of the back-propagation algorithm and mitigated the vanishing gradient problem \cite{he2016deep}.
\citet{tang2019deep} proposed further improvements to the $x$-vector system.
They introduced a hybrid structure based on TDNN and LSTM to generate complementary speaker information at different levels.
They also suggested a multi-level pooling strategy to collect the speaker information from global and local perspectives.
These advancements have significantly improved speaker verification systems' performance and paved the way for further developments in the field.

\citet{desplanques2020ecapa} propose a state-of-the-art architecture for speaker verification utilizing a Time Delay Neural Network (TDNN) called ECAPA-TDNN.
The paper presents a range of enhancements to the existing $x$-vector architecture that leverages recent developments in face verification and computer vision.
Specifically, the authors suggest three major improvements.
Firstly, they propose restructuring the initial frame layers into 1-dimensional Res2Net modules with impactful skip connections, which can better capture the relationships between different time frames.
Secondly, they introduce Squeeze-and-Excitation blocks to the TDNN layers, which help highlight the most informative channels and improve feature discrimination.
Lastly, the paper proposes channel attention propagation and aggregation to efficiently propagate attention weights through multiple TDNN layers, further enhancing the model's ability to discriminate between speakers.

Additionally, the paper presents a new approach that utilizes ECAPA-TDNN from the speaker recognition domain as the backbone network for a multiscale channel adaptive module.
The proposed method achieves promising results, demonstrating the effectiveness of the proposed architecture in speaker verification.
Overall, ECAPA-TDNN  offers a comprehensive solution to speaker verification by introducing several novel contributions that improve the existing $x$-vector architecture, which has been state-of-the-art in speaker verification for several years.
The proposed approach also achieves promising results, suggesting that the proposed architecture can effectively tackle the challenges of speaker verification.

The attention mechanism is a powerful method for obtaining a more discriminative utterance-level feature by explicitly selecting frame-level representations that better represent speaker characteristics.
Recently, the Transformer model with a self-attention mechanism has become effective in various application fields, including speaker verification.
The Transformer architecture has been extensively explored for speaker verification.
TESA \cite{mary2021s} is an architecture based on the Transformer's encoder, proposed as a replacement for conventional PLDA-based speaker verification to capture speaker characteristics better.
TESA outperforms PLDA on the same dataset by utilizing the next sentence prediction task of BERT \cite{devlin2018bert}.
\citet{zhu2021serialized} proposed a method to create fixed-dimensional speaker verification representation using a serialized multi-layer multi-head attention mechanism.
Unlike other studies that redesign the inner structure of the attention module, their approach strictly follows the original Transformer, providing simple but effective modifications.

## 5.4·Speaker Diarization

### 5.4.1·Task Description

Speaker diarization is a critical component in the analysis of multi-speaker audio data, and it addresses the question of "who spoke when." The term "diarize" refers to the process of making a note or keeping a record of events, as per the English dictionary.
A traditional speaker diarization system comprises several crucial components  that work together to achieve accurate and efficient speaker diarization.
In this section, we will discuss the different components of a speaker diarization system (\Cref{fig:sd}) and their role in achieving accurate speaker diarization.

- **Acoustic Features Extraction**: In the analysis of multi-speaker speech data, one critical component is the extraction of acoustic features \cite{anguera2012speaker,tranter2006overview}.
This process involves extracting features such as pitch, energy, and MFCCs from the audio signal.
These acoustic features play a crucial role in identifying different speakers by analyzing their unique characteristics.
- **Segmentation**: Segmentation is a crucial component in the analysis of multi-speaker audio data, where the audio signal is divided into smaller segments based on the silence periods between speakers \cite{anguera2012speaker,tranter2006overview}.
This process helps in reducing the complexity of the problem and makes it easier to identify different speakers in smaller segments
- **Speaker Embedding Extraction**: This process involves obtaining a low-dimensional representation of each speaker's voice, which is commonly referred to as speaker embedding.
This is achieved by passing the acoustic features extracted from the speech signal through a deep neural network, such as a CNN or RNN\cite{snyder2017deep}.
- **Clustering**: In this component, the extracted speaker embeddings are clustered based on similarity, and each cluster represents a different speaker \cite{anguera2012speaker,tranter2006overview}.
This process commonly uses unsupervised clustering algorithms, such as k-means clustering.
- **Speaker Classification**: In this component, the speaker embeddings are classified into different speaker identities using a supervised classification algorithm, such as SVM or MLP \cite{anguera2012speaker,tranter2006overview}.
- **Re-segmentation**: This component is responsible for refining the initial segmentation by adjusting the segment boundaries based on the classification results.
It helps in improving the accuracy of speaker diarization by reducing the errors made during the initial segmentation.

Various studies focus on traditional speaker diarization systems \cite{anguera2012speaker,tranter2006overview}.
This paper will review the recent efforts toward deep learning-based speaker diarizations techniques.

### 5.4.2·Dataset

- NIST SRE 2000 (Disk-8) or CALLHOME dataset: The NIST SRE 2000 (Disk-8) corpus, also referred to as the CALLHOME dataset, is a frequently utilized resource for speaker diarization in contemporary research papers.
Originally released in 2000, this dataset comprises conversational telephone speech (CTS) collected from diverse speakers representing a wide range of ages, genders, and dialects.
It includes 500 sessions of multilingual telephonic speech, each containing two to seven speakers, with two primary speakers in each conversation.
The dataset covers various topics, including personal and familial relationships, work, education, and leisure activities.
The audio recordings were obtained using a single microphone and had a sampling rate of 8 kHz, with 16-bit linear quantization.
- Directions into Heterogeneous Audio Research (DIHARD) Challenge and dataset: The DIHARD Challenge, organized by the National Institute of Standards and Technology (NIST), aims to enhance the accuracy of speech recognition and diarization in challenging acoustic environments, such as crowded spaces, distant microphones, and reverberant rooms.
The challenge comprises tasks requiring advanced machine-learning techniques, including speaker diarization, recognition, and speech activity detection.
The DIHARD dataset used in the challenge comprises over 50 hours of speech from more than 500 speakers, gathered from diverse sources like meetings, broadcast news, and telephone conversations.
These recordings feature various acoustic challenges, such as overlapping speech, background noise, and distant or reverberant speech, captured through different microphone setups.
To aid in the evaluation process, the dataset has been divided into separate development and evaluation sets.
The assessment metrics used to gauge performance include diarization error rate (DER), as well as accuracy in speaker verification, identification, and speech activity detection.
- Augmented Multi-party Interaction (AMI) database: The AMI database is a collection of audio and video recordings that capture real-world multi-party conversations in office environments.
The database was developed as part of the AMI project, which aimed to develop technology for automatically analyzing multi-party meetings.
The database contains over 100 hours of audio and video recordings of meetings involving four to seven participants, totaling 112 meetings.
The meetings were held in multiple offices and were designed to reflect the kinds of discussions that take place in typical business meetings.
The audio recordings were captured using close-talk microphones placed on each participant and additional microphones placed in the room to capture ambient sound.
The video recordings were captured using multiple cameras placed around the room.
In addition to the audio and video recordings, the database also includes annotations that provide additional information about the meetings, including speaker identities, speech transcriptions, and information about the meeting structure (e.g., turn-taking patterns).
The AMI database has been used extensively in research on automatic speech recognition, speaker diarization, and other related speech and language processing topics.
- VoxSRC Challenge and VoxConverse corpus: The VoxCeleb Speaker Recognition Challenge (VoxSRC) is an annual competition designed to assess the capabilities of speaker recognition systems in identifying speakers from speech recorded in real-world environments.
The challenge provides participants with a dataset of audio and visual recordings of interviews, news shows, and talk shows featuring famous individuals.
The VoxSRC encompasses several tracks, including speaker diarization, and comprises a development set (20.3 hours, 216 recordings) and a test set (53.5 hours, 310 recordings).
Recordings in the dataset may feature between one and 21 speakers, with a diverse range of ambient noises, such as background music and laughter.
To facilitate the speaker diarization track of the VoxSRC-21 and VoxSRC-22 competitions, VoxConverse, an audio-visual diarization dataset containing multi-speaker clips of human speech sourced from YouTube videos, is available, and additional details are provided on the project website \footnote{https://www.robots.ox.ac.uk/~vgg/data/voxconverse/}.
- LibriCSS: The LibriCSS corpus is a valuable resource for researchers studying speech separation, recognition, and speaker diarization.
The corpus comprises 10 hours of multichannel recordings captured using a 7-channel microphone array in a real meeting room.
The audio was played from the LibriSpeech corpus, and each of the ten sessions was subdivided into six 10-minute mini-sessions.
Each mini-session contained audio from eight speakers and was designed to have different overlap ratios ranging from 0\% to 40\%.
To make research easier, the corpus includes baseline systems for speech separation and Automatic Speech Recognition (ASR) and a baseline system that integrates speech separation, speaker diarization, and ASR.
These baseline systems have already been developed and made available to researchers.
- Rich Transcription Evaluation Series: The Rich Transcription Evaluation Series dataset is a collection of speech data used for speaker diarization evaluation.
The Rich Transcription Fall 2003 Evaluation (RT-03F) was the first evaluation in the series focused on "Who Said What" tasks.
The dataset has been used in subsequent evaluations, including the Second DIHARD Diarization Challenge, which used the Jaccard index to compute the JER (Jaccard Error Rate) for each pair of segmentations.
The dataset is essential for data-driven spoken language processing methods and calculates speaker diarization accuracy at the utterance level.
The dataset includes rules, evaluation methods, and baseline systems to promote reproducible research in the field.
The dataset has been used in various speaker diarization systems and their subtasks in the context of broadcast news and CTS data
- CHiME-5/6 challenge and dataset: The CHiME-5/6 challenge is a speech processing challenge focusing on distant multi-microphone conversational speech diarization and recognition in everyday home environments.
The challenge provides a dataset of recordings from everyday home environments, including dinner recordings originally collected for and exposed during the CHiME-5 challenge.
The dataset is designed to be representative of natural conversational speech.
The challenge features two audio input conditions: single-channel and multichannel.
Participants are provided with baseline systems for speech enhancement, speech activity detection (SAD), and diarization, as well as results obtained with these systems for all tracks.
The challenge aims to improve the robustness of diarization systems to variations in recording equipment, noise conditions, and conversational domains.
- MI dataset: The AMI database is a comprehensive collection of 100 hours of recordings sourced from 171 meeting sessions held across various locations.
It features two distinct audio sources – one recorded using lapel microphones for individual speakers and the other using omnidirectional microphone arrays placed on the table.
It is an ideal dataset for evaluating speaker diarization systems integrated with the ASR module.
AMI's value proposition is further enhanced by providing forced alignment data, which captures the timings at the word and phoneme levels and speaker labeling.
Finally, it's worth noting that each meeting session involves a small group of three to five speakers.

### 5.4.3·Models

Speaker diarization has been a subject of research in the field of audio processing, with the goal of separating speakers in an audio recording.
In recent years, deep learning has emerged as a powerful technique for speaker diarization, leading to significant advancements in this field.
In this article, we will explore some of the recent developments in deep learning architecture for speaker diarization, focusing on different modules of speaker diarization as outlined in Figure \ref{fig:sd}.
Through this discussion, we will highlight major advancements in each module.

#### Segmentation and Clustering

Speaker diarization systems typically use a range of techniques for segmenting speech, such as identifying speaker change, uniform speaker segmentation, ASR-based word segmentation, and supervised speaker turn detection.
However, each approach has its own benefits and drawbacks.
Uniform speaker segmentation involves dividing speech into segments of equal length, which can be difficult to optimize to capture speaker turn boundaries and include enough speaker information.
ASR-based word segmentation identifies word boundaries using automatic speech recognition, but the resulting segments may be too brief to provide adequate speaker information.
Supervised speaker turn detection, on the other hand, involves a specialized model that can accurately identify speaker turn timestamps.
While this method can achieve high accuracy, it requires labeled data for training.
These techniques have been widely discussed in previous research, and choosing the appropriate one depends on the specific requirements of the application.
- The authors in \cite{coria2021overlap} propose real-time speaker diarization system that combines incremental clustering and local diarization applied to a rolling window of speech data and is designed to handle overlapping speech segments.
The proposed pipeline is designed to utilize end-to-end overlap-aware segmentation to detect and separate overlapping speakers.
- In another related work, authors in \cite{zhang2022towards} introduce a novel speaker diarization system with a generalized neural speaker clustering module as the backbone.
- In a recent study conducted by \citet{park2019auto}, a new framework for spectral clustering is proposed that allows for automatic parameter tuning of the clustering algorithm in the context of speaker diarization.
The proposed technique utilizes normalized maximum eigengap (NME) values to determine the number of clusters and threshold parameters for each row in an affinity matrix during spectral clustering.
The authors demonstrated that their method outperformed existing state-of-the-art methods on two different datasets for speaker diarization.
- Bayesian HMM clustering of x-vector sequences (VBx) diarization approach, which clusters x-vectors using a Bayesian hidden Markov model (BHMM) \cite{landini2022bayesian}, combined with a ResNet101  (\citet{he2016deep}) $x$-vector extractor achieves superior results on CALLHOME \cite{diez2020optimizing}, AMI \cite{carletta2006ami} and DIHARD II \cite{ryant2019second} datasets

#### Speaker Embedding Extraction and Classification

- Attentive Aggregation for Speaker Diarization \cite{kwon2021adapting}: This approach uses an attention mechanism to aggregate embeddings from multiple frames and generate speaker embeddings.
The speaker embeddings are then used for clustering to identify speaker segments.
-  End-to-End Speaker Diarization with Self-Attention \cite{fujita2019end}: This method uses a self-attention mechanism to capture the correlations between the input frames and generates embeddings for each frame.
The embeddings are then used for clustering to identify speaker segments.
- \citet{wang2022similarity} present an innovative method for measuring similarity between speaker embeddings in speaker diarization using neural networks.
The approach incorporates past and future contexts and uses a segmental pooling strategy.
Furthermore, the speaker embedding network and similarity measurement model are jointly trained.
The paper extends this framework to target-speaker voice activity detection (TS-VAD) \cite{medennikov2020target}.
The proposed method effectively learns the similarity between speaker embeddings by considering both past and future contexts.
- Time-Depth Separable Convolutions for Speaker Diarization \cite{koluguri2022titanet}: This approach uses time-depth separable convolutions to generate embeddings for each frame, which are then used for clustering to identify speaker segments.
The method is computationally efficient and achieves state-of-the-art performance on several benchmark datasets.

#### Re-segmentation

- Numerous studies in this field centre around developing a re-segmentation strategy for diarization systems that can effectively handle both voice activity and overlapped speech detection.
This approach can also be a post-processing step to identify and assign overlapped speech regions accurately.
Notable examples of such works include those by \citet{bullock2020overlap} and \citet{bredin2021end}.

#### End-to-End Neural Diarization

In addition to the above work, end-to-end speaker diarization systems have gained the attention of the research community due to their ability to handle speaker overlaps and their optimization to minimize diarization errors directly.
In one such work, the authors propose end-to-end neural speaker diarization that does not rely on clustering and instead uses a self-attention-based neural network to directly output the joint speech activities of all speakers for each segment \cite{fujita2019end}.
Following the trend, several other works propose enhanced architectures based on self-attention \cite{lin2020self,yu2022auxiliary}.

## 5.5·Speech-to-Speech Translation

### 5.5.1·Task Description

Speech-to-text translation (ST) is the process of converting spoken language from one language to another in text form.
Traditionally, this has been achieved using a cascaded structure that incorporates automatic speech recognition (ASR) and machine translation (MT) components.
However, a more recent end-to-end (E2E)  method \cite{sung2019towards,salesky2019exploring,zhang2020adaptive,chen2020mam,han2021learning,zheng2021fused,ansari2020findings} has gained popularity due to its ability to eliminate issues with error propagation and high latency associated with cascaded methods \cite{sperber2020speech, chen2021specrec}.
The E2E method uses an audio encoder to analyze audio signals and a text decoder to generate translated text.

One notable advantage of ST systems is that they allow for more natural and fluent communication than other language translation methods.
By translating speech in real-time, ST systems can capture the subtleties of speech, including tone, intonation, and rhythm, which are essential for effective communication.
Developing ST systems is a highly intricate process that involves integrating various technologies such as speech recognition, natural language processing, and machine translation.
One significant obstacle in ST is the variation in accents and dialects across different languages, which can significantly impact the accuracy of the translation.

### 5.5.2·Dataset

There are numerous datasets available for the end-to-end speech translation task, with some of the most widely used ones being MuST-C \cite{cattoni2021must}, IWSLT \cite{scarton2019estimating}, and CoVoST 2 \cite{wang2020covost}.
These datasets cover a variety of languages, including English, German, Spanish, French, Italian, Dutch, Portuguese, Romanian, Arabic, Chinese, Japanese, Korean, and Russian.
For instance, TED-LIUM \cite{rousseau2012ted} is a suitable dataset for speech-to-text, text-to-speech, and speech-to-speech translation tasks, as it contains transcriptions and audio recordings of TED talks in English, French, German, Italian, and Spanish.
Another open-source dataset is Common Voice, which covers several languages, including English, French, German, Italian, and Spanish.
Additionally, VoxForge\footnote{http://www.voxforge.org/} is designed for acoustic model training and includes speech recordings and transcriptions in several languages, including English, French, German, Italian, and Spanish.
LibriSpeech \cite{panayotov2015librispeech} is a dataset of spoken English specifically designed for speech recognition and speech-to-text translation tasks.
Lastly, How2 \cite{duarte2021how2sign} is a multimodal machine translation dataset that includes speech recordings, text transcriptions, and video and image data, covering English, German, Italian, and Spanish.
These datasets have been instrumental in training state-of-the-art speech-to-speech translation models and will continue to play a crucial role in further advancing the field.

### 5.5.3·Models

End-to-end speech translation models are a promising approach to direct the speech translation field.
These models use a single sequence-to-sequence model for speech-to-text translation and then text-to-speech translation.
In 2017, researchers demonstrated that end-to-end models outperform cascade models[3].
One study published in 2019 provides an overview of different end-to-end architectures and the usage of an additional connectionist temporal classification (CTC) loss for better convergence \cite{bahar2019comparative}.
The study compares different end-to-end architectures for speech-to-text translation.
In 2019, Google introduced Translatotron \cite{jia2022translatotron}, an end-to-end speech-to-speech translation system.
Translatotron uses a single sequence-to-sequence model for speech-to-text translation and then text-to-speech translation.
No transcripts or other intermediate text representations are used during inference.
The system was validated by measuring the BLEU score, computed with text transcribed by a speech recognition system.
Though the results lag behind a conventional cascade system, the feasibility of the end-to-end direct speech-to-speech translation was demonstrated \cite{jia2022translatotron}.

In a recent publication from 2020, researchers presented a study on an end-to-end speech translation system.
This system incorporates pre-trained models such as Wav2Vec 2.0 and mBART, along with coupling modules between the encoder and decoder.
The study also introduces an efficient fine-tuning technique, which selectively trains only $20\%$ of the total parameters \cite{ye2021end}.
The system developed by the UPC Machine Translation group actively participated in the IWSLT 2021 offline speech translation task, which aimed to develop a system capable of translating English audio recordings from TED talks into German text.

E2E ST is often improved by pretraining the encoder and/or decoder with transcripts from speech recognition or text translation tasks \cite{di2019adapting,wang2020fairseq,zhang2020adaptive,xu2021stacked}.
Consequently, it has become the standard approach used in various toolkits \cite{inaguma2020espnet,wang2020fairseq,zhao2020neurst,zheng2021fused}.
However, transcripts are not always available, and the significance of pretraining for E2E ST is rarely studied.
\citet{zhang2022revisiting} explored the effectiveness of E2E ST trained solely on speech-translation pairs and proposed an algorithm for training from scratch.
The proposed system outperforms previous studies in four benchmarks covering 23 languages without pretraining.
The paper also discusses neural acoustic feature modeling, which extracts acoustic features directly from raw speech signals to simplify inductive biases and enhance speech description.

## 5.6·Speech Enhancement

### 5.6.1·Task Description

In situations where there is ambient noise present, speech recognition systems can encounter difficulty in correctly interpreting spoken language signals, resulting in reduced performance \cite{du2014robust}.
One possible solution to address this issue is the development of speech enhancement systems that can eliminate noise and other types of signal distortion from spoken language, thereby improving signal quality.
These systems are frequently implemented as a preprocessing step to enhance the accuracy of speech recognition and can serve as an effective approach for enhancing the performance of ASR systems in noisy environments.
This section will delve into the significance of speech enhancement technology in boosting the accuracy of speech recognition.

### 5.6.2·Dataset

One popular dataset for speech enhancement tasks is AISHELL-4, which comprises authentic Mandarin speech recordings captured during conferences using an 8-channel circular microphone array.
In accordance with \cite{fu2021aishell}, AISHELL-4 is composed of 211 meeting sessions, each featuring 4 to 8 speakers, for a total of 120 hours of content.
This dataset is of great value for research into multi-speaker processing owing to its realistic acoustics and various speech qualities, including speaker diarization and speech recognition

Another popular dataset used for speech enhancement is the dataset from Deep Noise Suppression (DNS) challenge \cite{reddy2020interspeech}, a large-scale dataset of noisy speech signals and their corresponding clean speech signals.
The DNS dataset contains over $10,000$ hours of noisy speech signals and over $1,000$ hours of clean speech signals, making it useful for training deep learning models for speech enhancement.
The Voice Bank Corpus (VCTK)  is another dataset containing speech recordings from 109 speakers, each recording approximately $400$ sentences.
The dataset contains clean and noisy speech recordings, making it useful for training speech enhancement models.
These datasets provide realistic acoustics, rich natural speech characteristics, and large-scale noisy and clean speech signals, making them useful for training deep learning models.

### 5.6.3·Models

Several Classical algorithms have been reported in the literature for speech enhancement, including spectral subtraction \cite{boll1979suppression}, Wiener and Kalman filtering \cite{lim1978all,scalart1996speech}, MMSE estimation \cite{ephraim1992bayesian}, comb filtering \cite{jin2009speech}, subspace methods \cite{hansen1997signal}.
Phase spectrum compensation \cite{paliwal2011importance}.
However, classical algorithms such as spectral subtraction and Wiener filtering approach the problem in the spectral domain and are restricted to stationary or quasi-stationary noise.

Neural network-based approaches inspired from other areas such as computer vision \cite{hou2018audio,gabbay2017visual,afouras2018conversation} and generative adversarial networks \cite{wu2019speech,lin2019speech,routray2022phase,fu2019metricgan} or developed for general audio processing tasks \cite{wang2020complex,giri2019attention} have outperformed the classical approaches.
Various neural network models based on different architectures, including fully connected neural networks \cite{xu2014regression}, deep denoising autoencoder \cite{lu2013speech}, CNN \cite{fu2016snr}, LSTM \cite{chen2015speech}, and Transformer \cite{koizumi2020speech} have effectively handled diverse noisy conditions.

Diffusion-based models have also shown promising results for speech enhancement \cite{lemercier2022storm,yen2022cold,lu2022conditional} and have led to the development of novel speech enhancement algorithms called Conditional Diffusion Probabilistic Model (CDiffuSE) that incorporates characteristics of the observed noisy speech signal into the diffusion and reverse processing \cite{lu2022conditional}.
CDiffuSE is a generalized formulation of the diffusion probabilistic model that can adapt to non-Gaussian real noises in the estimated speech signal.
Another diffusion-based model for speech enhancement is StoRM \cite{lemercier2022storm}, which stands for Stochastic Regeneration Model.
It uses a predictive model to remove vocalizing and breathing artifacts while producing high-quality samples using a diffusion process, even in adverse conditions.
StoRM has shown great ability at bridging the performance gap between predictive and generative approaches for speech enhancement.
Furthermore, authors in \cite{yen2022cold} propose cold diffusion process is an advanced iterative version of the diffusion process to recover clean speech from noisy speech.
According to the authors, it can be utilized to restore high-quality samples from arbitrary degradations.
Table \ref{performance:se} summarizing the performance of different speech enhancement algorithms on the Deep
Noise Suppression (DNS) Challenge dataset using different metrics.

## 5.7·Audio Super Resolution

### 5.7.1·Task Description

Audio super-resolution is a technique that involves predicting the missing high-resolution components of low-resolution audio signals.
Achieving this task can be difficult due to the continuous nature of audio signals.
Current methods typically approach super-resolution by treating audio as discrete data and focusing on fixed scale factors.
In order to accomplish audio super-resolution, deep neural networks are trained using pairs of low and high-quality audio examples.
During testing, the model predicts missing samples within a low-resolution signal.
Some recent deep network approaches have shown promise by framing the problem as a regression issue either in the time or frequency domain \cite{8462049}.
These methods have been able to achieve impressive results.

### 5.7.2·Datasets

This section provides an overview of the diverse datasets utilized in Audio Super Resolution literature.
One of the most frequently used datasets is the MUSDB18, specifically designed for music source separation and enhancement.
This dataset encompasses more than 150 songs with distinct tracks for individual instruments.
Another prominent dataset is UrbanSound8K, which comprises over, $8000$ environmental sound files collected from 10 different categories, making it ideal for evaluating Audio Super Resolution algorithms in noisy environments.
Furthermore, the VoiceBank dataset is another essential resource for evaluating Audio Super Resolution systems, comprising over 10,000 speech recordings from five distinct speakers.
This dataset offers a rich source of information for assessing speech processing systems, including Audio Super Resolution.
Another dataset, LibriSpeech, features more than 1000 hours of spoken words from several books and speakers, making it valuable for evaluating Audio Super Resolution algorithms to enhance the quality of spoken words.
Finally, the TED-LIUM dataset, which includes over 140 hours of speech recordings from various speakers giving TED talks, provides a real-world setting for evaluating Audio Super Resolution algorithms for speech enhancement.
By using these datasets, researchers can evaluate Audio Super Resolution systems for a wide range of audio signals and improve the generalizability of these algorithms for real-world scenarios.

### 5.7.3·Models

Audio super-resolution has been extensively explored using deep learning architectures \cite{rakotonirina2021self,yoneyama2022nonparallel,8462049,lee2021nu,han2022nu,birnbaum2019temporal,abdulatif2022cmgan,nguyen2022tunet,kim2022learning,liu2022neural}.
One notable paper by \citet{rakotonirina2021self} proposes a novel network architecture that integrates convolution and self-attention mechanisms for audio super-resolution.
Specifically, they use Attention-based Feature-Wise Linear Modulation (AFiLM) \cite{rakotonirina2021self} to modulate the activations of the convolutional model.
In another recent work by \citet{yoneyama2022nonparallel}, the super-resolution task is decomposed into domain adaptation and resampling processes to handle acoustic mismatch in unpaired low- and high-resolution signals.
To address this, they jointly optimize the two processes within the CycleGAN framework.

Moreover, the Time-Frequency Network (TFNet) \cite{8462049} proposed a deep network that achieves promising results by modeling the task as a regression problem in either time or frequency domain.
To further enhance audio super-resolution, the paper proposes a time-frequency network that combines time and frequency domain information.
Finally, recent advancements in diffusion models have introduced new approaches to neural audio upsampling.
Specifically, \citet{lee2021nu}, and \citet{han2022nu} propose NU-Wave 1 and 2 diffusion probabilistic models, respectively, which can produce high-quality waveforms with a sampling rate of 48kHz from coarse 16kHz or 24kHz inputs.
These models are a promising direction for improving audio super-resolution.
