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
