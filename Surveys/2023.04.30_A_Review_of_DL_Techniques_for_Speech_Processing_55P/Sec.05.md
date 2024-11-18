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
