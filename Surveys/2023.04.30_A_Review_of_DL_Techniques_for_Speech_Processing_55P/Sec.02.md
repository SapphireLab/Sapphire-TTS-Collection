# 2·Background

Before moving on to deep neural architectures, we discuss basic terms used in speech processing, low-level representations of speech signals, and traditional models used in the field.

## 2.1·Speech Signals

Signal processing is a fundamental discipline that encompasses the study of quantities that exhibit variations in space or time.
In the realm of signal processing, a quantity exhibiting spatial or temporal variations is commonly referred to as a signal.
Specifically, sound signals are defined as variations in air pressure.
Consequently, a speech signal is identified as a type of sound signal, namely pressure variations, generated by humans to facilitate spoken communication.
Transducers play a vital role in converting these signals from one form, such as air pressure, to another form, typically an electrical signal.

In signal processing, a signal that repetitively manifests after a fixed duration, known as a period, is classified as periodic.
The reciprocal of this period represents the frequency of the signal.
The waveform of a periodic signal defines its shape and concurrently determines its timbre, which pertains to the subjective perception of sound quality by humans.
To facilitate the processing of speech, speech signals are commonly digitized.
This entails converting them into a series of numerical values by measuring the signal's amplitude at consistent time intervals.
The sampling rate, defined by the number of samples collected per second, determines the granularity of this digitization process.

## 2.2·Speech Features

Speech features are numerical representations of speech signals that are used for analysis, recognition, and synthesis.
Broadly, speech signals can be classified into two categories: **time-domain** features and **frequency-domain** features.

### Time-Domain Features

**Time-domain** features are derived directly from the amplitude of the speech signal over time.
These are simple to compute and often used in real-time speech-processing applications.
Some common time-domain features include:

- **Energy**: Energy is a quantitative measure of the amplitude characteristics of a speech signal over time.
It is computed by squaring each sample in the signal and summing them within a specific time window.
This captures the overall strength and dynamics of the signal, revealing temporal variations in intensity.
The energy measure provides insights into segments with higher or lower amplitudes, aiding in speech recognition, audio segmentation, and speaker diarization.
It also helps identify events and transitions indicative of changes in vocal activity.
By quantifying amplitude variations, energy analysis contributes to a comprehensive understanding of speech signals and their acoustic properties.
- **Zero-Crossing Rate**: The zero-crossing rate indicates how frequently the speech signal crosses the zero-axis within a defined time frame.
It is computed by counting the number of polarity changes in the signal during a specific window.
- **Pitch**: Pitch refers to the perceived tonal quality in a speaker's voice, which is determined by analyzing the fundamental frequency of the speech signal.
The fundamental frequency can be estimated through the application of pitch detection algorithms \cite{rabiner1976comparative} or by utilizing autocorrelation techniques \cite{tan2003pitch}.
- **Linear Predictive Coding (LPC)**: Linear Predictive Coding (LPC) is a powerful technique that represents the speech signal as a linear combination of past samples, employing an autoregressive model.
The estimation of model parameters is accomplished through methods like the Levinson-Durbin algorithm \cite{castiglioni2005levinson}.
The obtained coefficients serve as a valuable feature representation for various speech-processing tasks.

### Frequency-Domain Features

**Frequency-domain** features are derived from the signal represented in the frequency domain also known as its spectrum.
A spectrum captures the distribution of energy as a function of frequency.
Spectrograms are two-dimensional visual representations capturing the variations in a signal's spectrum over time.
When compared against time-domain features, it is generally more complex to compute frequency-domain features as they tend to involve time-frequency transform operations such as Fourier transform.

- **Mel-spectrogram**: A Mel spectrogram, also known as a Mel-frequency spectrogram or Melspectrogram, is a representation of the short-term power spectrum of a sound signal.
It is widely used in audio signal processing and speech recognition tasks.
It is obtained by converting the power spectrum of a speech signal into a mel-scale, which is a perceptual scale of pitches based on the human auditory system's response to different frequencies.
The mel-scale divides the frequency range into a set of mel-frequency bands, with higher resolution in the lower frequencies and coarser resolution in the higher frequencies.
This scale is designed to mimic the non-linear frequency perception of human hearing.
To compute the Melspectrogram, the speech signal is typically divided into short overlapping frames.
For each frame, the Fast Fourier Transform (FFT) is applied to obtain the power spectrum.
The power spectrum is then transformed into the mel-scale using a filterbank that converts the power values at different frequencies to their corresponding mel-frequency bands.
Finally, the logarithm of the mel-scale power values is computed, resulting in the Melspectrogram.
Melspectrogram provides a time-frequency representation of the audio signal, where the time dimension corresponds to the frame index, and the frequency dimension represents the mel-frequency bands.
It captures both the spectral content and temporal dynamics of the signal, making it useful for tasks such as speech recognition, music analysis, and sound classification.
By using the Melspectrogram, the representation of the audio signal is transformed to a more perceptually meaningful domain, which can enhance the performance of various audio processing algorithms.
It is particularly beneficial in scenarios where capturing the spectral patterns and frequency content of the signal is important for the analysis or classification task at hand.
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: Mel-frequency cepstral coefficients (MFCCs) are a feature representation widely utilized in various applications such as speech recognition, gesture recognition, speaker identification, and cetacean auditory perception systems.
MFCCs capture the power spectrum of a sound over a short duration by utilizing a linear cosine transformation of a logarithmically-scaled power spectrum on a non-linear mel frequency scale.
The MFCCs consist of a set of coefficients that collectively form a Mel-frequency cepstrum ([Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)).
With just 12 parameters related to the amplitude of frequencies, MFCCs provide an adequate number of frequency channels to analyze audio, while still maintaining a compact representation.
The main objectives of MFCC extraction are to eliminate vocal fold excitation (F0) information related to pitch, ensure the independence of the extracted features, align with human perception of loudness and frequency, and capture the contextual dynamics of phones.
The process of extracting MFCC features involves A/D conversion, pre-emphasis filtering, framing, windowing, Fourier transform, Mel filter bank application, logarithmic operation, discrete cosine transform (DCT), and liftering.
By following these steps, MFCCs enable the extraction of informative audio features while avoiding redundancy and preserving the relevant characteristics of the sound signal.

### Other Speech Features

Other types of speech features include formant frequencies, pitch contour, cepstral coefficients, wavelet coefficients, and spectral envelope.
These features can be used for various speech-processing tasks, including speech recognition, speaker identification, emotion recognition, and speech synthesis.

In the field of speech processing, frequency-based representations such as Mel spectrogram and MFCC are widely used since they are more robust to noise as compared to temporal variations of the sound \cite{9955539}.
Time-domain features can be useful when the task warrants this information (such as pauses, emotions, phoneme duration, and speech segments).
It is noteworthy that the time-domain and frequency-domain features tend to capture different sets of information and thus can be used in conjunction to solve a task \cite{1165240,9053712,tang2021joint}.

## 2.3·Traditional Models for Speech Processing

Traditional speech representation learning algorithms based on shallow models utilize basic non-parametric models for extracting features from speech signals.
The primary objective of these models is to extract significant features from the speech signal through mathematical operations, such as Fourier transforms, wavelet transforms, and linear predictive coding (LPC).
The extracted features serve as inputs to classification or regression models.
The shallow models aim to extract meaningful features from the speech signal, enabling the classification or regression model to learn and make accurate predictions.

### Gaussian Mixture Models (GMMs)

Gaussian Mixture Models (GMMs) are powerful generative models employed to represent the probability distribution of a speech feature vector.
They achieve this by combining multiple Gaussian distributions with different weights.
GMMs have found widespread applications in speaker identification \cite{kinnunen2005real} and speech recognition tasks \cite{reynolds2003channel}.
Specifically, in speaker identification, GMMs are utilized to capture the distribution of speaker-specific features, enabling the recognition of individuals based on their unique characteristics.
Conversely, in speech recognition, GMMs are employed to model the acoustic properties of speech sounds, facilitating accurate recognition of spoken words and phrases.
GMMs play a crucial role in these domains, enabling robust and efficient analysis of speech-related data.

### Support Vector Machines (SVMs)

Support Vector Machines (SVMs) are a widely adopted class of supervised learning algorithms extensively utilized for various speech classification tasks \cite{smith2001speech}.
They are particularly effective in domains like speaker recognition  \cite{hatch2006within,solomonoff2004channel,solomonoff2005advances} and phoneme recognition \cite{campbell2003phonetic}.
SVMs excel in their ability to identify optimal hyperplanes that effectively separate different classes in the feature space.
By leveraging this optimal separation, SVMs enable accurate classification and recognition of speech patterns.
As a result, SVMs have become a fundamental tool in the field of speech analysis and play a vital role in enhancing the performance of speech-related classification tasks.

### Hidden Markov Models (HMMs)

Hidden Markov Models (HMMs) have gained significant popularity as a powerful tool for performing various speech recognition tasks, particularly ASR \cite{gales2008application, rabiner1989tutorial}.
In ASR, HMMs are employed to model the probability distribution of speech sounds by incorporating a sequential arrangement of hidden states along with corresponding observations.
The training of HMMs is commonly carried out using the Baum-Welch algorithm, a variant of the Expectation Maximization algorithm, which enables effective parameter estimation and model optimization [Wikipedia: Baum-Welch algorithm](http://en.wikipedia.org/wiki/Baum\%e2\%80\%93Welch\_algorithm).
By leveraging HMMs in speech recognition, it becomes possible to predict the most likely sequence of speech sounds given an input speech signal.
This enables accurate and efficient recognition of spoken language, making HMMs a crucial component in advancing speech recognition technology.
Their flexibility and ability to model temporal dependencies contribute to their widespread use in ASR and various other speech-related applications, further enhancing our understanding and utilization of spoken language.

### K-Nearest Neighbors (KNN)

The K-nearest neighbors (KNN) algorithm is a simple yet effective classification approach utilized in a wide range of speech-related applications, including speaker recognition \cite{sadjadi2014nearest} and language recognition.
The core principle of KNN involves identifying the K-nearest neighbors of a given input feature vector within the training data and assigning it to the class that appears most frequently among those neighbors.
This algorithm has gained significant popularity due to its practicality and intuitive nature, making it a reliable choice for classifying speech data in numerous real-world scenarios.
By leveraging the proximity-based classification, KNN provides a straightforward yet powerful method for accurately categorizing speech samples based on their similarities to the training data.
Its versatility and ease of implementation contribute to its widespread adoption in various speech-related domains, facilitating advancements in speaker recognition, language identification, and other applications in the field of speech processing.

### Decision Trees

Decision trees are widely employed in speech classification tasks as a class of supervised learning algorithms.
Their operation involves recursively partitioning the feature space into smaller regions, guided by the values of the features.
Within each partition, a decision rule is established to assign the input feature vector to a specific class.
The strength of decision trees lies in their ability to capture complex decision boundaries by hierarchically dividing the feature space.
By analyzing the values of the input features at each node, decision trees efficiently navigate the classification process.
This approach not only provides interpretability, but also facilitates the identification of key features contributing to the classification outcome.
Through their recursive partitioning mechanism, decision trees offer a flexible and versatile framework for speech classification.
They excel in scenarios where the decision rules are based on discernible thresholds or ranges of feature values.
The simplicity and transparency of decision trees make them a valuable tool for understanding and solving speech-related classification tasks.

### Summary

To summarize, conventional speech representation learning algorithms based on shallow models entail feature extraction from the speech signal, which is subsequently used as input for classification or regression models.
These algorithms have found extensive applications in speech processing tasks like speech recognition, speaker identification, and speech synthesis.
However, they have been progressively superseded by more advanced representation learning algorithms, particularly deep neural networks, due to their enhanced capabilities.