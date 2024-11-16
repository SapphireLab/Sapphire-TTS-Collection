# 3·Deep Learning Architectures and Their Applications in Speech Processing Tasks

Deep learning architectures have revolutionized the field of speech processing by demonstrating remarkable performance across various tasks.
With their ability to automatically learn hierarchical representations from raw speech data, deep learning models have surpassed traditional approaches in areas such as speech recognition, speaker identification, and speech synthesis.
These architectures have been instrumental in capturing intricate patterns, uncovering latent features, and extracting valuable information from vast amounts of speech data.
In this section, we delve into the applications of deep learning architectures in speech processing tasks, exploring their potential, advancements, and the impact they have had on the field.
By examining the key components and techniques employed in these architectures, we aim to provide insights into the current state-of-the-art in deep learning for speech processing and shed light on the exciting prospects it holds for future advancements in the field.

## 3.1·Recurrent Neural Networks (RNNs)

It is natural to consider Recurrent Neural Networks for various speech processing tasks since the input speech signal is inherently a dynamic process \cite{salehinejad2017recent}.
RNNs can model a given time-varying (sequential) patterns that were otherwise hard to capture by standard feedforward neural architectures.
Initially, RNNs were used in conjunction with HMMs where the sequential data is first modeled by HMMs while localized classification is done by the neural network.
However, such a hybrid model tends to inherit limitations of HMMs, for instance, HMM requires task-specific knowledge and independence constraints for observed states \cite{bourlard1994connectionist}.
To overcome the limitations inherited by the hybrid approach, end-to-end systems completely based on RNNs became popular for sequence transduction tasks such as speech recognition and text\cite{graves2012sequence, kawakami2008supervised}.
Next, we discuss RNN and it's variants:

### 3.1.1·RNN Models

#### Vanilla RNN

{Give input sequence of T states $(x_{1}, \ldots ,x_{T})$ with $x_i \in \mathbb{R}^d$, the output state at time $t$ can be computed as}

$$
    h_{t} = \mathcal{H}(W_{hh}h_{t-1}+W_{xh}x_{t}+b_h)
$$

$$
    y_{t} = W_{hy}h_{t}+b_y
$$

where $W_{hh}, W_{hx}, W_{yh}$ are weight matrices and $b_h, b_y$ are bias vectors.
$\mathcal{H}$ is non-linear activation functions such as Tanh, ReLU, and Sigmoid.
RNNs are made of high dimensional hidden states, notice $h_t$ in the above equation, which makes it possible for them to model sequences and help overcome the limitation of feedforward neural networks.
The state of the hidden layer is conditioned on the current input and the previous state, which makes the underlying operation recursive.
Essentially, the hidden state $h_{t-1}$ works as a memory of past inputs $\{x_k\}_{k=1}^{t-1}$ that influence the current output $y_t$.

#### Bidirectional RNNs

For numerous tasks in speech processing, it is more effective to process the whole utterance at once.
For instance, in speech recognition, one-shot input transcription can be more robust than transcribing based on the partial (i.e.
previous) context information \cite{graves2013speech}.
The vanilla RNN has a limitation in such cases as they are unidirectional in nature, that is, output $y_t$ is obtained from $\{x_k\}_{k=1}^{t}$, and thus, agnostic of what comes after time $t$.
Bidirectional RNNs (BRNNs) were proposed to overcome such shortcomings of RNNs \cite{schuster1997bidirectional}.
BRRNs encode both future and past (input) context in separate hidden layers.
The outputs of the two RNNs are then combined at each time step, typically by concatenating them together, to create a new, richer representation that includes both past and future context.

$$
\begin{aligned}
    \overrightarrow{h_{t}} &= \mathcal{H}(W_{\overrightarrow{hh}}\overrightarrow{h}_{t-1}+W_{\overrightarrow{xh}}x_{t}+b_{\overrightarrow{h}})\\
    \overleftarrow{h_{t}} &= \mathcal{H}(W_{\overleftarrow{hh}}\overleftarrow{h}_{t+1}+W_{\overleftarrow{xh}}x_{t}+b_{\overleftarrow{h}})\\
    y_{t} &= W_{\overrightarrow{hy}}\overrightarrow{h}_{t}+W_{\overleftarrow{hy}}\overleftarrow{h}_{t}+b_y
\end{aligned}
$$

where high dimensional hidden states $\overrightarrow{h}_{t-1}$ and $\overleftarrow{h}_{t+1}$ are hidden states modeling the forward context from $1,2,\ldots,t-1$ and backward context from $T, T-1, \ldots, t+1$, respectively.

#### Long Short-Term Memory (LSTM)

Vanilla RNNs are observed to face another limitation, that is, vanishing gradients that do not allow them to learn from long-range context information.
To overcome this, a variant of RNN, named as LSTM, was specifically designed to address the vanishing gradient problem and enable the network to selectively retain (or forget) information over longer periods of time \cite{hochreiter1997long}.
This attribute is achieved by maintaining separate purpose-built memory cells in the network: the long-term memory cell $c_t$ and the short-term memory cell $h_t$.
In Eq of RNN(h), LSTM redefines the operator $\mathcal{H}$ in terms of forget gate $f_t$, input gate $i_t$, and output gate $o_t$,

$$
\begin{aligned}
    i_t &= \sigma(W_{xi}x_t + W_{hi}h_{t-1}+W_{ci}c_{t-1}+b_i),\\
    f_t &= \sigma(W_{xf}x_t + W_{hf}h_{t-1}+W_{cf}c_{t-1}+b_f),\\
    c_t &= f_t\odot c_{t-1} + i_t\odot \tanh{(W_{xc}x_t+W_{hc}h_{t-1}+b_c)},\\
    o_t &= \sigma(W_{xo}x_t + W_{ho}h_{t-1}+W_{co}c_{t}+b_o),\\
    h_t &= o_t\odot \tanh{(c_t)},
\end{aligned}
$$

where $\sigma(x)=1/({1+e^{-x}})$ is a logistic sigmoid activation function.
$c_t$ is a fusion of the information from the previous state of the long-term memory $c_{t-1}$, the previous state of short-term memory $h_{t-1}$, and current input $x_t$.
$W$ and $b$ are weight matrices and biases.
$\odot$ is the element-wise vector multiplication or Hadamard operator.
Bidirectional LSTMs (BLSTMs) can capture longer contexts in both forward and backward directions \cite{graves2012sequence}.


#### Gated Recurrent Units (GRU)

Gated Recurrent Units (GRU) aim to be a computationally-efficient approximate of LSTM by using only two gates (vs three in LSTM) and a single memory cell (vs two in LSTM).
To control the flow of information over time, a GRU uses an update gate $z_t$ to decide how much of the new input to be added to the previous hidden state and a reset gate $r_t$ to decide how much of previous hidden state information to be forgotten.

$$
\begin{aligned}
    z_{t} &= \sigma(W_{xz}x_{t} + W_{hz}h_{t-1}),\\
    r_{t} &= \sigma(W_{xr}x_{t} + W_{hr}h_{t-1}), \\
    h_{t} &= (1-z_{t})\odot h_{t-1} + z_{t}\odot \tanh(W_{xh}x_{t} + W_{rh}(r_{t}\odot h_{t-1})),
\end{aligned}
$$

where $\odot$ is element-wise multiplication between the two vectors (Hadamard product).

RNNs and their variants are widely used in various deep learning applications like speech recognition, synthesis, and natural language understanding.
Although seq2seq based on recurrent architectures such as LSTM/GRU has made great strides in speech processing, they suffer from the drawback of slow training speed due to internal recurrence.
Another drawback of the RNN family is their inability to leverage information from long distant time steps accurately.

#### Connectionist Temporal Classification

Connectionist Temporal Classification (CTC) \cite{graves2012connectionist} is a scoring and output function commonly used to train LSTM networks for sequence-based problems with variable timing.
CTC has been applied to several tasks, including phoneme recognition, ASR, and other sequence-based problems.
One of the major benefits of CTC is its ability to handle unknown alignment between input and output, simplifying the training process.
When used in ASR  \cite{9688009,9747887,9053165}, CTC eliminates the need for manual data labeling by assigning probability scores to the output given any input signal.
This is particularly advantageous for tasks such as speech recognition and handwriting recognition, where the input and output can vary in size.
CTC also solves the problem of having to specify the position of a character in the output, allowing for more efficient training of the neural network without post-processing the output.
Finally, the CTC decoder can transform the neural network output into the final text without post-processing.

### 3.1.2·Application

The utilization of RNNs in popular products such as Google's voice search and Apple's Siri to process user input and predict the output has been well-documented \cite{he2017streaming,li2017acoustic}.
RNNs are frequently utilized in speech recognition tasks, such as the prediction of phonetic segments from audio signals \cite{papastratis2021speech}.
They excel in use cases where context plays a vital role in outcome prediction and are distinct from CNNs as they utilize feedback loops to process a data sequence that informs the final output \cite{papastratis2021speech}.

In recent times, there have been advancements in the architecture of RNNs, which have been primarily focused on developing end-to-end (E2E) models \cite{li2020towards,9746187} for ASR.
These E2E models have replaced conventional hybrid models and have displayed substantial enhancements in speech recognition \cite{li2020towards,li2021better}.
However, a significant challenge faced by E2E RNN models is the synchronization of the input speech sequence with the output label sequence \cite{graves2012sequence}.
To tackle this issue, a loss function called CTC \cite{graves2012connectionist} is utilized for training RNN models, allowing for the repetition of labels to construct paths of the same length as the input speech sequence.
An alternative method is to employ an Attention-based Encoder-Decoder (AED) model based on RNN architecture, which utilizes an attention mechanism to align the input speech sequence with the output label sequence.
However, AED models tend to perform poorly on lengthy utterances.

The development of Bimodal Recurrent Neural Networks (BRNN) has led to significant advancements in the field of Audiovisual Speech Activity Detection (AV-SAD) \cite{tao2019end}.
BRNNs have demonstrated immense potential in improving the performance of speech recognition systems, particularly in noisy environments, by combining information from various sources.
By integrating separate RNNs for each modality, BRNNs can capture temporal dependencies within and across modalities.
This leads to successful outcomes in speech-based systems, where integrating audio and visual modalities is crucial for accurate speech recognition.
Compared to conventional audio-only systems, BRNN-based AV-SAD systems display superior performance, particularly in challenging acoustic conditions where audio-only systems might struggle.

To enhance the performance of continuous speech recognition, LSTM networks have been utilized in hybrid architectures alongside CNNs \cite{passricha2019hybrid}.
The CNNs extract local features from speech frames that are then processed by LSTMs over time \cite{passricha2019hybrid}.
LSTMs have also been employed for speech synthesis, where they have been shown to enhance the quality of statistical parametric speech synthesis \cite{passricha2019hybrid}.

Aside from their ASR and speech synthesis applications, LSTM networks have been utilized for speech post-filtering.
To improve the quality of synthesized speech, researchers have proposed deep learning-based post-filters, with LSTMs demonstrating superior performance over other post-filter types \cite{coto2019improving}.
Bidirectional LSTM (Bi-LSTM) is another variant of RNN that has been widely used for speech synthesis \cite{fan2014tts}.
Several RNN-based analysis/synthesis models such as WaveNet \cite{oord2016wavenet}, SampleRNN \cite{mehri2016samplernn}, and Tacotron have been developed.
These neural vocoder models can generate high-quality synthesized speech from acoustic features without requiring intermediate vocoding steps.

## 3.2·Convolutional Neural Networks (CNNs)

Convolutional neural networks (CNNs) are a specialized class of deep neural architecture consisting of one or more pairs of alternating convolutional and pooling layers.
A convolution layer applies filters that process small local parts of the input, where these filters are replicated along the whole input space.
A pooling layer converts convolution layer activations to low resolution by taking the maximum filter activation within a specified
window and shifting across the activation map.
CNNs are variants of fully connected neural networks widely used for processing data with grid-like topology.
For example, time-series data (1D grid) with samples at regular intervals or images (2D grid) with pixels constitute a grid-like structure.

As discussed in [Sec.02](Sec.02.md), the speech spectrogram retains more information than hand-crafted features, including speaker characteristics such as vocal tract length differences across speakers, distinct speaking styles causing formant to undershoot or overshoot, etc.
Also, explicitly expressed these characteristics in the frequency domain.
The spectrogram representation shows very strong correlations in time and frequency.
Due to these characteristics of the spectrogram, it is a suitable input for a CNN processing pipeline that requires preserving locality in both frequency and time axis.
For speech signals, modeling local correlations with CNNs will be beneficial.
The CNNs can also effectively extract the structural features from the spectrogram and reduce the complexity of the model through weight sharing.
This section will discuss the architecture of 1D and 2D CNNs used in various speech-processing tasks.

### 3.2.1·CNN Model Variants

#### 2D CNN

Since spectrograms are two-dimensional visual representations, one can leverage CNN architectures widely used for visual data processing (images and videos) by performing convolutions in two dimensions.

The mathematical equation for a 2D convolutional layer can be represented as:

$$
y_{i,j}^{(k)}= \sigma\Big(\sum_{l=1}^{L} \sum_{m=1}^{M} x_{i+l-1,j+m-1}^{(l)} w_{l,m}^{(k)} + b^{(k)}\Big)
$$

Here, $x_{i,j}^{(l)}$ is the pixel value of the $l^{th}$ input channel at the spatial location $(i,j)$, $w_{l,m}^{(k)}$ is the weight of the $m^{th}$ filter at the $l^{th}$ channel producing the $k^{th}$ feature map, and $b^{(k)}$ is the bias term for the $k^{th}$ feature map.

The output feature map $y_{i,j}^{(k)}$ is obtained by convolving the input image with the filters and then applying an activation function $\sigma$ to introduce non-linearity.
The convolution operation involves sliding the filter window over the input image, computing the dot product between the filter and the input pixels at each location, and producing a single output pixel.

However, there are some drawbacks to using a 2D CNN for speech processing.
One of the main issues is that 2D convolutions are computationally expensive, especially for large inputs.
This is because 2D convolutions involve many multiplications and additions, and the computational cost grows quickly with the input size.

To address this issue, a 1D CNN can be designed to operate directly on the speech signal without needing a spectrogram.
1D convolutions are much less computationally expensive than 2D convolutions because they only operate on one dimension of the input.
This reduces the multiplications and additions required, making the network faster and more efficient.
In addition, 1D feature maps require less memory during processing, which is especially important for real-time applications.
A neural network's memory requirements are proportional to its feature maps' size.
By using 1D convolutions, the size of the feature maps can be significantly reduced, which can improve the efficiency of the network and reduce the memory requirements.

#### 1D CNN

1D CNN is essentially a special case of 2D CNN where the height of the filter is equal to the height the spectogram.
Thus, the filter only slides along the temporal dimension and the height of the resultant feature maps is one.
As such, 1D convolutions are computationally less expensive and memory efficient~\cite{KIRANYAZ2021107398}, as compared to 2D CNNs.
Several studies \cite{kiranyaz2015convolutional,Karita2019ImprovingTE,abdeljaber2017real} have shown that 1D CNNs are preferable to their 2D counterparts in certain applications.
For example, \citet{alsabhan2023human} found that the performance of predicting emotions with a 2D CNN model was lower compared to a 1D CNN model.

1D convolution is useful in speech processing for several reasons:

- Since, speech signals are sequences of amplitudes sampled over time, 1D convolution can be applied along temporal dimension to capture temporal variations in the signal.
- **Robustness to distortion and noise**:
  Since, 1D convolution allows local feature extraction, the resultant features are often resilient to global distortions of the signal.
  For instance, a speaker might be interrupted in the middle of an utterance.
  Local features would still produce robust representations for those relevant spans, which is key to ASR, among many speech processing task.
  On the other hand, speech signals are often contaminated with noise, making extracting meaningful information difficult.
  1D convolution followed by pooling layers can mitigate the impact of noise~\cite{hendrycks2018benchmarking}, improving speech recognition systems' accuracy.

The basic building block of a 1D CNN is the convolutional layer, which applies a set of filters to the input data.
A convolutional layer employs a collection of adjustable parameters called filters to carry out convolution operations on the input data, resulting in a set of feature maps as the output, which represent the activation of each filter at each position in the input data.
The size of the feature maps depends on the size of the input data, the size of the filters, and the number of filters used.
The activation function used in a 1D CNN is typically a non-linear function, such as the rectified linear unit (ReLU) function.

Given an input sequence $x$ of length $N$, a set of $K$ filters $W_k$ of length $M$, and a bias term $b_k$, the output feature map $y_k$ of the $k^{th}$ filter is given by

$$
    y_k[n]  = \ReLU(b_k + \sum_{m=0}^{M-1} W_k[m] * x[n-m])
$$

where $n$ ranges from $M-1$ to $N-1$, and $*$ denotes the convolution operation.
After the convolutional layer, the output tensor is typically passed through a pooling layer, reducing the feature maps' size by down-sampling.
The most commonly used pooling operation is the max-pooling, which keeps the maximum value from a sliding window across each feature map.

#### Summary

CNNs often replace previously popular methods like HMMs and GMM-UBM in various cases.
Moreover, CNNs possess the ability to acquire features that remain robust despite variations in speech signals resulting from diverse speakers, accents, and background noise.
This is made possible due to three key properties of CNNs: locality, weight sharing, and pooling.
The locality property enhances resilience against non-white noise by enabling the computation of effective features from cleaner portions of the spectrum.
Consequently, only a smaller subset of features is affected by the noise, allowing higher network layers a better opportunity to handle the noise by combining higher-level features computed for each frequency band.
This improvement over standard fully connected neural networks, which process all input features in the lower layers, highlights the significance of locality.
As a result, locality reduces the number of network weights that must be learned.

### 3.2.2·Application

CNNs have proven to be versatile tools for a range of speech-processing tasks.
They have been successfully applied to speech recognition \cite{nassif2019speech, 6857341}, including in hybrid NN-HMM models for speech recognition, and can be used for multi-class classification of words \cite{6288864}.
In addition, CNNs have been proposed for speaker recognition in an emotional speech, with a constrained CNN model presented in \cite{simic2022speaker}.

CNNs, both 1D and 2D, have emerged as the core building block for various speech processing models, including acoustic models \cite{schneider2019wav2vec,gulati2020conformer,kriman2020quartznet} in ASR systems.
For instance, in 2021, researchers from Facebook AI proposed wav2vec2.0 \cite{schneider2019wav2vec}, a hybrid ASR system based on CNNs for learning representations of raw speech signals that were then fed into a transformer-based language model.
The system achieved state-of-the-art results on several benchmark datasets.

Similarly, Google's VGGVox \cite{Nagrani17} used a CNN with VGG architecture to learn speaker embeddings from Mel spectrograms, achieving state-of-the-art results in speaker recognition.
CNNs have also been widely used in developing state-of-the-art speech enhancement and text-to-speech architectures.
For instance,
the architecture proposed in \cite{li2021real,tzinis2022remixit} for Deep Noise Suppression (DNS) \cite{reddy2020interspeech} challenge and Google's Tacotron2 \cite{shen2018natural} are examples of models that use CNNs as their core building blocks.
In addition to traditional tasks like ASR and speaker identification, CNNs have also been applied to non-traditional speech processing tasks like emotion recognition \cite{kakuba2022deep}, Parkinson's disease detection \cite{johri2019parkinson}, language identification \cite{singh2021spoken} and sleep apnea detection \cite{simply2019diagnosis}.
In all these tasks, CNN extracted features from speech signals and fed them into the task classification model.

## 3.3·Temporal Convolution Neural Networks

Recurrent neural networks, including RNNs, LSTMs, and GRUs, have long been popular for deep-learning sequence modeling tasks.
They are especially favored in the speech-processing domain.
However, recent studies have revealed that certain CNN architectures can achieve state-of-the-art accuracy in tasks such as audio synthesis, word-level language modelling, and machine translation, as reported in \cite{kalchbrenner2016neural,kalchbrenner2014convolutional,dauphin2017language}.
The advantage of convolutional neural networks is that they enable faster training by allowing parallel computation.
They can avoid common issues associated with recurrent models, such as the vanishing or exploding gradient problem or the inability to retain long-term memory.

In a recent study by \citet{bai2018empirical}, they proposed a generic Temporal Convolutional Neural Network (TCNN) architecture that can be applied to various speech-related tasks.
This architecture combines the best practices of modern CNNs and has demonstrated comparable performance to recurrent architectures such as LSTMs and GRUs.
The TCN approach could revolutionize speech processing by providing an alternative to the widely used recurrent neural network models.

### 3.3.1·TCNN Model Variants

The architecture of TCNN is based upon two principles:(1) There is no information “leakage” from future to past;(2) the architecture can map an input sequence of any length to an output sequence of the same length, similar to RNN.
TCN consists of dilated, causal 1D fully-convolutional layers with the same input and output lengths to satisfy the above conditions.
In other words, TCNN is simply a 1D fully-convolutional network (FCN) with casual convolutions as shown in \Cref{dilation}.

- Causal Convolution~\cite{oord2016wavenet}: Causal convolution convolves the input at a specific time point $t$ solely with the temporally-prior elements.
- Dilated Convolution~\cite{Yu2015MultiScaleCA}: By itself, causal convolution filters have a limited range of perception, meaning they can only consider a fixed number of elements in the past.
Therefore, it is challenging to learn any dependency between temporally distant elements for longer sequences.
Dilated convolution ameliorates this limitation by repeatedly applying dilating filters to expand the range of perception, as shown in \Cref{dilation}.
The dilation is achieved by uniformly inserting zeros between the filter weights.
Consider a 1-D sequence $x \in \mathbf{R}^{n}$ and a filter: $f: \{0,...,k-1\} \rightarrow \mathbf{R}$, the dilated convolution operation $F_d$ on an element $y$ of the sequence is defined as
$$
    F_d(y) = (x*_{d}f)(s) = \sum_{i=0}^{k-1}f(i).x_{y-d.i},
$$
where $k$ is filter size, $d$ is dilation factor, and $y-d.i$ is the span along the past.
The dilation step introduces a fixed step between every two adjacent filter taps.
When $d=1$, a dilated convolution acts as a normal convolution.
Whereas, for larger dilation, the filter acts on a wide but non-contiguous range of inputs.
Therefore, dilation effectively expands the receptive field of the convolutional networks.

#### 3.3.2·Application

Recent studies have shown that the TCNN architecture not only outperforms traditional recurrent networks like LSTMs and GRUs in terms of accuracy but also possesses a set of advantageous properties, including:

- Parallelism is a key advantage of TCNN over RNNs.
In RNNs, time-step predictions depend on their predecessors' completion, which limits parallel computation.
In contrast, TCNNs apply the same filter to each span in the input, allowing parallel application thereof.
This feature enables more efficient processing of long input sequences compared to RNNs that process sequentially.
- The receptive field size can be modified in various ways to enhance the performance of TCNNs.
For example, incorporating additional dilated convolutional layers, employing larger dilation factors, or augmenting the filter size are all effective methods.
Consequently, TCNNs offer superior management of the model's memory size and are highly adaptable to diverse domains.
- When dealing with lengthy input sequences, LSTM and GRU models tend to consume a significant amount of memory to retain the intermediate outcomes for their numerous cell gates.
On the other hand, TCNNs utilize shared filters throughout a layer, and the back-propagation route depends solely on the depth of the network.
This makes TCNNs a more memory-efficient alternative to LSTMs and GRUs, especially in scenarios where memory constraints are a concern.

TCNNs can perform real-time speech enhancement in the time domain \cite{pandey2019tcnn}.
They have much fewer trainable parameters than earlier models, making them more efficient.
TCNs have also been used for speech and music detection in radio broadcasts \cite{hung2022large,lemaire2019temporal}.
They have been used for single channel speech enhancement \cite{9601275,richter2020speech} and are trained as filter banks to extract features from waveform to improve the performance of ASR \cite{li2019single}.

## 3.4·Transformers

While recurrence in RNNs (\cref{sec:rnn}) is a boon for neural networks to model sequential data, it is also a bane as the recurrence in time to update the hidden state intrinsically precludes parallelization.
Additionally, although dedicated gated RNNs such as LSTM and GRU have helped to mitigate the vanishing gradient problem to some extent, it can still be a challenge to maintain long-term dependencies in RNNs.

Proposed by \citet{vaswani2017attention}, Transformer solved a critical shortcoming of RNNs by allowing parallelization within the training sample, that is, facilitating the processing of the entire input sequence at once.
Since then, the primary idea of using only the attention mechanism to construct an encoder and decoder has served as the basic recipe for many state-of-the-art architectures across the domains of machine learning.
In this survey, we use \textbf{transformer} to denote architectures that are inspired by Transformer \cite{devlin2018bert, brown2020language, radford2019language, radford2018improving, han2022survey}.
This section overviews the transformer's fundamental design proposed by \citet{vaswani2017attention} and its adaptations for different speech-related applications.

### 3.4.1·Basic Architecture

Transformer architecture \cite{vaswani2017attention} comprises an attention-based encoder and decoder, with each module consisting of a stack of identical blocks.
Each block in the encoder and decoder consists of two sub-layers: a multi-head attention (MHA) mechanism and a position-wise fully connected feedforward network as described in \Cref{fig:Transformer}.
The MHA mechanism in the encoder allows each input element to attend to every other element in the sequence, enabling the model to capture long-range dependencies in the input sequence.
The decoder typically uses a combination of MHA and encoder-decoder attention to attend to both the input sequence and the previously generated output elements.
The feedforward network in each block of the Transformer provides non-linear transformations to the output of the attention mechanism.
Next, we discuss operations involved in transformer layers, that is, multi-head attention and position-wise feedforward network:

#### Attention in Transformers

Attention mechanism, first proposed by \citet{bahdanau2014neural}, has revolutionized sequence modeling and transduction models in various tasks of NLP, speech, and computer vision \cite{galassi2020attention, cho2015describing, wang2016survey, chaudhari2021attentive}.
Broadly, it allows the model to focus on specific parts of the input or output sequence, without being limited by the distance between the elements.
We can describe the attention mechanism as the mapping of a query vector and set of key-value vector pairs to an output.
Precisely, the output vector is computed as a weighted summation of value vectors where the weight of a value vector is obtained by computing the compatibility between the query vector and key vector.
Let, each query $Q$ and key $K$ are $d_k$ dimensional and value $V$ is $d_v$ dimensional.
Specific to the Transformer, the compatibility function between a query and each key is computed as their dot product between scaled by $\sqrt{d_k}$.
To obtain the weights on values, the scaled dot product values are passed through a softmax function:

$$
    \Attn(\textbf{Q},\textbf{K},\textbf{V}) = \Softmax\left(\frac{\textbf{Q}\textbf{K}^{T}}{\sqrt{d_{k}}}\right)\textbf{V}
$$

{Here multiple queries, keys, and value vectors, are packed together in matrix form respectively denoted by $\textbf{Q} \in {\mathbb{R}^{N\times d_{k}}}$, $\textbf{K} \in {\mathbb{R}^{M\times d_{k}}}$, and $\textbf{V} \in {\mathbb{R}^{M\times d_{v}}}$.
\textit{N} and \textit{M} represent the lengths of queries and keys (or values).
Scaling of dot product attention becomes critical to tackling the issue of small gradients with the increase in $d_k$ \cite{vaswani2017attention}.}

Instead of performing single attention in each transformer block, multiple attentions in lower-dimensional space have been observed to work better \cite{vaswani2017attention}.
This observation gave rise to \textbf{Multi-Head Attention}: For $h$ heads and dimension of tokens in the model $d_m$, the $d_m$-dimensional query, key, and values are projected $h$ times to $d_k$, $d_k$, and $d_v$ dimensions using learnable linear projections.
Projection weights are neither shared across heads nor query, key, and values.
Each head performs attention operation as per \Cref{attention}.
The $h$ $d_v$-dimensional are concatenated and projected back to $d_{m}$ using another projection matrix:

$$
\begin{aligned}
    \MultiHead(\textbf{Q},\textbf{K},\textbf{V}) &= \Concat(\Head{}_{1},....\Head{}_{h})\textbf{W}^{O}, \\
    \text{with } \Head{}_{i} &=\Attn(\textbf{Q}\textbf{W}^{Q}_{i},\textbf{K}\textbf{W}^{K}_{i},\textbf{V}\textbf{W}^{V}_{i})
\end{aligned}
$$

Where $\textbf{W}^{Q}, \textbf{W}^{K} \in {\mathbb{R}^{d_{model}\times d_{k}}},  \textbf{W}^{V} \in \mathbb{R}^{d_{model}\times d_{v}}, \textbf{W}^{O} \in {\mathbb{R}^{hd_{v}\times d_{model}}}$ are learnable projection matrices.
Intuitively, multiple attention heads allow for attending to parts of the sequence differently (e.g., longer-term dependencies versus shorter-term dependencies).
Intuitively, multiple attention heads allow for attending in different representational spaces jointly.

#### Position-wise FFN

The position-wise FNN consists of two dense layers.
It is referred to position-wise since the same two dense layers are used for each positioned item in the sequence and are equivalent to applying two $1\times1$ convolution layers.
\paragraph{Residual Connection and Normalization}
Residual connection and Layer Normalization are employed around each module for building a deep model.
For example, each encoder block output can be defined as follows:

$$
H^{'} = \LayerNorm(\SelfAttention(X) + X)
$$

$$
H = \LayerNorm(\FFN(H^{'}) + H^{'})
$$

$\SelfAttention(.)$ denotes attention module with $\textbf{Q} = \textbf{K} = \textbf{V} = \textbf{X}$, where $\textbf{X}$ is the output of the previous layer.

Transformer-based architecture turned out to be better than many other architectures such as RNN, LSTM/GRU, etc.
One of the major difficulties when applying a Transformer to
speech applications that it requires more complex configurations
(e.g., optimizer, network structure, data augmentation) than the conventional RNN-based models.
 Speech signals are continuous-time signals with much higher dimensionality than text data.
This high dimensionality poses significant computational challenges for the Transformer architecture, originally designed for sequential text data.
Speech signals also have temporal dependencies, which means that the model needs to be able to process and learn from the entire signal rather than just a sequence of inputs.
Also, speech signals are inherently variable and complex.
The same sentence can be spoken differently and even by the same person at different times.
This variability requires the model to be robust to differences in pitch, accent, and speed of speech.

### 3.4.2·Application

Recent advancements in NLP which lead to a paradigm shift in the field are highly attributed to the foundation models that are primarily a part of the transformers category, with self-attention being a key ingredient \cite{bommasani2021opportunities}.
The recent models have demonstrated human-level performance in several professional and academic benchmarks.
For instance, GPT4 scored within the top 10\% of test takers on a simulated version of the Uniform Bar Examination \cite{OpenAI2023GPT4TR}.
While speech processing has not yet seen a shift in paradigm as in NLP owing to the capabilities of foundational models, even so, transformers have significantly contributed to advancement in the field including but not limited to the following tasks: automatic speech recognition, speech translation, speech synthesis, and speech enhancement, most of which we discuss in detail in \Cref{speech_processing_tasks}.

RNNs and Transformers are two widely adopted neural network architectures employed in the domain of Natural Language Processing (NLP) and speech processing.
While RNNs process input words sequentially and preserve a hidden state vector over time, Transformers analyze the entire sentence in parallel and incorporate an internal attention mechanism.
This unique feature makes Transformers more efficient than RNNs \cite{karita2019comparative}.
Moreover, Transformers employ an attention mechanism that evaluates the relevance of other input tokens in encoding a specific token.
This is particularly advantageous in machine translation, as it allows the Transformer to incorporate contextual information, thereby enhancing translation accuracy \cite{karita2019comparative}.
To achieve this, Transformers combine word vector embeddings and positional encodings, which are subsequently subjected to a sequence of encoders and decoders.
These fundamental differences between RNNs and Transformers establish the latter as a promising option for various natural language processing tasks \cite{karita2019comparative}.

A comparative study on transformer vs.
RNN   \cite{karita2019comparative} in speech applications found that transformer neural networks achieve state-of-the-art performance in neural machine translation and other natural language processing applications \cite{karita2019comparative}.
The study compared and analysed transformer and conventional RNNs in a total of 15 ASR, one multilingual ASR, one ST, and two TTS applications.
The study found that transformer neural networks outperformed RNNs in most applications tested.
Another survey of transformer-based models in speech processing found that transformers have an advantage in comprehending speech, as they analyse the entire sentence simultaneously, whereas RNNs process input words one by one.

Transformers have been successfully applied in end-to-end speech processing, including automatic speech recognition (ASR), speech translation (ST), and text-to-speech (TTS) \cite{li2019neural}.
In 2018, the Speech-Transformer was introduced as a no-recurrence sequence-to-sequence model for speech recognition.
To reduce the dimension difference between input and output sequences, the model's architecture was modified by adding convolutional neural network (CNN) layers before feeding the features to the transformer.
In a later study \cite{nakatani2019improving}, the authors proposed a method to improve the performance of end-to-end speech recognition models based on transformers.
They integrated the connectionist temporal classification (CTC) with the transformer-based model to achieve better accuracy and used language models to incorporate additional context and mitigate recognition errors.

In addition to speech recognition, the transformer model has shown promising results in TTS applications.
The transformer based TTS model generates mel-spectrograms, followed by a WaveNet vocoder to output the final audio results \cite{li2019neural}.
Several neural network-based TTS models, such as Tacotron 2, DeepVoice 3, and transformer TTS, have outperformed traditional concatenative and statistical parametric approaches in terms of speech quality \cite{li2019neural, shen2018natural, ping2017deep}.

One of the strengths of Transformer-based architectures for neural speech synthesis is their high efficiency while considering the global context \cite{gulati2020conformer,shi2020weak}.
The Transformer TTS model has shown advantages in training and inference efficiency over RNN-based models such as Tacotron 2 \cite{shen2018natural}.
The efficiency of the Transformer TTS network can speed up the training about 4.25 times \cite{li2019neural}.
Moreover, Multi-Speech, a multi-speaker TTS model based on the Transformer \cite{li2019neural}, has demonstrated the effectiveness of synthesizing a more robust and better quality multi-speaker voice than naive Transformer-based TTS.

In contrast to the strengths of Transformer-based architectures in neural speech synthesis, large language models based on Transformers such as BERT \cite{devlin2018bert}, GPT \cite{radford2018improving}, XLNet \cite{yang2019xlnet}, and T5 \cite{raffel2020exploring} have limitations when it comes to speech processing.
One of the issues is that these models require discrete tokens as input, necessitating using a tokenizer or a speech recognition system, introducing errors and noise.
Furthermore, pre-training on large-scale text corpora can lead to domain mismatch problems when processing speech data.
To address these limitations, dedicated frameworks have been developed for learning speech representations using transformers, including wav2vec \cite{schneider2019wav2vec}, data2vec \cite{baevski2022data2vec}, Whisper \cite{radford2022robust}, VALL-E \cite{wang2023neural}, Unispeech \cite{wang2021unispeech}, SpeechT5 \cite{ao2021speecht5} etc.
We discuss some of them as follows.

- Speech representation learning frameworks, such as wav2vec, have enabled significant advancements in speech processing tasks.
One recent framework, w2v-BERT \cite{wang2019bridging}, combines contrastive learning and MLM to achieve self-supervised speech pre-training on discrete tokens.
Fine-tuning wav2vec models with limited labeled data has also been demonstrated to achieve state-of-the-art results in speech recognition tasks \cite{baevski2019vq}.
Moreover, XLS-R \cite{babu2021xls}, another model based on wav2vec 2.0, has shown state-of-the-art results in various tasks, domains, data regimes, and languages, by leveraging multilingual data augmentation and contrastive learning techniques on a large scale.
These models learn universal speech representations that can be transferred across languages and domains, thus representing a significant advancement in speech representation learning.
-  Transformers have been increasingly popular in the development of frameworks for learning representations from multi-modal data, such as speech, images, and text.
Among these frameworks, Data2vec \cite{baevski2022data2vec} is a self-supervised training approach that aims to learn joint representations to capture cross-modal correlations and transfer knowledge across modalities.
It has outperformed other unsupervised methods for learning multi-modal representations in benchmark datasets.
However, for tasks that require domain-specific models, such as speech recognition or speaker identification, domain-specific models may be more effective, particularly when dealing with data in specific domains or languages.
The self-supervised training approach of Data2vec enables cost-effective and scalable learning of representations without requiring labeled data, making it a promising framework for various multi-modal learning applications.
- The field of speech recognition has undergone a revolutionary change with the advent of the Whisper model \cite{radford2022robust}.
This innovative solution has proven to be highly versatile, providing exceptional accuracy for various speech-related tasks, even in challenging environments.
The Whisper model achieves its outstanding performance through a minimalist approach to data pre-processing and weak supervision, which allows it to deliver state-of-the-art results in speech processing.
The model is capable of performing multilingual speech recognition, translation, and language identification, thanks to its training on a diverse audio dataset.
Its multitasking model can cater to various speech-related tasks, such as transcription, voice assistants, education, entertainment, and accessibility.
One of the unique features of Whisper is its minimalist approach to data pre-processing, which eliminates the need for significant standardization and simplifies the speech recognition pipeline.
The resulting models generalize well to standard benchmarks and deliver competitive performance without fine-tuning, demonstrating the potential of advanced machine learning techniques in speech processing.
- Text-to-speech synthesis has been a topic of interest for many years, and recent advancements have led to the development of new models such as VALL-E \cite{wang2023neural}.
VALL-E is a novel text-to-speech synthesis model that has gained significant attention due to its unique approach to the task.
Unlike traditional TTS systems, VALL-E treats the task as a conditional language modelling problem and leverages a large amount of semi-supervised data to train a generalized TTS system.
It can generate high-quality personalized speech with a 3-second acoustic prompt from an unseen speaker and provides diverse outputs with the same input text.
VALL-E also preserves the acoustic environment and the speaker's emotions about the acoustic prompt, without requiring additional structure engineering, pre-designed acoustic features, or fine-tuning.
Furthermore, VALL-E X \cite{zhang2023speak} is an extension of VALL-E that enables cross-lingual speech synthesis, representing a significant advancement in TTS technology.

The timeline highlights the development of large transformer based models for speech processing is shown in \Cref{fig:timeline}.
The size of the models has grown exponentially, with significant breakthroughs achieved in speech recognition, synthesis, and translation.
These large models have set new performance benchmarks in the field of speech processing, but also pose significant computational and data requirements for training and inference.
