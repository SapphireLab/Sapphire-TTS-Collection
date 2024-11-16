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
In \Cref{rnn_h}, LSTM redefines the operator $\mathcal{H}$ in terms of forget gate $f_t$, input gate $i_t$, and output gate $o_t$,

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
