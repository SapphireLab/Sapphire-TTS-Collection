# 4·Speech Representation Learning

The process of speech representation learning is essential for extracting pertinent and practical characteristics from speech signals, which can be utilized for various downstream tasks such as speaker identification, speech recognition, and emotion recognition.
While traditional methods for engineering features have been extensively used, recent advancements in deep-learning-based techniques utilizing supervised or unsupervised learning have shown remarkable potential in this field.
Nonetheless, a novel approach founded on self-supervised representation learning has surfaced, aiming to unveil the inherent structure of speech data and acquire representations that capture the underlying structure of the data.
This approach surpasses traditional feature engineering methods and can significantly increase the accuracy to a considerable extent and effectiveness of downstream tasks.
The primary objective of this new paradigm is to uncover informative and meaningful features from speech signals and outperform existing approaches.
Therefore, this approach is considered a promising direction for future research in speech representation learning.

This section provides a comprehensive overview of the evolution of speech representation learning with neural networks.
We will examine various techniques and architectures developed over the years, including the emergence of unsupervised representation learning methods like autoencoders, generative adversarial networks (GANs), and self-supervised representation learning frameworks.
We will also examine the difficulties and constraints associated with these techniques, such as data scarcity, domain adaptation, and the interpretability of learned representations.
Through a comprehensive analysis of the advantages and limitations of different representation learning approaches, we aim to provide insights into how to harness their power to improve the accuracy and robustness of speech processing systems.

## 4.1·Supervised Learning

In supervised representation learning, the model is trained using annotated datasets to learn a mapping between input data and output labels.
The set of parameters that define the mapping function is optimized during training to minimize the difference between the predicted and true output labels in the training data.
The goal of supervised representation learning is to enable the model to learn a useful representation or features of the input data that can be used to accurately predict the output label for new, unseen data.
For instance, supervised representation learning in speech processing using CNNs learn speech features from spectrograms.
CNNs can identify patterns in spectrograms relevant to speech recognition, such as those corresponding to different phonemes or words.
Unlike CNNs, which typically require spectrogram input, RNNs can directly take in the raw speech signals as input and learn to extract features or representations that are relevant for speech recognition or other speech-processing tasks.
Learning speaker representations typically involves minimizing a loss function.
\citet{chung2020defence} compares their effectiveness for speaker recognition tasks, we distill it in \Cref{lossfunction} to present an overview of commonly used loss functions.
Additionally, a new angular variant of the prototypical loss is introduced in their work.
Results from extensive experimental validation on the VoxCeleb1 test set indicate that the GE2E and prototypical networks outperform other models in terms of performance.

### 4.1.1·Deep Speaker Representation


Speaker representation is a critical aspect of speech processing, allowing machines to analyze and process various parts of a speaker's voice, including pitch, intonation, accent, and speaking style.
In recent years, deep neural networks (DNNs) have shown great promise in learning robust features for speaker recognition.
This section reviews deep learning-based techniques for speaker representation learning that have demonstrated significant improvements over traditional methods.

These deep speaker representations can be applied to a range of speaker-recognition tasks beyond verification and identification, including diarization \cite{wang2018speaker,zhang2019fully,larcher2021speaker}, voice conversion \cite{wu2020one,lin2021s2vc,chou2019one}, multi-speaker TTS \cite{saito2021perceptual,paul2021universal,xue2022ecapa}, speaker adaptation \cite{chorowski2019unsupervised} etc.
To provide a comprehensive overview, we analyzed deep embeddings from the perspectives of input raw \cite{jung2019rawnet,ravanelli2018speaker} or mel-spectogram \cite{snyder2018x}, network architecture \cite{lin2020wav2spk,desplanques2020ecapa}, temporal pooling strategies \cite{monteiro2019combining}, and loss functions \cite{snell2017prototypical,chung2020defence,wang2018cosface}.
In the following subsection, we introduce two representative deep embeddings: $d$-vector \cite{variani2014deep} and $x$-vector \cite{snyder2017deep,snyder2018x}.
These embeddings have been widely adopted recently and have demonstrated state-of-the-art performance in various speaker-recognition tasks.
By understanding the strengths and weaknesses of different deep learning-based techniques for speaker-representation learning, we can better leverage their power to improve the accuracy and robustness of speaker-recognition systems.

- d-vector technique, proposed by Variani et al.
(2014) \cite{variani2014deep}, serves as a frame-level speaker embedding method, as illustrated in Figure \ref{fig:dvectors}.
In this approach, during the training phase, each frame within a training utterance is labeled with the speaker's true identity.
This transforms the training process into a classification task, where a maxout Deep Neural Network (DNN) classifies the frames based on the speaker's identity.
The DNN employs softmax as the output layer to minimize the cross-entropy loss between the ground-truth frame labels and the network's output.
During the testing phase, the $d$-vector technique extracts the output activation of each frame from the last hidden layer of the DNN, serving as the deep embedding feature for that frame.
To generate a compact representation called the $d$-vector, the technique computes the average of the deep embedding features from all frames within an utterance.
The underlying hypothesis is that the compact representation space developed using a development set can effectively generalize to unseen speakers during the testing phase \cite{variani2014deep}.
- x-vector \cite{snyder2017deep,snyder2018x} is a segment-level speaker embedding and an advancement over the $d$-vector method as it incorporates additional modeling of temporal information and phonetic information in speech signals, resulting in improved performance compared to the $d$-vector.
$x$-vector employs an aggregation process to move from frame-by-frame speaker labeling to utterance-level speaker labeling as highlighted in \Cref{fig:xvectors}.
The network structure of the $x$-vector is depicted in a figure, which consists of time-delay layers for extracting frame-level speech embeddings, a statistical pooling layer for concatenating mean and standard deviation of embeddings as a segment-level feature, and a standard feedforward network for classifying the segment-level feature to its speaker.
$x$-vector is the segment-level speaker embedding generated from the feedforward network's second-to-last hidden layer.
The authors in \cite{yang2022data,9287426} have also discovered the significance of data augmentation in enhancing the performance of the $x$-vector.