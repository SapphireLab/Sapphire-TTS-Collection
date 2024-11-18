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

## 4.2·Unsupervised Learning

Unsupervised representation learning for speech processing has gained significant emphasis over the past few years.
Similar to visual modality in CV and text modality in NLP, speech i.e.
audio modality introduces unique challenges.
Unsupervised speech representation learning is concerned with learning useful speech representations without using annotated data.
Usually, the model is first pre-trained on the task where plenty of data is available.
The model is then fined tuned or used to extract input representations for a small model, specifically targeting tasks with limited data.

One approach to addressing the unique challenges of unsupervised speech representation learning is to use probabilistic latent variable models (PLVM), which assume an unknown generative process produces the data and enables the learning of rich structural representations and reasoning about observed and unobserved factors of variation in complex datasets such as speech within a probabilistic framework.
PLVM specified a joint distribution $p(x,z)$ over unobserved stochastic latent variable \textit{z} and observed variables \textit{x}.
By factorizing the joint distribution into modular components, it becomes possible to learn rich structural representations and reason about observed and unobserved factors of variation in complex datasets such as speech within a probabilistic framework.

The likelihood of a PLVM given a data \textit{x} can be written as

$$
    p(x) = \int p(x|z) p(z) dz.
$$

Probabilistic latent variable models provide a powerful way to learn a representation that captures the underlying relationships between observed and unobserved variables, without requiring explicit supervision or labels.
These models involve unobserved latent variables that must be inferred from the observed data, typically using probabilistic inference techniques such as Markov Chain Monte Carlo (MCMC) methods.
In the context of representation learning, Variational autoencoders (VAE) are commonly used with latent variable models for various speech processing tasks, leveraging the power of probabilistic modeling to capture complex patterns in speech data.

## 4.3·Semi-Supervised Learning

Semi-supervised learning can be viewed as a process of optimizing a model using both labeled and unlabeled data.
The set of labeled data points, denoted by $X_L$, contains $N_L$ items, where each item is represented as $(x_i, y_i)$ with $y_i$ being the label of $x_i$.
On the other hand, the set of unlabeled data points, denoted by $X_U$, consists of $N_U$ items, represented as $x_{N_L+1}, x_{N_L+2}, ..., x_{N_L+N_U}$.

In semi-supervised learning, the objective is to train a model $f_{\theta}$ with parameters $\theta$ that can minimize the expected loss over the entire dataset.
The loss function $L(y, f_{\theta}(x))$ is used to quantify the deviation between the model's prediction $f_{\theta}(x)$ and the ground truth label $y$.
The expected loss can be mathematically expressed as:

$$
L(y, f_{\theta}(x)) = E_{(x,y) \sim p_{data}(x,y)}[L(y,f_{\theta}(x))]
$$

where $p_{data}(x,y)$ is the underlying data distribution.In semi-supervised learning, the loss function is typically decomposed into two parts: a supervised loss term that is only defined on the labeled data, and an unsupervised loss term that is defined on both labeled and unlabelled data.
The supervised loss term is calculated as follows:

$$
    \mathcal{L}_{sup} = \frac{1}{N_{L}} \sum_{(x,y) \in X_{L}} L(y, f_{\theta}(x))
$$

The unsupervised loss term leverages the unlabelled data to encourage the model to learn meaningful representations that capture the underlying structure of the data.
One common approach is to use a regularization term that encourages the model to produce similar outputs for similar input data.
This can be achieved by minimizing the distance between the output of the model for two similar input data points.
One such regularization term is the entropy minimization term, which can be expressed as:

$$
    \mathcal{L}_{unsup} =  \frac{1}{N_{U}} \sum_{(x_{i}) \in X_{U}} \sum_{j=1}^{|y|} p_{\theta}(y_{j},x_{i}) \log p_{\theta} (y_{j},x_{i})
$$

where $p_{\theta}(y_j|x_i)$ is the predicted probability of the $j$-th label for the unlabelled data point $x_i$.
Finally the overall objective function for semi-supervised learning can be expressed as $\mathcal{L} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{unsup} $, $\alpha$ is a hyperparameter that controls the weight of the unsupervised loss term.
The goal is to find the optimal parameters $\theta$ that minimize this objective function.
Semi-supervised learning involves learning a model from both labelled and unlabelled data by minimizing a combination of supervised and unsupervised loss terms.
By leveraging the additional unlabelled data, semi-supervised learning can improve the generalization and performance of the model in downstream tasks.

Semi-supervised learning techniques are increasingly being employed to enhance the performance of DNNs across a range of downstream tasks in speech processing, including ASR, TTS, etc.
The primary objective of such approaches is to leverage large unlabelled datasets to augment the performance of supervised tasks that rely on labelled datasets.
The recent advancements in speech recognition have led to a growing interest in the integration of semi-supervised learning methods to improve the performance of ASR and TTS systems \cite{zhang2020pushing,baskar2019semi,9795080,kahn2020self,xu2021self,9207023}.
This approach is particularly beneficial in scenarios where labelled data is scarce or expensive to acquire.
In fact, for many languages around the globe, labelled data for training ASR models are often inadequate, making it challenging to achieve optimal results.
Thus, using a semi-supervised learning model trained on abundant resource data can offer a viable solution that can be readily extended to low-resource languages.

Semi-supervised learning has emerged as a valuable tool for addressing the challenges of insufficient annotations and poor generalization \cite{hady2013semi}.
Research in various domains, including image quality assessment \cite{liu2019exploiting}, has demonstrated that leveraging both labelled and unlabelled data through semi-supervised learning can lead to improved performance and generalization.
In the domain of speech quality assessment, several studies \cite{serra2021sesqa} have exploited the generalization capabilities of semi-supervised learning to enhance performance.

Moreover, semi-supervised learning has gained significant attention in other areas of speech processing, such as end-to-end speech translation \cite{pino2020self}.
By leveraging large amounts of unlabelled data, semi-supervised learning approaches have demonstrated promising results in improving the performance and robustness of speech translation models.
This highlights the potential of semi-supervised learning to address the limitations of traditional supervised learning approaches in a variety of speech processing tasks.

## 4.4·Self-Supervised Representation Learning (SSRL)

Self-supervised representation learning (SSRL) is a machine learning approach that focuses on achieving robust and in-depth feature learning while minimizing reliance on extensively annotated datasets, thus reducing the annotation bottleneck \cite{ericsson2022self,lee2022self}.
SSRL comprises various techniques that allow models to be trained without needing human-annotated labels \cite{ericsson2022self,lee2022self}.
One of the key advantages of SSRL is its ability to operate on unlabelled datasets, which reduces the need for large annotated datasets \cite{ericsson2022self,lee2022self}.
In recent years, self-supervised learning has progressed rapidly, with some methods approaching or surpassing the efficacy of fully supervised learning methods.
Self-supervised learning methods typically involve pretext tasks that generate pseudo labels for discriminative model training without actual labeling.
The difference between self-supervised representation learning and unsupervised representation is highlighted in \cref{fig:unsupervised}.
In contrast to unsupervised representation learning, SSRL techniques are designed to generate these pseudo labels for model training.
The ability of SSRL to achieve robust and in-depth feature learning without relying heavily on annotated datasets holds great promise for the continued development of machine learning techniques.

SSRL differs from supervised learning mainly in terms of its data requirements.
While supervised learning relies on labeled data, where the model learns from input-output pairs, SSL generates its own labels from the input data, eliminating the need for labeled data \cite{lee2022self}.
The SSL approach trains the model to predict a portion of the input data, which is then utilized as a label for the task at hand \cite{lee2022self}.
Although SSRL is an unsupervised learning technique, it seeks to tackle tasks commonly associated with supervised learning without relying on labeled data \cite{lee2022self}.

### 4.4.1·Generative Models

This method involves instructing a model to produce samples resembling the input data without explicitly learning the labels, creating valuable representations applicable to other tasks.
The detailed architecture for generative models with three different variants is shown in \Cref{fig:ssrl-gen}.
The earliest self-supervised method, predicting masked inputs using surrounding data, originated from the text field in 2013 with word2vec.
The continuous bag of words (CBOW) concept of word2vec predicts a central word based on its neighbors, resembling ELMo and BERT's masked language modeling (MLM).
These non-autoregressive generative approaches differ in their use of advanced structures, such as bidirectional LSTM (for ELMo) and transformer (for BERT), with recent models producing contextual embeddings.
In the context of the speech, Mockingjay \cite{liu2020mockingjay} applied masking to all feature dimensions in the speech domain, whereas TERA \cite{liu2021tera} applied to mask only to a particular subset of feature dimensions.
The summary of generative self-supervised approaches along with the data used for training the models are outlined in \Cref{SSRL:gen}.
We further discuss different generative approaches as highlighted in \Cref{fig:ssrl-gen} as follows:

#### Auto-encoding Models

Auto-encoding Models have garnered significant attention in the domain of self-supervised learning, particularly Autoencoders (AEs) and Variational Autoencoders (VAEs).
AEs consist of an encoder and a decoder that work together to reconstruct input while disregarding less important details, prioritizing the extraction of meaningful features.
VAEs, a probabilistic variant of AEs, have found wide-ranging applications in the field of speech modeling.
Furthermore, the vector-quantized variational autoencoder (VQ-VAE) \cite{van2017neural} has been developed as an extended generative model.
The VQ-VAE introduces parameterization of the posterior distribution to represent discrete latent representations.
Remarkably, the VQ-VAE has demonstrated notable success in generative spoken language modeling.
By combining a discrete latent space with self-supervised learning, its performance is further improved.

#### Autoregressive models

Autoregressive generative self-supervised learning uses autoregressive prediction coding technique \cite{chung2019unsupervised}  to model the probability distribution of a sequence of data points.
This approach aims to predict the next data point in a sequence based on the previous data points.
Autoregressive models typically use RNNs or a transformer architecture as a basic model.
The authors in paper \cite{oord2016wavenet} introduce a generative model for raw audio called WaveNet, based on PixelCNN \cite{van2016conditional}.
To enhance the model's ability to handle long-range temporal dependencies, the authors incorporate dilated causal convolutions \cite{oord2016wavenet}.
They also utilize Gated Residual blocks and skip connections to improve the model's expressivity.

#### Masked Reconstruction

The concept of masked reconstruction is influenced by the masked language model (MLM) task proposed in BERT \cite{devlin2018bert}.
This task involves masking specific tokens in input sentences with learned masking tokens or other input tokens, and training the model to reconstruct these masked tokens from the non-masked ones.
Recent research has explored similar pretext tasks for speech representation learning that help models develop contextualized representations capturing information from the entire input, like the DeCoAR model \cite{ling2020decoar}.
This approach assists the model in comprehending input data better, leading to more precise and informative representations.

### 4.4.2·Contrastive Models

The technique involves training a model to differentiate between similar and dissimilar pairs of data samples, which helps the model acquire valuable representations that can be utilized for various tasks, as shown on \Cref{fig:ssrl-con}.
The fundamental principle of contrastive learning is to generate positive and negative pairs of training samples based on the comprehension of the data.
The model must learn a function that assigns high similarity scores to two positive samples and low similarity scores to two negative samples.
Therefore, generating appropriate samples is crucial for ensuring that the model comprehends the fundamental features and structures of the data.
\Cref{SSRL:con} outlines popular contrastive self-supervised models used for different speech-processing tasks.
We discuss Wav2Vec 2.0 since it has achieved state-of-the-art results in different downstream tasks.

- Wav2Vec 2.0 \cite{baevski2020wav2vec} is a framework for self-supervised learning of speech representations that is one of the current state-of-the-art models for ASR \cite{baevski2020wav2vec}.
The training of the model occurs in two stages.
Initially, the model operates in a self-supervised mode during the first phase, where it uses unlabelled data and aims to achieve the best speech representation possible.
The second phase is fine-tuning a particular dataset for a specific purpose.
Wav2Vec 2.0 takes advantage of self-supervised training and uses convolutional layers to extract features from raw audio.

In the speech field, researchers have explored different approaches to avoid overfitting, including augmentation techniques like Speech SimCLR \cite{jiang2020speech} and the use of positive and negative pairs through methods like Contrastive Predictive Coding (CPC) (\citet{ooster2019improving}), Wav2vec (v1, v2.0) (\citet{schneider2019wav2vec}), VQ-wav2vec (\citet{baevski2019vq}), and Discrete BERT \cite{baevski2019effectiveness}."
"In the graph field, researchers have developed approaches like Deep Graph Infomax (DGI) (Velickovic et al., 2019 \cite{velivckovic2018deep}) to learn representations that maximize the mutual information between local patches and global structures while minimizing mutual information between patches of corrupted graphs and the original graph's global representation.

### 4.4.3·Predictive Models

In training predictive models, the primary concept involves creating simpler objectives or targets to minimize the need for data generation.
However, the most critical and difficult aspect is ensuring that the task's difficulty level is appropriate for the model to learn effectively.
Predictive SSRL methods have been leveraged in ASR through transformer-based models to acquire meaningful representations \cite{baevski2019effectiveness,hsu2021hubert,liu2021tera} and have proven transformative in exploiting the growing abundance of data \cite{gao2023self}.
\Cref{SSRL:pred} highlight popularly used SSRL methods along with the data used for training these models.
In the following section we breifly discuss three popular predictive SSRL approaches used widely in various downstream tasks.

- The direct application of BERT-type training to speech input presents challenges due to the unsegmented and unstructured nature of speech.
To overcome this obstacle, a pioneering model known as Discrete BERT \cite{baevski2019effectiveness} has been developed.
This model converts continuous speech input into a sequence of discrete codes, facilitating code representation learning.
The discrete units are obtained from a pre-trained vq-wav2vec model \cite{baevski2019vq}, and they serve as both inputs and targets within a standard BERT model.
The architecture of Discrete BERT, illustrated in \Cref{fig:ssrl-pred} (a), incorporates a softmax normalized output layer.
During training, categorical cross-entropy loss is employed, with a masked perspective of the original speech input utilized for predicting code representations.
Remarkably, the Discrete BERT model has exhibited impressive efficacy in self-supervised speech representation learning.
Even with a mere 10-minute fine-tuning set, it achieved a Word Error Rate (WER) of 25\% on the standard test-other subset.
This approach effectively tackles the challenge of directly applying BERT-type training to continuous speech input and holds substantial potential for significantly enhancing speech recognition accuracy.
- The HuBERT \cite{hsu2021hubert} and TERA  \cite{liu2021tera} models are two self-supervised approaches for speech representation learning.
HuBERT uses an offline clustering step to align target labels with a BERT-like prediction loss, with the prediction loss applied only over the masked regions as outlined in \Cref{fig:ssrl-pred} (b).
This encourages the model to learn a combined acoustic and language model over the continuous inputs.
On the other hand, TERA is a self-supervised speech pre-training method that reconstructs acoustic frames from their altered counterparts using a stochastic policy to alter along various dimensions, including time, frequency, and tasks.
These alterations help extract feature-based speech representations that can be fine-tuned as part of downstream models.

Microsoft has introduced UniSpeech-SAT \cite{chen2022unispeech} and WavLM \cite{chen2022wavlm} models, which follow the HuBERT framework.
These models have been designed to enhance speaker representation and improve various downstream tasks.
The key focus of these models is data augmentation during the pre-training stage, resulting in superior performance.
WavLM model has exhibited outstanding effectiveness in diverse downstream tasks, such as automatic speech recognition, phoneme recognition, speaker identification, and emotion recognition.
It is worth highlighting that this model currently holds the top position on the SUPERB leaderboard \cite{DBLP:journals/corr/abs-2105-01051}, which evaluates speech representations' performance in terms of reusability.

Self-supervised learning has emerged as a widely adopted and effective technique for speech processing tasks due to its ability to train models with large amounts of unlabeled data.
A comprehensive overview of self-supervised approaches, evaluation metrics, and training data is provided in Table \ref{SSRL:pred} for speech recognition, speaker recognition, and speech enhancement.
Researchers and practitioners can use this resource to select appropriate self-supervised methods and datasets to enhance their speech-processing systems.
As self-supervised learning techniques continue to advance and refine, we can expect significant progress and advancements in speech processing.
