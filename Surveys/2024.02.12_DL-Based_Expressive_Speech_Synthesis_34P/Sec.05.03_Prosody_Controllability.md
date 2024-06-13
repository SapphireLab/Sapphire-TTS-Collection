
### 5.3 Prosody Controllability 韵律可控性

> Text-to-speech is a one-to-many mapping problem, i.e.,for one piece of text there could be many valid prosody patterns because of speaker-specific variations.
> Accordingly, providing a kind of controllability over prosody-related features in synthesized speech is essential for generating expressive speech with different variations.

> However, it’s not always easy to mark-up prosody or even to define boundaries between prosody events, i.e., duration boundaries can vary depending on segmentation,pitch contour prediction is error-prone, and prosody features may not always correlate well with what listeners perceive.
> Several studies in literature have addressed the controllability issue in terms of selecting an emotion/style class or intensity level and adjusting prosody-related features at different speech levels.
> In this section, we discuss studies considering prosody controllability.

文本到语音转换是一个一一对应的问题，即对于一段文本，由于说话人特定的变化，可能存在许多有效的韵律模式。因此，在合成的语音中提供对韵律相关特征的控制对于生成具有不同变化的表达性语音至关重要。
然而，标记韵律或定义韵律事件之间的界限并不总是容易的，即持续时间界限可能会根据分段而变化，音高轮廓预测容易出错，韵律特征并不总是与听众感知相关。
在文献中，有几项研究从选择情感/风格类别或强度级别以及在不同的语音级别调整韵律相关特征的角度解决了可控性问题。在本节中，我们将讨论考虑韵律可控性的研究。

#### 5.3.1 Modeling‑specific prosody styles 

> This group of studies provides individual representations of expressive styles/emotions, enabling the control of prosody in synthesized speech by offering the ability to select from available representations or adjust their values.
> In some studies [55] [70] [116], style is modeled at a single speech/text level, while in other studies [68] [79] [133] a multi-level or hierarchical model of expressive styles is used to allow for a better capture of prosody variation in expressive speech.
> In single-level prosody modeling approaches, [55] is one of the early studies that extends a baseline with fine-grained control over the speaking style/prosody of synthesized speech.
> The proposed modification involves adding an embedding network with temporal structure to either the speech-side or text-side of the TTS model.
> Accordingly, the resulting prosody embedding is of variable length, and it is used to condition input to either encoder or decoder based on the position of the embedding network.
> Speech-side prosody embedding provides adjustment of prosody at frame-level, while text-side prosody embedding enables phoneme-level prosody control.
> Single-level prosody embeddings can be converted into discrete embeddings as in [70] [116].
> Discrete prosody representations are easier to control and analyze and provide a better interpretation of prosodic styles.

这些研究提供了表达风格/情感的个体表示，使得在合成语音中控制语调成为可能，通过提供从可用表示中选择或调整其值的能力。在某些研究中，风格在单个语音/文本级别建模，而在其他研究中，使用多级或分层模型来更好地捕捉表达性语音中的语调变化。

在单级语调建模方法中，[55]是早期研究之一，它扩展了基线，对合成语音的说话风格/语调进行细粒度控制。提出的修改涉及在TTS模型的语音侧或文本侧添加具有时间结构的嵌入网络。

因此，得到的语调嵌入是可变长度的，并且根据嵌入网络的位置用于条件输入到编码器或解码器。语音侧语调嵌入提供帧级语调调整，而文本侧语调嵌入允许音素级语调控制。

单级语调嵌入可以转换为离散嵌入，如[70] [116]。离散语调表示更容易控制和分析，并提供语调风格的更好解释。

> In [116], a word-level prosody embedding is proposed based on decision trees and a GMM.
> A word-level reference encoder is first used to obtain word-level prosody embedding from reference audio.
> A binary decision tree is employed to cluster embeddings with their identities based on their phonetic information.
> Prosody embeddings of words in each leaf node will differ only in their prosodies.
> Then prosody embeddings of each leaf can be clustered via a GMM model where clusters represent prosody tags.
> If the applied GMM consists of five components and a tree of ten leaf nodes, a set of 50 prosody tags is produced.
> At inference time, prosody tags can be selected manually or via a prosody predictor that is trained to select appropriate prosody tags based on input text.

> In [70], an audiobook speech synthesis model is proposed.
> The model uses a character-acting-style extraction module based on ResCNN [165] to extract different character acting styles from the input speech.
> Discrete character-level styles are obtained via vector quantization(VQ) [149], which maps them to a codebook, limiting the number of styles.
> At inference, the discrete character-acting-styles are predicted via a style predictor.
> The character-level style predictor uses both character embeddings from Skip-Gram [166] and text-based features from RoBERTa [167] as input.

在[116]中，基于决策树和GMM提出了一个词级语调嵌入。首先使用词级参考编码器从参考音频中获取词级语调嵌入。使用二叉决策树根据其音素信息对嵌入进行聚类。每个叶节点中单词的语调嵌入仅在语调上有所不同。然后，可以通过GMM模型对每个叶的语调嵌入进行聚类，其中聚类表示语调标签。如果应用的GMM包含五个组件和一个十个叶节点的树，则会产生50个语调标签。在推理时，语调标签可以手动选择或通过训练以根据输入文本选择适当语调标签的语调预测器选择。

在[70]中，提出了一种有声读物语音合成模型。该模型使用基于ResCNN [165]的角色扮演风格提取模块从输入语音中提取不同的角色扮演风格。通过矢量量化(VQ) [149]获得离散角色级风格，将其映射到码本，限制风格数量。在推理时，通过风格预测器预测离散角色扮演风格。角色级风格预测器使用来自Skip-Gram [166]的角色嵌入和来自RoBERTa [167]的文本特征作为输入。

> Regarding multi-level prosody modeling, some studies propose enhancing prosody control in the baseline models [74] [75] [77] by modifying their single-level prosody modeling to multiple levels.
> For instance, [133] proposes a hierarchical structure of [75] with multiple GST layers.
> Three GST layers are employed in the proposed model, each consisting of 10 tokens, which were found to yield better token interpretation.
> Tokens of the first and second layers were found to learn different speakers and styles, but these representations were not easily interpreted.
> Interestingly, the tokens in the third layer were able to generate higher quality samples with more distinct and interpretable styles.
> Specifically, third-layer styles exhibit clear differences in their features, including pitch, stress, speaking rate, start offset, rhythm, pause position, and duration.
> Model in [77] is further extended in [68] with three VAEs to generate three different levels (utterance, phrase,and word) of latent variables with varying time resolutions.
> Acoustic features and linguistic features are passed as input to the three VAEs.
> Initially, a conditional prior(CP) is applied to learn a distribution for sampling utterance-level latent variables based on linguistic features from the input text.
> The generated latent variables are passed to other levels via auto-regressive (AR) latent converters that convert latent variables from coarser-level to finer-level with input text condition.
> In fact, the utterance-level latent variables can be used to control the generated speech styles, regardless of latent variables of other levels, as they are predicted based on the utterance-level latent variables.

> The Controllable Expressive Speech Synthesis (ConEx)model in [79] proposes modeling prosody at two levels,utterance-level (global) and phone-level (local), using reference encoders [74].
> However, the global prosody embedding is used to condition the local prosody embedding, resulting in an integrated prosody embedding.

> The local embeddings are 3D vectors that are converted into discrete local prosody embeddings (codes) via vector quantization (VQ) [149].
> At inference time, the integrated prosody embedding is predicted by an auto-regressive (AR) prior model trained to predict categorical distributions for each of the discrete codes utilizing global prosody embedding and the phoneme embedding as inputs.
> While global prosody embedding can be obtained from training samples or from an audio reference, local prosody embeddings for a given global prosody embedding are achieved via the AR prior model.
> Fine-grained prosody control can be achieved by selecting a specific phoneme to start adjusting prosody from.
> The AR prior model will first generate the top k prosody options for this phoneme.
> Then, the local prosody sequence will be generated autoregressively for each of the first top k options by the AR prior model.

#### 5.3.2 Modeling‑specific prosody features 

> This group of studies provides individual representations of prosody-related features.
> Control over prosody of the synthesized speech is provided via selecting or adjusting a specific representation of a specific prosody-related feature.
> Some studies in this direction model prosody features at the global or utterance-level [97] [128], while other studies propose modeling at fine-grained levels [48] [63] [71] [122] [138], such as phoneme, syllable, or word-level.

> The STYLER model [97], for example, employs multiple style encoders to factor speech style into several components, including duration, pitch, speaker, energy, and noise.
> This structure enables STYLER to generate controllable expressive speech by adjusting each of the individually modeled features.
> Furthermore, with the explicit noise encoding, other encoders can be constrained to exclude noise information as a style factor, and thus the model can generate clean speech even with noisy references.
> Adjusting the style factors, various styles of speech can be generated from STYLER.

这些研究提供了与韵律相关的特征的个体表示。通过选择或调整特定韵律相关特征的特定表示来控制合成的语音的韵律。在这个方向上，一些研究在全局或句子级别建模韵律特征[97] [128]，而其他研究则提出在细粒度级别建模[48] [63] [71] [122] [138]，例如音素、音节或单词级别。

例如，STYLER模型[97]使用多个风格编码器将语音风格分解为几个组成部分，包括持续时间、音高、说话者、能量和噪声。这种结构使STYLER能够通过调整每个单独建模的特征来生成可控的表达性语音。此外，通过显式噪声编码，其他编码器可以被约束以排除噪声信息作为风格因素，因此模型可以生成干净的语音，即使使用嘈杂的参考。通过调整风格因素，可以从STYLER生成各种风格的语音。


> Adjusting several features at fine-grained levels can be a difficult task. 
> For example, [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md) provides fine-grained control over pitch range, duration,energy, which are modeled at the phone-level (phone-wise), and it is not easy to adjust these features to achieve a specific prosodic output. 
> Raitio and Seshadri [128] improves [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md) with an utterance-wise (coarse-grained) prosody model using an additional variance adaptor.
> That second variance adaptor is the same as the original one, but it models five features at the utterance-level: pitch, pitch range, duration, energy,and spectral tilt.
> These features are then concatenated with the corresponding output of the first variance adaptor.
> Such utterance-wise prosody model enables easier control of prosody while still allowing modification at the phone-level.
> To control high-level prosody,a bias is added to the corresponding utterance-wise prosody predictions.
> A phone-level prosody control is achieved by directly modifying the phone-wise features.
> Fine-grained control over a specific prosody-feature can also be required specially for strong speaking styles.
> To that end, in [71], a predictor is proposed to predict F0, energy, and duration features at the phoneme-level.
> During inference, the predicted features are generated based on the input text alone; however, they can also be provided externally and modified as desired.
> Furthermore, two prosody modeling levels are proposed in [63]: the local level (word-level) and global level (utterance-wise).
> The global prosody embedding is the emotion embedding obtained by a reference-based encoder.
> The local prosody embedding is obtained from a predictor of the F0 features at the word-level with global prosody embedding and the phoneme embedding as inputs.
> Both embeddings are then passed to a multi-style encoder to form the final multi-style prosody embedding.
> Therefore, modifying the predicted F0values can provide control of prosody at the utterance,word, and phoneme levels.

在细粒度级别调整多个特征可能是一项困难的任务。例如，FastSpeech2 在音素级别（音素级）提供对音高范围、持续时间和能量的细粒度控制，并且不容易调整这些特征以实现特定的韵律输出。Raitio和Seshadri[128]通过使用额外的变异适配器改进FastSpeech2，该变异适配器在句子级别（粗粒度）使用韵律模型。第二个变异适配器与原始变异适配器相同，但在句子级别建模五个特征：音高、音高范围、持续时间、能量和频谱倾斜。然后，将这些特征与第一个变异适配器的相应输出连接起来。这种句子级别的韵律模型允许更容易地控制韵律，同时仍然允许在音素级别进行修改。为了控制高级别韵律，向相应的句子级别韵律预测添加偏差。通过直接修改音素级特征来实现音素级韵律控制。

对于强烈的说话风格，可能需要对特定韵律特征进行细粒度控制。为此，[71]中提出了一种预测器来预测音素级别的F0、能量和持续时间特征。在推理过程中，根据输入文本生成预测特征；但是，它们也可以从外部提供并根据需要进行修改。此外，[63]中提出了两个韵律建模级别：局部级别（单词级别）和全局级别（句子级别）。全局韵律嵌入是从基于参考的编码器获得的情绪嵌入。局部韵律嵌入是从具有全局韵律嵌入和音素嵌入作为输入的预测器获得的音素级别的F0特征。然后，将这两个嵌入传递到一个多风格编码器，以形成最终的多风格韵律嵌入。因此，修改预测的F0值可以提供对句子、单词和音素级别的韵律的控制。

> More flexibility in controlling the F0 feature is provided in the controllable deep auto-regressive model(C-DAR) model [138] which allows for F0 contour adjustment by the user.
> To achieve this goal, three strategies are used: 1) context awareness by conditioning the model on the preceding and following speech during training, 2) conditioning the model on some random segments of ground truth F0, and 3) predicting F0 values in reverse order.
> Additionally, several text-based features are used as input to the model, including word embeddings derived from BERT, V/UV label,one-hot vector for the nearby punctuation, and phoneme encodings.
> At inference, F0 values specified by the user are used as alternatives for the ground truth F0 segments, and the model predicts the rest of the utterance’s F0 contour through context awareness.
> Discrete fine-grained representations for prosody features as in [48] [122] are also useful to limit the number of the obtained representations.
> Both studies [48] [122]utilize VQ [149] to map each prosody embedding to the closest discrete representation from a predefined codebook.
> In [48], a pitch predictor is used to predict character-level continuous pitch representation using character embeddings from the text encoder as input.
> Zhang et al.[122], however, produces syllable-level prosody embeddings from a reference encoder that takes F0, intensity,and duration features from reference audio as input.
> The resulting prosody embeddings are then mapped to a pre-defined codebook to extract b discrete prosody codes.

在可控深度自回归模型（C-DAR）模型[138]中提供了对F0特征的更多控制灵活性，该模型允许用户调整F0轮廓。为了实现这一目标，使用了三种策略：1）通过在训练期间将模型条件化在先前和随后的语音上来实现上下文感知，2）将模型条件化在地面真F0的一些随机片段上，以及3）以相反的顺序预测F0值。此外，将几个基于文本的特征用作模型的输入，包括从BERT派生的词嵌入、V/UV标签、附近标点符号的一热向量和音素编码。在推理时，用户指定的F0值被用作地面真F0片段的替代品，并且模型通过上下文感知预测整个句子的F0轮廓。

与[48] [122]中一样，离散细粒度表示对于韵律特征也很有用，可以限制获得的表示的数量。两项研究[48] [122]都使用VQ[149]将每个韵律嵌入映射到预定义码本中最近的离散表示。在[48]中，使用音高预测器来预测使用文本编码器作为输入的字符嵌入的字符级连续音高表示。然而，张等人[122]从参考编码器产生音节级韵律嵌入，该参考编码器将参考音频的F0、强度和持续时间特征作为输入。然后，将得到的韵律嵌入映射到一个预定义的码本，以提取离散的韵律代码。

> Resulting prosody codes in [48] represent the pitch and other suprasegmental information that can be adjusted via a specific bias value to generate speech with different pitch accents.
> The codes in [122], can be interpreted as representing some prosody features such as pitch and duration.
> The prosody variation at the syllable-level can be manually controlled by assigning each syllable the desired prosody code from the codebook.
> In [125], ToBI features, which involve a set of conventions used for transcribing and annotating speech prosody, are used.
> The applied ToBI features are four word-level tags: pitch accents, boundary tones, phrase accents, and break indices.
> The extracted ToBI tags are used as input to TTS model.
> Simultaneously, a ToBI predictor is trained to predict these prosody tags based on grammatical and semantic information extracted from the input text using a self-supervised language model.
> The resulting model had the ability to control the stress,intonation, and pause of the generated speech to sound natural, utilizing only ToBI tags from the text-based predictor.

在[48]中，得到的韵律代码表示音高和其他超音段信息，可以通过特定的偏置值进行调整，以生成具有不同音高重音的语音。在[122]中，代码可以解释为表示一些韵律特征，如音高和持续时间。可以通过将每个音节分配从码本中所需的韵律代码来手动控制音节级别的韵律变化。

在[125]中，使用了ToBI特征，这是一种用于转录和注释语音韵律的约定集。应用的ToBI特征是四个单词级别的标签：音高重音、边界音、短语重音和停顿索引。提取的ToBI标签用作TTS模型的输入。同时，训练了一个ToBI预测器来预测这些韵律标签，基于使用自监督语言模型从输入文本中提取的语法和语义信息。

最终模型能够仅使用文本预测器中的ToBI标签来控制生成的语音的强调、语调和停顿，使其听起来自然。


#### 5.3.3 Modeling prosody strength 

> This group of studies focus on regulating the strength of emotion or prosody.
> For instance, [61] utilizes the distance between emotion embeddings and the neutral emotion embeddings to identify scalar values for emotion intensity.
> It proposes a phoneme-level emotion embedding and a fine-grained emotion intensity.
> The emotion embedding is first obtained via a reference encoder.

> The emotion intensity is then generated by an intensity extractor that takes the emotion embedding as input.
> The intensity extractor produces intensity as a scalar value based on the distance between the emotion embedding and the centroid of a pre-defined cluster for neutral emotion embeddings.
> The resulting emotion intensity values are quantized into pseudo-labels that serve as the index for an intensity embedding table.
> Another method for learning emotion strength values in an unsupervised manner is by using ranking functions.
> Studies [27] [31] [33] [64] utilize a ranking function-based method named relative attributes [89] for this purpose.
> In[33], prosody is modeled at three levels: global-level representation by emotion embedding, utterance-level represented by prosody embedding from a reference-based encoder, and the local-level represented by emotion strength.
> The study trains an emotion strength extractor at the syllable-level based on input speech utilizing the ranking function.
> Simultaneously, a predictor of emotion strength is trained based on features extracted from input text via BERT model.
> Besides changing emotion label and emotion reference audio, the model provides manual control of the emotion strength values in the synthesized speech.

> Alternatively, the reference encoder in [31] functions as a ranking function to learn a phoneme-level emotion strength (descriptor) sequence.
> The proposed ranking function [89] receives its input from fragments of target reference audio obtained via a forced alignment model to phoneme boundaries.
> The OpenSMILE [139] tool is then used to extract 384-dimensional emotion-related features from these reference speech fragments as input to the ranking function.
> Similarly, the proposed ranking function in [27] takes a set of acoustic features extracted from the input speech via OpenSMILE tool but at the utterance-level as input.
> The ranking function leverages the difference between neutral samples and samples associated with each emotion class in the dataset.
> The training process is formulated as solving a max-margin optimization problem.
> The resulting emotion strength scalars can be manually adjusted or predicted based on text or reference speech.

> In [64], both emotion class and emotion strength value are obtained via a joint emotion predictor based only on the input text.
> The input to the predictor is features extracted from input text via the Generative Pre-trained Transformer (GPT)-3 [88].
> Emotion class and emotion strength are the two outputs of the predictor where the former is represented as a one-hot encoded vector and the latter is presented as a scalar value.
> Emotion labels and emotion strength values which are also obtained via[89], are used as ground truth for predictor training.

> Another ranking method is proposed in **"Cross-Speaker Emotion Transfer by Manipulating Speech Style Latents"** using the ranking support vector machine.
> The model generates style embedding and speaker embedding via two separate encoders.
> Both style and speaker embeddings at inference time are represented by centroids of each single speaker and style embeddings.
> However, a linear SVM is trained with the model to provide the ability for style embedding adjustment.
> The proposed SVM model is trained to classify between neutral emotion and a specific emotion embedding, where the learned hyperplane is utilized to move(scale) the style vectors in a direction towards/opposite to the hyperplane.

> Another type of control that contributes to generating speech with a better representation of local prosodic variation is introduced in [124].
> The proposed model suggests an unsupervised approach to obtain word-level prominence and phrasal boundary strength features.
> For this purpose, continuous wavelet transform (CWT) [168]is utilized to extract continuous estimates of word prominence and boundary information from the audio signal.

> First, the three prosodic signals f0, energy, and duration are extracted and combined as input to the CWT.
> Then,the combined signal is decomposed via CWT into scales that represent prosodic hierarchy.
> Word and phrase-level prosody are then obtained by following ridges or valleys across certain scales.
> The continuous word prominence and boundary estimates are achieved via the integration of the resulting lines aligned with the textual information.
> With manually identified intervals, the continuous values of prominence and boundary strength are then discretized.

#### 5.3.4 Prosody clustering 

> In this section, methods for selecting the appropriate prosody embedding for the referenced-based ETTS models are described.
> To begin with, clustering methods are utilized in [57] [58] to generate representative prosody embeddings for each emotion class when the GST-TTS model is trained with a labeled dataset.
> Initially,the resulting emotion embeddings are clustered in a 2d space.
> In [57], the centroid of each cluster is used as the weights of the GSTs to generate emotion embedding for each emotion class.
> In [58], the weight vector that represents each emotion cluster is obtained by considering the inter and intra distances between emotion embedding clusters.
> Specifically, an algorithm is used for minimizing each embedding distance to the target emotion cluster and maximizing its distance to other emotion clusters.

> Similarly, clustering algorithms are applied in [112] [113] to achieve discrete prosody embeddings but for two specific prosody-related features.
> The two studies employ K-means algorithm to cluster F0 and duration features extracted for each phoneme.
> The centroids of the clusters are then used as discrete F0 and duration values/tokens for each phoneme. work [112] applies a balanced clustering method with duration features to overcome degradation in voice quality that appeared in [113] during duration control.
> Moreover, to keep phonetic and prosodic information separate during training, an attention unit is introduced to map prosody tokens to decoder hidden states and generate prosody context vectors.
> The resulting discrete tokens for F0 and duration features provide a fine-grained level of control over prosody by changing the corresponding prosodic tokens for each phoneme.

> In [105], a cross-domain SER model with the GST-TTS model is proposed to obtain emotion embeddings for an unlabeled dataset.
> The cross-domain SER model is trained using two datasets including: 1) an SER dataset (source) labeled with emotions, and 2) a TTS dataset (target) that is not labeled.
> Simultaneously, the SER model trains an emotion classifier that generates soft labels for the unlabeled TTS dataset.
> These soft labels are then used to train an extended version of the baseline in[74] with an emotion predictor.
> In the training process,the weights of the style tokens layer are passed as input to the predictor, which employs the learned soft labels as ground truth values.
> At inference time, weights vectors for each emotion class are averaged to obtain the emotion class embedding.
> However, since the predicted labels for the TTS dataset are soft labels, and thus not entirely reliable, only the top K samples with the highest posterior probabilities are selected.
