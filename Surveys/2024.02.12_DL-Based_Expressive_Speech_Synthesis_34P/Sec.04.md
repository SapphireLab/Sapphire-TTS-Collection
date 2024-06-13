
## 4.无监督学习方法

> Due to the limited availability and challenges associated with collecting or preparing labeled datasets of expressive speech, as discussed in Section 6, many researchers tend to resort to unsupervised approaches for generating expressive speech. Within these approaches, models are trained to extract speaking styles or emotions from expressive speech data through unsupervised methods.
> Unsupervised models typically utilize reference speech as an input to the TTS model, which extracts a style or prosody embedding which is then used to synthesize speech resembling the input style reference. In the literature, three primary structures emerge as baseline models for unsupervised ETTS models: including reference encoders, global style tokens, and variational autoencoders, which are explained in the following three sections. In addition, we identify the recent TTS models that utilize in-context learning as another group of unsupervised approaches. The last subcategory under the unsupervised approaches involves other individual approaches. We then provide a general summary of all the unsupervised approaches reviewed in this work in Table 3.

由于表达性语音的标注数据集的收集或准备相关的有限可用性和挑战, 许多研究人员倾向于采用无监督方法用于生成表达性语音. 在这些方法中, 模型通过无监督方法被训练用于从表达性语音数据中提取说话风格或情感. 无监督模型通常使用参考语音作为输入传递给 TTS 模型, 该模型提取风格或韵律嵌入, 之后用于合成类似于输入风格参考的语音. 
现有的文献中出现了三个主要结构作为无监督 ETTS 模型的基线模型: Reference Encoders, Global Style Tokens, VAEs.
此外我们认为近期使用上下文学习的 TTS 模型为其他一组无监督学习方法.最后一个子类别还涉及到其他个别方法.
我们在表格三种提供了本文回顾的所有无监督方法的一般性总结.

|序号|组别|TTS 模型|韵律级别|


### 4.1 Direct Reference Encoding 

> The main approach, based on a reference or prosody encoder, can be traced back to an early Google paper[74]. The paper suggests using a reference encoder to produce a low-dimensional embedding for a given style reference audio, which is called a prosody embedding.
> This encoder takes spectrograms as input to represent the reference audio. The generated prosody embedding is then concatenated with the text embedding derived from the text encoder of a Seq2Seq TTS model such as [Tacotron](../../Models/TTS2_Acoustic/2017.03.29_Tacotron.md), [Tacotron2](../../Models/TTS2_Acoustic/2017.12.16_Tacotron2.md). Figure 6 shows reference encoder integrated to the TTS model.

基于参考或韵律编码器的主要方法可以回溯到 Google 的一篇论文 [74] Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron. 该文献建议使用参考编码器以为给定风格参考音频生成低维嵌入, 称为韵律嵌入. 这一编码器将频谱作为输入来表示参考音频. 生成的韵律嵌入会与 Seq2Seq TTS 模型如 Tacotron 的文本编码器导出的文本嵌入相拼接. 图六展示了参考编码器集成到 TTS 模型.

> Various features have been employed in the literature as inputs for the reference encoder. For example, in the work [85], MFCC features extracted using the openSMILE toolkit [139] are fed into one of the encoders within its style extraction model, which is composed of a multi-modal dual recurrent encoder (MDRE). In another study [31], the reference encoder is proposed as a ranking function model, aimed at learning emotion strength at the phoneme level. This model leverages the OpenSMILE toolkit to extract 384-dimensional emotion-related features from segments of reference audio, derived using a forced alignment model for phoneme boundaries. Furthermore, in work [63], a word-level prosody embedding is generated. This is achieved by extracting phoneme-level F0 features from reference speech using the WORLD vocoder [140] and an internal aligner operating with the input text.

文献中已经使用了各种特征作为参考编码器的输入.
- 文献 [85] 使用 OpenSMILE 工具箱提取 MFCC 特征被输入到其风格提取模型的一个编码器中. 该模型有一个多模态对偶循环编码器组成.
- 文献 [31] 参考编码器被作为一个排序函数模型, 旨在音素级别学习情感强度. 这个模型利用 OpenSMILE 工具箱从参考音频片段中提取和情感相关的 384 维特征, 这些片段是对音素边界采用强制对齐模型获得的.
- 文献 [63] 生成了基于单词级别的韵律嵌入. 这是通过使用 WORLD 声码器和与输入问题一起操作的内置对齐器从参考音频中提取音素级别的 F0 特征来实现的.

> A prosody-aware module is proposed in [37] which extracts other prosody-related features. The prosody-aware module consists of an encoder, an extractor, and a predictor. The encoder receives the three phoneme-level features including logarithmic fundamental frequency(LF0), intensity, and duration from the extractor as input and generates the paragraph prosody embedding with the assistance of an attention unit. Simultaneously, the predictor is trained to predict these features at inference time based on the input text embedding only.

文献 [37] 提出了韵律感知模块用于提取其他韵律相关特征. 该模块由一个编码器, 一个提取器和一个预测器组成. 编码器接收来自提取器的三个音素级别特征包括对数基频 LF0, 强度和时长作为输入, 并借助注意力单元生成段落韵律嵌入. 同时预测器被训练用于在推理时仅输入文本嵌入来预测这些特征.

> In Daft-Exprt TTS model [118], the prosody encoder receives pitch, energy and spectrogram as input. The prosody encoder then uses FiLM conditioning layers[141] to carry out affine transformations to the intermediate features of specific layers in the TTS model. A slightly modified version of the [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md) model is utilized in this work where the phoneme encoder,prosody predictor and the decoder are the conditioned components. The prosody predictor is similar to the variance adaptor of [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md) but without the length regulator, and it estimates pitch, energy and duration at phoneme-level.

在 Draft-Exprt TTS 模型中, 韵律编码器接受音高, 能量和频谱作为输入. 然后韵律编码器使用 FiLM 条件层对 TTS 模型的特定层的中间特征执行仿射变换. 这项工作使用了 FastSpeech2 的稍微修改版本, 其中音素编码器, 韵律预测器和解码器都是条件化组件. 韵律预测器和 FastSpeech2 的方差适配器相似但没有长度调节器, 且其在音素水平估计音高, 能量和时长.

> A pre-trained Wav2Vec model [142] has also been utilized for extracting features from the reference waveform.

文献 [142] 采用预训练 Wav2Vec 模型用于从参考波形中提取特征. 

> These features serve as input to the reference encoders of the proposed [Emo-VITS]() model, which integrates an emotion network into the VITS model [143] to enhance expressive speech synthesis. In fact, the emotion network in the Emo-VITS model comprises two reference encoders. The resulting emotion embeddings from these encoders are then combined through a feature fusion module that employs an attention mechanism. Wav2vec2.0-derived features from the reference waveform in this work are particularly suitable for attention-based fusion and contribute to reducing the textual content within the resulting embeddings.

Emo-VITS 将这些特征作为 Emo-VITS 的参考编码器的输入, 它将情感网络集成到 VITS 模型中用于增强表达性语音合成. 实际上, 情感网络包含了两个参考编码器. 这些编码器的情感嵌入输出之后通过一个特征融合模块进行结合然后应用注意力机制. 由 Wave2Vec 2.0 从参考音频导出的特征尤其适合基于注意力的融合, 且有助于在结果嵌入中减少文本内容.

> In contrast, [60] proposes a an image style transfer module to generate input for reference encoder. The concept of image style transfer involves altering the artistic style of an image from one domain to another while retaining the image’s original content [144]. In specific research, the style reconstruction module from VGG-19[145], a deep neural network primarily used for image classification, is employed to extract style-related information from the Mel-spectrogram used as input image. Subsequently, the output of this module is fed into the reference encoder to generate the style embedding.

文献 [60] 提出了一种图像风格转换模块用于生成参考编码器的输入. 图像风格迁移的概念涉及到图像的艺术风格从一个领域的转化到另一个领域, 同时保留图像原始内容. 具体而言来自 VGG-19 的风格重构模块, 一个主要用于图像分类的神经网络, 将梅尔频谱视为输入图像从中提取和风格相关的信息. 之后这个模块的输出传递到参考编码器中以生成风格嵌入.

### 4.2 Latent Features via Variational Auto‑Encoders 通过 VAE 获取隐特征

> The goal of TTS models under this is to map input speech from the higher dimensional space to a well-organized and lower-dimensional latent space utilizing variational auto-encoders (VAEs) [146]. VAE is a generative model that is trained to learn the mapping between observed data x and continuous random vectors z in an unsupervised manner. In detail, VAEs learn a Gaussian distribution denoted as the latent space from which the latent vectors representing the given data x can be sampled. A typical variational autoencoder consists of two components. First, the encoder learns the parameters of the z vectors (latent distribution), namely the mean $\mu(x)$ and variance $\sigma^2(x)$, based on the input data x. Second,the decoder regenerates the input data x based on latent vectors z sampled from the distribution learned by the encoder. In addition to the reconstruction loss between the model input and the data, variational autoencoders are also trained to minimize a latent loss, which ensures that the latent space follows a Gaussian distribution.

在这类 TTS 模型中, 目标是利用 VAE 将来自于高维空间的语音映射到组织良好且维度较低的隐空间. VAE 是一种生成模型, 通过无监督学习的方式学习观察数据 $x$ 和连续随机向量 $z$ 之间的映射. 具体来说, VAE 学习一个记为隐空间的高斯分布, 从中可以采样到能表示给定数据 $x$ 的隐向量. 一个典型的变分自编码器由两部分组成: 一是编码器基于输入数据学习 $z$ 向量即隐分布的参数: 均值和方差, 二是解码器基于解码器学习到的分布中采样的隐变量 $z$ 重新生成输入数据. 除了模型输出和数据之间的重构损失, 变分自编码器还需要最小化一个潜在损失, 使得隐空间服从高斯分布.

> Utilizing VAEs in expressive TTS models as shown by Fig. 7, allows for mapping the various speech styles within the given dataset to be encoded as latent vectors,often referred to as prosody vectors, within this latent space. During inference, these latent vectors can be sampled directly or with the guidance of reference audio from the VAE’s latent space. Furthermore, the latent vectors offer the advantage of disentangling prosody features,meaning that some specific dimensions of these vectors independently represent single prosody features such as pitch variation or speaking rate. Disentangled prosody features allow for better prosody control via manipulating the latent vectors with different operations such as interpolation and scaling [77]. 

在表达性 TTS 中使用 VAEs 如图七所示, 允许将给定数据集中各种语音风格编码为隐变量, 通常称为韵律向量. 在推理时这些隐变量可以从隐空间直接采样或在参考音频的指导下采样. 此外, 隐变量还提供了分离韵律特征的优势, 意味着这些向量某些特定维度独立地表示单个韵律特征, 如音高变化或语速. 分离的韵律特征通过某些操作以进行更好的韵律控制, 如插值和缩放.

> The two early papers, [76] [77], can be regarded as the baseline for latent feature-based approaches. The former study [76] introduces VAE within the VoiceLoop model [147], while the latter [77]incorporates VAE into [Tacotron2](../../Models/TTS2_Acoustic/2017.12.16_Tacotron2.md) as an end-to-end TTS model for expressive speech synthesis.

文献 [76] [77] 可以视为基于隐特征方法的基线模型. [76] 将 VAE 引入 VoiceLoop 模型, [77] 将 VAE 集成到 Tacotron2 中作为表达性语音合成的端到端 TTS 模型.

> In the same direction of modeling the variation of the prosodic features in expressive speech, studies [109] [110] propose a hierarchical structure for the baseline variational autoencoder, known as Clockwork Hierarchical Variational AutoEncoder (CHiVE). Both the encoder and decoder in the CHiVE model have several layers to capture prosody at different levels based on the input text’s hierarchical structure. Accordingly, linguistic features are also used alongside acoustic features as input to the model’s encoder. The model’s layers are dynamically clocked at specific rates: sentence, words, syllables, and phones. The encoder hierarchy goes from syllables to the sentence level, while the decoder hierarchy is in the reversed order.

表达性语音中建模韵律特征变化的方面, 文献 [109], [110] 为基线 VAE 提出了一个层次结构, 即 CHiVE. 编码器和解码器都有数层, 基于输入文本的层次结构在不同级别捕获韵律. 因此除了声学特征之外, 语言特征也作为模型编码器的输入. 模型的层以特定的速率动态计时: 句子, 单词, 音节和音素. 编码器的层次结构从音节到句子, 而解码器则相反;

> The CHiVE-BERT model in [110], differs from the main model in [109] as it utilizes BERT [148] features for input text at the word-level. Since the features extracted by the BERT model incorporate both syntactic and semantic information from a large language model, CHiVE-BERT model is expected to have improved the prosody generation.

ChiVE-BERT 模型使用 BERT 特征作为单词级别的输入文本. 由于 BERT 模型提取的特征包含大型语言模型的语法和语义信息, 所以 CHiVE-BERT 模型预计将提高韵律的生成.

> Other studies [DiffProsody](../../Models/E2E/2023.07.31_DiffProsody.md) [53] propose Vector-Quantized Variational Auto-Encoder (VQ-VAE) to achieve discretized latent prosody vectors. In vector quantization(VQ) [149], latent representations are mapped from the prosody latent space to a codebook of a limited number of prosody codes. Specifically, during training, the nearest neighbor lookup algorithm is applied to find the nearest codebook vector to the output of the reference encoder and used to condition TTS decoder. 

DiffProsody [53] 提出矢量量化 VAE 用于实现离散的隐韵律向量. 在 VQ 中, 隐表示从韵律潜在空间映射到有限数量的韵律代码的码本. 特别地在训练时, 最近邻查找算法应用于查找参考编码器输出最近的码本向量, 并用于条件 TTS 模型. 

> To further improve the quality of latent prosody vectors and consequently the expressiveness of the generated speech, [DiffProsody](../../Models/E2E/2023.07.31_DiffProsody.md) proposes a diffusion-based VQ-VAE model.

为了进一步提高隐韵律向量的质量和后续生成语音的表达性, Diff-Prosody 提出了一种基于扩散的 VQ-VAE 模型.

> In the proposed model a prosody generator that utilizes a denoising diffusion generative adversarial networks (DDGANs) [150] is trained to generate the prosody latent vectors based only on text and speaker information. At inference time, the prosody generator is used to produce prosody vectors based on input text and with no need for an audio reference which improves both quality and speed of speech synthesis.

在提出的模型中, 一个韵律生成器使用去噪扩散对抗生成模型仅使用文本和说话人信息生成韵律隐变量. 在推理时, 韵律生成器用于基于输入文本的韵律向量而无需音频参考, 从而提升语音合成的质量和速度.

> While most of the studies in this category follow the baseline model and use mel-spectrograms to represent the reference audio, other studies extract correlated prosody features as input to the VAE. For instance, frame-level F0, energy, and duration features are extracted from the reference speech as basic input for the hierarchical encoder of the CHiVE model [109]. These same features are also used as input for the VAE encoder in work [35], but at the phoneme level. In work [68], multi-resolution VAEs are employed, each with acoustic and linguistic input vectors. The acoustic feature vectors for each encoder include 70 mel-cepstral coefficients, log F0value, a voiced/unvoiced value, and 35 mel-cepstral analysis aperiodicity measures.

此类的大部分研究都遵循基线模型并使用梅尔频谱表示参考音频, 其他研究提取相关韵律特征作为 VAE 的输入. 例如 帧级别的 F0, 能量, 时长特征从参考音频中提取, 作为 ChiVE 模型层次编码器的输入. 这些相同特征同样在文献 [35] 中使用, 但是是音素级别.

文献 [68] 采用了多分辨率 VAEs, 每个都有声学和语言输入向量. 声学特征包括 70 个梅尔频谱系数, 对数 F0 值, 有声/无声值和 35 个 mel-cepstral 分析非周期度量.

### 4.3 Global Style Tokens 

> The Global Style Tokens (GST) approach for expressive synthesis was first introduced in [75]. The paper proposes a framework to learn various speaking styles (referred to as style tokens) in an unsupervised manner within an end-to-end TTS model. The proposed approach can be seen as a soft clustering method that learns soft style clusters for expressive styles in an unlabeled dataset. In detail, GST, as shown by Fig. 8, extends the approach introduced in [74] by passing the resulting style embedding from the reference encoder to an attention unit,which functions as a similarity measure between the style embedding and a bank of randomly initialized tokens.
> During training, the model learns the style tokens and a set of weights, where each style embedding is generated via a weighted sum of the learned tokens. In fact, the obtained weights represent how each token contributes to the final style embedding. Therefore, each token will represent a single style or a single prosody-related feature, such as pitch, intensity, or speaking rate.
> At inference time, a reference audio can be passed to the model to generate its corresponding style embedding via a weighted sum of the style tokens. Alternatively, each individual style token can be used as a style embedding.
> In addition, GSTs offer an enhanced control over the speaking style through various operations. These include manual weight refinement, token scaling with different values, or the ability to condition different parts of the input text with distinct style tokens.

用于表达性语音合成的全局风格标记方法在文献 [75] 中被首次提出. 该文献提出了一个框架用于端到端 TTS 模型中以无监督学习的方式用于学习各种说话风格 (称为风格标记). 该方法可以视为一种软聚类方法用于无标签数据集中学习软风格簇. 具体地 GST 如图八所示, 将文献 [74] 的方法进行了扩展, 将参考编码器输出的风格嵌入传递给注意力单元, 作为风格嵌入和一组随机初始标记的相似性度量. 在训练时, 模型学习风格标记和一组权重, 其中每个风格嵌入可以通过学习到的标记进行加权和获得. 实际上, 获得的权重表示每个标记对于最终风格嵌入的贡献. 因此每个标记将表示单个风格或单个韵律相关的特征, 例如音高, 强度或语速. 在推理时, 参考音频传递给模型通过风格标记的加权和用于生成对应风格嵌入. 或者每个单独的风格标记作为风格嵌入.
此外 GSTs 提供了通过各种操作增强对说话风格的控制, 包括手动权重细化, 标记按不同值缩放或使用不同风格标记条件话输入文本的不同部分.

> The GST-TTS model can be further enhanced by modeling different levels of prosody to improve both expressiveness and control over the generated speech. For instance, [46] proposes a fine-grained GST-TTS model where word-level GSTs are generated to capture local style variations (WSVs) through a prosody extractor. The WSV extractor consists of a reference encoder and a style token layer, as described in [75], along with an attention unit to produce the word-level style token.

> In [133] a hierarchical structure of multi-layer GSTs with residuals is proposed. The model employs three GST layers, each with 10 tokens, resulting in a better interpretation of the tokens of each level. Upon tokens analysis, it was found that the first-layer tokens learned speaker representations, while the second-layer tokens captured various speaking style features such as pause position, duration, and stress. The third-layer tokens, however, were able to generate higher-quality samples with more distinct and interpretable styles. Similarly, in[50], a multi-scale GST extractor is proposed to extract speaking style at different levels. This extractor extracts style embeddings from the reference mel-spectrogram using three style encoders at global, sentence, and sub word levels, and combines their outputs to form the multi-scale style embedding.

GST-TTS 模型能够通过建模不同级别的韵律来提升生成语音的表达性和控制.

- 文献 [46] 提出了细粒度 GST-TTS 模型, 通过韵律提取器生成了词级别的 GSTs 以捕获局部风格变化 (WSVs). WSV 提取器由参考编码器和风格标记层组成, 如文献 [75] 所述, 以及用于生成单词级别的注意力单元.
- 文献 [133] 具有残差的多层 GSTs 的层次结构, 应用三层 GST 层, 每个有 10 个标记, 从而得到各个级别标记的更佳解释. 标记分析发现第一层标记学习到说话人表示, 第二层捕获了各种说话风格例如停顿位置, 时长和强调. 第三层标题能够生成高质量的样本, 具有更明显和可解释的风格. 
- 类似地 [50], 一个多尺度 GST 提取器用于提取不同级别的说话风格, 该提取器使用三种风格编码器按全局, 句子和单词级别从参考梅尔频谱中提取风格嵌入, 并将它们的输出结合以形成多尺度的风格嵌入.

> With only a small portion of the training dataset labeled with emotions, **"End-to-End Emotional Speech Synthesis Using Style Tokens  and Semi-Supervised Training"** proposes a semi-supervised GST model for generating emotional speech. The model applies a cross-entropy loss between the one-hot vectors representing the emotion labels and the weights of GSTs,in addition to the GST-TTS reconstruction loss. The semi-GST model is trained on a dataset in which only 5%of the samples are labeled with emotion classes, while the rest of the dataset is unlabeled. After training, each style token represents a specific emotion class from the training dataset and can be used to generate speech in the corresponding emotion.

当训练集只有一小部分带有情感标签时, 文献 [026] 提出了一种半监督的 GST 模型用于生成情感语音. 该模型应用表示情感标签的独热编码和 GSTs 的权重之间的交叉熵损失, 以及 GST-TTS 重构损失. 这个 semi-GST 模型在训练集只有 5% 的样本具有情感类别的情况下训练, 训练后每个风格标记表示训练集中一个特定的情感类别并且能用于对应情感生成语音.

> Furthermore, in [92], a speech emotion recognition(SER) model is proposed with the GST-TTS to generate emotional speech while acquiring only a small labeled dataset for training. The paper formulates the training process as reinforcement learning (RL). In this frame-work, the GST-TTS model is treated as the agent, and its parameters serve as the policy. The policy aims to predict the emotional acoustic features at each time step, where these features represent the actions. The pre-trained SER model then provides feedback on the predicted features through emotion recognition accuracy, which represents the reward. The policy gradient strategy is employed to perform backpropagation and optimize the TTS model to achieve the maximum reward.

文献 [92] 一个语音情感识别 SER 模型被提出和 GST-TTS 模型结合用于生成情感语音, 只需要很小部分带标签的数据集用于训练. 该文献将训练过程形式化为强化学习. 在此架构下, GST-TTS 模型视为智能体, 其参数作为策略. 策略旨在预测每个时间步的情感声学特征, 这些特征表示动作. 预训练 SER 模型通过情感识别精度提供反馈, 即奖励. 策略梯度策略用于优化 TTS 模型以达到最大奖励.

> In contrast, the Mellotron model [114] introduces a unique structure for the GSTs, enabling Mellotron to generate speech in various styles, including singing styles, based on pitch and duration information extracted from the reference audio. This is achieved by obtaining a set of explicit and latent variables from the reference audio. Explicit variables (text, speaker, and F0contour) capture explicit audio information, while latent variables (style tokens and attention maps) capture the latent characteristics of speech that are hard to extract explicitly.

文献 [114] Mellotron 引入 GSTs 独特结构, 使得 Mellotron 能够基于从参考音频中提取的音高和时长信息生成各种风格的语音, 包括歌唱风格. 这通过从参考音频中获取显式和隐式变量实现. 显式变量捕获显式音频信息, 隐式变量捕获语音中难以显式提取的隐藏特征.

### 4.4.基于上下文学习的方法

> These is a group of recent TTS models that are trained on a large amounts of data using in-context learning strategy. During in-context learning (also called prompt engineering), the model is trained to predict missing data based its context. In other words, the model is trained with a list of input-output pairs formed in a way that represents the in-context learning task. After training, the model should be able to predict the output based on a given input.

近期有一组 TTS 模型通过上下文学习策略在大量数据上进行训练. 在上下文学习 (或提示工程) 中, 模型被训练用于基于上下文预测缺失数据. 换句话说模型通过表示上下文学习任务的输入输出对列表进行训练, 训练后模型能够预测给定输入的输出.

> For the TTS task, the provided style reference (referred to as prompt) is considered as part of the entire utterance to be synthesized. The TTS model training task is to generate the rest of this utterance following the style of the provided prompt as shown by Fig. 9. By employing this training strategy, recent TTS models such as [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [NaturalSpeech2 (2022)](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md), and [Voicebox (2023)](../../Models/Speech_LLM/2023.06.23_VoiceBox.md) are capable of producing zero-shot speech synthesis using only a single acoustic prompt. Furthermore, these models demonstrate the ability to replicate speech style/emotion from a provided prompt ([NaturalSpeech2 (2022)](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md), [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md)) or reference ([Voicebox (2023)](../../Models/Speech_LLM/2023.06.23_VoiceBox.md)) to the synthesized speech.

对于 TTS 任务, 提供的风格参考 (即提示) 被考虑为要合成的整个发言的一部分. TTS 模型训练任务即遵循提示的风格生成这个发言剩下的部分. 通过应用这种训练策略, 近期 TTS 模型例如 VALL-E, NaturalSpeech2 和 VoiceBox 能够使用单个声学提示进行零次语音合成. 此外, 这些模型说明了从提供的提示或参考复制语音风格/情感到合成语音的能力.

> In [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), a language model is trained on tokens from [Encodec (2022)](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md), and the input text is used to condi-tion the language model. Specifically, the Encodec model tokenizes audio frames into discrete latent vectors/codes,where each audio frame is encoded with eight codebooks.
> VALL-E employs two main models: the first one is an auto-regressive (AR) model that predicts the first code of each frame, and the second is non-auto-regressive (NAR)model that predicts the other seven codes of the frame.

VALL-E 一个语言模型在 Encodec 的标记上训练, 且输入文本用于条件化语言模型. 特别地, Encodec 模型将音频帧离散化为离散的隐向量/代码, 每个音频帧由八个码本进行编码. VALL-E 应用两个主要模型一个是自回归模型能够预测每帧的第一个编码, 第二个是非自回归模型用于预测其他七个编码.

Instead of discrete tokens used in VALL-E, [NaturalSpeech2 (2022)](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md) represents speech as latent vectors from a neural audio codec with residual vector quantizers. The latent vectors are then predicted via a diffusion model,conditioned on input text, pitch from a pitch predictor,and input speech prompt.
> 和 VALL-E 不同, NaturalSpeech 2 使用具有残差向量量化的神经音频编解码器将语音表示为隐向量. 隐向量之后通过扩散模型进行预测, 根据输入问题, 音高预测器的音高和输入语音提示进行条件化.

> Another example of in-context training is [Voicebox (2023)](../../Models/Speech_LLM/2023.06.23_VoiceBox.md) which is a versatile generative model for speech trained on a large amount of multilingual speech data. The model is trained on a text-guided speech infilling task, which gives it the flexibility to perform various speech tasks such as zero-shot TTS, noise removal, content editing,and diverse speech sampling. Voicebox is modeled as a non-autoregressive (NAR) flow-matching model with the ability to consider future context.

另一个基于上下文训练的例子是 Voicebox, 是一个在大量多语言语音数据上训练的语音的通用生成模型. 该模型在文本引导的语音填充任务上进行训练, 这使其能够执行各种语音任务, 例如零次 TTS, 去噪, 内容编辑和多样化的语音采样. Voicebox 被建模为一个非自回归的流匹配模型, 能够考虑未来上下文.


### 4.5 Other Approaches 其他方法

> This category containes reviewed papers that propose individual techniques or methods which cannot be categorized under any of the previously mentioned unsupervised approaches. 

这个类别包含了一些不能归类为之前提到的任何无监督方法的单独技术和方法的总结.

> For instance, in [121], a neural encoder is introduced to encode the residual error between the predictions of a trained average TTS model and the ground truth speech. The encoded error is then used as a style embedding that conditions the decoder of the TTS model to guide the synthesis process. 

文献 [121] 引入神经编码器来编码一个训练好的平均 TTS 模型预测和真实语音之间的残差误差. 然后编码的误差被用作风格嵌入用于条件化 TTS 模型的解码器以指导合成过程.

> Raitio and Seshadri [128] improves prosody modeling of [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md) model with an additional variance adaptor for utterance-wise prosody modeling. 

文献 [128] 通过使用额外的方差适配器用于语调韵律建模, 以提升 FastSpeech2 模型的韵律建模.

> As context information is strongly related to speech expressivity, [45] proposes using multiple self-attention layers in [Tacotron2](../../Models/TTS2_Acoustic/2017.12.16_Tacotron2.md) encoder to better capture the con-text information in the input text. The outputs of these layers in the encoder are combined through either direct aggregation (concatenation) or weighted aggregation using a multi-head attention layer. 

由于上下文信息和语音表达性强相关, 文献 [45] 在 Tacotron 编码器中使用多个自注意力层用于更好地捕获输入文本的内容信息. 这些层的输出通过直接聚合 (拼接) 或加权聚合 (多头注意力层) 进行结合.

> Additionally, there are some papers that propose using only input text to obtain prosody-related representations/embeddings without any style references, and those are further discussed in Section 5.2.4.

此外, 有些文献提出只使用输入文本用于获得韵律相关表示/嵌入, 无需风格参考, 这在后续的 5.2.4 中进行讨论.
