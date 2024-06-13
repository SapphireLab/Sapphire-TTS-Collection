
## 3.监督学习方法


> Supervised approaches refer to models that are trained on datasets with emotion labels.
> Those labels guide model training, enabling it to learn accurate weights.
> Early deep learning-based expressive speech synthesis systems were primarily supervised models that utilized labeled speech exhibiting various emotions (such as sadness, happiness, and anger) or speaking styles (such as talk-show, newscaster, and call-center).
> Note that the term style has also been used to refer to a set of emotions or a mixture of emotions and speaking styles ([ST-TTS](), [68] [78] [79]. 
> **Generally, the structure of early conventional TTS models was built upon two primary networks: one for predicting duration and the other for predicting acoustic features.
> These acoustic features were then converted to speech using vocoders.
> Both networks receive linguistic features extracted from the input text.** 
> In supervised ETTS approaches, speech labels (emotions and/or styles) are represented in the TTS model as either input features or as separate layers, models, or sets of neurons for each specific label.
> The following sections explain these three representations in detail then we provide a general summary of the supervised approaches reviewed in this work in Table 2.

监督学习方法是指在带有情感标签的数据集上训练的模型。这些标签指导模型训练以确保能够学习到准确的权重。
早期基于深度学习的表达性语音合成系统主要是监督模型，他们使用带有各种情感 （如悲伤、快乐和愤怒）或说话风格（如脱口秀、新闻播音和呼叫中心）标签的语音。注意，“风格”这一术语同样用于指代一组情感或情感和说话风格的混合。
**通常早期传统 TTS 模型的结构是建立在两个主要网络之上：一个用于预测时长；另一个用于预测声学特征，这些声学特征之后通过声码器转换为语音。这两个网络都接收从输入文本中提取出的的语义特征。**
在监督 ETTS 方法中，语音标签（情感与、或风格）在 TTS 模型中被表示为输入特征或为每个特定标签的单独层、模型或神经元集合。以下小节将详细解释这三种表示，然后在表格二中提供本文回顾的监督方法的概述。

|引用序号|算法简称|输入|情感标签表示|是否支持情绪转移|TTS 模型|
|:-:|:-:|---|---|:-:|:-:|
|80||语言特征+情感标签|独热编码||DL-SPSS, HMM|
|65||语言特征+情感标签|独热编码/单独层|√|DL-SPSS|
|66||语言特征+情感标签|感知向量/矩阵||DL-SPSS|
|41||语言特征+情感标签|独热编码||DL-SPSS|
|42||语言特征+情感标签|依赖层||DL-SPSS|
|81||语言特征+情感标签|独热编码/神经元集合|√|DL-SPSS|
|43||语言特征+情感标签|独热编码/依赖层/独立模型||DL-SPSS|
|82||语言特征+情感标签|独热编码|√|DL-SPSS|
|83||音素序列+语言模型特征+情感标签|嵌入向量||Encoder-Dttention-Decoder|
|28, 78||语言特征+情感标签|独热编码/依赖层/独立模型|√|DL-SPSS|
|26||音素序列+梅尔频谱+情感标签|独热编码 GSTs 权重的真实值||Tacotron2|
|27||音素序列+语言特征+情感标签|嵌入向量||Tacotron2|
|84||语言特征+情感标签|嵌入与其他数据标签联合||DL-SPSS|
|85||语言特征+韵律特征+情感标签|分类器的真实值||DL-SPSS|
|86||音素序列+情感标签|嵌入向量||Transformer TTS|
|32, 36||字符序列+梅尔频谱+情感标签|分类器的真实值||Tacotron2|
|69||语言特征+情感标签|独热编码/依赖层|√|DL-SPSS|
|34||音素序列+梅尔频谱+情感标签|分类器的真实值||Tacotron2|
|64||字符序列+语言模型特征+情感标签|分类器的真实值||Tacotron2|
|39, 87||音素序列+梅尔频谱+情感标签|分类器的真实值||Tacotron2|

### 3.1 Labels as Input Features 标签作为输入特征

> The most straightforward method for representing emotion labels of annotated datasets as input to the TTS model is by using a one-hot vector.
> This approach entails using a vector with a size equivalent to the number of available labels.
> In this vector, a value of (1) is assigned to the index corresponding to the label ID, while all other values are set to (0).
> Many early ETTS models [43] [56] [65] [69] [78] [80] [82] [84] advocated for this direct representation of emotion labels in order to generate speech encompassing various emotions.

用于表示带有注释的数据集的情感标签作为 TTS 模型输入的最直接方法是使用独热编码。这种方法需要使用和可用标签数量长度相同的向量。在这个向量内，将值为 1 分配给标签 ID 对应的索引，其他值为 0. 许多早期 ETTS 模型提倡这种直接表示情感标签的方法以生成包含各种情绪的语音。

> The one-hot emotion vector, also referred to as a style/emotion code in some studies [43] [78] [80] [82], is concatenated with the input linguistic features of the model.

独热编码在某些文献中也被称为风格/情感编码，和模型的输入语言特征进行拼接。

> When dealing with large number of labels, the one-hot representation becomes both high-dimensional and sparse.
> Moreover, in other scenarios, merging label vectors with input features instead of concatenation can lead to length mismatch issues. 

当处理大量标签时，独热编码表示变得高维且稀疏。而且在其他情况下，将标签向量和输入特征合并而不是拼接会导致长度不匹配问题。

> In both situations, the embedding layer offers a solution by creating a continuous representation for each label, known as embedding vectors.
> Unlike the one-hot vector, which is constrained in size based on the number of labels, an emotion embedding can have any dimension, regardless of the number of available labels.

在这两种情况下，嵌入层提供了一种解决方案：通过给每个标签创建一个连续的表示，即嵌入向量。和独热编码受到标签数量的限制不同，情感嵌入可以有任意的维度，和可用标签数量无关。

> For instance, in [84], each sample in the training dataset has three separated labels including speaker, style(emotion), and cluster.
> In this context, the cluster value indicates the consistency in speech quality of a given speaker and style pair.
> If one-hot vector is used to represent each unique combined label of each sample, the resulting label vector will be high dimensional (which in this case is 67).
> Therefore, the three one-hot vectors representing the given three labels are combined and passed as input to an embedding layer to reduce its dimension (in this case 15).
> On a different note, [41] utilizes an embedding layer to expand concise binary one-hot label vectors to match with the dimensions of the input features to be added together as input to the TTS model.

例如, 在文献 [84] 中, 训练数据集的每个样本有三个独立的标签, 包括说话人, 风格 (情感) 和聚类类别. 在这种情况下, 聚类类别值表明了同时给定说话人和风格在语音质量方面的一致性. 如果独热编码用于表示每一个唯一拼接的标签, 那么标签向量会变得非常高维. 因此这三个独热编码向量分别表示给定的三个标签结合并输入到嵌入层进行降维. 而文献 [41] 使用嵌入层将简洁的二进制独热标签向量扩展到匹配输入特征的维度, 以便相加作为 TTS 模型的输入.

> To address the potential disparities between a talker’s intent and a listener’s perception when annotating emotional samples, in [66], a different methodology for representing labels is introduced.
> In the context of N emotion classes, each sample from the talker may be perceived by the listener as one of the N emotions.
> In response to this, the paper suggests the adoption of a singular vector termed the ’perception vector,’ with N dimensions.
> This vector represents how samples from a specific emotion class are distributed among the N emotions, based on the listener’s perception.
> Furthermore, in the context of multiple listeners, each emotion class can be represented as a confusion matrix that captures the diverse perceptions of samples belonging to that emotion class by multiple listeners.

为了解决在注释情感样本时说话人意图和倾听者的感知之间的潜在差异, 文献 [66] 引入了表示标签的不同方法. 在具有 N 个情绪类别的情况下, 来自说话人的每个样本可能被倾听者感知为这 $N$ 个情绪的其中之一. 对此, 文献 [66] 建议采样一个名为"感知向量"的单个 N 维向量. 这一向量表示特定情绪类别的样本如何根据倾听者的感知在 N 个情绪上分布. 此外, 在多倾听者的情况下, 每个情绪类可以表示为一个混淆矩阵, 捕获由多个倾听者提供的属于该情绪类别的样本的多样性感知.

### 3.2 Labels as Separate Layers/Models 标签作为单独层/模型

> In this approach, to represent emotion or style labels in TTS models, each label is associated with either a separate instance of the DNN model, an emotion-specific layers, or a set of emotion-specific neurons within a layer.
> Initially, the model is trained using neutral data, which typically has larger size.
> Subsequently, in the first approach, multiple copies of the trained model are fine-tuned using emotion-specific data of small size [43] [78].
> In the second approach, instead of creating an individual model for each emotion, only specific model layers (usually the uppermost or final layers) from the employed DNN model are assigned to each emotion [43] [65] [69] [78] as shown by Fig. 5.
> While shared layers are adjusted during training using neutral data, output layers corresponding to each emotion are modified exclusively when the model is trained with data from the respective emotion.

在这种方法中, TTS 模型内为了表示情感或风格标签, 每个标签要么和 DNN 模型的单独示例即一个特定情感层, 要么和一层内的特定情感神经元集合相关联. 
首先, 模型使用通常尺寸较大的中性数据进行训练. 
第一种方法, 对多个已经训练好的模型的副本分别使用小尺寸的特定情感数据进行微调; 
第二种方法, 不为每种情感创建单独模型, 而是只将所使用的 DNN 模型中的特定层 (通常是最上层/最终层) 分配给每种情感. 如图五所示. 使用中性数据训练时共享层会进行调整, 对应每种情感的输出层仅在模型使用相应情感的数据进行训练是进行修改.

> Alternatively, when dealing with limited data for certain emotions/styles, the model can initially undergo training for emotions with large amount of data.
> Following this step, the weights of the shared layers within the model are fixed, and only the weights of the top layers are fine-tuned using the limited, emotion-specific data [42]. 

当处理某些情感或风格的有限数据时, 模型可以先为具有大量数据的情感进行训练. 完成后将共享层的权重固定, 只有最顶层的权重使用有限的, 特定情感的数据进行微调. 如文献 [42]. 

> Another method for representing emotion labels involves allocating specific neurons from a layer within the DNN model for each emotion.
> In this approach, the hidden layers of the model could be expanded by introducing new neurons.
> Then, as outlined in [81], particular neurons from this expanded set are assigned to represent each distinct emotion.
> Importantly, the associated weights of these specific neuron subsets are adjusted solely during the processing of data relevant to the corresponding emotion.
> Furthermore, by substituting the subset of neurons dedicated to a particular emotional class with a different set, the model becomes capable of generating speech imbued with the desired emotional class.
> This capability holds true even for new speakers who only possess neutral data, and in this case, it is known as **expression/emotion transplantation**.

其他表示情感标签的方法是将 DNN 模型层中特定的神经元分配给每种情感. 这种方法可以通过引入新的神经元来扩展模型的隐藏层. 如文献 [81] 从扩展的神经元集合中分配特定神经元来表示每种不同的情感. 重要的是只有在处理和相应情感相关的数据时, 这些特定神经元子集的关联权重才会进行调整. 此外, 通过加入专门用于某种特定情感类别的神经元, 模型可以生成具有所需情感类的语音.
这种能力对仅有中性数据的新说话人也成立, 这称为表达/情感移植.

### 3.3 Labels for Emotion Predictors/Classifiers 标签用于情感预测器/分类器

> Another common approach to utilize emotion labels is to use them directly or via emotion predictor or classifier to support the process of extracting emotion/prosody embedding. 

另一种常见的使用情感标签的方法是直接使用它们或者通过情感预测器/分类器以支持提取情感/韵律嵌入的过程.

> For example, in **"End-to-End Emotional Speech Synthesis Using Style Tokens  and Semi-Supervised Training"** emotion labels represented as one-hot vectors are used as targets for the weight vectors of GSTs (explained in Section 4.3) where a cross entropy loss between the two vectors is added to the total loss function.
> Yoon et al. [64] proposes a joint emotion predictor based on the Generative Pre-trained Transformer (GPT)-3 [88].
> The proposed predictor produces two outputs including emotion class and emotion strength based on features extracted from input text by (GPT)-3.
> A joint emotion encoder is then used to encode the predictor outputs into a joint emotion embedding.
> The joint emotion predictor is trained with the guidance of the emotion labels and emotion strength values obtained via a <algo>ranking support vector machine (RankSVM)</algo[89].

文献 [026] 中情感标签表示成独热向量, 作为 GSTs 权重向量的目标值, 这两个向量之间的交叉熵损失被添加到总损失函数中;
文献 [64] 提出了基于 GPT-3 的联合情感预测器, 这个预测器基于 GPT-3 从输入文本中提取的特征产生两个输出包括情感类别和情感强度. 
联合情感编码器将预测器的输出编码为一个联合情感嵌入. 
联合情感预测器在通过 RankSVM 获得的情感标签和情感强度值得指导下进行训练.

> In [32], an emotion classifier is used to produce more discriminative emotion embeddings.
> Initially, the input Mel-spectrogram features from the reference-style audio and those predicted by the proposed TTS model are passed to two reference encoders (explained in Section 4.1) to generate reference embeddings.
> Both embeddings are then fed to two emotion classifiers, which consist of intermediate fully connected (FC) layers.
> The output of the second FC layer from both classifiers is considered as the emotion embedding.
> Apart from the loss of the classifiers, an additional loss function is established between the resulting emotion embeddings from the two classifiers.
> Similarly, an emotion classifier is also employed in [36] to reduce irrelevant information in the generated emotion embedding from an emotion encoder with reference speech (Mel-spectrogram) as input.

文献 [32] 使用情感分类器用于产生更具有区分性得情感嵌入. 
首先从参考风格音频的输入梅尔频谱特征和 TTS 模型的相应预测传递到两个参考编码器中以生成参考嵌入. 
两个嵌入之后都输入到两个情感分类器中, 由中间全连接层组成. 
分类器的第二个全连接层的输出被视为情感嵌入. 
除了分类器的损失之外, 还建立了两个分类器产生的情感嵌入结果之间的附加损失函数. 
文献 [36] 使用了情感分类器用于减少带有参考语音 (梅尔频谱) 的情感编码器生成的情感嵌入的无关信息.

> Several other studies [34] [36] [39] that support multiple speakers also suggest utilizing a speaker classifier in addition to the emotion classifier.
> This approach aims to improved the speaker embedding derived from speaker encoders.
> Moreover, these studies introduce an adversarial loss between the speaker encoder and the emotion classifier using a gradient reversal layer (GRL) [90].
> The purpose of this is to minimize the potential transfer of emotion-related information into the speaker embedding.
> The GRL technique involves updating the weights of the speaker encoder by utilizing the inverse of the gradient values obtained from the emotion classifier during the training process.

文献 [34] [36] [39] 支持多说话人的研究也建议使用说话人分类器以及情感分类器. 
这一方法旨在改善说话人编码器导出的说话人嵌入. 
此外这些研究通过使用一个 GRL 引入了说话人编码器和情感分类器的对抗损失. 
目的是最小化情感相关的信息转移到说话人嵌入中.
GRL 技术涉及在训练过程中使用从情感分类器获得的梯度值的逆来更新说话人编码器的权重.
