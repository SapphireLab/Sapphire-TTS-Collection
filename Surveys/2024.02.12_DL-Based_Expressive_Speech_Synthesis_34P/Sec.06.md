# Section 06: Datasets and Open Source Codes

Deep learning models, including TTS models, rely heavily on the availability of data in terms of size and diversity.
Furthermore, the quality of synthesized speech by TTS models is closely tied to the quality and size of the data used for model training.
ETTS models face even greater challenges in this regard, as they require data to be not only high-quality and clean but also to accurately represent the numerous available speaking styles and emotions.

深度学习模型, 包括文本转语音模型, 很依赖数据在数量和多样性方面的可用性.
此外, 文本转语音模型合成的语音质量和模型训练所用数据的质量和多少高度相关.
表达性文本转语音模型在这一方面甚至面临更加艰难的挑战, 因为它们需要的数据不仅要高质量且干净, 还需要能准确地表示许多可用的说话风格和情感.

A main limitation in the domain of expressive speech synthesis is the inadequate availability of expressive speech datasets.
Although there are several emotional and expressive speech datasets publicly available, they still fall short in terms of size, accuracy, and diversity required to train effective ETTS models.
As a result, current ETTS models still suffer from performance degradation and poor generalization.
In [170], which introduces a recent multi-emotion and multi-speaker dataset, it provides a concise summary of the majority of the available emotional speech datasets.
Additionally, there are open expressive datasets, such as the Blizzard datasets [171], which are larger in size but lack any labels.
Furthermore, widely used TTS datasets, such as LJSpeech [172] with a single speaker, and VCTK [173], and LibriTTS [174] with multiple speakers, are also employed in expressive TTS models to address prosodic features generally, control issues, or learning different speakers’ styles.
In Table 6, we list the main open-source databases used in the papers covered by this review.

表达性语音合成领域的一个主要限制是表达性语音数据集的可用性不足.
虽然有数个情感和表达性语音数据集公开可用, 但在训练表达性文本转语音模型所需的大小, 精度和多样性方面仍然不足.
因此, 现有的表达性文本转语音模型仍然会有性能退化和泛化能力差的现象.
在文献 170 中引入了最近的多情感和多说话人数据集, 其提供了大多数可用情感语音数据集的简要总结.

此外, 还有一些开放的表达性数据集, 例如 Blizzard 数据集, 规模较大但缺乏标签.
另外被广泛应用的文本转语音数据集, 例如单说话人的 LJSpeech, 多说话人的 VCTK 和 LibriTTS, 通常也被用于表现性文本转语音模型以处理韵律特征, 控制问题或学习不同说话人风格等.

表格 06 列出了本综述涵盖的论文主要使用的开源数据库.
注: 说话人无标注的默认为单说话人.

<!-- #TODO add files links -->

|数据库|语言|说话人|情感|
|---|:-:|:-:|:-:|
|Blizzard Challenge 2012, 2013, 2016, 2019| 英语 |  | |
|VCTK|英语|多||
|LibriSpeech|英语|多||
|IEMOCAP|英语|多|多|
|CMUARCTIC|英语|多||
|LJSpeech|英语|||
|LibriTTS|英语|多||
|Chinese Standard Mandarin Speech Copus (CSMSC) | 汉语|
|Aishell-3|汉语|
|Korean Emotional Speech (KES) dataset|韩语||多|
|English conversation corpus (ECC)|英语|多||
|IndicTTS database |印度|||
|Emotional Speech Dataset (ESD)|英语/汉语|多|多|
|Japanese Kamishibai and audiobook corpus (J-KAC)|日语|||
|Multilingual Libri Speech|多语种|多||
|Korean Single Speaker (KSS) | 韩语 |多||

Numerous internal expressive speech datasets are utilized in many studies in literature.
Some of these datasets are of large size and exhibit good quality, with high diversity, including multiple speakers, styles, and emotions.
However, they are not open to the research community.
Additionally, replicating the work presented in these studies or making further improvements is challenging.
Constructing an expressive speech dataset is, in fact, a demanding endeavor compared to collecting neutral speech datasets, due to several factors.

文献中有许多研究使用了内部表达性语音数据集.
其中有一些数据集有很大规模, 且具有良好质量, 高多样性, 包括多个说话人, 风格和情感.
然而, 这些数据集并不对研究社区开放.
此外, 复制这些研究成果或进行进一步改进也面临着挑战.
构造一个表达性语音数据集, 实际上与收集中性语音数据集相比, 要复杂得多, 因为存在多种因素.

First of all, differences among speakers in portraying different speech styles or emotions pose the first challenge.
Some speakers may overact, while others may misinterpret or blend acting styles or emotions.
Secondly, there are variations in emotional interpretation among different listeners who annotate the same expressive speech, which can impact the accuracy and consistency of these datasets.
Notably, [66] highlighted the differences in emotional reception among listeners for the same utterance, as explained in [Section 3.1]().

<!-- #TODO add links to sec3.1 -->

首先, 说话人在描述不同语音风格和情感时的不同就是第一个挑战.
一些说话人可能夸张表演, 而另一些可能误解或混合了表演风格和情感.
齐次, 不同听众对于同一个表达性语音的标注存在变化, 会影响这些数据集的精度和一致性.
值得注意的是, 文献 66 指出, 不同听众对于同一句话的情感接收差异, 如 3.1 节所述.

Moreover, the wide range of human emotions and speaking styles introduces further complexities.
In the literature, emotions are defined and classified based on various criteria [175].
One common classification approach distinguishes between discrete emotions, which are considered basic emotions recognizable through facial expressions and biological processes, and dimensional emotions, which are identified based on dimensions such as valence and arousal [176, 177].
A well-known study conducted by Paul Ekman and Carroll Izard [178] involved cross-cultural studies and identified six main basic emotions, including anger, disgust, fear, happiness, sadness, and surprise.
In fact, although the available emotional datasets diverge in the set of emotions they cover, as shown in [170], most of the emotions considered in these datasets belong to the six basic emotion classes identified by [178].

此外, 人类情感和说话风格的广泛范围引入了更多的复杂性.
文献中基于各种评判标准对情感进行了定义并分类, 参见 175.
一个常用的分类方法用于区分离散情感和维度情感, 离散情感被认为是通过面部表情和生理过程可识别的基本情感, 而维度情感是基于效价和唤醒等维度来识别的. 参见 176 177.
一项著名的研究是 Paul Ekman 和 Carroll Izard 进行的跨文化研究, 确定了六种主要的基本情感, 包括愤怒, 厌恶, 恐惧, 快乐, 悲伤和惊讶.
实际上, 尽管可用的情感数据集在所涵盖的情感集合上有所不同, 如 170 所示, 但数据集考虑的大多数情感都属于这六种基本情感类型.

Additionally, when considering different languages and multiple speakers, the challenge becomes more intricate.
However, with the new trend that introduces the language modeling approach to the field of speech synthesis, it becomes possible to train TTS models on a large amount of data using an in-context learning strategy.
This vast amount of data provides diversity in speakers, speaking styles, and prosodies, and it can be used for training despite noisy speech and inaccurate transcriptions.
In fact, recent TTS models based on language modeling, such as VALL-E [22], NaturalSpeech 2 [18], and Voicebox [25], have been successful in various speech-related tasks, especially zero-shot speech synthesis.
Besides, they have shown promising results in expressive speech synthesis, as they are able to replicate the speech style and emotion provided in a single input acoustic prompt to the synthesized speech.

此外, 当考虑到不同语言和多个说话人时, 挑战变得更加复杂.
然而, 随着将语言建模方法引入语音合成领域的新趋势, 使用上下文学习策略在大规模数据集上训练文本转语音模型成为可能.
这些大量的数据提供了说话人, 说话风格和韵律的多样性, 即便存在嘈杂的噪音和不准确的转录也可以用于训练.
实际上, 近期的基于语言建模的文本转语音模型, 例如 [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [NaturalSpeech 2](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md), [Voicebox](../../Models/_tmp/2023.06.23_VoiceBox.md) 等, 在各种语音相关任务上都取得了成功, 尤其是零样本语音合成.
此外, 它们在表现性语音合成方面也取得了令人振奋的成果, 因为它们能够复制单个输入音频提示的语音风格和情感.

As for open-source codes, several implementations and repositories are publicly available.
Table 7 list some main open-source implementations for expressive speech synthesis models.

对于开源代码, 有几种实现和仓库是公开可用的.
表格 07 列举了表达性语音合成模型的一些主要的开源实现.


|模型|链接|
|---|---|
|Espnet| [Github](https://github.com/espnet/espnet)|
|coqui| [Github](https://github.com/coqui-ai)|
|Mozilla| [Github](https://github.com/mozilla)|
|NeMo (NVidia)| [Github](https://github.com/NVIDIA/NeMo)|
|espeak-ng| [Github](https://github.com/espeak-ng/espeak-ng)|
|marytts| [Github](https://github.com/marytts/)|
|CSTR-Edinburgh| [Github](https://github.com/CSTR-Edinburgh)|
|Hugging Face| [HuggingFace](https://huggingface.co/docs/transformers/main/en/tasks/text-to-speech) / [HF-Mirror](https://hf-mirror.com/docs/transformers/main/en/tasks/text-to-speech)|