# 6.Evaluations: 评估

<details>
<summary>展开原文</summary>

Similar to TextLMs, SpeechLMs have a wide range of capabilities, making it challenging to compare different SpeechLMs.
Consequently, it's essential to evaluate SpeechLMs from various perspectives to determine their effectiveness.
In this section, we review the commonly used methods and benchmarks for evaluating SpeechLMs.
We categorize these evaluation methods into automatic and human assessments, each containing distinct evaluation aspects.

</details>
<br>

类似于文本语言模型, 语音语言模型也有广泛的能力, 使得对比不同的语音语言模型变得具有挑战性.

## 6.1.Automatic (Objective) Evaluation: 自动评估 (客观评估)

<details>
<summary>展开原文</summary>

Automatic evaluation methods are essential for providing quick and consistent assessments of SpeechLMs.
These methods typically rely on quantitative metrics that can be computed without human intervention.
Below, we outline some of the most commonly used automatic evaluation techniques.

</details>
<br>

自动评估方法对于提供快速且一致的语音语言模型评估至关重要.
这些方法通常依赖于定量指标, 这些指标可以在不需人工干预的情况下计算.
下面, 我们简要介绍一些最常用的自动评估技术.

### Representation Evaluation: 表示评估

<details>
<summary>展开原文</summary>

Representation (embedding) is a crucial component in SpeechLMs (and TextLMs).
It refers to how input data, such as speech or text, is transformed into a format that the model can understand and process.
Effective representation lays a solid foundation for models to understand lexical, syntax, and contextual information, which are vital for generating coherent and contextually relevant outputs.

In the context of SpeechLMs, representation evaluation focuses on how well the model encodes speech features into meaningful vectors.
[GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md) uses between-speaker ABX score to measure the embedding similarity.
It quantifies how well-separated the phonetic categories are.
Specifically, It works by comparing three sound samples: two from the same category (A) and one from a different category (B).
The test measures how often the system correctly identifies that two sounds from category A are more similar to each other than one sound from A is to a sound from B.
Another way of evaluating representations is through speech resynthesis ([GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md)).
Specifically, an input speech is encoded into tokens and then synthesized back to speech.
Then, word error rate (WER) or character error rate (CER) can be computed on the ASR results of the input and resynthesized speech.
This measures the information loss caused by discretizing the input speech into speech tokens, thereby evaluating the robustness of the latent representations.

</details>
<br>

表示 (嵌入) 是语音语言模型 (和文本语言模型) 的重要组成部分.
它指的是如何将输入数据, 如语音或文本, 转换为模型可以理解和处理的格式.
有效的表示为模型理解词汇, 语法和上下文信息提供了坚实的基础, 这对于生成连贯且上下文相关的输出至关重要.

在语音语言模型的背景下, 表示评估关注的是模型如何将语音特征编码为有意义的向量.
[GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md) 使用不同说话人的 ABX 得分来衡量嵌入的相似性.
它量化了音素类别之间的分离程度.
具体来说, 它通过比较三个音频样本: 两个来自同一类别 (A) 的样本和一个来自不同类别 (B) 的样本.
这一测试衡量了系统正确识别两个来自类别 A 的音频样本比一个来自 A 的样本更相似于来自 B 的样本的频率.
另一种评估表示的方法是通过语音重新合成 ([GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md)).
具体来说, 输入语音编码为 Token, 然后再合成回语音.
然后, 字错误率 (WER) 或 字符错误率 (CER) 可以在输入和合成语音的 ASR 结果之间计算.
这衡量了将输入语音离散化为语音 Token 所导致的信息损失, 从而评估潜在表示的鲁棒性.

### Linguistic Evaluation: 语言学评估

<details>
<summary>展开原文</summary>

Linguistics, including lexical, syntactic, and semantic evaluation methods, assess the model’s ability to generate and understand the rules for constructing words, sentences, and meaningful contents.
These evaluations focus on the correctness and appropriateness of word choices, the grammatical structure of the outputs, and the coherence and relevance of the generated content.
In terms of benchmark datasets, [sWUGGY (2020)](../../Evaluations/2020.11.23_sWUGGY.md) assesses at the lexical level by determining if the model can distinguish a real word from a (real, non-real) word pair.
sBLIMP ([sWUGGY (2020)](../../Evaluations/2020.11.23_sWUGGY.md)) evaluates at the syntactic level by determining if the model can identify the grammatically correct sentence from a (grammatical, ungrammatical) sentence pair.
Spoken StoryCloze ([TWIST (2023)](../../Models/SpeechLM/2023.05.22_TWIST.md)) evaluates semantic comprehension by assessing the model's capability to select the genuine ending of a story from a pair of ending choices.
All the evaluation is conducted by comparing the model's negative log-likelihood of the data pair.

</details>
<br>

语言学, 包括词汇, 句法, 语义评估方法, 评估模型生成和理解构建词汇, 句子和有意义内容的规则的能力.
这些评估关注词选择的正确性和适当性, 输出的语法结构, 以及生成内容的连贯性和相关性.
在基准数据集方面, [sWUGGY (2020)](../../Evaluations/2020.11.23_sWUGGY.md) 通过判断模型能否区分真实词和 (真实, 非真实) 词对来评估词汇水平.
sBLIMP ([sWUGGY (2020)](../../Evaluations/2020.11.23_sWUGGY.md)) 通过判断模型能否识别 (语法正确, 语法错误) 句子对来评估句法水平.
Spoken StoryCloze ([TWIST (2023)](../../Models/SpeechLM/2023.05.22_TWIST.md)) 通过判断模型能否正确选择故事的真实结尾来评估语义理解能力.
所有评估都通过比较模型对数据对的负对数似然来进行.

### Paralinguistic Evaluation: 副语言学评估

<details>
<summary>展开原文</summary>

In contrast to linguistic evaluation, paralinguistic evaluation focuses on the non-verbal aspects of communication that accompany speech.
Some works choose to utilize paralinguistic tokens alongside semantic tokens to enhance the paralinguistic abilities of SpeechLMs ([pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md); [SpiRit-LM (2024)](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md)), so one way is to evaluate the paralinguistic tokens.
[pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md) measures the correctness, consistency, and expressiveness of the prosodic tokens.
Correctness evaluates the model's ability to generate accurate prosodic profiles by calculating the minimal mean absolute error (min-MAE) of the prosodic tokens from 20 generated samples against the prosodic tokens from the reference, consistency is assessed through the Pearson correlation between the mean values of the prompt prosodic and its generated continuation prosodic tokens, and expressiveness is measured by the standard deviation of the generated prosody token values, with the expectation that it matches the variability of the ground truth.
We note that the same metrics can also be applied to other paralinguistic tokens.
Instead of evaluating from the token level, [SpiRit-LM (2024)](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) propose to measure on the perceptual level.
They introduced a speech-text sentiment preservation benchmark (STSP), which requires the model to generate a text or speech sequence of tokens that preserves the sentiment of the prompt.
A sentiment classifier is used to assess the sentiment in the generated speech.
It should be noted that although they only apply the preservation approach on sentiment, this idea can be generalized to other paralinguistic features, such as timbre or prosody.

</details>
<br>

副语言学评估与语言学评估相比, 关注的是伴随语音交流的非语言方面.
一些研究选择利用副语言 Token 与语义 Token 一起, 以增强语音语言模型的副语言能力 ([pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md); [SpiRit-LM (2024)](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md)), 因此, 一种方法是评估副语言 Token.
[pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md) 衡量了韵律 Token 的正确性, 一致性, 和表达性.
- 正确性通过计算生成的韵律 Token 与参考韵律 Token 之间的最小平均绝对误差来评估模型生成准确韵律轮廓的能力.
- 一致性通过提示韵律和其生成的延续韵律 Token 的平均值之间的 Pearson 相关系数来评估.
- 表达性通过生成韵律 Token 值的标准差来评估, 期望其与真值方差相匹配.

我们注意到相同的指标可以用于其他副语言 Token.

除了从 Token 级别进行评估, [SpiRit-LM (2024)](../../Models/SpeechLM/2024.02.08_SpiRit-LM.md) 提出了在感知层面上衡量.
他们引入了一个语音-文本情感保留基准 (STSP), 要求模型生成一个文本或语音序列 Token, 该 Token 能够保留提示语音的情感.
情感分类器用于评估生成的语音的情感.
需要注意的是, 虽然他们只在情感上应用了保留方法, 但这个想法可以泛化到其他副语言特征, 如音色或韵律.

### Generation Quality and Diversity: 生成质量和多样性

<details>
<summary>展开原文</summary>

Quality and diversity are two crucial aspects of model generation.
Typically, there is a trade-off between these dimensions when sampling model responses at different temperatures, so [GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md) suggests using the Area Under the Curve (AUC) with various temperature values.
Specifically, AUC on perplexity and VERT are employed to assess these factors, where VERT represents the geometric mean of the ratio of k-grams in the generated speech that appears at least once.
Additionally, the ChatGPT score can be utilized to evaluate the quality of the generated speech.
In this process, the generated speech is transcribed using state-of-the-art ASR models and then sent to ChatGPT for quality (and diversity) assessment.

</details>
<br>

质量和多样性是模型生成的两个重要方面.
通常, 在不同温度下采样模型响应时, 这两个维度之间需要权衡, 因此 [GSLM (2021)](../../Models/SpeechLM/2021.02.01_GSLM.md) 建议使用不同温度值下的曲线下面积.
具体来说, 困惑度和 VERT 的 AUC 用于评估这些音素, 其中 VERT 表示生成语音中至少出现一次的 K-Gram 的比率的几何平均.

此外, ChatGPT 得分可用于评估生成语音的质量.
在这一过程中, 生成的语音可以使用最先进的 ASR 模型进行转录, 然后发送到 ChatGPT 进行质量 (和多样性) 评估.

### Downstream Evaluation: 下游评估

<details>
<summary>展开原文</summary>

Downstream evaluation refers to evaluating the ability of SpeechLMs to perform specific tasks, such as ASR, TTS, Speaker Identification, etc.
The evaluation can be performed on pre-trained models by adding few-shot example(s) at the start of the prompt or on the instruction-tuned models by directly instructing them to do so.
[SUPERB (2021)](../../Evaluations/2021.05.03_SUPERB.md) is a benchmark containing a wide range of downstream tasks that can be performed by SpeechLMs.

</details>
<br>

下游评估指的是评估语音语言模型对特定任务的能力, 如 ASR, TTS, 说话人识别等.
评估可以通过在提示开头添加少量示例对预训练模型进行, 或者通过直接指示指令微调模型来执行.
[SUPERB (2021)](../../Evaluations/2021.05.03_SUPERB.md) 是一个包含广泛的下游任务的基准, 这些任务可以由语音语言模型进行.

## 6.2.Human (Subjective) Evaluation: 人工评估 (主观评估)

<details>
<summary>展开原文</summary>

Human evaluation plays a crucial role in assessing the performance of SpeechLMs, as ultimately, speech is designed to be heard and perceived by humans.
This type of evaluation relies on human judgment to assess the quality of the outputs generated by SpeechLMs.
Below, we outline several commonly used human evaluation methods.

</details>
<br>

人工评估在评估语音语言模型的性能中扮演着重要角色, 因为最后, 语音是为人类所听和感知的.
这种类型的评估依赖于人类的判断来评估语音语言模型生成的输出的质量.
下面, 我们简要介绍一些常用的人工评估方法.

### Mean Opinion Score: 平均意见得分

<details>
<summary>展开原文</summary>

Mean opinion score (MOS) is a widely used metric in the field of speech evaluation that quantifies the perceived quality of speech output as judged by human listeners.
Typically, a group of evaluators listens to a series of audio samples generated by the SpeechLM and rates each sample on a predefined scale, often from 1 (poor quality) to 5 (excellent quality).

MOS is calculated by averaging the scores given by all evaluators for each audio sample, providing a single score that reflects the overall quality as perceived by humans.
Variations of MOS focus on different aspects of speech quality, including MMOS, PMOS, and SMOS ([pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md); [SpeechGPT-Gen (2024)](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md)).
They evaluate the aspects of naturalness, prosody, and timbre similarity of the given speech, respectively.

Typically, evaluating naturalness or timbre similarity involves collecting human opinions.
However, this process can be complicated due to the challenges of recruiting participants and gathering their evaluations.
As a result, researchers often turn to machine-based evaluations.
They commonly employ neural network models specifically trained for these tasks.
For instance, a naturalness prediction model ([NISQA (2021)](../../Evaluations/2021.04.19_NISQA.md)) can assess the naturalness of generated outputs, while a speaker identification model can evaluate timbre similarity.

</details>
<br>

平均意见得分是语音评估领域中广泛使用的度量, 它衡量的是人类听众认为语音输出的质量.
通常, 一组评估者听取由语音语言模型生成的音频样本, 并根据预定义的评分标准 (通常是 1 (差) 到 5 (优)) 对每个音频样本进行打分.

MOS 通过计算所有评估者对每个音频样本的评分平均值来计算, 提供一个反映人类感知总体质量的单一得分.
MOS 的变体关注语音质量的不同方面, 包括 MMOS, PMOS, 和 SMOS ([pGSLM (2021)](../../Models/SpeechLM/2021.09.07_pGSLM.md); [SpeechGPT-Gen (2024)](../../Models/SpeechLM/2024.01.24_SpeechGPT-Gen.md)).
它们分别评估给定的语音的自然性, 韵律, 和音色相似性.

通常, 评估自然性或音色相似性涉及收集人类意见.
然而, 这一过程可能因招募参与者和收集评估而变得复杂.
因此, 研究人员通常转向基于机器的评估.
他们通常使用专门为此任务训练的神经网络模型.
例如, 自然性预测模型 ([NISQA (2021)](../../Evaluations/2021.04.19_NISQA.md)) 可以评估生成的输出的自然性, 而说话人识别模型可以评估音色相似性.
