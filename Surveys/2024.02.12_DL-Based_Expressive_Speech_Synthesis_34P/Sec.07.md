# Section 07: Evaluation Metrics

> An essential step in building any generative model is to evaluate its performance and compare it to the previous state-of-the-art models.
> In addition to using the same datasets, standard evaluation metrics are also needed to compare different approaches with each other.
> While evaluation metrics applied for general TTS models focus on speech quality in terms of intelligibility and naturalness, the assessment of ETTS models’ performance goes beyond that, focusing on other aspects.
> Evaluation metrics of ETTS models measure more aspects like emotion or style expressiveness, prosodic features, and controllability over all these aspects.
> Tables 8 and 9 list the common objective and subjective metrics applied for evaluating TTS models’ performance, respectively.

建立任意生成式模型的本质步骤是评估模型的性能, 并和先前的 SOTA 模型进行对比.
除了使用相同的数据集, 还需要标准的评价指标用于比较不同的方法.
除了对一般主要关注语音质量的文本转语音模型在可理解度和自然度方面的评价指标, 表达性文本转语音模型的评估还要更进一步, 着重于其他方面.
表达性文本转语音模型的评价指标需要度量更多的方面如情感, 风格表现性, 韵律特征, 以及对所有这些方面的控制能力.
表 08 和 09 列出了用于评估文本转语音模型性能的常见客观和主观指标.

| Objective Metric (客观指标) | Description (描述) |
| --- | --- |
| [Mel-Cepstral Distortion (MCD)](../../Evaluations/MCD.md) <br> 梅尔倒谱失真 | Sums the squared differences between the Mel-Frequency Cepstrum Coefficients (MFCC) from the ground truth and synthesized sample. <br>真实样本和合成样本的梅尔频率倒谱系数的均方误差. |
| [Gross Pitch Error (GPE)](../../Evaluations/GPE.md) <br> 总音高误差 | Calculates percentage of voiced frames that deviate in pitch by more than 20% compared to the ground truth samples.<br> 计算音高偏离真实样本超过 20% 的有声帧的比例. |
| [Voice Decision Error (VDE)](../../Evaluations/VDE.md) <br> 发声决策误差| Measures the difference of voiced/unvoiced decision between the ground truth and the synthesized sample.<br> 真实样本和合成样本在有声无声部分的准确性.|
| [F0 Frame Error (FFE)](../../Evaluations/FFE.md) <br> F0 帧误差 | Combines GPE and VDE by measuring the percentage of frames that either contain a 20% pitch error (GPE) or a voicing decision error (VDE) in ground truth and synthesized samples.<br> 结合 GPE 和 VDE.|
| [Word Error Rate (WER)](../../Evaluations/WER.md) <br> 词错误率 | Measures word error rate of the synthesized speech’s transcription with respect to the input text. Public automatic speech recognition (ASR) models are used for transcribing synthesized speech.<br> 衡量合成语音的文字转录的词错误率. 用公开的自动语音识别模型进行文字转录. |
| [Band APeriodicity Distortion (BAPD)](../../Evaluations/BAPD.md) <br> 频带非周期失真 | Measures over linearly spaced band aperiodicity coefficients between the ground truth and the synthesized samples.<br>衡量真实样本和合成样本之间的线性频带非周期系数 |
| [Root Mean Square Error (RMSE)](../../Evaluations/RMSE.md) <br> 均方根误差|Measure the root mean square error of F0 or energy of the synthesized samples compared to their ground truth. 度量真实样本和合成样本在 F0 或能量之间的均方根误差. |

| Subjective Metric (主观指标) | Description (描述) |
| --- | --- |
| [Mean Opinion Score (MOS)](../../Evaluations/MOS.md) <br> 平均意见得分 | Listeners to scores quality (naturalness and intelligibility) of synthesized speech with a five-point scoring system. 听众用五分制评分系统对合成的语音进行评分 (自然度和可理解度). |
| [Comparison Mean Opinion score (CMOS)](../../Evaluations/CMOS.md) <br> 比较平均意见得分 | Compares MOS values between models under test and the baseline via comparing ground truth and synthetic samples from each model. <br> 比较真实样本和每个模型地合成样本得到 MOS 值, 然后比较测试模型和基线模型的 MOS 值. |
| [Differential Mean Opinion Score (DMOS)](../../Evaluations/DMOS.md) <br> 差异平均意见得分 | Listeners score samples from one to five based on its similarity to a specific emotion or style. <br> 听众根据样本与特定情感或风格地相似度.|
| AB Preference Test <br>AB 偏好测试 | Listeners score same sentence synthesized by the two models and select the one that fulfills the given condition more than the other. <br> 听众对两个模型合成同一句话进行评分, 并选择更符合给定条件的那个. |
| ABX Preference Test <br> ABX 偏好测试 | Listeners hear three samples A, B and X ,where X represents the target speech, and they should score the one that is more close to target speech. <br> 听众听到三个样本, X 为目标语音, 听众应选择最接近目标语音的那个. |
| [MUltiple Stimuli Hidden Reference and Anchor (MUSHRA)](../../Evaluations/MUSHRA.md) <br> 多刺激隐藏参考和锚点测试| Listeners are presented with mixed samples including synthesized sample, natural speech samples (named proper reference) and total loss sample (named anchor). Listeners score each sample from 0 to 100 through a double-blind listening test.<br> 听众在双盲听音测试中, 对包含合成样本, 自然语音样本 (正确参考), 总损失样本 (锚点) 的混合样本进行评分, 分数从 0 ~ 100. |

> In fact, all the mentioned objective and subjective evaluation metrics have been applied by the studies covered in this review.
> However, in many studies, these metrics have been applied differently to assess aspects related to expressivity.
> In other words, these metrics have been applied to samples representing different emotions, speaking styles, and their varying levels of strength or intensity.
> Furthermore, samples can represent various speech synthesis scenarios, such as parallel/non-parallel style transfer and seen/unseen styles or speakers.

实际上, 前面提到的所有客观指标和主观指标已经用于本文所涵盖的研究.
然而, 在很多研究中, 这些指标被不同地应用于评估与表现力相关的方面.
换句话说, 这些指标被应用于代表不同情感, 说话风格, 不同强度的样本.
此外, 样本可以代表各种语音合成场景, 例如并行或非并行风格迁移和已知/未知风格或说话者.

> On the other hand, various additional methods have been proposed in the papers to evaluate either the effectiveness of the proposed models or the expressiveness of the synthesized speech.
> For instance, emotion and style classifiers as in [^1], [^2] and speech emotion recognition (SER) models as in [^3], [^4] which are used to measure classification accuracy, reflecting the efficiency of the proposed model in generating emotional speech.
> Furthermore, visualization and plotting of different prosodic features, variables, or embeddings are also employed in several studies [^5], [^6], [^7], [^8], [^9], [^10] to evaluate the expressivity of generated samples and compare different approaches or synthesizing scenarios.
> Additionally, ablation studies as in [^3], [^11] have also been conducted to measure the effectiveness of each component in the proposed model and how it affects the expressivity of generated speech.

另一方面, 文献提出了各种额外方法用于评估所提方法的有效性或合成语音的表达性.
例如, 情感和风格分类器, 语音情感识别模型用于衡量分类精度, 反映所提模型在生成情感语音方面的效率.
此外, 可视化和绘制不同韵律特征, 变量或嵌入的图标也被在一些研究中使用来评估生成样本的表达性和比较不同方法或合成场景.
另外, 文献中的消融实验用于衡量所提模型中每个组件的有效性和如何影响生成语音的表达性.

[^1]: An Effective Style Token Weight Control Technique for End-To-End Emotional Speech Synthesis
[^2]: Language Model-Based Emotion Prediction Methods for Emotional Speech Synthesis Systems
[^3]: Cross-Speaker Emotion Transfer by Manipulating Speech Style Latents
[^4]: Emotional Speech Synthesis with Rich and Granularized Control
[^5]: Controlling Emotion Strength with Relative Attribute for End-To-End Speech Synthesis
[^6]: FluentTTS: Text-Dependent Fine-Grained Style Control for Multi-Style TTS
[^7]: Ctrl-P: Temporal Control of Prosodic Variation for Speech Synthesis
[^8]: Emotion Controllable Speech Synthesis Using Emotion-Unlabeled Dataset with the Assistance of Cross-Domain Speech Emotion Recognition
[^9]: Controllable Speech Synthesis by Learning Discrete Phoneme-Level Prosodic Representations
[^10]: Learning Syllable-Level Discrete Prosodic Representation For Expressive Speech Generation
[^11]: Styler: Style Factor Modeling with Rapidity and Robustness via Speech Decomposition for Expressive and Controllable Neural Text To Speech