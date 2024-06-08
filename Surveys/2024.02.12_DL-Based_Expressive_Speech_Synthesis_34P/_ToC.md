# Deep Learning-Based Expressive Speech Synthesis: A Systematic Review of Approaches, Challenges, and Resources

- 标题: Deep Learning-Based Expressive Speech Synthesis: A Systematic Review of Approaches, Challenges, and Resources
- 作者: 
  - [Huda Barakat](../../Authors/Huda_Barakat.md)
  - [Oytun Turk](../../Authors/Oytun_Turk.md)
  - [Cenk Demiroglu](../../Authors/Cenk_Demiroglu.md)
- 发表:
  - EURASIP Journal on Audio, Speech, and Music Processing (2024)
  - DOI: https://doi.org/10.1186/s13636-024-00329-7


## Abstract·摘要

> Speech synthesis has made significant strides thanks to the transition from machine learning to deep learning models. 
> Contemporary **Text-To-Speech (TTS)** models possess the capability to generate speech of exceptionally high quality, closely mimicking human speech.
> Nevertheless, given the wide array of applications now employing TTS models, mere high-quality speech generation is no longer sufficient. 
> Present-day TTS models must also excel at producing expressive speech that can convey various speaking styles and emotions, akin to human speech.
> Consequently,researchers have concentrated their efforts on developing more efficient models for expressive speech synthesis in recent years.
> This paper presents a systematic review of the literature on expressive speech synthesis models published within the last 5 years, with a particular emphasis on approaches based on deep learning.
> We offer a comprehensive classification scheme for these models and provide concise descriptions of models falling into each category.
> Additionally, we summarize the principal challenges encountered in this research domain and outline the strategies employed to tackle these challenges as documented in the literature.
> In the Section 8, we pinpoint some research gaps in this field that necessitate further exploration.
> Our objective with this work is to give an all-encompassing overview of this hot research area to offer guidance to interested researchers and future endeavors in this field.

语音合成技术因从机器学习到深度学习模型的转变而取得了显著进步.
当代的文本转语音模型能够生成质量极高的, 几乎媲美人类的语音. 
但是鉴于现在有众多应用正在采用文本转语音模型, 仅仅生成高质量的语音已经不再足够.
现今的文本转语音模型还必须擅长生成富有表现力的语音, 能够传达类似于人类语音的多种说话风格和情感.
因此, 近年来研究人员集中精力于开发更高效的富有表现力的语音合成模型.

本文对过去五年内发表的关于富有表现力的语音合成模型的文献进行了系统性的回顾, 特别关注基于深度学习的方法.
我们为这些模型提供了一个全面的分类方案, 并对每个类别中的模型进行了简要的描述.
此外, 我们总结了在这一研究领域遇到的主要挑战, 并概述了文献中记录的应对这些挑战的策略.

在第 8 节中, 我们指出了该领域中需要进一步探索的一些研究空白.

我们进行这项工作的目的是为这一热门研究领域提供一个全面的概述, 为对这一领域感兴趣的研究人员和未来研究提供指导.

文章内容:
- [ ] [Sec.01](Sec.01.md)
- [ ] [Sec.02](Sec.02.md)
- [ ] [Sec.03](Sec.03.md)
- [ ] [Sec.04](Sec.04.md)
- [ ] [Sec.05](Sec.05.md)
- [x] [Sec.06](Sec.06.md)
- [x] [Sec.07](Sec.07.md)
- [ ] [Sec.08](Sec.08.md)

## Sec.09: Conclusions·结论

> This paper presents the findings of our systematic literature review on expressive speech synthesis over the past 5 years.
> The main contribution of this article is the development of a comprehensive taxonomy for DL-based approaches published in this field during that specific time frame.
> The approaches are classified into three primary categories based on the learning method, followed by models within each category.
> Further subcategories are identified at the lower levels of the taxonomy, considering the methods and structures applied to achieve expressiveness in synthesized speech.
> In addition to the ETTS approaches taxonomy, we provide descriptions of the main challenges in the ETTS field and proposed solutions from the literature.
> Furthermore, we support the reader with brief summaries of ETTS datasets, performance evaluation metrics, and some open-source implementations.
> The significance of our work lies in its potential to serve as an extensive overview of the research conducted in this area from different aspects, benefiting both experienced researchers and newcomers in this active research domain.

本文介绍了我们对过去 5 年内表现力语音合成领域的系统性文献综述的研究结果.
本文的主要贡献是开发了一个全面的分类法, 用于描述在该特定时间段内发表的基于深度学习（DL）的方法.
这些方法根据学习方式被分为三个主要类别, 每个类别下又包含具体的模型.
在分类法的较低层次, 进一步确定了子类别, 这些子类别考虑了实现合成语音表现力的方法和结构.
除了ETTS方法的分类法之外, 我们还描述了ETTS领域的主要挑战以及文献中提出的解决方案.
此外, 我们为读者提供了ETTS数据集, 性能评估指标以及一些开源实现的简要概述.
我们工作的意义在于, 它有可能作为一个全面的概述, 从不同方面展示该领域的研究, 对经验丰富的研究人员和活跃研究领域的新手都有益处.

> Some main directions for future work in this area involve collection of large expressive datasets in different languages, going from acted expressive style to realistic style.
> Further evaluation metrics are still needed in this area for assessing models’ performance such as evaluation of prosody controllability.
> Efficient metrics are also required for monitoring performance and guiding loss evaluation during the training process.
> These need to be lightweight and fast in order not to slow down training but still reliable.
> Another suggestion for future investigations is to take cultural differences in perception of expressions into account for multi-language, multi-speaker expressive TTS applications.
> Moreover, as speech is just one modality for expressions, multi-modal approaches that combine facial expressions, eye movements, body movements, gestures, non-verbal clues, etc., will be required to reach human-level expressiveness.
> Training several modalities together could be beneficial as the model can transfer useful information from one modality to another in a self-supervised fashion.

该领域未来工作的一些主要方向包括收集不同语言的大型表现力数据集, 从表演风格过渡到真实风格.
该领域仍然需要进一步的评估指标来评估模型的性能, 例如韵律可控性的评估.
还需要有效的指标来监控性能并在训练过程中指导损失评估.
这些指标需要轻量级且快速, 以免减慢训练速度, 但仍然可靠.
未来研究的另一个建议是考虑多语言, 多说话者表现力TTS应用中对表达感知的文化差异.
此外, 由于语音只是表达的一种方式, 因此需要多模态方法, 这些方法结合了面部表情, 眼神移动, 身体动作, 手势, 非言语线索等, 以达到人类水平的表现力.
同时训练多个模态可能是有益的, 因为模型可以在自我监督的方式下将一个模态的有用信息传递给另一个模态.