# Amphion: An Open-Source Audio, Music and Speech Generation Toolkit

<details>
<summary>基本信息</summary>

- 标题: Amphion: An Open-Source Audio, Music and Speech Generation Toolkit
- 作者: 
  - 01 [Xueyao Zhang](../../../Authors/Xueyao_Zhang_(张雪遥).md)
  - 02 [Liumeng Xue](../../../Authors/Liumeng_Xue_(薛浏蒙).md)
  - 03 [Yicheng Gu](../../../Authors/Yicheng_Gu.md)
  - 04 [Yuancheng Wang](../../../Authors/Yuancheng_Wang_(王远程).md)
  - 05 [Haorui He](../../../Authors/Haorui_He.md)
  - 06 [Chaoren Wang](../../../Authors/Chaoren_Wang.md)
  - 07 [Xi Chen](../../../Authors/Xi_Chen.md)
  - 08 [Zihao Fang](../../../Authors/Zihao_Fang.md)
  - 09 [Haopeng Chen](../../../Authors/Haopeng_Chen.md)
  - 10 [Junan Zhang](../../../Authors/Junan_Zhang.md)
  - 11 [Tze Ying Tang](../../../Authors/Tze_Ying_Tang.md)
  - 12 [Lexiao Zou](../../../Authors/Lexiao_Zou.md)
  - 13 [Mingxuan Wang](../../../Authors/Mingxuan_Wang.md)
  - 14 [Jun Han](../../../Authors/Jun_Han.md)
  - 15 [Kai Chen](../../../Authors/Kai_Chen.md)
  - 16 [Haizhou Li](../../../Authors/Haizhou_Li.md)
  - 17 [Zhizheng Wu](../../../Authors/Zhizheng_Wu_(武执政).md)
- 机构:
  - [香港中文大学 (深圳)](../../../Institutions/CHN-CUHK_香港中文大学.md)
  - Shanghai AI Lab
  - Shenzhen Research Institute of Big Data
- 时间:
  - 2023.12.15 ArXiv v1
  - 2024.02.22 ArXiv v2
- 链接:
  - [ArXiv](https://arxiv.org/abs/2312.09911)
  - [Github](https://github.com/open-mmlab/amphion)
- 标签:
  - [开源](../../../Tags/OpenSource.md)
- 页数: 13
- 引用: ?
- 被引: 3

</details>

## Abstract

<details>
<summary>原文</summary>

> ***Amphion*** is an open-source toolkit for Audio, Music, and Speech Generation, targeting to ease the way for junior researchers and engineers into these fields.
> It presents a unified framework that is inclusive of diverse generation tasks and models, with the added bonus of being easily extendable for new incorporation.
> The toolkit is designed with beginner-friendly workflows and pre-trained models, allowing both beginners and seasoned researchers to kick-start their projects with relative ease.
> Additionally, it provides interactive visualizations and demonstrations of classic models for educational purposes.
> The initial release of ***Amphion v0.1*** supports a range of tasks including Text to Speech (TTS), Text to Audio (TTA), and Singing Voice Conversion (SVC), supplemented by essential components like data preprocessing, state-of-the-art vocoders, and evaluation metrics.
> This paper presents a high-level overview of ***Amphion***.

</details>
<br>

***Amphion*** 是一个开源工具包, 旨在简化音频, 音乐和语音生成领域的新手研究人员和工程师的入门过程.
它提供了一个统一的框架, 涵盖了多种生成任务和模型, 并且易于扩展以整合新的内容.
该工具包设计了适合初学者的友好工作流程和预训练模型, 使得初学者和经验丰富的研究人员都能相对轻松地启动他们的项目.
此外, 它还提供了经典模型的交互式可视化和演示, 以供教育目的使用.
***Amphion v0.1*** 的初始版本支持一系列任务, 包括文本到语音（TTS）, 文本到音频（TTA）和歌唱声音转换（SVC）, 并辅以数据预处理, 最先进的声码器和评估指标等基本组件.
本文对 ***Amphion*** 进行了高层概述.

## 1.Introduction

<details>
<summary>原文</summary>

> The development of deep learning (LeCun et al., 2015) has greatly improved the performance of generative models (Hinton and Salakhutdinov, 2006; Kingma and Welling, 2014; Goodfellow et al., 2020; Ho et al., 2020; Vaswani et al., 2017).
> Leveraging these models has enabled researchers and practitioners to explore innovative possibilities, leading to notable progress and breakthroughs across various fields, including computer vision (Rombach et al., 2022) and natural language processing (Radford et al., 2019; Brown et al., 2020; Ren et al., 2019, 2020; Shen et al., 2018, 2024).
> The potential in tasks related to audio1, music, and speech generation has spurred the scientific community to actively publish new models and ideas (Liu et al., 2023a,b; Ren et al., 2019, 2020; Shen et al., 2018, 2024).

> There is an increasing presence of both official and community-driven open-source repositories that replicate these models.
> **However, the quality of these repositories varies significantly** (Stamelos et al., 2002; Borges and Valente, 2018; Aghajani et al., 2020; Wolf et al., 2020), and **they are often scattered, focusing on specific papers**.
> These scattered repositories introduce several obstacles to junior researchers or engineers who are new to the research area.
> First, attempts to replicate an algorithm using different implementations or configurations can result in inconsistent model functionality or performance (Paszke et al., 2019; Wolf et al., 2020; Lhoest et al., 2021).
> Second, while many repositories focus on the model architectures, they often neglect crucial steps such as detailed data pre-processing, feature extraction, model training, and systematic evaluation.
> This lack of systematic guidance poses substantial challenges for beginners (Georgiou et al., 2022), who may have limited technical expertise and experience in training large-scale models.
> **In summary, the scattered nature of these repositories hampers efforts towards reproducible research and fair comparisons among models or algorithms**.

> Motivated by that, we introduce ***Amphion***, an open-source platform dedicated to the north-star objective of “Any to Audio” (Figure 1).
> The initiative aims to facilitate the conversion of any input into an audible audio signal.
> Compared with the existing open-source toolkits as presented in Table 1, ***Amphion*** integrates audio, music, and speech generation into a unified framework that provides shared workflow across all models, including dataset processing, feature extraction, model training, inference, evaluation, and open-source pre-trained models.
> To aid junior engineers and researchers in understanding the internal mechanism of generative models, ***Amphion*** provides visualizations and interactive demonstrations of classic models.
> In a nutshell, ***Amphion***, **with its foundation in education, holds four unique strengths that set it apart from current open-source tools or repositories, in particular**,
> - **Unified Framework**: 
> ***Amphion*** provides a unified framework for audio, music, and speech generation and evaluation.
> It is designed to be adaptable, flexible, and scalable, supporting the integration of new models.
> - **Beginner-friendly End-to-End Workflow**: 
> ***Amphion*** offers a beginner-friendly end-to-end workflow with straightforward documentation and instructions.
> It establishes itself as an accessible one-stop research platform suitable for both novices and experienced researchers, including plenty of engineering know-how.
> - **Open Pre-trained Models**: 
> To promote reproducible research, ***Amphion*** commits to strict standards to the release of pre-trained models.
> In partner with industry, ***Amphion*** aims to make large-scale pre-trained models widely available for various applications.
> - **Visualization and Interactivity**: 
> ***Amphion*** provides visualization tools to interactively illustrate the internal processing mechanism of classic models.
> This provides an invaluable resource for educational purposes and for facilitating understandable research.

> The ***Amphion v0.1*** toolkit, now available under the MIT license,2 has supported a diverse array of generation tasks.
> This paper presents a high-level overview of the ***Amphion v0.1*** toolkit.

</details>
<br>

深度学习的发展（LeCun等人, 2015年）极大地提升了生成模型的性能（Hinton和Salakhutdinov, 2006年; Kingma和Welling, 2014年; Goodfellow等人, 2020年; Ho等人, 2020年; Vaswani等人, 2017年）.
利用这些模型, 研究人员和实践者得以探索创新的可能性, 导致包括计算机视觉（Rombach等人, 2022年）和自然语言处理（Radford等人, 2019年; Brown等人, 2020年; Ren等人, 2019年, 2020年; Shen等人, 2018年, 2024年）在内的多个领域取得了显著进展和突破.
与音频, 音乐和语音生成相关的任务潜力激发了科学界积极发布新模型和想法（Liu等人, 2023a, b; Ren等人, 2019年, 2020年; Shen等人, 2018年, 2024年）.

官方和社区驱动的开源存储库越来越多地复制了这些模型.
然而, 这些存储库的质量差异很大（Stamelos等人, 2002年; Borges和Valente, 2018年; Aghajani等人, 2020年; Wolf等人, 2020年）, 并且它们通常分散, 专注于特定的论文.
这些分散的存储库为刚接触该研究领域的初级研究人员或工程师带来了几个障碍.
首先, 尝试使用不同的实现或配置复制算法可能导致模型功能或性能不一致（Paszke等人, 2019年; Wolf等人, 2020年; Lhoest等人, 2021年）.
其次, 尽管许多存储库关注模型架构, 但它们往往忽视了数据预处理, 特征提取, 模型训练和系统评估等关键步骤.
这种缺乏系统指导对初学者构成了重大挑战（Georgiou等人, 2022年）, 他们可能在训练大规模模型方面技术专长和经验有限.
总之, 这些存储库的分散性阻碍了可重复研究和模型或算法之间公平比较的努力.

受此启发, 我们推出了 ***Amphion***, 这是一个致力于“任何到音频”（图1）北极星目标的开源平台.
该倡议旨在促进将任何输入转换为可听音频信号.

与表1中呈现的现有开源工具包相比, ***Amphion*** 将音频, 音乐和语音生成集成到一个统一的框架中, 为所有模型提供共享的工作流程, 包括数据集处理, 特征提取, 模型训练, 推理, 评估和开源预训练模型.
为了帮助初级工程师和研究人员理解生成模型的内部机制, ***Amphion*** 提供了经典模型的可视化和交互演示.

简而言之, ***Amphion***, 以其教育为基础, 拥有四个独特的优势, 使其与当前的开源工具或存储库区分开来, 特别是：
- **统一框架**：***Amphion*** 提供了一个统一的音频, 音乐和语音生成及评估框架.
它设计为可适应, 灵活和可扩展, 支持新模型的集成.
- **对初学者友好的端到端工作流程**：***Amphion*** 提供了一个对初学者友好的端到端工作流程, 具有直接的文档和说明.
它建立为一个适合新手和经验丰富的研究人员的一站式研究平台, 包括大量的工程知识.
- **开放的预训练模型**：为了促进可重复研究, ***Amphion*** 致力于严格的标准来发布预训练模型. 与行业合作伙伴一起, ***Amphion***旨在使大规模预训练模型广泛可用于各种应用.
- **可视化和交互性**：***Amphion*** 提供了可视化工具, 交互式地说明经典模型的内部处理机制. 这为教育目的和促进可理解的研究提供了宝贵的资源.

***Amphion v0.1*** 工具包现在以MIT许可证提供, 已经支持了多种生成任务.
本文提供了 ***Amphion v0.1*** 工具包的高级概述.


## 2.Unified Framework

<details>
<summary>原文</summary>

> The north-star goal of ***Amphion*** is to unify various audible waveform generation tasks.
> To make the framework adaptable and flexible to varied forms of tasks, from the perspective of input, we formulate audio generation tasks into three categories,
> 1. **Text to Waveform**: The input consists of discrete textual tokens, which strictly constrain the content of the output waveform.
> The representative tasks include Text to Speech (TTS) (Tan et al., 2021) and Singing Voice Synthesis (SVS)3 (Liu et al., 2022a).
> 2. **Descriptive Text to Waveform**: The input consists of discrete textual tokens, which generally guide the content or style of the output waveform.
> The representative tasks include Text to Audio (TTA) (Kreuk et al., 2022; Yang et al., 2023) and Text to Music (TTM) (Copet et al., 2023).
> 3. **Waveform to Waveform**: Both the input and output are continuous waveform signals.
> The representative tasks include Voice Conversion (VC) (Sisman et al., 2021), Singing Voic Conversion (SVC) (Liu et al., 2021; Zhang et al., 2023), Emotion Conversion (EC) (Zhou et al., 2022), Accent Conversion (AC) (Felps et al., 2009), and Speech Translation (ST) (Song et al., 2023a).

> This section will introduce the system architecture design and multiple classic generation tasks that are released in v0.1.

</details>
<br>

***Amphion*** 的北极星目标是统一各种可听波形生成任务.
为了使框架能够适应和灵活处理不同形式的任务, 从输入的角度, 我们将音频生成任务分为三类：

1. **文本到波形**：输入由离散的文本令牌组成, 严格限制输出波形的内容.
代表性任务包括文本到语音（TTS）（Tan等人, 2021年）和歌唱声音合成（SVS）（Liu等人, 2022a）.

2. **描述性文本到波形**：输入由离散的文本令牌组成, 通常指导输出波形的内容或风格.
代表性任务包括文本到音频（TTA）（Kreuk等人, 2022年; Yang等人, 2023年）和文本到音乐（TTM）（Copet等人, 2023年）.

3. **波形到波形**：输入和输出都是连续的波形信号.
代表性任务包括语音转换（VC）（Sisman等人, 2021年）, 歌唱声音转换（SVC）（Liu等人, 2021年; Zhang等人, 2023年）, 情感转换（EC）（Zhou等人, 2022年）, 口音转换（AC）（Felps等人, 2009年）和语音翻译（ST）（Song等人, 2023a）.


本节将介绍v0.1中发布的系统架构设计和多个经典生成任务.


### 2.1.System Architecture Design

<details>
<summary>原文</summary>

> To achieve the north-star objective, the ***Amphion*** architecture is designed with the following three principles,
> - **To establish a fair comparison platform for a multitude of models**.
> In particular, ***Amphion*** provides consistent data preprocessing, unified underlying training framework, and unified vocoders to create a fair experimental environment for comparison.
> - **To save practitioners from the tedious and repetitive task of “reinventing the wheel"**.
> ***Amphion*** abstracts various classic components, such as feature extraction, neural network modules, and batch sampling, into a unified infrastructure.
> - **To facilitate the unification of research across the entire audio domain**.
> In the context of today’s era of big data and large-scale pre-trained models, a unified data preprocessing, model extraction, and training framework can facilitate the development of the entire audio field.

> In particular, the system architecture design is presented in Figure 2. 
> From the bottom up,
> 1. We unify the data processing (Dataset, Feature Extractor, Sampler, and DataLoader), the optimization algorithms (Optimizer, Scheduler, and Trainer), and the common network modules (Module) as the infrastructure for all the audio generation tasks.
> 2. For each specific generation task, we unify its data/feature usage (TaskLoader), task framework (TaskFramework), and training pipeline (TaskTrainer).
> 3. Under each generation task, for every specific model, we specify its architecture (ModelArchitecture) and training pipeline (ModelTrainer).
> 4. Finally, we provide a textitrecipe of each model for users.
> We unify the recipe format across all models, ensuring it is as self-contained and novice-friendly as possible.
> Besides, we provide the visualizations to demonstrate the internal mechanisms of some typical models.
> On top of pre-trained models, we also offer interactive demos for users to explore.

</details>
<br>

为了实现北极星目标, ***Amphion*** 的架构设计遵循以下三个原则：

- **建立多个模型的公平比较平台**.
特别是, ***Amphion*** 提供一致的数据预处理, 统一的底层训练框架和统一的声码器, 为比较创造一个公平的实验环境.
- **节省实践者从繁琐和重复的“重新发明轮子”任务中**.
***Amphion*** 将各种经典组件, 如特征提取, 神经网络模块和批量采样, 抽象为一个统一的基础设施.
- **促进整个音频领域研究的统一**.
在大数据和大规模预训练模型的今天, 统一的数据预处理, 模型提取和训练框架可以促进整个音频领域的发展.


特别是, 系统架构设计如图2所示.
从下至上, 
1. 我们将数据处理（数据集, 特征提取器, 采样器和数据加载器）, 优化算法（优化器, 调度器和训练器）和常用网络模块（模块）统一为所有音频生成任务的基础设施.
2. 对于每个特定的生成任务, 我们统一其数据/特征使用（任务加载器）, 任务框架（任务框架）和训练流程（任务训练器）.
3. 在每个生成任务下, 对于每个特定模型, 我们指定其架构（模型架构）和训练流程（模型训练器）.
4. 最后, 我们为用户提供每个模型的文本it食谱.
我们统一所有模型的食谱格式, 确保它尽可能自包含且对初学者友好.

此外, 我们提供可视化来展示一些典型模型的内部机制.
在预训练模型之上, 我们还为用户提供交互式演示, 供其探索.

### 2.2.Classic Audio Generation Tasks

<details>
<summary>原文</summary>

> ***Amphion v0.1*** toolkit includes a representative from each of the three generation task categories (namely TTS, TTA, and SVC) for integration.
> This ensures that ***Amphion***’s framework can be conveniently adaptable to other audio generation tasks during future development.
> Notably, most audio generation models usually adopt a two-stage generation process, where they generate some intermediate acoustic features (e.g.
> Mel Spectrogram) in the first stage, and then generate the final audible waveform using a Vocoder in the second stage (Figure 3d).
> Motivated by that, ***Amphion v0.1*** also integrates a variety of vocoder models.

> Specifically, the pipelines of these classic audio tasks are designed as follows:
> - **Text to Speech**: TTS aims to convert written text into spoken speech (Tan et al., 2021).
> The traditional TTS considers only textual tokens as input (Wang et al., 2017; Ren et al., 2020; Kim et al., 2021).
> Recently, zero-shot TTS attracts more attentions from the research community.
> In addition to text, zero-shot TTS requires a reference audio as a prompt (Figure 3a).
> By utilizing in-context learning techniques, it can imitate the timbre and speaking style of the reference audio (Shen et al., 2024; Wang et al., 2023a).
> - **Text to Audio**: TTA aims to generate sounds that are semantically in line with descriptions (Kreuk et al., 2022; Yang et al., 2023).
> It usually requires a pre-trained text encoder like T5 (Kale and Rastogi, 2020) to capture the global information of the input descriptive text first, and then utilizes an acoustic model, such as diffusion model (Huang et al., 2023a; Liu et al., 2023a; Wang et al., 2023b), to synthesize the acoustic features (Figure 3b).
> - **Singing Voice Conversion**: SVC aims to transform the voice of a singing signal into the voice of a target singer while preserving the lyrics and melody (Huang et al., 2023b).
> To empower the reference speaker to sing the source audio, the main idea is to extract the speaker-specific representations from the reference, extract the speaker-agnostic representations (including semantic and prosody features) from the source, and then synthesize the converted features using acoustic models (Zhang et al., 2023) (Figure 3c).
>
> Current supported tasks, models and algorithms of ***Amphion v0.1*** are presented in Table 2.

</details>
<br>

***Amphion v0.1***工具包包括来自三个生成任务类别（即TTS, TTA和SVC）的代表性任务, 以确保***Amphion***的框架在未来发展中可以方便地适应其他音频生成任务.
值得注意的是, 大多数音频生成模型通常采用两阶段生成过程, 其中在第一阶段生成一些中间声学特征（例如梅尔频谱图）, 然后在第二阶段使用声码器生成最终的可听波形（图3d）.
受此启发, ***Amphion v0.1***还集成了各种声码器模型.

具体来说, 这些经典音频任务的流程设计如下：
- **文本到语音**：TTS旨在将书面文本转换为口语（Tan等人, 2021年）.
传统的TTS仅考虑文本令牌作为输入（Wang等人, 2017年; Ren等人, 2020年; Kim等人, 2021年）.
最近, 零样本TTS引起了研究界的更多关注.
除了文本, 零样本TTS需要一个参考音频作为提示（图3a）.
通过利用上下文学习技术, 它可以模仿参考音频的音色和说话风格（Shen等人, 2024年; Wang等人, 2023a）.

- **文本到音频**：TTA旨在生成与描述语义一致的声音（Kreuk等人, 2022年; Yang等人, 2023年）.
它通常需要一个预训练的文本编码器, 如T5（Kale和Rastogi, 2020年）, 首先捕捉输入描述文本的全局信息, 然后利用声学模型, 如扩散模型（Huang等人, 2023a; Liu等人, 2023a; Wang等人, 2023b）, 来合成声学特征（图3b）.

- **歌唱声音转换**：SVC旨在将歌唱信号的声音转换为目标歌手的声音, 同时保留歌词和旋律（Huang等人, 2023b）.
为了使参考说话者能够唱出源音频, 主要思想是从参考中提取说话者特定的表示, 从源中提取说话者无关的表示（包括语义和韵律特征）, 然后使用声学模型（Zhang等人, 2023年）合成转换后的特征（图3c）.

当前支持的任务, 模型和算法***Amphion v0.1***在表2中呈现.


## 3.Beginner-Friendly End-to-End Workflow

<details>
<summary>原文</summary>

> ***Amphion*** provides a beginner-friendly, end-to-end workflow structured into five stages: data preprocessing, feature extraction, model training, model inference and evaluation, and visualization.
> Notably, ***Amphion*** workflow incorporates plenty of engineering know-how for training models on largescale datasets such as Libri-light (Kahn et al., 2020) and MLS (Pratap et al., 2020).
> This enables novice researchers to replicate advanced, proprietary systems.
> Figure 4 presents the primary workflow of ***Amphion***, outlined briefly as follows,
> - **Data Preprocessing**: ***Amphion*** offers preimplemented, dataset-specific preprocessing scripts for a wide range of academic datasets in audio, music, and speech generation research (Table 2), enabling a seamless transition from dataset-specific structures to ***Amphion***’s unified data format.
> - **Feature Extraction**: After preprocessing, ***Amphion*** allows for both offline and on-the-fly feature extraction.
> Offline extraction, which occurs before model training, saves extracted features locally for later use in training.
> However, for large-scale datasets, it is infeasible to perform the lengthy offline extraction.
> Motivated by that, ***Amphion*** offers an on-the-fly extraction that processes raw waveforms in batches and utilizes GPU as much as possible to accelerate feature extraction during model training.
> - **Model Training**: ***Amphion*** enhances training efficiency by supporting multi-GPU training with a Distributed Data Parallel (DDP) trainer (Gugger et al., 2022), dynamic batch sizes, and other userfriendly training infrastructures, simplifying the replication of state-of-the-art models.
> - **Model Inference and Evaluation**: ***Amphion*** provides various pre-trained model checkpoints of top-performing vocoders, one-click evaluation scripts, and interactive demonstrations, enabling high-quality generation and easy experimentation.
> The supported evaluation metrics can be seen in Table 2.
> - **Visualization**: Lastly, ***Amphion*** introduces a unique interactive visual analysis of some classical models for educational purposes, helping newcomers understand their inner workings.

</details>
<br>

***Amphion***提供了一个对初学者友好的端到端工作流程, 分为五个阶段：数据预处理, 特征提取, 模型训练, 模型推理和评估, 以及可视化.
值得注意的是, ***Amphion***工作流程包含了大量关于在大规模数据集（如Libri-light（Kahn等人, 2020年）和MLS（Pratap等人, 2020年））上训练模型的工程知识.
这使得初学者研究人员能够复制先进的专有系统.
图4展示了***Amphion***的主要工作流程, 简要概述如下：

- **数据预处理**：***Amphion***提供了针对音频, 音乐和语音生成研究中广泛使用的学术数据集的预实现, 数据集特定的预处理脚本（表2）, 实现了从数据集特定结构到***Amphion***统一数据格式的无缝过渡.
- **特征提取**：预处理后, ***Amphion***支持离线和实时特征提取.
离线提取在模型训练之前进行, 将提取的特征本地保存以供后续训练使用.
然而, 对于大规模数据集, 进行耗时的离线提取是不可行的.
受此启发, ***Amphion***提供了实时提取, 该提取处理原始波形批次, 并尽可能利用GPU在模型训练期间加速特征提取.
- **模型训练**：***Amphion***通过支持多GPU训练与分布式数据并行（DDP）训练器（Gugger等人, 2022年）, 动态批量大小和其他用户友好的训练基础设施, 提高了训练效率, 简化了复制最先进模型的过程.
- **模型推理和评估**：***Amphion***提供了各种顶级性能声码器的预训练模型检查点, 一键评估脚本和交互式演示, 实现了高质量生成和易于实验.
支持的评估指标可以在表2中看到.
- **可视化**：最后, ***Amphion***引入了一种独特的交互式可视化分析, 用于教育目的, 帮助新来者理解经典模型的内部工作原理.

## 4.Open Pre-trained Models

<details>
<summary>原文</summary>

> Pre-trained checkpoints can empower practitioners to reproduce experimental results and build applications.
> Public pre-trained models could facilitate innovation and learning for practitioners.
> For that purpose, ***Amphion*** has released a variety of models pertaining to Audio, Music, and Speech Generation, such as TTS, TTA, SVC, and Vocoder.
> Our standards for releasing pre-trained models are designed to ensure transparency, reproducibility, and usability for the community.
> Each model is released with a model card that includes not only the checkpoint but also detailed descriptions of the following criteria:
> - **Model Metadata**: 
> Detail the model architecture and the number of parameters.
> - **Training Datasets**: 
> List all the training corpus and their sources.
> - **Training Configuration**: 
> Detail the training hyberparameters (like batch size, learning rate, and number of training steps) and the computational platform.
> - **Evaluation Results**: 
> Display the evaluation results and the performance comparison to other typical baselines.
> - **Usage Instructions**: 
> Instruct how to inference and fine-tune based on the pre-trained model.
> - **Interactive Demo**: 
> Provide an online interactive demo for users to explore.
> - **License**: 
> Clear the licensing details including how the model can be utilized, shared, and modified.
> - **Ethical Considerations**: 
> Address ethical considerations related to the model’s application, focusing on privacy, consent, and bias, to encourage responsible usage.

> As of now, the released pre-trained models of ***Amphion v0.1*** are listed in Table 3.
> We are partnering with industry for larger models on larger datasets.

</details>
<br>

预训练的检查点可以帮助实践者重现实验结果并构建应用程序.
公开的预训练模型可以促进实践者的创新和学习.
为此, ***Amphion***已经发布了与音频, 音乐和语音生成相关的各种模型, 如TTS, TTA, SVC和声码器.
我们发布预训练模型的标准旨在确保对社区的透明度, 可重复性和可用性.
每个模型都附带一个模型卡, 其中不仅包括检查点, 还包括以下标准的详细描述：

- **模型元数据**：
详细说明模型架构和参数数量.

- **训练数据集**：
列出所有训练语料库及其来源.

- **训练配置**：
详细说明训练超参数（如批量大小, 学习率和训练步骤数）和计算平台.

- **评估结果**：
显示评估结果和与其他典型基线的性能比较.

- **使用说明**：
指导如何基于预训练模型进行推理和微调.

- **交互式演示**：
为用户提供在线交互式演示以供探索.

- **许可证**：
明确许可证详情, 包括模型如何被使用, 共享和修改.

- **伦理考虑**：
解决与模型应用相关的伦理考虑, 重点关注隐私, 同意和偏见, 以鼓励负责任的使用.


截至目前, ***Amphion v0.1*** 发布的预训练模型列于表3中.
我们正在与行业合作, 针对更大规模的数据集开发更大的模型.


## 5.Visualization and Interactivity

<details>
<summary>原文</summary>

> Aiming to serve as a platform for exploring and understanding generative models with educational purposes, ***Amphion*** provides intuitive and engaging visualization tools for classic models or algorithms. 

> Visualizations act as a bridge between abstract theoretical concepts and their practical applications, enabling users to see the “how" and “why" behind model outputs (Hohman et al., 2018).
> By providing a visual representation of model architectures, data flow, and parameter effects, ***Amphion*** demystifies the inner workings of generative models, making the technology more approachable. ***Amphion v0.1*** has developed the visualization system for diffusion model under SVC task, namely SingVisio4 (Xue et al., 2024).
> The interface of the interactive visualization is shown in the appendix.
> This tool offers a dynamic and interactive visualization of the generation process in diffusion models, illustrating the gradual denoising of the noisy spectrum and its evolution into a clear spectrum that embodies the target singer’s timbre.

> ***Amphion*** also utilizes online platforms like Gradio5 to support a series of interactive demos.
> With these interactive demos, users can input text or select predefined prompts, and the underlying models will generate corresponding audio outputs instantly.
> The interactive interface provides users with immediate feedback and allows them to modify input parameters, select options, or make choices using drop-down menus, check-boxes, or sliders, depending on the specific task and model being utilized.
> We provide an interactive demo example in the appendix.

> ***Amphion***’s visualization capabilities stand out as one of its most unique features, designed to demystify complex generative models for those at the beginning of their research journey.
> By merging the capabilities of Gradio with the visualization features of ***Amphion***, the toolkit provides a seamless and interactive experience for users to explore and comprehend the inner workings of generative models.
> This combination of visualizations and interactivity empowers researchers, engineers, and enthusiasts to experiment, iterate, and gain insights into the complex world of audio generation.

</details>
<br>

旨在作为探索和理解具有教育目的的生成模型的平台, ***Amphion*** 为经典模型或算法提供了直观且引人入胜的可视化工具.
可视化作为抽象理论概念与其应用之间的桥梁, 使用户能够看到模型输出的“如何”和“为什么”（Hohman等人, 2018年）.
通过提供模型架构, 数据流和参数影响的视觉表示, ***Amphion*** 揭示了生成模型的内部工作原理, 使技术更易于接近.
***Amphion v0.1*** 已经为SVC任务下的扩散模型开发了可视化系统, 即SingVisio4（Xue等人, 2024年）.
交互式可视化的界面显示在附录中.
该工具提供了一个动态和交互式的扩散模型生成过程可视化, 展示了噪声频谱逐渐去噪并演变成体现目标歌手音色的清晰频谱的过程.


***Amphion*** 还利用Gradio5等在线平台支持一系列交互式演示.
通过这些交互式演示, 用户可以输入文本或选择预定义的提示, 底层模型将立即生成相应的音频输出.
交互式界面为用户提供即时反馈, 并允许他们修改输入参数, 选择选项, 或使用下拉菜单, 复选框或滑块进行选择, 具体取决于正在使用的特定任务和模型.
我们在附录中提供了一个交互式演示示例.


***Amphion*** 的可视化能力是其最独特的功能之一, 旨在为研究旅程初期的用户揭示复杂的生成模型.
通过将 Gradio 的功能与 ***Amphion*** 的可视化特性相结合, 该工具包为用户提供了一个无缝且交互式的体验, 以探索和理解生成模型的内部工作原理.
这种可视化和交互性的结合使研究人员, 工程师和爱好者能够实验, 迭代并深入了解音频生成的复杂世界.


## 6.Experiments

<details>
<summary>原文</summary>

> We compare the performance of ***Amphion v0.1*** on three tasks, namely TTS, TTA and SVC, and the vocoder with public open repositories.

> We use both objective and subjective evaluations to evaluate different tasks.
> The objective evaluation metrics are presented in Table 2.
> Regarding the Mean Opinion Score (MOS) and the Similarity Mean Opinion Score (SMOS) tests, listeners are required to grade ranging from 1 (“Bad”) to 5 (“Excellent”) and from 1 ("Different speaker, sure") to 5 ("Same speaker, sure") individually with an interval of 1 to evaluate the generated audio’s overall quality and the similarity to the reference speaker; regarding the Naturalness and Similarity tests, listeners are required to grade ranging from 1 (“Bad”) to 5 (“Excellent”) and from 1 ("Different speaker, sure") to 4 ("Same speaker, sure") individually with an interval of 1 to evaluate the conversed singing utterance’s naturalness and the similarity to the reference singer.
> All the subjective evaluation score listed are within a 95% Confidence Interval.

</details>
<br>

我们比较了 ***Amphion v0.1*** 在三个任务（即TTS, TTA和SVC）以及声码器与公共开源存储库的性能.
我们使用客观和主观评估来评估不同的任务.
客观评估指标在表2中呈现.
关于平均意见得分（MOS）和相似性平均意见得分（SMOS）测试, 听众需要从1（“差”）到5（“优秀”）和从1（“不同说话者, 确定”）到5（“相同说话者, 确定”）分别以1的间隔对生成的音频的整体质量和与参考说话者的相似性进行评分; 关于自然度和相似性测试, 听众需要从1（“差”）到5（“优秀”）和从1（“不同说话者, 确定”）到4（“相同说话者, 确定”）分别以1的间隔对转换后的歌唱话语的自然度和与参考歌手的相似性进行评分.
所有列出的主观评估得分都在95%的置信区间内.


### 6.1.Text to Speech

#### 6.1.1.Results of Classic TTS

<details>
<summary>原文</summary>

> We evaluate the performance of ***Amphion v0.1*** TTS following the evaluation pipeline in ***Amphion v0.1***.
> Specifically, we construct testing sets with 100 text transcriptions and then generate the corresponding speech using ***Amphion v0.1*** VITS and other four popular open-source speech synthesis toolkits, including Coqui TTS6 (VITS is used in this system), SpeechBrain7 (FastSpeech 2 is used because FastSpeech 2 is the most advanced model in this system), TorToiSe8 (an autoregressive and diffusion-based model), and ESPnet9 (VITS is used in this system).
> The evaluation results are shown in Table 4.
> The evaluation results show that the performance of ***Amphion v0.1*** VITS is comparable to existing open-source systems.

</details>
<br>

我们评估了 ***Amphion v0.1*** TTS的性能, 遵循了 ***Amphion v0.1*** 中的评估流程.
具体来说, 我们构建了包含100个文本转录的测试集, 然后使用 ***Amphion v0.1*** VITS和其他四个流行的开源语音合成工具包生成相应的语音, 包括Coqui TTS6（在此系统中使用VITS）, SpeechBrain7（使用FastSpeech 2, 因为FastSpeech 2是此系统中最先进的模型）, TorToiSe8（一种自回归和基于扩散的模型）和ESPnet9（在此系统中使用VITS）.
评估结果显示在表4中.
评估结果表明, ***Amphion v0.1*** VITS的性能与现有的开源系统相当.


#### 6.1.2.Results of Zero-shot TTS

<details>
<summary>原文</summary>

> We also evaluate the speech quality and speaker similarity of ***Amphion v0.1*** zero-shot TTS systems alongside other open-source zero-shot TTS models.
> We use the samples from NaturalSpeech2 (Shen et al., 2024) demo page10 as our test set.
> We compare our ***Amphion v0.1*** NaturalSpeech2 with YourTTS (Casanova et al., 2022), and VALLE (Wang et al., 2023a) from the reproduced VALLE repository11.
> The results are shown in Table 5.
> The evaluation results show that ***Amphion v0.1*** NatrualSpeech2 is significantly better than YourTTS in terms of speech quality and speaker similarity.
> Compared with the open-source reproduced VALLE, we have comparable speaker similarity, and the difference in speech quality may come from the difference in training data.

</details>
<br>

我们还评估了 ***Amphion v0.1*** 零样本TTS系统的语音质量和说话者相似性, 以及与其他开源零样本TTS模型的比较.
我们使用NaturalSpeech2（Shen等人, 2024年）演示页面10的样本作为我们的测试集.
我们将我们的 ***Amphion v0.1*** NaturalSpeech2与YourTTS（Casanova等人, 2022年）和VALLE（Wang等人, 2023a）从重现的VALLE存储库11进行了比较.
结果显示在表5中.
评估结果表明, ***Amphion v0.1*** NatrualSpeech2在语音质量和说话者相似性方面明显优于YourTTS.
与开源重现的 VALL-E 相比, 我们在说话者相似性方面具有可比性, 而语音质量的差异可能来自训练数据的差异.


### 6.2.Text to Audio

<details>
<summary>原文</summary>

> We use inception score (IS), Fréchet Distance (FD), and Kullback–Leibler Divergence (KL) to evaluate our text-to-audio generation model.
> FD measures the fidelity between the generated samples and target samples.
> IS measures the quality and diversity of the generated samples.
> KL measures the correlation between output samples and the target samples.
> FD, IS, and KL are based on the state-of-the-art audio classification model PANNs (Kong et al., 2020).
> We use the test set of AudioCaps as our test set.
> The evaluation results of ***Amphion v0.1*** TTA are shown in Table 6.
> The results demonstrate that our ***Amphion v0.1*** AudioLDM system achieves similar results to state-of-the-art models.

</details>
<br>

我们使用初始得分（IS）, 弗雷歇距离（FD）和库尔贝克-莱布勒散度（KL）来评估我们的文本到音频生成模型.
FD衡量生成样本与目标样本之间的保真度.
IS衡量生成样本的质量和多样性.
KL衡量输出样本与目标样本之间的相关性.
FD, IS和KL基于最先进的音频分类模型PANNs（Kong等人, 2020年）.
我们使用AudioCaps的测试集作为我们的测试集.
***Amphion v0.1*** TTA 的评估结果显示在表6中.
结果表明, 我们的 ***Amphion v0.1*** AudioLDM系统与最先进的模型取得了相似的结果.


### 6.3.Singing Voice Conversion

<details>
<summary>原文</summary>

> To evaluate the effectiveness of the SVC models of ***Amphion v0.1***, we adopt the in-domain evaluation task of the Singing Voice Conversion Challenge (SVCC) 202312 (Huang et al., 2023b).
> Specifically, there are 48 evaluated singing utterances, with 24 from a male and 24 from a female.
> Our task is to convert them into two target singers (one male and one female).
> For the training data, we utilize five datasets: Opencpop (Wang et al., 2022), SVCC training data (Huang et al., 2023b), VCTK (Yamagishi et al., 2019), OpenSinger (Huang et al., 2021), and M4Singer (Zhang et al., 2022).
> There are 83.1 hours of speech and 87.2 hours of singing data in total.
> We compare the DiffWaveNetSVC (Zhang et al., 2023) of ***Amphion v0.1*** with SoftVC VITS (SVC-Develop-Team, 2023), which is one of the most popular open-source project in the SVC area.
> The MOS evaluation results for ***Amphion v0.1*** SVC are shown in Table 7.
> We can see the DiffWaveNetSVC model of ***Amphion v0.1*** owns better performance in both naturalness and speaker similarity.

</details>
<br>

为了评估 ***Amphion v0.1*** 的SVC模型的有效性, 我们采用了2023年歌唱声音转换挑战（SVCC）12（Huang等人, 2023b）的域内评估任务.
具体来说, 有48个评估的歌唱话语, 其中24个来自男性, 24个来自女性.
我们的任务是将它们转换为两个目标歌手（一男一女）.
对于训练数据, 我们使用了五个数据集：Opencpop（Wang等人, 2022年）, SVCC训练数据（Huang等人, 2023b）, VCTK（Yamagishi等人, 2019年）, OpenSinger（Huang等人, 2021年）和M4Singer（Zhang等人, 2022年）.
总共有83.1小时的语音和87.2小时的歌唱数据.
我们将 ***Amphion v0.1*** 的 DiffWaveNetSVC（Zhang等人, 2023年）与 SoftVC VITS（SVC-Develop-Team, 2023年）进行了比较, 后者是SVC领域最受欢迎的开源项目之一.
***Amphion v0.1*** SVC 的 MOS 评估结果显示在表7中.
我们可以看到, ***Amphion v0.1*** 的DiffWaveNetSVC模型在自然度和说话者相似性方面具有更好的性能.


### 6.4.Vocoder

<details>
<summary>原文</summary>

> We compare the ***Amphion v0.1*** Vocoder to the two widely used open-source HiFi-GAN checkpoints.
> One is the UNIVERSAL_V1 from the official HiFi-GAN repository13; the other is the libritts_hifigan.v1 used by the ESPnet14.
> All of the checkpoints are trained on around 600 hours of speech data.
> The whole evaluation set and the test set of LibriTTS are used for evaluation, with a total of 20,306 utterances.
> Objective evaluation is conducted with M-STFT, PESQ, F0RMSE, and FPC.
> The results are illustrated in Table 8.
> With the assistance of additional guidance from Time-Frequency Representation-based Discriminators (Gu et al., 2024), the ***Amphion v0.1*** HiFi-GAN achieves superior performance in spectrogram reconstruction and F0 modeling.

</details>
<br>

我们将 ***Amphion v0.1*** 声码器与两个广泛使用的开源HiFi-GAN检查点进行了比较.
一个是官方HiFi-GAN存储库13中的UNIVERSAL_V1; 另一个是ESPnet14使用的libritts_hifigan.v1.
所有检查点都是在约600小时的语音数据上训练的.
整个评估集和LibriTTS的测试集用于评估, 总共有20,306个话语.
客观评估是通过M-STFT, PESQ, F0RMSE和FPC进行的.
结果显示在表8中.
在基于时频表示的判别器（Gu等人, 2024年）的额外指导下, ***Amphion v0.1*** HiFi-GAN在频谱图重建和F0建模方面取得了卓越的性能.


## 7.Conclusions

<details>
<summary>原文</summary>

> This paper presented ***Amphion***, an open-source toolkit dedicated to audio, music, and speech generation. ***Amphion***’s primary objective is to facilitate reproducible research and serve as a stepping stone for junior researchers and engineers entering the field of audio, music, and speech generation.
> Besides focusing on specific generation tasks, ***Amphion*** provides visualizations of classic models or architectures to empower users with a deeper understanding of the underlying processes.

> Large-scale datasets and large-scale models have proven to achieve state-of-the-art performance in In the future, ***Amphion*** plans to various tasks. release a few large-scale dataset in the area of audio, music and speech generation area.
> Also, we plan to partner with industry for large-scale and production-oriented pre-trained models.
> Since the release of ***Amphion*** in Dec 2023, ***Amphion*** has received more than 3, 500 stars on GitHub and received a significant number of requests and feedback.
> We will actively work on ***Amphion*** to serve junior researchers and engineers.

</details>
<br>

本文介绍了 ***Amphion***, 一个致力于音频, 音乐和语音生成的开源工具包.
***Amphion*** 的主要目标是促进可重复的研究, 并为进入音频, 音乐和语音生成领域的初级研究人员和工程师提供一个跳板.
除了专注于特定的生成任务外,  ***Amphion*** 还提供了经典模型或架构的可视化, 以使用户更深入地理解底层过程.


大规模数据集和大规模模型已被证明在各种任务中实现了最先进的表现.
未来,  ***Amphion*** 计划在音频, 音乐和语音生成领域发布一些大规模数据集.
我们还计划与行业合作, 开发大规模和面向生产的预训练模型.
自2023年12月发布 ***Amphion*** 以来,  ***Amphion*** 在GitHub上获得了超过3,500颗星, 并收到了大量的请求和反馈.
我们将积极致力于 ***Amphion*** , 为初级研究人员和工程师服务.


## Limitations

<details>
<summary>原文</summary>

> In its initial release, ***Amphion*** has certain limitations that call for further exploration and enhancement.
> While ***Amphion v0.1*** has successfully incorporated Text to Speech, Text to Audio, Singing Voice Conversion, and a variety of Vocoders, there remains a multitude of audio generation tasks yet to be integrated.
> Moreover, the audio processing field is progressing at a rapid pace, with new models and techniques being developed on a continuous basis.
> Ensuring that ***Amphion*** stays up-to-date and seamlessly integrates these innovations will be an ongoing challenge that needs to be addressed in future iterations.

</details>
<br>

在其初始发布中, ***Amphion*** 存在某些限制, 需要进一步探索和增强.
虽然 ***Amphion v0.1*** 已成功整合了文本到语音, 文本到音频, 歌唱声音转换和各种声码器, 但仍有大量音频生成任务尚未集成.
此外, 音频处理领域正在快速发展, 新的模型和技术不断被开发.
确保 ***Amphion*** 保持最新并无缝集成这些创新将是未来迭代中需要解决的持续挑战.


## Ethical Considerations

<details>
<summary>原文</summary>

> In providing an open-source toolkit for audio, music, and speech generation, we strive to balance innovation with ethical responsibility.
> We acknowledge the ethical implications associated with our model’s ability to generate a wide array of audio signals.
> Such capabilities, while powerful, carry potential risks of misuse, such as the production of misinformation, deepfake audio, or harmful content.
> We fervently advocate for the responsible use of our code and pre-trained models, emphasizing the imperative of adhering to regulatory standards.
> Our commitment to ethical considerations is not merely an afterthought, but rather a guiding principle in our pursuit of advancing the field of audio signal generation.

</details>
<br>

在提供用于音频, 音乐和语音生成的开源工具包时, 我们努力在创新与道德责任之间取得平衡.
我们承认我们的模型能够生成广泛音频信号的道德影响.
虽然这些能力很强大, 但它们也带来了潜在的滥用风险, 例如产生虚假信息, 深度伪造音频或有害内容.
我们强烈主张负责任地使用我们的代码和预训练模型, 强调遵守监管标准的必要性.
我们对道德考虑的承诺不仅仅是一个事后考虑, 而是我们在推进音频信号生成领域追求中的指导原则.
