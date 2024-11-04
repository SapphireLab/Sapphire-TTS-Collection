# 举世无双语音合成 VITS 发展历程

- 原文: <https://zhuanlan.zhihu.com/p/474601997>
- 注意: 基于原文整理如下

VITS 经典项目
- FAIRSeq 版本 [MMS](https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/examples/mms/README.md) ![Star](https://img.shields.io/github/stars/facebookresearch/fairseq)
- WeNet 版本 [WeTTS](https://github.com/wenet-e2e/wetts) ![Star](https://img.shields.io/github/stars/wenet-e2e/wetts?style=social)
- 树莓派版本 [Piper](https://github.com/rhasspy/piper) ![Star](https://img.shields.io/github/stars/rhasspy/piper?style=social)
- C++ 版本 [SummerTTS](https://github.com/huakunyang/SummerTTS) ![Star](https://img.shields.io/github/stars/huakunyang/SummerTTS?style=social)

VITS 流式推理
- [VITS Chinese](https://github.com/PlayVoice/vits_chinese) ![Star](https://img.shields.io/github/stars/PlayVoice/vits_chinese?style=social)

VITS 优化项目
- [MB-iSTFT-VITS (ICASSP2023)](https://github.com/MasayaKawamura/MB-iSTFT-VITS) ![Star](https://img.shields.io/github/stars/MasayaKawamura/MB-iSTFT-VITS?style=social)

VITS 歌声合成
- [VISinger (ICASSP2023)](https://github.com/zhangyongmao/VISinger2) ![Star](https://img.shields.io/github/stars/zhangyongmao/VISinger2?style=social)

VITS 变声系列
- [Update at 2023.10.03] [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc) ![Star](https://img.shields.io/github/stars/svc-develop-team/so-vits-svc) [Archive]
- [Update at 2024.09.05] [Retrieval-based-Voice-Conversion-WebUI (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) ![Star](https://img.shields.io/github/stars/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Update at 2024.10.08] [Voice Changer](https://github.com/w-okada/voice-changer) ![Star](https://img.shields.io/github/stars/w-okada/voice-changer) (日语)
- [Update at 2024.10.10] [So-VITS-SVC-Fork](https://github.com/voicepaw/so-vits-svc-fork) ![Star](https://img.shields.io/github/stars/voicepaw/so-vits-svc-fork) [Archive]
- [Update at 2024.04.23] [Whisper-VITS-SVC](https://github.com/PlayVoice/whisper-vits-svc) ![Star](https://img.shields.io/github/stars/PlayVoice/whisper-vits-svc)
- [Update at 2022.11.13] [Stella VC](https://github.com/sophiefy/Sovits) ![Star](https://img.shields.io/github/stars/sophiefy/Sovits) [Closed]

VITS 语音克隆
- [MyShell.AI/OpenVoice](https://github.com/myshell-ai/OpenVoice) ![Star](https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social)
- [MyShell.AI/MeloTTS](https://github.com/myshell-ai/MeloTTS) ![Star](https://img.shields.io/github/stars/myshell-ai/MeloTTS)
- [VITS Fast Fine-Tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning) ![Star](https://img.shields.io/github/stars/Plachtaa/VITS-fast-fine-tuning?style=social)
- [VITS Simple API](https://github.com/Artrajz/vits-simple-api) ![Star](https://img.shields.io/github/stars/Artrajz/vits-simple-api?style=social)
- [FishAudio/BERT-VITS2](https://github.com/fishaudio/Bert-VITS2) ![Star](https://img.shields.io/github/stars/fishaudio/Bert-VITS2?style=social)

VITS 代码解读
- [VITS 论文与代码详解 by 鲁东大学于泓](../../Courses/Bilibili-于泓/2021.07.11_智能语音处理.md)
- [So-VITS-SVC 5.0 代码详解 by 鲁东大学于泓](../../Courses/Bilibili-于泓/2021.07.11_智能语音处理.md)

## 时间线

| 事件 | 简称 | 标题 | 机构 | 发表 | 链接 | 目的 |
|:-:|:-:|---|---|---|---|---|
| 2021.06.11 | VITS | Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | 韩国科学院 | ICML | [Github](https://github.com/jaywalnut310/vits)<br>![Star](https://img.shields.io/github/stars/jaywalnut310/vits) | |
| 2021.06.21 | Glow-WaveGAN | Glow-WaveGAN：Learning Speech Representations from GAN-based Variational Auto-Encoder For High Fidelity Flow-based Speech Synthesis | 西北工业大学，腾讯 AI 实验室 | InterSpeech | |
| 2021.10.15 | ESPnet2-TTS | ESPnet2-TTS Extending the Edge of TTS Research | ESPnet, CMU, 东京大学等| | [Github](https://github.com/espnet/espnet/tree/master/espnet2/gan_tts/vits) | 对先进的语音合成系统进行评估，尤其是 VITS；ESPnet 提供的 152 个预训练模型（ASR+TTS）中有 48 为 VITS 语音合成模型。|
| 2021.10.17 | VISinger | | 西北工业大学、网易伏羲 AI 实验室 | | | 基于 VITS 实现的歌声合成系统 |
| 2021.12.04 | YourTTS | YourTTS：Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone | 开源机构 Coqui-AI | | [Github](https://edresson.github.io/YourTTS/) | 基于 VITS 实现跨语言语音合成和声音转换 |


## TODO (Merge Later)

2021 年 12 月 23 日 语音合成专题学术论坛：

机构：CCF 语音对话与听觉专委会

在会议中，微软亚洲研究院主管研究员谭旭博士，透露基于 VITS 实现的构建录音水平的文本到语音合成系统：DelightfulTTS 2 (Blizzard Challenge 2021/Ongoing)，论文还未公开



2022年3月30日 VoiceMe：TTS中的个性化语音生成

论文：VoiceMe: Personalized voice generation in TTS

代码：https://github.com/polvanrijn/VoiceMe

机构：University of Cambridge etc

目的：使用来自最先进的说话人验证模型（SpeakerNet）的说话人嵌入来调节TTS模型。展示了用户可以创建与人脸、艺术肖像和卡通照片非常匹配的声音；使用wav2lip合成口型。



2022年3月30日 Nix-TTS：VITS模型的加速

论文：Nix-TTS: An Incredibly Lightweight End-to-End Text-to-Speech Model via Non End-to-End Distillation

代码：https://github.com/choiHkk/nix-tts

演示：https://github.com/rendchevi/nix-tts

机构：Amazon (UK) etc

目的：使用VITS作为教师模型，使用Nix-TTS作为学生模型，大约得到3倍的加速



2022年5月10日 NaturalSpeech：具有人类水平质量的端到端文本到语音合成

论文：NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality

机构：Microsoft Research Asia & Microsoft Azure Speech Xu Tan

目的：通过几个关键设计来增强从文本到后文本的能力，降低从语音到后文本的复杂性，包括音素预训练、可微时长建模、双向前/后建模以及VAE中的记忆机制。



2022年6月2日 AdaVITS: Tiny VITS

论文：AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation

机构：西工大&&腾讯

目的：用于低计算资源的说话人自适应；提出了一种基于iSTFT的波形构造解码器，以取代原VITS中资源消耗较大的基于上采样的解码器；引入了NanoFlow来共享流块之间的密度估计；将语音后验概率（PPG）用作语言特征；



2022年6月27日

论文：End-to-End Text-to-Speech Based on Latent Representation of Speaking Styles Using Spontaneous Dialogue

目的：语音上下文对话风格；两个阶段进行训练：第一阶段，训练变分自动编码器（VAE）-VITS，从语音中提取潜在说话风格表示的风格编码器与TTS联合训练。第二阶段，训练一个风格预测因子来预测从对话历史中综合出来的说话风格。以适合对话上下文的风格合成语音。



2022年6月27日 Sane-TTS

论文：SANE-TTS: Stable And Natural End-to-End Multilingual Text-to-Speech

机构：MINDsLab Inc，KAIST

目的：跨语言克隆；引入了说话人正则化丢失，在跨语言合成过程中提高了语音的自然度，并引入了域对抗训练。在持续时间预测器中用零向量代替说话人嵌入，稳定了跨语言推理。



2022年7月6日 Glow-WaveGAN 2

论文：Glow-WaveGAN 2: High-quality Zero-shot Text-to-speech Synthesis and Any-to-any Voice Conversion

演示：https://leiyi420.github.io/glow-wavegan2/

机构：腾讯

目的：零资源语音克隆，任意到任意的变声；使用通用预训练大模型WaveGAN，替换VAE和HIFIGAN；



2022年7月14日 CLONE

论文：Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech

演示：https://xcmyz.github.io/CLONE/

机构：字节、清华

目的：【VITS cannot control prosody.】一对多映射问题；缺乏真实声学特征的监督；归一化流的变分自动编码器来建模语音中的潜在韵律信息；双并行自动编码器，在训练期间引入对真实声学特征的监督；



2022年7月 nix-tts

名称：End-To-End SpeechSynthesis system with knowledge distillation

代码：https://github.com/choiHkk/nix-tts

目的：vits知识蒸馏，模型压缩



2022年9月 interspeech_2022

论文：TriniTTS: Pitch-controllable End-to-end TTS without External Aligner

机构：现代汽车、卡梅伦

目的：VITS架构中添加基音控制；去掉Flow，加速；



2022年10月6日 无标注训练

论文：Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus

代码：https://github.com/hcy71o/TransferTTS

机构：三星等

目的：使用大规模无标注语料训练TTS；使用wav2vec2.0;



2022年10月28日 基于VITS架构的变声

论文：FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion

代码：https://github.com/olawod/freevc

目的：本文采用了端到端的VITS框架来实现高质量的波形重构，并提出了无需文本标注的干净内容信息提取策略。通过在WavLM特征中引入信息瓶颈，对内容信息进行分解，并提出基于谱图大小调整的数据增强方法，以提高提取内容信息的纯度。



2022年10月31日 VITS加速

论文：Lightweight and High-Fidelity End-to-End Text-to-Speech with Multi-Band Generation and Inverse Short-Time Fourier Transform

代码：https://github.com/MasayaKawamura/MB-iSTFT-VITS

机构： University of Tokyo, Japan,LINE Corp., Japan.

目的：比VITS快4.1倍，音质无影响；1）用简单的iSTFT部分地替换计算上最昂贵的卷积（2倍加速），2）PQMF的多频带生成来生成波形。



2022年10月31日 Period VITS情感TTS

论文：Period VITS: Variational Inference with Explicit Pitch Modeling for End-to-end Emotional Speech Synthesis

机构： University of Tokyo, Japan,LINE Corp., Japan.

目的：解码器中使用NSF，情感表达准确



2022年11月8日 VISinger 2

论文：VISinger 2: High-Fidelity End-to-End Singing Voice Synthesis Enhanced by Digital Signal Processing Synthesizer

机构：School of Computer Science, Northwestern Polytechnical University, Xi’an, China, DiDi Chuxing, Beijing, China

目的：NSF+VISinger



2023年1月 VITS onnx推理代码

代码：https://github.com/rhasspy/piper

机构：Rhasspy

目的：可导出onnx模型的VITS训练代码；C++推理代码；提供安装包，和预训练模型；支持平台 desktop Linux && Raspberry Pi 4；



2023年2月 VITS 变声 QuickVC

论文：QuickVC: Many-to-any Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion

代码：https://github.com/quickvc/QuickVoice-Conversion

目的：SoftVC + VITS + iSTFT



2023年 wetts vits产品化

代码：GitHub - wenet-e2e/wetts: Production First and Production Ready End-to-End Text-to-Speech Toolkit

功能：前端处理，onnx，流式VITS？~



2023年02月27日 端到端音调可控TTS的无基频变音调推理

论文：PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS

机构：VITS团队

代码：https://github.com/anonymous-pits/pits

目的：PITS在VITS的基础上，结合了Yingram编码器、Yingram解码器和对抗式的移频合成训练来实现基音可控性。



2023年1月 语音克隆

论文：HierSpeech: Bridging the Gap between Text andSpeech by Hierarchical Variational Inference usingSelf-supervised Representations for Speech Synthesis

机构：Korea University

代码：https://github.com/CODEJIN/HierSpeech

目的：利用自我监督的语音表示作为额外的语言表示，以弥合文本和语音之间的信息差距。HierSpeech达到了+0.303 比较平均意见得分，音素错误率从9.16%降低到5.78%。可以利用自我监督的语音来适应新的说话人而没有标注。



2022年12月01日 zero-short语音克隆

论文：SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech

机构：Seoul National University & Samsung

代码：https://github.com/hcy71o/SNAC

主页：https://byoungjinchoi.github.io/snac/

目的：基于微软的说话人自适应器；在VITS的Flow层中嵌入adapter，实现zero-short语音克隆；我们通过引入一个说话人归一化仿射耦合（SNAC）层来改进先前的说话人条件化方法，该层允许以零拍方式利用基于归一化的条件化技术来合成看不见的说话人语音。



2023年4月01日 zero-short语音克隆

论文：NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers

机构：微软

代码：https://github.com/lucidrains/naturalspeech2-pytorch

代码：https://github.com/rishikksh20/NaturalSpeech2

代码：GitHub - CODEJIN/NaturalSpeech2

代码：https://github.com/adelacvg/NS2VC

目的：捕获人类语音的多样性，诸如说话者身份、韵律和风格，如唱歌；利用神经音频编解码器与残余向量量化器得到量化的潜向量，并使用扩散模型以文本输入为条件来生成这些潜向量；设计了一个语音提示机制，以促进扩散模型学习上下文和时长与音高预测；仅使用语音提示进行新颖的零拍歌唱合成；



2023年5月 大规模语音技术 MMS

论文：MMS: Scaling Speech Technology to 1000+ languages

机构：facebook

代码：https://github.com/facebookresearch/fairseq/tree/main/examples/mms

目的：1107种语言，vits语合音成：https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html



2023年5月 zero-shot vits

论文：Automatic Tuning of Loss Trade-offs without Hyper-parameter Search in End-to-End Zero-Shot Speech Synthesis

代码：https://github.com/cnaigithub/Auto_Tuning_Zeroshot_TTS_and_VC

目的：设计一个zero-shot vits框架；vits loss比较多，loss的平衡对质量影响大，所以提出自动均衡loss的方案。



2023年5月 Stochastic Pitch Prediction

论文：Stochastic Pitch Prediction Improves the Diversity and Naturalness of Speech in Glow-TTS

目的：随机F0预测，并嵌入到Flow结构，提高语音多样性和自然度



2023年6月 XPhoneBERT for vits

论文：XPhoneBERT：一种用于文本到语音转换的音素表示的预训练多语言模型

代码：https://github.com/VinAIResearch/XPhoneBERT

代码：https://github.com/ORI-Muchim/PolyLangVITS

目的：phone进TextEncoder后，第一步经过PhoneBERT输出embedding；大幅提升韵律；支持中文，但不太好；



2023年6月 文本描述的VITS风格控制：强悍

论文：PromptStyle: Controllable Style Transfer for Text-to-Speech with Natural Language Descriptions

机构：西工大等

演示：https://promptstyle.github.io/PromptStyle

目的：使用文本描述合成语音风格



2023年7月 IWSLT (口语语言翻译国际会议)，VITS零样本语音克隆

NPU-MSXF队以3.84分取得主观评测综合第一名的优异成绩，同时在语音质量得分上明显高于其他参赛队伍。

文章链接：喜报！实验室队伍获得 IWSLT 2023 语音到语音翻译赛道冠军

论文：https://aclanthology.org/2023.iwslt-1.29.pdf

同时也实现了英文语音到中文语音的零样本说话人音色迁移，即在翻译结果的语音中保留原始语种说话人的音色；使用神经网络瓶颈特征（BN ）作为中间表征，这是因为 BN 可以较好的解耦语言和声学特征，有助于我们实现说话人音色的迁移。在 BN 到语音的还原中，我们采用 VITS[5,6] 框架，并使用主办方提供的数据预训练了一个 ECAPA-TDNN[7] 模型用于提取说话人特征。



2023 年 8 月 VITS2：

论文：VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design

演示：https://vits-2.github.io/demo/

代码：https://github.com/p0p4k/vits2_pytorch 非官方

简介：VITS原作者续作；四个改进点：

1，a stochastic duration predictor trained through adversarial learning

2，monotonic alignment Search with gaussian Noise

3，normalizing flows improved by utilizing the transformer block

4，a speaker-conditioned text encoder to model multiple speakers' characteristics better.



2023年8月 dc-comix-tts

论文：DCComix TTS: An End-to-End Expressive TTS with Discrete Code Collaborated with Mixer

代码：https://github.com/lakahaga/dc-comix-tts

简介：目标是一句话克隆；使用说活人嵌入和风格嵌入作为全局条件，克隆音色和说话风格。



2023年9月

论文：Towards Improving the Expressiveness of Singing Voice Synthesis with BERT Derived Semantic Information

简介：改进VISinger，使用从预先训练的BERT提取的歌词的文本表示作为模型的额外语义信息输入，引入能量预测器来稳定合成的声音，为了减弱跑调问题，重新设计音高预测器以预测真实音符音高比。



2023年10月

论文：VITS-based Singing Voice Conversion System with DSPGAN post-processing for SVCC2023

简介：歌唱声音转换挑战赛（SVCC 2023）排名第一和第二的自然度和相似性



2023年10月

论文：VITS-Based Singing Voice Conversion Leveraging Whisper and multi-scale F0 Modeling

简介：歌唱声音转换挑战赛（SVCC 2023）跨领域任务中，自然度和发音人相似度分别排名第一和第二



2023年11月

论文：SponTTS: modeling and transferring spontaneous style for TTS

演示：https://kkksuper.github.io/SponTTS/

简介：在第一阶段，采用条件变分自动编码器（CVAE）从BN特征中捕获自发韵律，将自发韵律嵌入FastSpeech模型预测声学特征。在第二阶段，采用了VITS将第一阶段基于自发风格的声学特征转换为目标说话人的语音。



2023年11月

论文：DINO-VITS: Data-Efficient Noise-Robust Zero-Shot Voice Cloning via Multi-Tasking with Self-Supervised Speaker Verification Loss

演示：https://vc-research-team.github.io/dino-vits

机构：华为

简介：提出了一种半监督的zero-shot语音克隆方法，该方法将基于HuBERT的语音转换系统适配到语音克隆任务；在声纹识别系统和语音克隆系统（unit-based VITS）的联合训练中，采用自监督DINO损失。



2023年11月

论文：HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis

代码：https://github.com/sh-lee-prml/HierSpeechpp （推荐）

简介：一个快速和强大的zero-shot语音合成器的文本到语音（TTS）和语音转换（VC）；对于文本到语音，采用了文本到向量框架，生成文本表示和韵律提示，文本表示由一个自监督的语音表示和F0表示组成。HierSpeech ++ 根据生成的矢量F0和语音提示生成语音；分层变分自编码器可以是一个强大的zero-shot语音合成器，因为它优于基于LLM和基于扩散的模型。



2023年11月

论文：Transduce and Speak: Neural Transducer for Text-to-Speech with Semantic Token Prediction

论文：https://arxiv.org/abs/2311.02898

简介：2024年1月，改名：Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction



2023年12月

论文：OpenVoice: Versatile Instant Voice Cloning

网址：https://myshell.ai/

代码：https://github.com/myshell-ai/OpenVoice 已释放

简介：这是一种多功能的语音克隆方法，只需要参考speaker的一个简短的音频片段就可以复制他们的声音并生成多种语言的语音。

1）灵活的语音风格控制。OpenVoice支持对语音风格的精细控制，包括情感、口音、节奏、停顿和语调，以及复制参考说话者的音色。

1) Zero-Shot跨语言语音克隆。OpenVoice可以将语音克隆到一种新的语言中，而无需该语言的任何非母语训练数据。

2) OpenVoice在计算上也很高效，成本比商业上可用的API低几十倍。



2024年1月

论文：Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction

标题：基于语义词预测的两阶段文语转换神经传感器（类似GPT+VITS）

代码：https://github.com/scutcsq/Neural-Transducers-for-Two-Stage-Text-to-Speech-via-Semantic-Token-Prediction (非官方)

摘要：将TTS管道分为：语义级序列到序列（seq2seq）建模（AR-Conformer）和细粒度声学建模（VITS）；利用从wav2vec2.0嵌入获得的离散语义令牌。这种解耦的框架降低了TTS的训练复杂度，同时允许每个阶段专注于语义和声学建模。For a robust and efficient alignment modeling, we employ a neural transducer named token transducer for the semantic token prediction, benefiting from its hard monotonic alignment constraints.



2024年3月

论文：PAVITS: Exploring Prosody-aware VITS for End-to-End Emotional Voice Conversion

标题： PAVITS：探索韵律感知的VITS用于端到端情感语音转换

链接：https://arxiv.org/abs/2403.01494

摘要：旨在实现情感语音转换（EVC）的两个主要目标：高内容的自然性和高情感的自然性。引入了一个情感描述子来描述不同语音情感的细微韵律变化。提出了一个文本的韵律特征预测器。引入了韵律对齐损失来建立两种不同模态的潜在韵律特征之间的联系。