# 上海交通大学 X-LANCE 实验室

- 主页: https://x-lance.sjtu.edu.cn

## 学者列表

- Qixi Zheng
- Yushen Chen
- Zhikang Niu
- Ziyang Ma
- Kai Yu 俞凯
- Xie Chen 陈谐

## 论文列表

| 时间 | 类型 | 简称 | 标题 |
| --- | --- | --- | --- |
| 2025.05.31 | Codec | MagiCodec | MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation |
| 2025.05.28 | Benchmark | ERSB | Towards General Discrete Speech Codec for Complex Acoustic Environments: A Study of Reconstruction and Downstream Task Consistency |
| 2025.05.26 | Trick | Fast F5-TTS (EPSS) | Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling |
| 2025.05.26 | Trick | A-DMA | Accelerating Diffusion-based Text-to-Speech Model Training with Dual Modality Alignment |
| 2025.05.25 | | | Towards Reliable Large Audio Language Model |
| 2025.05.22 | | TFC | Unlocking Temporal Flexibility: Neural Speech Codec with Variable Frame Rate |
| 2025.05.19 | Benchmark | MMAR | MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix |
| 2025.04.29 | | | Towards Flow-Matching-based TTS without Classifier-Free Guidance |
| 2025.04.17 | | EmoVoice | EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting |
| 2025.04.14 | | PALLE | Pseudo-Autoregressive Neural Codec Language Models for Efficient Zero-Shot Text-to-Speech Synthesis |
| 2025.03.11 | | YuE | YuE: Scaling Open Foundation Models for Long-Form Music Generation |
| 2025.03.03 | | Spark-TTS | Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens |
| 2025.02.25 | Benchmark | URO-Bench | URO-Bench: A Comprehensive Benchmark for End-to-End Spoken Dialogue Models |
| 2025.02.10 | Survey | | Recent Advances in Discrete Speech Tokens: A Review |
| 2024.10.09 | Flow Matching TTS | F5-TTS | F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching |

## 论文内容

### Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling

- 简称: ***Fast F5-TTS***, ***EPSS***
- 版本:
  - 2025.05.26 [ArXiv:2505.19931v1](https://arxiv.org/abs/2505.19931)
- 链接:
  - 示例: [Github.IO](https://fast-f5-tts.github.io)
  - 代码: 表明开源
- 作者: Qixi Zheng, Yushen Chen, Zhikang Niu, Ziyang Ma, Xiaofei Wang (王晓飞, 微软), 俞凯, 陈谐
- 摘要:
  近期基于流匹配的 TTS 模型如 Voicebox, E2-TTS, F5-TTS 备受关注. 这些模型需要通过多个采样步骤从噪声重构语音, 使得推理速度成为关键挑战, 而减少采样步骤数量可以极大提升推理效率.
  本文介绍 ***Fast F5-TTS***, 无需重新训练即可加速基于流匹配 TTS 模型推理速度的方法.
  通过分析 F5-TTS 的采样轨迹, 发现存在冗余步骤并提出了 ***经验性剪枝步采样 (Empirically Pruned Step Sampling, EPSS)***, 非均匀时间步采样策略能够有效减少采样步数.
  在 RTX3090 上可以实现七步生成, 推理 RTF 为 0.030, 比原始 F5-TTS 快四倍. 此外 EPSS 技术也在 E2-TTS 模型上表现优异, 展现出强大的泛化能力.

现有 TTS 模型可以分为自回归和非自回归两类, 虽然合成质量高, 但推理速度仍面临挑战.
自回归由于序列生成过程存在长推理延迟, VALL-E 合成一秒需要顺序生成 75 个 Token.
非自回归中的扩散模型可以以并行方式生成整个梅尔频谱的所有 Token, 但需要多个采样步从噪声重构语音, NaturalSpeech2 需要 150 步.
采用最优传输的流匹配 (FM-OT) 通过直接传输轨迹减少所需步数, Voicebox 需要 64 步生成, E2-TTS 和 F5-TTS 进一步减少到 32 步.
NaturalSpeech3 使用因子分解扩散模型降低步数到 30, FlashSpeech 通过 LCM 模型实现 2 步生成.
此外蒸馏方法如 Rectified Flow 和分布匹配蒸馏 (Distribution Matching Distillation, DMD) 被证明也能加速推理.

本文主要关注流匹配模型, 提出即插即用方法, 即通过 EPSS 策略加速 FM-OT 类 TTS 模型.
F5-TTS 的摇摆采样增强了在固定的函数评估数量 (Number of Function Evaluations, NFE) 条件下的性能, EPSS 则在保持模型性能的同时进一步减少 NFE.

流匹配技术是学习一个时间依赖的向量场 $v_t: [0,1]\times \mathbb{R}^d \to \mathbb{R}^d$, 用于生成一个流 $\phi_t$ 将样本从简单先验分布 $p_0$ 转换到目标数据分布 $p_1\approx q$.
这一变换通过求解如下 ODE 实现:
$$
\text{d}\phi_t(x) = v_t(\phi_t(x))\text{d}t,\quad \phi_0(x)=x.
$$

设 $p_t: [0,1]\times \mathbb{R}^d\to \mathbb{R}^{>0}$ 为 $\phi_t$ 的概率路径, $u_t$ 是目标向量场. 向量场 $v_t$ 由神经网络参数化, 通过条件流匹配损失 (Conditional Flow Matching Loss, CFM) 进行回归训练.
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}\| v_t(x;\theta) - u_t(x|x_1) \|^2,\quad x_1\sim q(x_1), x\sim p_t(x|x_1)
$$

流匹配 TTS 模型可以分为两种主要方法: 梅尔频谱生成和语音 Token 生成. F5-TTS 是梅尔频谱生成类, 主要关注此类.

- 在训练时, 模型在文本引导语音填充任务上训练, 即给定周围的语音 $(1-m)\odot x_1$ 和全部文本 $z$ 作为条件 $c$ 来预测该语音片段 $m\odot x_1$. 使用最优传输路径 $\phi_t(x) = (1-t) x + t x_1$, 可以将 CFM 损失重写为:
  $$
  \mathcal{L}_{CFM}(\theta) = \mathbb{E}\| m\odot (v_t(x_t,c;\theta) - (x_1-x_0))\|^2,\quad x_t=(1-t)x_0 + t x_1.
  $$

- 在推理时, 给定采样的噪声 $x_0$ 和条件 $c$ (待生成文本和参考语音), 目标 $x_1=\phi_1(x_0)$ 可以求解 ODE 获得:
  $$
  x_1 = x_0 + \int_0^1 v_t(\phi_t(x_0),c;\theta)\text{d}t.
  $$
  然后使用 ODE 求解器如 Euler 方法或中点法数值求解. NFE 指的是求解器需要评估神经网络 $v_t$ 的次数. 无分类器引导 (Classifier-Free-Guidance, CFG) 技术被广泛使用, 因此不将 NFE 值翻倍. 这些求解器将连续时间区间划分为 NFE 个区间, 时间步长由调度策略 $\pi(t)$ 确定, 通常为均匀调度.

F5-TTS 是基于 FM-OT 和 DiT 的完全 NAR 模型, 使用 ConvNeXt 用于文本建模, 它引入了 Sway Sampling 策略, 非均匀调度提升采样性能, 可以在相同 NFE 下获得更高效率:
$$
\text{SS}(t;s) = t + s (\cos(\dfrac{\pi}{2}t)+t-1)
$$

大多数流匹配模型使用均匀时间步调度进行采样, 即 $t_k = k / \text{NFE}$, 尽管能生成高质量样本, 但需要函数评估次数较多导致成本较高.
对 F5-TTS 进行分析, 从高斯噪声进行的一帧梅尔频谱生成步骤, 应用 PCA 到 100 维梅尔特征项进行降维, 可以观察到两个关键现象:
- 复杂初始阶段: 早期轨迹表现出显著的曲率, 可能是由于模型在接近噪声的输入上运行, 导致流的方向不确定.
- 简化后续阶段: 随着轨迹推进, 其变得几乎呈线性, 表明模型输入已经包含足够信息使得流的方向良好定义.

从现象上反映了应在初始阶段细化步骤进行精确求解, 后续步骤使用更大的步长. 此外 ODE 求解可被视为马尔可夫过程, 后续结果依赖于前一时刻的结果, 也强调了初始阶段精确求解的重要性.

因此本文提出了 EPSS 策略, 基于经验观察对不必要的步骤进行剪枝. 例如 7-NFE EPSS 配置 (0, 1/16, 1/8, 3/16, 1/4, 1/2, 3/4, 1) 相比 32-NFE 可以减少 78% 的计算成本但保持合成质量.

### F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

- 简称: ***F5-TTS***
- 版本:
  - 2024.10.09 ArXiv:2505.19931v1
  - 2024.10.15 ArXiv:2505.19931v2
  - 2025.05.20 ArXiv:2505.19931v3
- 链接:
  - 示例: [Github.IO](https://swivid.github.io/F5-TTS/)
  - 代码: [Github](https://github.com/SWivid/F5-TTS/) ![Star](https://img.shields.io/github/stars/SWivid/F5-TTS)
- 作者: Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, 俞凯, 陈谐
- 摘要:
  本文介绍 ***F5-TTS***, 基于流匹配和 DiT 的完全非自回归 TTS 系统. 无需如时长模型, 文本编码器和音素对齐等复杂设计, 文本输入使用填充 Token 进行补足到和语音输入相同长度, 然后应用去噪过程进行语音生成, 这是 E2-TTS 证明可行的方法.
  然而 E2-TTS 的原始设计收敛慢且稳健性低, 为了解决这些问题, 首先用 ConvNeXt 用于精细化文本表示, 使其更易与语音对齐, 然后进一步提出推理时摇摆采样策略, 显著提升模型性能和效率.
  这些设计可以实现更快的训练和推理 RTF 为 0.15, 是现有 SoTA 扩散 TTS 模型的极大提升.
  在公开 100K 小时多语言数据集上训练, F5-TTS 展现出了高度自然和表现力的零样本能力.
