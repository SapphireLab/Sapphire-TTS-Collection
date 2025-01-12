# 第 2 章·大语言模型架构

- [原文](https://github.com/ZJU-LLMs/Foundations-of-LLMs/blob/main/%E3%80%8A%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E3%80%8B%E6%95%99%E6%9D%90/%E3%80%8A%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E3%80%8B%E5%88%86%E7%AB%A0%E8%8A%82%E5%86%85%E5%AE%B9/%E7%AC%AC2%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.pdf)

## 引言

随着数据资源和计算能力的爆发式增长, 语言模型的参数规模和性能表现实现了质的飞跃, 迈入了**大语言模型 (Large Language Model, LLM)** 的新时代.
凭借着庞大的参数量和丰富的训练数据, 大语言模型不仅展现出了强大的泛化能力, 还催生了新智能的涌现, 勇立生成式人工智能(Artificial Intelligence Generated Content, AIGC) 的浪潮之巅.

当前, 大语言模型技术蓬勃发展, 各类模型层出不穷.
这些模型在广泛的应用场景中已经展现出与人类比肩甚至超过人类的能力, 引领着由 AIGC 驱动的新一轮产业革命.

本章将深入探讨大语言模型的相关背景知识, 并分别介绍 **Encoder-Only**, **Encoder-Decoder** 以及 **Decoder-Only** 三种主流模型架构.
通过列举每种架构的代表性模型, 深入分析它们在网络结构, 训练方法等方面的主要创新之处.
最后, 本章还将简单介绍一些非 Transformer 架构的模型, 以展现当前大语言模型研究百花齐放的发展现状.

## 2.1.大数据+大模型→新智能

#TODO

## 2.2.大语言模型架构概览

在语言模型的发展历程中, [Transformer](../../Models/_Transformer/2017.06.12_Transformer.md) 框架的问世代表着一个划时代的转折点.
其独特的**自注意力 (Self-Attention) 机制**极大地提升了模型**对序列数据的处理能力**, 在**捕捉长距离依赖关系**方面尤为出色.
此外, Transformer 框架**对并行计算的支持**极大地加速了模型的训练过程.

当前绝大多数大语言模型均以 Transformer 框架为核心, 并进一步演化出了三种经典架构: **Encoder-Only**, **Encoder-Decoder**, **Decoder-Only**.
这三种架构在设计和功能上各有不同.

### 2.2.1.基本概念

#### Encoder-Only 架构

Encoder-Only 架构仅选取了 Transformer 中的编码器 Encoder 部分, 用于接受输入文本并生成与上下文相关的特征.
Encoder-Only 架构包含三个部分:
- 输入编码部分: 分词, 向量化, 添加位置编码三个过程;
  - 原始输入文本被分词器 Tokenizer 拆解为 Token 序列;
  - 通过词表和词嵌入 (Embedding) 矩阵映射为向量序列, 确保文本信息得以数字化表达;
  - 位置编码 (Positional Encoding) 添加到每个向量序列中, 以保留文本中单词的顺序信息.
- 特征编码部分:
  - 由多个相同的编码器块堆叠而成, 每个编码器块包含自注意力模块和全连接前馈模块;
  - 前一部分得到的向量序列依次通过这些编码模块, 进一步提取和深化文本特征.
- 任务处理部分:
  - 在预训练阶段使用全连接层作为输出头, 用于完成掩码预测等任务;
  - 在下游任务适配阶段, 输出头会根据具体任务需求进行定制.
    - 情感分析/主题分类等判别任务: 只需添加一个分类器直接输出判别结果;
    - 文本摘要生成等生成任务: 只需添加一个全连接层逐个预测后续的 Token. (注: 每次生成新 Token 时需要重新计算整个输入序列的表示, 增加了计算成本, 也可能导致生成文本缺乏连贯性)

#### Encoder-Decoder 架构

Encoder-Decoder 架构在 Encoder-Only 架构的基础上引入了一个解码器, 并采用交叉注意力机制来实现编码器和解码器之间的有效交互, 弥补 Encoder-Only 架构在生成任务上的不足.
Decoder 部分包含三个部分:
- 输出编码部分: 和 Encoder 的输入编码结构相同, 含分词, 向量化, 位置编码三个过程.
- 特征解码部分: 和特征编码部分高度相似, 包括**掩码自注意力 (Masked Self-Attention) 模块**, 交叉注意力模块, 全连接前馈模块.
  - 掩码自注意力模块: 确保模型只关注上文, 不会预见未来信息, 从而在无下文泄露的条件下进行自回归的训练和推理;
  - 交叉注意力模块: 负责处理从编码模块向解码模块传递相关信息.
- 输出生成部分:
  - 线性层 + Softmax 层将特征解码后的向量转换为词表上的概率分布, 并从这个分布中采样得到最合适的 Token 作为输出.

训练阶段:
- 数据样本: 输入文本和真实输出文本 (Ground Truth).
- 输入文本首先被输入编码部分转化为向量序列, 在特征编码模块进一步处理转化为上下文表示.
- 输出文本添加特殊的开始 Token `[START]` 在输出编码部分转化为向量序列, 并行输入到特征解码模块.
- 使用 Teacher Forcing 技术, 在每轮预测时, 使用真实输出文本中的已知部分作为输入, 并结合从最后一个编码块得到的上下文信息来预测下一个 Token, 计算预测的 Token 和真实 Token 之间的损失, 通过反向传播更新参数.

推理阶段:
- 无真实输出文本, 输出序列原始状态只有开始 Token `[START]`, 无需分词器.
- 模型通过自回归的方式, 在每轮采样生成 Token 后拼接到输出序列中, 用于下一轮预测.
- 这一过程循环进行直到生成特定的结束 Token `[END]` 或达到模型设定的最大输出长度.
- 每轮的输入依赖上一轮的采样结果, 只能一步步串行输出.

#### Decoder-Only 架构

为了有效缩减模型的规模以及降低整体的计算复杂度, Decoder-Only 架构摒弃了 Encoder 部分以及交叉注意力模块.
在这种架构下, 模型仅使用解码器来构建语言模型, 利用自回归机制在给定上文的情况下生成流畅且连贯的下文.

Decoder-Only 架构包含三个部分:
- 输入编码部分: 分词器, 词嵌入矩阵, 位置编码;
- 特征解码部分: 忽略交叉注意力子模块;
- 输出生成部分: 线性层 + Softmax 层, 用于生成下一个 Token.

### 2.2.2.功能对比

#### 注意力矩阵

注意力矩阵 (Attention Matrix) 是 Transformer 中的核心组件, 用于计算输入序列中各个 Token 之间的依赖关系.
通过注意力机制, 模型可以在处理当前 Token 时, 灵活地关注序列中其他 Token 所携带的信息, 决定了这一过程中哪些 Token 能够相互影响.

- Encoder-Only 架构: 注意力矩阵来自自注意力模块, 用于捕捉输入序列中各个 Token 之间的关系. 整个矩阵呈现出完全的注意力, 即对每个 Token 的理解都依赖于整个输入序列中的所有 Token, 即**双向注意力机制**. 能够同时利用前后文信息, 深入理解复杂的语义联系和上下文依赖.
- Encoder-Decoder 架构: 注意力矩阵有编码器的自注意力, 解码器的掩码自注意力, 交叉注意力三种机制.
  - 编码器自注意力: 完全的注意力, 用于生成输入序列的全面上下文表示.
  - 解码器掩码自注意力: 下三角注意力矩阵, 确保生成当前 Token 时模型只关注之前生成的 Token.
  - 交叉注意力: 解码器始终能够动态地参考编码器生成的完整上下文表示, 确保输入和输出序列高度相关且连贯. 即生成 $y_{i}$ 时参考 $x_{1}\sim x_{n} + y_{1}\sim y_{i-1}$.
- Decoder-Only 架构: 掩码自注意力模块, 只能依赖于已经生成的历史 Token 信息, 整个矩阵呈现出下三角的注意力, 即**单向注意力机制**.

#### 适用任务

由于模型设计和注意力矩阵的差距, 在同等参数规模下, 三种架构的模型在适用任务上各有倾向.
- Encoder-Only 架构:
  - 特点: 双向注意力机制允许预测每个 Token 时充分考虑序列的上下文信息, 捕捉丰富的语义和依赖关系.
  - 任务: 适合**自然语言理解 (Natural Language Understanding, NLU) 任务**, 如情感分析和文本分类等任务.
  - 缺点: 缺少解码器组件, 无法直接生成所需目标序列. 在**自然语言生成 (Natural Language Generation, NLG) 任务**上表现可能不如专门设计的的生成模型.
- Encoder-Decoder 架构:
  - 特点: 添加解码器能够基于编码器输出的上下文表示逐步生成序列, 使得模型可以有效地处理复杂的输入条件, 并生成相关且连贯的高质量内容.
  - 任务: 适合处理各种复杂的**有条件生成任务**. 如机器翻译, 文本摘要, 问答系统等需要同时理解输入并生成相应输出的场景.
  - 缺点: 模型规模以及参数量庞大的问题.
- Decoder-Only 架构:
  - 特点: 删除编码器部分, 降低模型本身计算复杂度. 使用掩码操作确保在每个时间步生成当前 Token 时只能访问先前的 Token, 通过自回归生成机制从起始 Token 开始逐步生成文本. 大规模预训练数据使得该架构的模型能够生成高质量连贯的文本.
  - 任务: 适合处理**无条件文本生成任务**, 如自动故事生成, 新闻文章生成等.
  - 缺点: 模型规模有限时由于缺乏编码器的双向上下文信息, 模型在理解复杂输入数据时存在一定局限性, 表现可能不如 Encoder-Decoder 架构.

### 2.2.3.历史演变

在不同的历史阶段, 暗中模型架构分别展示了自身的优势.
随着模型规模以及数据规模的显著增长, Decoder-Only 架构模型逐渐占据上风, 以其强大的任务泛化性能展现出成为大一统架构的潜力.
以 GPT-3, GPT-4 等为代表的大型 Decoder-Only 语言模型已经发展出了与人类媲美甚至超越人类的记忆, 推理以及处理复杂任务的能力.

在大语言模型的早期发展阶段 (2018 年), BERT 和 GPT-1 分别作为 Encoder-Only 和 Decoder-Only 架构的代表几乎同时出现. 但受限于当时的模型参数规模, **BERT 的强大上下文理解能力比 GPT-1 初阶的文本生成能力更为亮眼**. 这使得 Encoder-Only 架构得到更广泛的探索和应用.

但随着用户对机器翻译等生成任务需求的增加, Encoder-Only 架构逐渐无法满足直接生成的需求, 因而被逐渐冷落.
2019 年末诞生了一众 Encoder-Decoder 架构的模型, 能够有效处理序列到序列的生成任务, 逐渐成为主流.
随着算力资源的急速发展, 研究人员开始寻求不断提升参数量来激发更强的生成能力.
得益于参数易扩展性, Decoder-Only 架构的模型参数量开始急剧扩充, 文本生成能力大幅提升.
2021 年后, 在 GPT-3 等模型的推动下, Decoder-Only 架构开始占据主流, 甚至主导了大语言模型的发展.
尽管如此, Encoder-Decoder 架构仍然活跃于开源社区中, 不断被探索和改进.
Encoder-Only 架构在 BERT 带来最初的爆炸性增长后, 关注度有所下降, 但仍在部分判别任务中发挥重要作用.

总的来讲, 大语言模型的主流架构经历了从 Encoder-Only 到 Encoder-Decoder 再到 Decoder-Only 的演变, 而引发这种更迭趋势的原因可能是模型本身生成能力以及计算效率上的差异.

## 2.3.Encoder-Only LLM

Encoder-Only 架构的核心在于能够覆盖输入所有内容的**双向编码模型 (Bidirectional Encoder Model)**.
在处理输入序列时, 双向编码模型融合了从左到右的正向注意力和从右往左的反向注意力, 能够充分捕捉每个 Token 的上下文信息, 因此也被称为具有全面的注意力机制.
由于其上下文感知能力和动态表示的优势, 双向编码器显著提升了自然语言处理任务的性能.

和 Word2Vec 和 GloVe **为每个词提供固定向量表示的静态编码方式**不同, 双向编码器为每个词生成**动态的上下文嵌入 (Contextual Embedding)**, 这种嵌入依赖于输入序列的具体上下文, 使得模型能够更加精确地理解词与词之间的依赖性和语义信息, 有效处理词语的多义性问题.
这种动态表示使得双向编码器**在句子级别的任务上表现出色**, 显著超过了静态词嵌入方法的性能. [^20]

Encoder-Only 虽然不直接生成文本, 但其生成的上下文嵌入对于深入理解输入文本的结构和含义至关重要. 这些模型在需要深度理解和复杂推理的自然语言处理任务中展现出卓越的能力.
当前 BERT [^11] 及其变体 RoBERTa [^23], ALBERT [^17] 等都是基于 Encoder-Only 架构的主流大语言模型.

### BERT

[BERT (Bidirectional Encoder Representations from Transformers)](../../Models/TextLM/2018.10.11_BERT.md) 是 Google AI 于 2018 年 10 月提出的 Encoder-Only 架构的预训练语言模型.
其核心创新在于通过双向编码模型狠毒挖掘文本的上下文信息, 从而为各种下游任务提供优秀的上下文嵌入.

BERT 模型的结构和 Transformer 中的编码器几乎一致: 多个编码模块堆叠而成, 每个模块包含多头自注意力模块和全连接前馈模块.
根据参数量不同 BERT 模型有 BERT-Base 和 BERT-Large 两个版本:
- BERT-Base: 12 个编码模块, 隐藏层维度 768, 自注意力头数 12, 总参数量 1.1 亿;
- BERT-Large: 24 个编码模块, 隐藏层维度 1024, 自注意力头数 16, 总参数量 3.4 亿.

预训练:
- 数据: BERT 使用小说数据集 BookCorpus [^46] (含约 8 亿个 Token) 和英语维基百科数据集 (含约 25 亿个 Token) 进行预训练, 总计约 33 亿个 Token, 总数据量达到 15 GB 左右.
- 任务: 在预训练任务上, BERT 开创性地提出了**掩码语言建模 (Masked Language Modeling, MLM)** 和**下句预测 (Next Sentence Prediction, NSP)** 两种任务来学习生成上下文嵌入.

具体来说, BERT 先基于给定的原始文本构造多个样本序列, 每个样本序列由原始文本中的两个句子组成, 这两个句子有 50% 的概率是来自原文的连续句, 50% 的概率是随机挑选的两个句子. 然后对构造出来的样本序列进行分词, 并在序列开头添加特殊 Token `[CLS]`, 在每个句子结尾添加特殊 Token `[SEP]`.
- `[CLS]` 用于聚合整个序列的信息;
- `[SEP]` 用于明确句子之间的界限.

BERT 利用处理后的序列进行下句预测任务, 利用模型判断样本序列中的两个句子是否为连续的. 这个任务训练 BERT 识别和理解句子之间的关系, 捕捉句子层面的语义特征. 对于理解文本的逻辑流和句子之间的关联性有很大帮助, 特别是在问答和自然语言推理等需要理解文档层次结构的自然语言处理任务.

最后 BERT 随机选择样本序列中大约 15% 的 Token 进行掩码, 将其替换为特殊 Token `[MASK]` 或者随机单词.
模型需要预测这些被替换的 Token 的原始内容. 这个过程类似于完形填空, 要求模型根据周围的上下文信息来推断缺失的 Token.
预测过程中使用的交叉熵损失函数驱动了 BERT 模型中参数的优化, 使其能够学习到文本的双向上下文表示.

值得注意的是, 在 MLM 任务的训练过程中, BERT 仅对那些被随即替换的 Token 进行学习, 即只计算这些 Token 的预测损失来更新模型参数.

通过这两种预训练任务的结合, 使 BERT 在理解语言的深度和广度上都有显著提升. BERT 不仅能够捕捉到 Token 的细粒度特征, 还能够把握长距离的依赖关系和句子间的复杂联系, 为各种下游任务提供了坚实的语言理解基础.

BERT 的输出是输入中所有 Token 的向量表示, 因此总长度不固定, 无法直接应用于各类下游任务.
为此, BERT 设计了 `[CLS]` Token 来提取整个输入序列的聚合表示.
`[CLS]` Token 是专门为分类和汇总任务设计的特殊 Token, 即 Classification Token. 通过注意力机制, `[CLS]` 汇总了整个输入序列的信息, 生成一个固定长度的向量表示, 从而实现对所有 Token 序列信息的概括, 以便处理各种下游任务.

- 在文本分类任务中, 可以将输出中 `[CLS]` 标签对应的向量提取出来, 传递给全连接层, 从而用于分类. 例如判断整个句子的情绪.
- 在问答系统任务中, 需要输入问题和一段相关文本: `[CLS]` 问题 `[SEP]` 文本 `[SEP]`, 最后提取出 `[CLS]` 对应的向量, 并传递给两个全连接层, 用于判断答案是否存在于相关文本. 如果存在则两个全连接层分别用于输出答案的起始和结束位置, 从而准确提取出问题的答案.
- 在语义相似度任务中, 需要计算两端或多段文本之间的语义相似度. 可以构造 `[CLS]` 文本1 `[SEP]` 文本2 `[SEP]` 并结合一个线性层来直接输出两个文本之间的相似度, 也可以直接提取 `[CLS]` 对应向量, 利用额外相似度度量来计算多段文本之间的相似度.

### RoBERTa

[RoBERTa (Robustly Optimized BERT Pretraining Approach)](../../Models/TextLM/2019.07.26_RoBERTa.md) 由 Facebook/Meta AI 研究院于 2019 年 07 月提出, 旨在解决 BERT 在训练程度上不充分的问题, 以提升预训练语言模型的性能.
RoBERTa 在 BERT 的基础上采用了更大的数据集, 更长的训练时间以及更细致的超参数调整来优化预训练过程, 从而提高模型在各种自然语言处理任务上的性能和鲁棒性.

RoBERTa 在结构上和 BERT 基本一致, 也有两个版本:
- RoBERTa-Base: 与 BERT-Base 对标, 12 个编码模块, 隐藏层维度 768, 自注意力头数 12, 总参数量 1.2 亿;
- RoBERTa-Large: 与 BERT-Large 对标, 24 个编码模块, 隐藏层维度 1024, 自注意力头数 16, 总参数量 3.5 亿.

预训练:
- 数据: RoBERTa 在 BERT 原有的小说数据集 BookCorpus 和英语维基百科数据集的基础上, 添加了新闻数据集 CC-News (76 GB 的新闻文章), 网页开放数据集 OpenWebText (38 GB 的网页文本内容), 故事数据集 Stories (31 GB 的故事文本), 总数据量达到约 160 GB.
- 任务: RoBERTa 移除了 BERT 的下句预测任务, 将静态掩码语言建模任务改为动态掩码语言建模任务.
具体而言, BERT 在数据预处理期间对句子进行掩码, 随后在每个训练 Epoch 中, 掩码位置不再变化. RoBERTa 将训练数据复制成十个副本, 分别进行掩码. 同样训练 40 个 Epoch, BERT 在静态掩码后的文本训练 40 次, RoBERTa 将十个不同掩码后的副本分别训练 4 次, 从而增加模型训练的多样性, 有助于模型学习到更丰富的上下文信息.

这些改进使得 RoBERTa 在理解上下文和处理长文本方面表现出色, 尤其是在捕捉细微的语义差异和上下文依赖性方面.

### ALBERT

[ALBERT (A Lite BERT)](../../Models/TextLM/2019.09.26_ALBERT.md) 是由 Google Research 团队于 2019 年 09 月提出的轻量级 BERT 模型, 旨在通过**参数共享**和**嵌入分解技术**来减少模型的参数量和内存占用, 从而提高训练和推理效率.
BERT 模型的参数较多使其不仅难以训练, 推理时间也较长.

ALBERT 在设计过程通过参数因子分解技术和跨层参数共享技术在相同的模型架构下显著减少了参数数量, 使其在资源受限的环境下更加实用, 处理大规模数据集和复杂任务时更加高效, 并降低了模型部署和维护成本.

**参数因子分解**: Embedding 层的输出向量维度 E 和隐藏层的向量维度 H 是一致的. 具体来说 BERT 的词表大小为 30000, 隐藏层维度为 768, 因此 BERT 的 Embedding 层参数量为 VH, 大约为 2,304万.
ALBERT 将 Embedding 层矩阵进行分解, 将词表对应的独热编码通过低维投影曾投影到维度 E, 再上投影会隐状态维度 H. 具体来说, Embedding 层维度为 128, 则参数量为 VE+EH, 大约为 394 万, 约为 BERT 的 1/6.
对于具有更大隐藏层向量维度 H 的 Large 版本, ALBERT 节省参数空间的优势更加明显, 能够将参数量压缩至 1/8 左右.

**跨层参数共享**: BERT-Base 中十二层编码模块的参数都是独立训练的, 而 ALBERT 为了降低模型参数量, 只学习第一层编码模块的参数, 然后直接共享给其他所有层. 在一定程度上牺牲了模型性能, 但显著提升了参数存储空间的压缩比.

ALBERT 有四个版本:
- ALBERT-Base: 12 个编码模块, 嵌入分解维度为 128, 隐藏层维度 768, 自注意力头数 12, 总参数量 0.12 亿;
- ALBERT-Large: 24 个编码模块, 嵌入分解维度为 128, 隐藏层维度 1024, 自注意力头数 16, 总参数量 0.18 亿;
- ALBERT X-Large: 12 个编码模块, 嵌入分解维度为 128, 隐藏层维度 2048, 自注意力头数 16, 总参数量 0.6 亿;
- ALBERT XX-Large: 12 个编码模块, 嵌入分解维度为 128, 隐藏层维度 4096, 自注意力头数 64, 总参数量 2.2 亿.

预训练:
- 数据: ALBERT 使用和 BERT 相同数据集进行预训练.
- 任务: 采用的预训练任务上保留了 BERT 的掩码语言建模任务, 并将下句预测任务替换为**句序预测 (Sentence Order Prediction, SOP)**, 即判断拼接的两个连续句子的顺序是正序还是反序.

### ELECTRA

[ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)](../../Models/TextLM/ELECTRA.md) 是由 Google Brain 和斯坦福大学的研究人员于 2020 年 03 月提出的另一种 BERT 变体, 旨在解决大规模预训练语言模型中的效率和可扩展性问题.
通过使用生成器-判别器架构, ELECTRA 能够更高效地利用预训练数据, 提高了模型在下游任务中的表现.

ELECTRA 提出三个版本:
- ELECTRA-Small: 生成器和判别器都是 12 个编码模块堆叠而成, 隐藏层维度为 256, 自注意力头数量为 4, 参数量均为 0.14 亿;
- ELECTRA-Base: 生成器和判别器都是 12 个编码模块堆叠而成, 隐藏层维度为 768, 自注意力头数量为 12, 参数量均为 1.1 亿;
- ELECTRA-Large: 生成器和判别器都是 24 个编码模块堆叠而成, 隐藏层维度为 1024, 自注意力头数量为 16, 参数量均为 3.3 亿.

预训练:
- 数据: Small 和 Base 版本使用 BERT 相同数据集; Large 版本使用大规模网页数据集 ClueWeb, CommonCrawl, Gigaword, 总计 330 亿 Token.
- 任务: ELECTRA 在 BERT 原有的掩码语言建模基础上结合了生成对抗网络的思想, 采用了一种生成器-判别器结构.
  - 生成器: 能进行掩码预测的模型 (BERT 模型), 负责将掩码后的文本恢复原状.
  - 判别器: 替换词检测 (Replaced Token Detection, RTD) 预训练任务, 负责检测生成器输出的内容中的每个 Token 是否为原文内容.

BERT 只有 15% 固定比例 Token 被掩码, ELECTRA 则不局限于固定比例, 判别器会判断生成器输出的所有 Token 是否被替换过, 因此能够更好地学习文本的上下文嵌入.

### 小结

经典模型参数大小止步于 6.6 亿, 预训练任务也主要服务于自然语言理解. 没有继续寻求参数量上的突破, 通常只专注于判别任务.

|模型|发布时间|参数量 (亿)|语料规模|预训练任务|
|:-:|:-:|:--:|:--:|:--|
|BERT|2018.10|Base 1.1<br>Large 3.4|~15 GB|MLM+NSP|
|RoBERTa|2019.07|Base 1.5<br>Large 3.5|160 GB|Dynamic MLM|
|ALBERT|2019.09|Base 0.12<br>Large 0.18<br>X-Large 0.6<br>XX-Large 2.2|~15 GB|MLM+SOP|
|ELECTRA|2020.03|Small 0.28<br>Base 2.2<br>Large 6.6|20~200 GB|RTD|

## 2.4.Encoder-Decoder LLM

Encoder-Decoder 架构在 Encoder-Only 架构的基础上引入 Decoder 组件, 以完成机器翻译等序列到序列 (Sequence to Sequence, Seq2Seq) 任务.

- 编码器和 Encoder-Only 架构相同, 输入序列通过编码器部分转变为固定大小的上下文向量, 包含了输入序列的丰富语义信息.
- 解码器由多个解码模块堆叠而成, 每个解码模块由掩码自注意力模块, 交叉注意力模块, 全连接前馈模块组成.
  - 掩码自注意力模块引入掩码机制来防止未来信息的泄露, 确保解码过程的自回归特性, 它也被称为因果注意力.
  - 交叉注意力模块实现编码器和解码器之间的信息交互, 对于生成和输入序列高度相关的输出十分重要.

通过自注意力和交叉注意力的结合, Encoder-Decoder 架构能够高效地编码输入信息并生成高质量的输出序列.

### T5

Google Research 团队于 2019 年 10 月提出了一种基于 Encoder-Decoder 架构的大型预训练语言模型 [T5 (Text-To-Text Transfer Transformer)](../../Models/TextLM/2019.10.23_T5.md), 采用统一的文本到文本的转换范式来处理多种任务, 避免对不同任务进行定制化设计.

T5 模型的核心思想是**将多种自然语言处理任务统一到一个文本转文本的生成式框架中**.
在这一框架下, T5 通过不同的输入前缀来指示模型执行不同任务, 然后生成相应的任务输出.
这种方法可以视为早期的提示 (Prompt) 技术的运用, 通过构造合理的输入前缀, T5 模型能够引导自身针对特定任务进行优化, 而无需对模型架构进行根本性的改变. 这种灵活性和任务泛化能力显著提高了模型的实用性, 以轻松适应各类新任务.

T5 模型和原始 Transformer 架构相同, 根据不同的参数提供了五个不同版本:
- T5-Small: 6 个编码模块, 6 个解码模块, 隐藏层维度 512, 自注意力头数量 8, 总参数量 6000 万;
- T5-Base: 12 个编码模块, 12 个解码模块, 隐藏层维度 768, 自注意力头数量 12, 总参数量 2.2 亿;
- T5-Large: 24 个编码模块, 24 个解码模块, 隐藏层维度 1024, 自注意力头数量 16, 总参数量 7.7 亿 (对标 BERT-Large);
- T5-3B: T5-Large 自注意力头扩大到 32, 全连接层维度扩大四倍, 总参数量 28 亿;
- T5-11B: T5-3B 自注意力头扩大到 128, 全连接层维度扩大四倍, 总参数量 110 亿.

预训练
- 数据: 从大规模网页数据集 Common Crawl 提取数据经过严格清理和过滤, 获得 Colossal Clean Crawled Corpus (C4) 数据集, 规模达到约 750 GB.
- 任务: Span Corruption 预训练任务. T5 对整个被遮挡的连续文本片段进行预测. 这些片段可能包括连续的短语或子句, 是具有完整意义的语义单元. 该任务要求模型不仅要理解局部词汇的表面形式, 还要能捕捉更深层次的句子结构和上下文之间的复杂依赖关系.

下游任务
- 基于预训练阶段学习到的大量知识和新提出的文本转文本的统一生成式框架, T5 模型可以在完全零样本 (Zero-Shot) 的情况下, 利用 Prompt 工程技术直接适配到多种下游任务.
- T5 也可以通过微调来适配到特定的任务, 不过此时需要针对下游任务收集带标签数据, 也需要更多计算资源和训练时间, 因此通常只被应用与那些对精度要求极高且任务本身较为复杂的应用场景.

T5 模型的统一生成框架不仅简化了不同自然语言处理任务之间的转换流程, 也为大语言模型的发展提供了新方向.

T5 也衍生了许多变体, 以改进其性能:
- mT5: 扩展了对 100 多种语言的支持;
- T0: 多任务训练增强了零样本学习能力;
- Flan-T5: 通过指令微调进一步提升模型灵活性和效率.

### BART

[BART (Bidirectional and Auto-Regressive Transformers)](../../Models/TextLM/2019.10.29_BART.md) 是由 Meta AI 研究院于 2019 年 10 月提出的一个 Encoder-Decoder 架构模型.
和 T5 将多种任务集成到统一框架不同, BART 旨在通过多样化的预训练任务来提升模型在文本生成和文本理解的表现.

BART 的模型同样和原始 Transformer 架构相同, 提供了两个版本:
- BART-Base: 6 个编码模块, 6 个解码模块, 隐藏层维度 768, 自注意力头数量 12, 总参数量 1.4 亿;
- BART-Large: 12 个编码模块, 12 个解码模块, 隐藏层维度 1024, 自注意力头数量 16, 总参数量 4 亿.

预训练

- 数据: 和 RoBERTa 相同数据 (BookCorpus + 英文维基百科 + CC-News + OpenWebText + Stories), 规模达到约 160 GB.
- 任务: 以重建被破坏的文本为目标, 包括以下五种任务破坏文本, 然后训练模型来恢复文本. 锻炼了模型对文本结构和语义的深入理解, 增强在面对不完整或损坏信息时的鲁棒性.
  - Token 掩码 (Token Masking) 任务: 类似 BERT 的 MLM 任务, 随机采样一部分 Token 替换为 `[MASK]`, 由模型推断被删除的 Token 的内容;
  - Token 删除 (Token Deletion) 任务: 在原始文本随机删除一部分 Token, 由模型推断被删除的 Token 的位置和内容;
  - 连续文本填空 (Text Infilling) 任务: 类似 T5 预训练任务, 选择的连续段长度服从泊松分布 $\lambda=3$, 整体替换为 `[MASK]`, 由模型推断一段 Span 及其长度的能力;
  - 句子打乱 (Sentence Permutation) 任务: 给定文本拆分为多句并随机打乱, 由模型推理前后句关系;
  - 文本旋转 (Document Rotation) 任务: 随机选择一个 Token 作为开头进行旋转, 由模型找到文本合理起始点.

下游任务
- 在预训练结束后, 可以对 BART 进行微调以将预训练阶段学到的语言知识迁移到具体的应用场景中, 适配多种下游任务.

BART 也衍生了许多变体:
- mBART: 处理跨语言文本生成任务.

### 小结

基于 Encoder-Decoder 架构的大语言模型, 在生成任务中展示了良好的性能表现.

|模型|发布时间|参数量 (亿)|语料规模|
|:-:|:-:|:--:|:--:|
|T5|2019.10|Small 0.6<br>Base 2.2<br>Large 7.7<br>3B 28<br>11B 110|~750 GB|
|mT5|2020.10|3~130|9.7 TB|
|T0|2021.10|30~110|~400 GB|
|BART|2019.10|Base 1.4<br>Large 4.0|160 GB|
|mBART|2020.06|0.4~6.1|~1 TB|

## 2.5.Decoder-Only LLM

在开放式 (Open-Ended) 生成任务中, 输入序列通常较为简单, 甚至没有具体明确的输入, 因此维持一个完整的编码器来处理这些输入并不是必要的.
对于这种任务, Encoder-Decoder 架构可能显得过于复杂且缺乏灵活性.
在这种情况下, Decoder-Only 架构表现得更为优异.

Decoder-Only 架构通过自回归方法逐字生成文本, 不仅保持了长文本的连贯性和内在一致性, 且在缺乏明确输入或复杂输入的情况下, 能够更自然流畅地生成文本.
此外 Decoder-Only 架构由于去除了编码器部分, 使模型更加轻量化, 从而加快了训练和推理的速度.

Decoder-Only 发展历史:
- Decoder-Only 架构模型的概念最早可以追溯到 2018 年发布的 GPT-1 模型, 但当时以 BERT 为代表的 Encoder-Only 架构模型在各项任务中展现出卓越性能, Decoder-Only 架构并没有受到足够关注.
- 直到 2020 年 GPT-3 的突破性成功, 使得 Decoder-Only 架构开始被广泛应用于各种大语言模型中. 如 OpenAI 的 GPT 系列, Meta 的 LLaMA 系列等.
- GPT 系列起步最早, 在性能上也成为时代标杆, 但自 ChatGPT 后逐渐走向闭源, LLaMA 系列起步较晚, 但以出色性能和始终开源也占据了一席之地.

### GPT 系列

GPT (Generative Pre-trained Transformer) 系列模型是由 OpenAI 开发的一系列基于 Decoder-Only 架构的大语言模型.
自 2018 年问世以来, GPT 系列模型经历了快速发展, 在模型规模和预训练范式上不断演进, 引领了大语言模型发展的浪潮.
其演进历程可以划分为五个阶段, 其参数规模和预训练语料规模呈现激增的趋势. 虽然后续进入闭源模式, 但根据 Scaling Law 有理由猜测后续版本在规模和预训练语料规模上也有所增长.

|模型|发布时间|参数量 (亿)|语料规模|
|:-:|:-:|:--:|:--:|
|GPT-1|2018.06|1.17|~5 GB|
|GPT-2|2019.02|1.24/3.55/7.74/15|40 GB|
|GPT-3|2020.05|1.25/3.5/7.62/13/27/67/130/1750|1 TB|
|ChatGPT|2022.11|-|-|
|GPT-4|2023.03|-|-|
|GPT-4o|2024.05|-|-|

#### GPT-1

预训练:
- 数据: BookCorpus (~8 亿 Token, 5 GB)
- 任务: Next Token Prediction (下一词预测), 基于给定的上下文预测下一个可能出现的 Token. 以自回归的方法不断完成下一词任务, 从而有效地完成文本生成任务. 这种策略使得模型可以在不需要人为构造大量带标签数据的前提下, 学习到大量语言的 "常识", 学会生成连贯且上下文相关的文本. 不仅提高了模型的泛化能力, 而且减少了对标注数据的依赖.

下游任务:
- 尽管 GPT-1 模型在预训练后展现了一定潜力, 但其任务泛化能力仍受限于当时的训练数据量和模型参数数量.
为了提升模型在特定下游任务上的表现, 通常需要进一步的有监督微调.

微调过程涉及到使用针对特定任务的标注数据来优化模型的参数, 其中模型的输入输出都是文本序列的形式.
- 文本分类: 接收文本作为输入, 根据预定义的类别标签, 对文本进行分类;
- 文本相似度评估: 衡量两端文本之间的相似性, 量化它们的内容和语义相似度;
- 多项选择题解答: 理解文本和选项内容, 从给定选项中识别并输出最合适的答案.

GPT-1 具备原生的文本生成能力, 但受限于训练数据量和模型参数数量, 其生成能力还不足以用于解决实际问题.
此外由于单向注意力机制的限制, 全面理解上下文的能力有所欠缺, 而几个月后的 BERT 强大的上下文嵌入能力遮盖了 GPT-1.
虽然 GPT-1 在当时的实用性不如 BERT, 但它作为 Decoder-Only 架构的开端, 为后续大语言模型的发展拉开了序幕.

#### GPT-2

GPT-2 延续了 GPT-1 的架构, 并在此基础上进一步加大了参数数量.

GPT-2 发布了四个版本:
- GPT-2 Small: 参数接近 GPT-1/BERT-Base, 12 个编码模块, 隐藏层维度 768, 自注意力头 12, 总参数 1.24 亿;
- GPT-2 Medium: 参数接近 BERT-Large, 24 个编码模块, 隐藏层维度 1024, 自注意力头 16, 总参数 3.55 亿;
- GPT-2 Large: 36 个编码模块, 隐藏层维度 1280, 自注意力头 20, 总参数 7.74 亿;
- GPT-2 XL: 48 个编码模块, 隐藏层维度 1600, 自注意力头 25, 总参数 15 亿.

预训练:
- 数据: WebText 数据集 (40 GB 精心清洗的网络文本)
- 任务: 下一词预测
- GPT-2 语言理解能力得到了显著增强, 接触到了更多样化的语言使用场景, 还学到了更复杂的语言表达方式.

下游任务:
- GPT-2 的任务泛化能力得到了改善, 在某些任务上可以不进行微调, 直接进行零样本学习, 大大增加了处理下游任务的灵活性, 降低了下游任务适配的成本.

#### GPT-3

[GPT-3](../../Models/TextLM/2020.05.28_GPT-3.md) 在模型规模和预训练预料上进一步提升, 并涌现出了优良的上下文学习 (In-Context Learning, ICL) 能力. 在这一能力加持下, GPT-3 可以在不进行微调的情况下, 仅通过任务描述或少量示例即可完成多样化下游任务.

GPT-3 设计了多个不同参数规模的版本, 以满足不同应用场景的需求:
- GPT-3 Small: 解码模块 × 12, 隐藏层维度 768, 自注意力头 12, 总参数 1.25 亿;
- GPT-3 Medium: 解码模块 × 24, 隐藏层维度 1024, 自注意力头 16, 总参数 3.5 亿;
- GPT-3 Large: 解码模块 × 24, 隐藏层维度 1536, 自注意力头 16, 总参数 7.62 亿;
- GPT-3 XL: 解码模块 × 24, 隐藏层维度 2048, 自注意力头 24, 总参数 13 亿.
- GPT-3 2.7B: 解码模块 × 32, 隐藏层维度 2560, 自注意力头 32, 总参数 27 亿.
- GPT-3 6.7B: 解码模块 × 32, 隐藏层维度 4096, 自注意力头 32, 总参数 67 亿.
- GPT-3 13B: 解码模块 × 40, 隐藏层维度 5120, 自注意力头 40, 总参数 130 亿.
- GPT-3 175B: 解码模块 × 96, 隐藏层维度 12288, 自注意力头 96, 总参数 1750 亿.

预训练:
- 数据: 更大规模更多样化的互联网文本数据集 (1 TB), CommonCrawl + WebText + BookCorpus + Wikipedia etc
- 任务: 下一词预测

下游任务:
- GPT-3 模型涌现出良好的上下文学习能力, 使其可以在无需微调的情况下， 仅通过在输入文本中明确任务描述和提供少量示例, 便能够执行多种下游任务.

#### InstructGPT

在 GPT-3 基础上, OpenAI 进一步推出了一系列衍生模型, 通过特定的训练方法:
- Codex: 十亿行代码继续预训练 (Continual Pre-Training), 以有效处理代码生成任务;
- InstructGPT: 采用用户偏好对齐 (User Intent Alignment), 具备良好的指令跟随能力, 是 ChatGPT 的前身;

InstructGPT 引入了**人类反馈强化学习 (Reinforcement Learning from Human Feedback, RLHF)**, 显著提升了模型对用户指令的响应能力.
RLHF 旨在缓解模型在遵行用户指令时可能出现的不准确性和不可靠性, 以使模型生成的内容更符合人类的要求.
在 RLHF 中, 人类评估者首先提供关于模型输出质量的反馈, 然后使用这些反馈来微调模型:
- 有监督微调: 收集大量 "问题-人类回答" 作为训练样本, 对大语言模型进行微调;
- 训练奖励模型: 针对每个输入, 让模型生成多个候选输出, 并由人工对其进行质量评估和排名, 构成偏好数据集, 用于训练奖励模型使其可以对输出是否符合人类偏好进行打分;
- 强化学习微调: 基于上一步中得到的奖励模型, 使用强化学习方法优化监督微调的语言模型. 即模型生成输出, 奖励模型打分, 强化学习算法根据评分微调参数, 提高高质量输出的概率.

经过 RLHF 训练得到的 InstructGPT 的性能通常优于 GPT-3, 尤其是在需要精确遵循用户指令的场景中.
它生成的回答更贴合用户的查询意图, 有效减少了无关或误导性内容的生成.
但是 RLHF 的计算成本十分高昂:
- 奖励模型的训练过程复杂且耗时;
- 单独训练语言模型和奖励模型外, 需要协调这两个模型进行多模型联合训练, 该过程也复杂且耗时.

为了克服 RLHF 在计算效率上的缺陷, 斯坦福大学于 2023 年提出了[直接偏好优化 (Direct Preference Optimization, DPO)](../../Modules/RLHF/DPO.md).
DPO 直接利用人类偏好数据来训练模型, 省略了单独构建奖励模型以及应用复杂强化学习算法的步骤.
- 该方法首先收集包含多个响应的人类偏好数据, 并从中标记出最优和次优响应.
- 然后微调模型以提高模型选择最优响应的概率, 同时降低次优响应的概率.
- 显著简化了人类反馈对齐的流程, 提高了训练效率和模型稳定性. 虽然处理复杂人类偏好时可能稍逊于 RLHF, 但 DPO 在计算效率上的优势使其在多个领域上得到了广泛应用.

#### ChatGPT~GPT-4

OpenAI 于 2022 年 11 月发布了 Chat Generative Pre-trained Transformer (ChatGPT), 允许用户通过 OpenAI 提供的网页端或 API 轻松使用预训练后的模型, 无需本地部署, 标志着一种新的服务模型 LLMaaS (Language Model as a Service) 的出现.

OpenAI 于 2023 年 03 月发布了 GPT-4 模型, 在理解复杂语境, 捕捉语言细微差别, 生成连贯文本等任务上进一步提升, 并且能够更有效地处理数学问题等高级认知任务. 此外还引入了图文双模态的支持, 扩展了在图像描述和视觉问题解答等应用领域的可能性.

OpenAI 于 2024 年 05 月提出了 GPT-4o, 大幅提升了响应速度, 降低延迟还增强了多模态多语言支持.

### LLaMA 系列

**LLaMA (Large Language Model Meta AI)** 是由 Meta AI 开发的一系列大语言模型, 其模型权重在非商业许可证下向学术界开放.

LLaMA 借鉴了 GPT 系列的设计理念, 同时在技术细节上进行了创新和优化.
两者的主要区别在于:
- GPT 系列升级主线聚焦于模型规模和预训练语料的同步提升;
- LLaMA 在模型规模上保持相对稳定, 更专注于提升预训练数据的规模.

|模型|发布时间|参数规模 (亿)|预训练语料|
|:-:|:-:|:-:|:-:|
|LLaMA-1|2023.02|67/130/325/652|~5 TB|
|LLaMA-2|2023.07|70/130/340/700|~7 TB|
|LLaMA-3|2024.04|80/700|~50 TB|

#### LLaMA1 模型

[LLaMA1](../../Models/TextLM/2023.02.27_LLaMA.md) 是 MetaAI 于 2023 年 02 月推出的首个大语言模型.
其在 Chinchilla 扩展法则的指引下, 实践 "小模型+大数据" 的理念, 以大规模的优质数据训练较小的模型. 相对较小的参数规模可以获得更快的推理速度, 使其可以更好地应对计算资源有限的场景.

预训练:
- 数据: CommonCrawl, T5 提出的 C4 数据集, 来自 Github & Wikipedia & Gutenberg & Books3 & ArXiv & StackExchange 等多种来源的数据, 总数据量高达 5TB.
- 模型架构: LLaMA1 采用了与 GPT 系列同样的网络架构. 但是在 Transformer 原始词嵌入模块, 注意力模块, 全连接前馈模块上进行了优化.
  - 在词嵌入模块上, 为了提高词嵌入质量, LLaMA1 参考了 GPTNeo 的做法, 使用旋转位置编码 (Rotary Position Embedding, RoPE) 替代了原有的绝对位置编码从而增强位置编码的表达能力, 增强模型对序列顺序的理解.
  - 在注意力模块上, LLaMA1 参考了 PaLM 的做法, 将 ReLU 激活函数改为 SwiGLU 激活函数, 并且在进行自注意力操作前对 Q, K 添加旋转位置编码.
  - 在全连接前馈模块上, LLaMA1 借鉴了 GPT-3 的 Pre-Norm 层正则化策略, 将正则化应用于自注意力和前馈网络的输入. (RMS LayerNorm)

LLaMA1 模型推出了四个版本的模型:
- LLaMA1-7B: 32 解码块, 4096 隐藏层维度, 32 自注意力头, 总参数 67 亿.
- LLaMA1-13B: 40 解码块, 5120 隐藏层维度, 40 自注意力头, 总参数 130 亿.
- LLaMA1-32B: 60 解码块, 6656 隐藏层维度, 52 自注意力头, 总参数 325 亿.
- LLaMA1-65B: 80 解码块, 8192 隐藏层维度, 64 自注意力头, 总参数 652 亿.

#### LLaMA2 模型

[LLaMA2](../../Models/TextLM/2023.07.18_LLaMA2.md) 是 MetaAI 于 2023 年 07 月推出的第二代大语言模型.

LLaMA2 在 LLaMA1 的基础上
- 数据:
  - 预训练阶段进一步优化和扩充了训练数据, 将语料库规模扩展到约 7 TB, 实现了对更丰富语言和领域资源的覆盖.
  - 使用了大规模且公开的指令微调数据集对模型进行有监督微调.
  - 训练了 RLHF 奖励模型, 基于 PPO 以及拒绝采样进行强化学习进行更新.
- 模型架构: LLaMA2 继承了架构, 也推出了四个版本.
  - LLaMA2-34B 和 LLaMA2-70B 额外增加了**分组查询注意力 (Grouped Query Attention, GQA)**

LLaMA2 模型推出了四个版本的模型:
- LLaMA2-7B: 32 解码块, 4096 隐藏层维度, 32 自注意力头, 总参数 70 亿.
- LLaMA2-13B: 40 解码块, 5120 隐藏层维度, 40 自注意力头, 总参数 130 亿.
- LLaMA2-34B: 60 解码块, 6656 隐藏层维度, 52 自注意力头, 总参数 340 亿.
- LLaMA2-70B: 80 解码块, 8192 隐藏层维度, 64 自注意力头, 总参数 700 亿.

#### LLaMA3 模型

[LLaMA3](../../Models/TextLM/2024.07.31_LLaMA3.md) 是 MetaAI 于 2024 年 04 月推出的第三代大语言模型.

- 数据:
  - 挑选了规模高达 50 TB 的预训练语料, 是 LLaMA2 的 7 倍, 包含丰富的代码数据以增强模型的逻辑推理能力, 涵盖了超过 5% 的非英文数据, 覆盖 30 多种语言, 显著扩展了模型的跨语言处理能力.
  - 同样进行了 RLHF, 这一策略已被证明能显著提升模型能力.
- 模型架构: LLaMA3 几乎和 LLaMA2 几乎完全相同, 只是在分词阶段将字典长度扩大了三倍, 极大提升了推理效率.
  - 减少了中文字符等语言元素被拆分为多个 Token 的情况, 有效降低了总体 Token 数量, 从而提高了模型处理语言的连贯性和准确性.
  - 扩大字典有助于减少对具有完整意义的语义单元进行分割, 使模型在处理文本时可以更准确地捕捉语义和上下文, 提高生成文本的流畅性和连贯性.
  - LLaMA3-8B 和 LLaMA3-70B 均采用了分组查询注意力机制.

#### 衍生模型

##### 性能改进类

通过微调继续提升 LLaMA 模型的性能.
- Alpaca 基于 GPT-3.5 生成的指令遵循样例数据对 LLaMA1 进行微调, 以较小的模型规模实现了与 GPT-3.5 相媲美的性能.
- Vicuna 利用 ShareGPT 平台累积的日常对话数据微调 LLaMA1 模型, 进一步提升其对话能力.
- Guanaco 模型在微调 LLaMA1 引入 QLoRA 技术, 显著降低微调的时间成本, 提高微调效率.

##### 垂域任务类

- CodeLLaMA 模型在 LLaMA2 的基础上利用大量公开代码数据进行微调, 使其能更好地适应自动化代码生成, 错误检测, 代码优化等任务.
- LawGPT 模型通过 30w 条法律问答对 LLaMA1 进行指令微调, 显著增强了对法律内容的处理能力.
- GOAT 模型通过 Python 脚本生成的数学题库对 LLaMA1 进行微调, 提高解决各类数学题的准确率.
- Cornucopia 模型利用金融问答数据进行微调, 增强金融问答效果.

##### 多模态任务类

通过整合视觉模态编码器和跨模态对齐组件, 将 LLaMA 模型扩展到多模态任务上.
- LLaVA 在 Vicuna 基础上利用 CLIP 提取图像特征并利用一个线性投影层实现图片和文本之间的对齐.
- MiniGPT 在 Vicuna 基础上使用 VIT-G/14 以及 Q-Former 作为图像编码器, 并同样使用线性投影层来实现图片和文本之间的对齐, 展现了多模态任务处理能力.

## 2.6.非 Transformer 架构

Transformer 架构是当前大语言模型的主流模型架构, 具备构建灵活, 易并行, 易扩展等优势.
但是其**并行输入的机制会导致模型规模随输入序列长度平方增长**, 导致在处理长序列时面临计算瓶颈.

为了提高计算效率和性能, 解决 Transformer 在长序列处理中的瓶颈问题, 可以选择基于 RNN 的语言模型.
RNN 在生成输出时, 只考虑之前的隐藏状态和当前输入, 理论上可以处理无限长的序列.
然而传统 RNN 在处理长序列时可能难以捕捉到长期依赖关系, 且面临梯度消失或梯度爆炸问题.

为了克服这些问题, 近年来研究者提出了两类 RNN 变体: 状态空间模型 (State Space Model, SSM) 和测试时训练 (Test-Time Training, TTT). 这两种范式可以实现**关于训练长度的线性时间复杂度**, 且避免了传统 RNN 中存在的问题.

### 状态空间模型 SSM

状态空间模型 (State Space Model, SSM) 范式可以有效处理长文本中存在的长程依赖性 (Long-Range Dependencies, LRDs) 问题, 并有效降低模型计算和内存开销.

SSM 的思想源自于控制理论中的动力系统.
- 通过一组状态变量来捕捉系统状态随着时间的连续变化, 这种连续时间的表示方法天然地适用于描述长时间范围内的依赖关系.
- 具有递归和卷积的离散化表示形式, 既能在推理时通过递归更新高效处理序列数据, 又能在训练时通过卷积捕捉全局依赖关系.

设状态为 $x(t)\in \mathbb{C}^n$, 输入为 $u(t)\in \mathbb{C}^m$, 输出为 $y(t)\in \mathbb{C}^p$
- 状态矩阵 $A\in \mathbb{C}^{n\times n}$;
- 控制矩阵 $B\in \mathbb{C}^{n\times m}$;
- 输出矩阵 $C\in \mathbb{C}^{p\times n}$;
- 命令矩阵 $D\in \mathbb{C}^{p\times m}$;

SSM 的系统方程为:

$$
\begin{aligned}
x'(t) &= A x(t) + B u(t) \\
y(t) &= C x(t) + D u(t)
\end{aligned}
$$

- 第一个方程为状态方程, 描述了系统状态如何基于输入和前一个状态进行变化, 其计算出的是状态关于时间的导数;
- 第二个方程为输出方程, 描述了系统状态如何转化为输出. 注 $D$ 矩阵表示残差连接可以忽略.

上述方程可以视为 SSM 系统方程的连续形式, 适用于对连续数据的处理, 但是在训练和推理都非常慢.

为了提高 SSM 的处理效率, 通常需要对方程进行**离散化**操作.
该步骤将系统方程从连续形式转换为递归形式和卷积形式, 从而提升整个 SSM 架构效率.
用如梯形法等方式计算积分, 可以得到离散化后**递归 SSM**.

$$
\begin{aligned}
x_{k} &= \bar{A} x_{k-1} + \bar{B} u_k \\
y_{k} &= \bar{C} x_{k}
\end{aligned}
$$

其中:
$$
\begin{aligned}
\bar{A} &= (I-\dfrac{\Delta}{2}A)^{-1} (I+\dfrac{\Delta}{2}A)\\
\bar{B} &= (I-\dfrac{\Delta}{2}A)^{-1} \Delta B \\
\bar{C} &= C\\
\Delta &= t_{n+1}-t_n
\end{aligned}
$$

递归 SSM 类似于 RNN, 具有 RNN 的优缺点, 适用于顺序数据的处理, 能够实现与序列长度呈线性复杂度的高效推理, 但是无法并行训练, 当面临长序列时存在梯度消失或爆炸问题.

将递归 SSM 进行迭代, 可以得到卷积 SSM.

$$
\begin{aligned}
x_k = \bar{A}^k \bar{B} u_0 + \bar{A}^{k-1}\bar{B} u_1 + \cdots + \bar{B} u_k\\
y_k &= \bar{C} x_k = \bar{C}\bar{A}^k \bar{B} u_0 + \bar{C}\bar{A}^{k-1}\bar{B} u_1 + \cdots + \bar{C}\bar{B} u_k
\end{aligned}
$$

可以看到输出 $y_k$ 是状态输入 $u_k$ 的卷积结果, 卷积核为:
$$
\bar{K}_k = (\bar{C}\bar{B}, \bar{C}\bar{A}\bar{B},\cdots,\bar{C}\bar{A}^{k}\bar{B})
$$

$$
y_k = \bar{K}_k \star u_k
$$

卷积核由 SSM 的矩阵参数决定, 由于这些参数在整个序列的处理过程是固定的, 被称为时不变性. 使得 SSM 能够一致地处理不同时间步长的数据, 进行高效并行化训练.
但由于上下文长度固定, 卷积 SSM 在进行自回归任务时延迟长且计算消耗大.

结合离散 SSM 形式, 可以选择**训练时使用卷积形式, 推理时使用递归形式**.

现有的 SSM 架构之间的主要区别在于 SSM 方程的离散化方式/A 矩阵的定义.
- S4 (Structured State-Space Model) 是 SSM 的变体, 使用 HiPPO 矩阵初始化 A 矩阵, 在处理长序列数据时表现优异.

#### RWKV

![中文文档](https://www.rwkv.cn/docs)

RWKV (Recurrent Weighted Key Value) 架构是基于 SSM 范式的创新架构. 核心机制 WKV 的计算可以看作是两个 SSM 的比.

RWKV 的设计结合了 RNNs 和 Transformers 的优点, 既保留了推理阶段的高效性, 又实现了训练阶段的并行化. (RWKV-v4)

模型的核心模块有两个: 时间混合模块和通道混合模块.

- 时间混合模块主要处理序列中不同时间步之间的关系,
- 通道混合模块主要关注同一时间步内不同特征通道之间的交互.

这两个模块的设计基于四个基本元素: 接收向量 R, 键向量 K, 值向量 V, 权重向量 W.

模块中共有的操作是 Token 位移, 该步通过对当前时间步和前一时间步的输入进行线性插值, 以确保模型对序列中时间变化的敏感性.
在时间混合模块中, 接收向量 R 负责接收并整合来自序列历史的信息, 权重 W 表示位置权重衰减, 键向量 K 和值向量 V 类似注意力机制中的 K 和 V, 分别用于匹配和携带信息.
具体来说, 时间混合模块将当前步和前一步的输入进行线性组合, 通过线性投影得到 R K V 向量, 然后通过 WKV 机制确保每个通道的权重随着时间推移逐步衰减; 最后将表示过去信息的 $\sigma(R)$ 和当前信息的 $WKV$ 通过输出门控机制进行整合.

可以发现 WKV 机制是核心部分, 权重 W 是一个和通道相关的时间衰减向量, 用于捕捉时间序列中不同时间步之间的依赖关系, 实现每个通道权重随着时间向后逐步衰减的效果.

#TODO
