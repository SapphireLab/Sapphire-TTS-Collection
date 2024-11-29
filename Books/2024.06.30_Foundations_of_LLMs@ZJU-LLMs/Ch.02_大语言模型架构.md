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
