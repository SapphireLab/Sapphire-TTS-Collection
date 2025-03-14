# 5·Streaming, Duplex, and Interaction

<table><tr><td width="50%">

Streaming, full-duplex technology, and interactions, are crucial elements for enhancing the interactive capabilities of spoken dialogue models because they directly impact the system's responsiveness, the fluidity of natural interaction, and its ability to handle complex interactions.
Unlike text language models, spoken dialogue models require real-time processing of user input.
Streaming allows the system to instantly acquire and process speech data; full-duplex technology enables both the system and user to speak simultaneously, enhancing the naturalness of interaction; and handling of interactions provides the model with the ability to recognize and adapt to various conversational contexts, making the dialogue more intelligent and realistic.
Building on early explorations, GPT-4o's advanced spoken dialogue capabilities have ignited a surge of research interest.
With real-time voice processing and natural conversational interaction, these models offer users a seamless and efficient communication experience.
However, achieving these capabilities requires deep research into model architecture, data collection, system design, and training methods.
The model needs to be carefully designed and optimized in terms of real-time performance, stability, and response speed.
At the same time, duplex technology is an indispensable key implementation, which ensures that the voice model has both "ears" and "mouths".

Next, we will first discuss the streaming processing method in Section 5.1, then introduce the key technologies of duplex communication and explains how to handle interaction to improve user experience in Section 5.2.

</td><td>

流式, 完全双工技术, 和交互, 是增强口语对话模型的交互能力的关键元素, 因为它们直接影响了系统的响应能力, 自然交互的流畅度, 以及处理复杂交互的能力.
和文本语言模型不同, 口语对话模型要求实时处理用户输入.
- **流式**允许系统即时获取和处理语音数据;
- **完全双工技术**使系统和用户可以同时发言, 增强了交互的自然性;
- **交互处理**为模型提供了识别和适应各种会话上下文的能力, 使对话更智能和真实.

建立在早期探索的基础之上, GPT-4o 的先进口语对话能力点燃了研究的热潮.

结合实时声音处理和自然对话交互, 这些模型为用户提供了无缝且高效的沟通体验.
然而, 实现这些能力需要深入研究模型架构, 数据收集, 系统设计和训练方法.
这些模型需要仔细设计和优化, 以确保实时性能, 稳定性, 响应速度.

同时, 双工技术是不可或缺的关键实现, 确保声音模型同时具有 "耳朵" 和 "嘴巴".

下面, 我们将在 5.1 节首先讨论流式处理方法;
然后, 在 5.2 节介绍双工通信的关键技术, 并解释如何处理交互以提高用户体验.

</td></tr></table>

## 5.1·Streaming Spoken Dialogue Models: 流式口语对话模型

<table><tr><td width="50%">

The core of streaming speech models lies in their "real-time" and "continuous" capabilities, meaning they can process input and generate output simultaneously without waiting for complete input.
This includes two main aspects:
- **Streaming Understanding**.
The model can process audio input as the user speaks, without needing to wait for the user to finish entirely, allowing it to align more naturally with the flow of conversation.
- **Streaming Generation**.
This concept refers to the model's ability to generate output without waiting for all intermediate hidden states.
Instead, it can produce output progressively as processing occurs, which improves responsiveness and allows for smoother, more efficient interactions.

These streaming capabilities allow the model to perform more fluidly in real-time interactions, providing a seamless communication experience for users.
We will explore streaming techniques in both end-to-end and cascaded spoken dialogue models, discussing the implementation methods of streaming in each system and highlighting their similarities and differences.

</td><td>

流式语音模型的核心在于其 "实时" 和 "连续" 的能力, 这意味着它们可以同时处理输入和生成输出, 而不需要等待完整的输入.

这包含两个主要方面:
- **流式理解**: 模型可以处理用户说话的音频输入, 不需要等待用户完全说完, 使其更自然地与对话流对齐.
- **流式生成**: 这一概念指的是模型的生成输出的能力, 不需要等待所有中间隐藏状态. 它可以随着处理的进行逐步产生输出, 提高响应实现更丝滑更有效的交互.

这些流式能力使得模型在实时交互中表现得更流畅, 为用户提供无缝的交流体验.

我们将探索端到端和级联口语对话模型中的流式技术, 探讨流式在每种系统中的实现方法, 并强调它们的相似和不同之处.

</td></tr></table>

### 5.1.1·Streaming End-to-End Spoken Dialogue Models: 流式端到端口语对话模型

<table><tr><td width="50%">

End-to-end streaming spoken dialogue models often leverage the knowledge of pre-trained text language models alongside an audio tokenizer, employing an tokenizer-detokenizer architecture to process and output audio signals.
Based on the concepts of streaming input and output discussed above, end-to-end models also require specific design considerations to enable streaming capabilities.
These designs center around the model’s input and output handling and can be distilled into three core techniques: causal convolution, causal attention mechanisms, and queue management.

</td><td>

端到端流式口语对话模型通常利用了预训练文本语言模型的知识和音频分词器, 采用分词器-反分词器架构处理和输出音频信号.
基于上面讨论的流式输入和输出的概念, 端到端模型还要求具体的设计考虑, 以启用流式能力.
这些设计以模型的输入和输出处理为中心, 可以总结为三个核心技术:
- 因果卷积 (Causal Convolution)
- 因果注意力机制 (Causal Attention Mechanisms)
- 队列管理 (Queue Management)

</td></tr></table>

#### Causal Convolution: 因果卷积

<table><tr><td width="50%">

[Causal Convolution [12]](../../Models/_Basis/2018.03.04_TCN.md) is a specialized form of convolution widely used in time-series processing, especially suitable for streaming speech models.
The key feature of causal convolution is that the current output depends only on the current and past inputs, without being influenced by future inputs, thereby strictly respecting temporal order.
Unlike regular convolution, causal convolution achieves this by "shifting" the convolution kernel to avoid accessing future information.
In a one-dimensional time series, if the convolution kernel size is $k$, a standard convolution would use data from $(t - k/2)$ to $(t + k/2)$ at the current time step $t$.
Causal convolution, however, pads the input on the left with $k-1$ zeros so that the kernel only uses data from $t - k + 1$ to $t$, aligning the kernel to only consider current and past inputs.
This padding ensures that each layer's output depends solely on current and prior information, maintaining causality.
To further expand the model’s receptive field while preserving causality, dilated causal convolution can be used.
This technique introduces gaps within the kernel by inserting zeros between weights, effectively expanding the convolution’s range.
This allows the model to capture longer dependencies in the data without increasing latency, which is particularly useful for streaming applications.
In streaming spoken dialogue models, causal convolution plays a critical role in:
- **Ensuring real-time processing**.
Causal convolution allows the model to compute outputs without accessing future frames, enabling real-time processing by generating outputs as input is received, which is essential for streaming.
- **Reducing latency**.
By not requiring future input data, causal convolution significantly lowers the latency in speech models, making it more suitable for real-time interaction applications, such as voice assistants and live translation.

</td><td>

[因果卷积 [12]](../../Models/_Basis/2018.03.04_TCN.md)是时间序列处理中广泛使用的卷积的一种特殊形式, 特别适用流式语音模型.
因果卷积的关键特征是当前输出仅依赖于当前和过去输入, 而不受未来输入的影响, 因此严格遵循时间顺序.
和常规卷积不同, 因果卷积通过 "移动" 卷积核来避免访问未来信息以实现这一点.

在一维时间序列中, 如果卷积核大小为 $k$, 则标准卷积将在当前时间步 $t$ 使用 $(t - k/2)$ 到 $(t + k/2)$ 之间的数据.
因果卷积对输入左填充 $k-1$ 个零, 因此卷积核仅使用 $t - k + 1$ 到 $t$ 之间的数据, 使得卷积核仅考虑当前和过去输入.
这种填充确保每个层的输出仅依赖于当前和先前信息, 保持因果性.

为了在保持因果性的同时扩展模型的感受野, 可以使用**膨胀因果卷积 (Dilated Causal Convolution)**.
这一技术通过在权重之间插入零来在卷积核中引入间隙, 有效扩展了卷积的范围.
这使得模型在不增加延迟的情况下捕获更长的依赖关系, 特别适用于流式应用.

在流式口语对话模型中, 因果卷积扮演着至关重要的角色:
- **确保实时处理**: 因果卷积允许模型在不访问未来帧的情况下计算输出, 实现实时处理, 当输入接收时就能生成输出, 这对于流式至关重要.
- **降低延迟**: 由于不需要未来输入数据, 因果卷积在语音模型中显著降低延迟, 适用于如语音助手和实时翻译等实时交互应用.

</td></tr></table>

#### Causal Attention: 因果注意力

<table><tr><td width="50%">

Causal Attention is a specialized form of the attention mechanism designed to ensure that each position in a sequence can only attend to previous positions, thus preserving the temporal order crucial for streaming models.
This approach ensures that the model’s current output depends only on past and present information, preventing any “leakage” of future information, which is essential for real-time processing tasks.
In causal attention, the attention mask is typically used to achieve causality.
By applying a mask that blocks connections to future time steps, the model restricts each token’s receptive field to only the tokens before it.
Specifically, a lower triangular mask is applied to the attention matrix, setting values to negative infinity for positions corresponding to future tokens.
This masking technique ensures that the model’s predictions for each time step only consider current and past inputs, thereby adhering to a strict causal structure.
In streaming speech models, causal attention plays a significant role in enabling real-time interaction.
Unlike standard attention, which requires access to the entire sequence, causal attention can operate incrementally.
As new inputs are processed, the model can generate outputs without waiting for future context.

</td><td>

因果注意力是注意力机制的一种特殊形式, 设计用于确保序列中的每个位置智能关注之前的位置, 因此保留时序对于流式模型至关重要.
这种方法确保了模型当前输出仅依赖于过去和现在信息, 阻止未来信息的任何"泄露", 这对于实时处理任务很重要.

在因果注意力中, 注意力掩码通常用于实现因果性.
通过应用阻塞到未来时间步的连接的掩码, 模型将每个 Token 的感受野限制为仅前面的 Token.
具体来说, 会应用下三角掩码到注意力矩阵, 将对应于未来 Token 的位置的值设置为负无穷.
这种掩码技术确保模型对每个时间步的预测仅考虑当前和过去输入, 因此遵循严格的因果结构.

在流式语音模型中, 因果注意力是实现实时交互的关键要素.
和标准注意力要求访问整个序列不同, 因果注意力可以增量地运行.
当新输入被处理时, 模型可以生成输出而无需等待未来上下文.

</td></tr></table>

#### [Queue Management: 队列管理 [220]](../../Models/SpeechCodec/2023.05.26_AudioDec.md)

<table><tr><td width="50%">

Audio streams are typically split into frames, then processed in sequence via a queue management system that ensures real-time, orderly processing.

Some end-to-end models, such as [LLaMA-Omni [57]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md), [Mini-Omni [222]](../../Models/SpokenDialogue/2024.08.27_Mini-Omni.md) and [Mini-Omni2 [223]](../../Models/SpokenDialogue/2024.10.15_Mini-Omni2.md), employ non-streaming ASR model Whisper as an audio encoder components.
These models have made improvements on the output side to reduce latency.
- **Mini-Omni**.
Mini-Omni use a generation strategy delayed parallel decoding is a that layer-by-layer delays during audio token generation.
This allows the model to generate text and multiple audio tokens simultaneously at each step, accelerating streaming audio generation and ensuring low-latency real-time output.
- **Llama-Omni**.
Llama-Omni incorporates a non-autoregressive streaming speech decoder that leverages connectionist temporal classification (CTC) to directly generate a sequence of discrete audio tokens as the response.
- [IntrinsicVoice [248]](../../Models/SpokenDialogue/2024.10.09_IntrinsicVoice.md)
IntrinsicVoice introduced GroupFormer module to group speech tokens, reducing the length of speech sequences to match that of text sequences.
This approach accelerates inference, alleviates the challenges of long-sequence modeling, and effectively narrows the gap between speech and text modalities.We think they cannot be considered fully streaming because they are not designed to be streaming on the input side.
- [Moshi [44]](../../Models/SpokenDialogue/2024.09.17_Moshi.md)
In contrast, Moshi references the architecture of SpeechTokenizer to train a streaming codec from scratch, serving as the audio tokenizer-detokenizer.
The entire model, including the codec, transformer, and attention mechanism, is built on a causal structure.
- [OmniFlatten [246]](../../Models/SpokenDialogue/2024.10.23_OmniFlatten.md)
OmniFlatten proposes chunk-based processing of text and speech along with gradual learning techniques and data handling to reduce turn-taking delays, such as response delays when users finish speaking or interrupt the system.
These models have achieved true streaming capabilities and established a foundation for diverse, bidirectional interactions.

</td><td>

音频流通常被分割成帧, 然后以序列形式通过队列管理系统确保实时有序的处理.

一些端到端模型, 例如 [LLaMA-Omni [57]](../../Models/SpokenDialogue/2024.09.10_LLaMA-Omni.md), [Mini-Omni [222]](../../Models/SpokenDialogue/2024.08.27_Mini-Omni.md) 和 [Mini-Omni2 [223]](../../Models/SpokenDialogue/2024.10.15_Mini-Omni2.md), 采用非流式 ASR 模型 Whisper 作为音频编码器组件.
这些模型在输出段上进行了改进, 以减少延迟.
- **Mini-Omni**: Mini-Omni 使用延迟并行解码策略, 在音频 Token 生成时逐层延迟.
这种策略允许模型同时生成文本和多个音频 Token, 加快流式音频生成速度, 并确保低延迟实时输出.
- **Llama-Omni**: Llama-Omni 融合了非自回归流式语音解码器, 利用**连接时序分类 (CTC)** 直接生成音频 Token 序列作为响应.
- [IntrinsicVoice [248]](../../Models/SpokenDialogue/2024.10.09_IntrinsicVoice.md): IntrinsicVoice 引入 GroupFormer 模块, 将语音 Token 进行分组, 减少语音序列的长度与文本序列的长度匹配.
这种方法加快推理速度, 缓解长序列建模的挑战, 有效缩小语音和文本模态之间的差距.

我们认为它们不能被认为是完全流式的, 因为它们没有在输入端设计为流式.

- [Moshi [44]](../../Models/SpokenDialogue/2024.09.17_Moshi.md): 相比之下, Moshi 参考 SpeechTokenizer 的架构, 重新训练一个流式编解码器, 作为音频 Tokenizer-DeTokenizer.
  整个模型, 包括编解码器, Transformer, 和注意力机制, 都建立在因果结构之上.
- [OmniFlatten [246]](../../Models/SpokenDialogue/2024.10.23_OmniFlatten.md): OmniFlatten 提出基于块的文本和语音处理, 并采用渐进学习技术和数据处理, 以减少轮次交换延迟, 例如用户完成说话或中断系统时的响应延迟.

这些模型实现了真正的流式能力, 并为多样化的双向交互奠定了基础.

</td></tr></table>

### 5.1.2·Streaming Cascaded Spoken Dialogue Models: 流式级联口语对话模型

<table><tr><td width="50%">

Consistent with the above, ensuring streaming capability in a model relies on designing both input and output for streaming.
Due to its cascaded nature, a cascaded model typically relies on external streaming ASR and TTS components, placing the streaming responsibility on these ASR and TTS modules.

In [Wang et al. [211]](../../Models/_Full/2024.05.29_A_Full-Duplex_Speech_Dialogue_Scheme_Based_on_Large_Language_Models.md), comparative studies were conducted on the streaming ASR model [U2++ Conformer](../../Models/ASR/2021.06.10_U2++.md), streaming TTS model [XTTS-v2 [21]](../../Models/SpeechLM/2024.06.07_XTTS.md), non-streaming ASR model **Whisper**, and non-streaming TTS model [VITS2 [109]](../../Models/E2E/2023.07.31_VITS2.md).
The combination of streaming components achieved the lowest latency and significantly contributed to interactive interruption capabilities.

</td><td>

和前文一致, 确保模型中的流式能力依赖于设计流式输入和输出.
由于模型的级联性质, 级联模型通常依赖于外部的流式 ASR 和 TTS 组件, 将流式责任放在这些 ASR 和 TTS 模块上.

在 [Wang et al. [211]](../../Models/_Full/2024.05.29_A_Full-Duplex_Speech_Dialogue_Scheme_Based_on_Large_Language_Models.md), 对流式 ASR 模型 [U2++ Conformer [219]](../../Models/ASR/2021.06.10_U2++.md), 流式 TTS 模型 [XTTS-v2 [21]](../../Models/SpeechLM/2024.06.07_XTTS.md), 非流式 ASR 模型 Whisper, 非流式 TTS 模型 VITS ([VITS2 [109]](../../Models/E2E/2023.07.31_VITS2.md)) 进行了比较研究.

流式组件的组合实现了最低延迟, 并为交互中断能力提供了显著贡献.

</td></tr></table>

## 5.2·Duplex Technology and Interaction: 双工技术和交互

### 5.2.1·Duplex Technology: 双工技术

<table><tr><td width="50%">

The term Duplex originates from the field of communications, used to describe interaction modes between two parties in data transmission.
Depending on the type of communication, duplex is divided into half-duplex and full-duplex.

With the development of audio processing and generation technology , the concept of duplex has been introduced to speech systems, especially within the context of speech language models.
Here, duplex doesn’t just refer to signal transmission but emphasizes the synchronization and natural interaction in human-computer dialogue.
Specifically, within model architecture, it means that the model must retain its ability to perceive external input even while generating a response---essentially, the ability to listen while speaking.

</td><td>

双工 (Duplex) 一词源自通信领域, 用于描述数据传输双方之间的交互模式.
根据通信类型, 双工可以分为半双工和全双工.

随着音频处理和生成技术的发展, 双工的概念被引入语音系统, 特别是在语音语言模型的语境中.
此时双工不只是指代信号传输, 还强调了人机对话中的同步和自然互动.
具体来说, 在模型架构中, 它意味着模型必须保持其接收外部输入的能力甚至是在生成响应时, 也就是在说话的时候保持监听的能力.

</td></tr></table>

![](Images/Fig.08.png)

#### Simplex: 单工

<table><tr><td width="50%">

In simplex communication, data flows in only one direction.
The speaker can send data, while the listener can only receive it.
As shown in Figure 08(a), the robot continuously transmits audio, while the user has no ability to respond.
This fixed-direction, one-way communication has the limitation of lacking interactivity.

</td><td>

在单工通信中, 数据仅在一个方向流动.
说话人可以发送数据, 但听众只能接收它.

如图 08(a) 所示, 机器人连续地传输音频, 而用户却无能力作出反应.
这种固定方向, 单向的通信有着缺乏互动性的局限性.

</td></tr></table>

#### Half-Duplex: 半双工

<table><tr><td width="50%">

In half-duplex communication, data flows in both directions but not simultaneously.
The two parties must take turns speaking and listening.
As illustrated in Figure 08(b), the user speaks first, followed by a response delay during which the robot "thinks" before replying.
The robot’s response occurs only after the user has finished speaking, and vice versa.
This turn-taking method is similar to using a walkie-talkie, where each party can only transmit after the other has finished, limiting efficiency.Half-duplex is a common mode in early voice interaction systems.
In a typical half-duplex interaction, there are noticeable pauses in the conversation; the user and the system cannot “speak” simultaneously, making the conversation feel less smooth, much like communication through a walkie-talkie.
For example, voice assistants like Siri use wake words or button presses to trigger the dialogue and require the speaker to finish a complete sentence before responding.
These systems typically adopt an ASR-LM-TTS cascaded structure and are often constrained by cascade delays and the turn-based nature of text language models.
Although this interaction method is simple and easy to implement, it can feel rigid and disjointed in natural conversational settings, with notable latency.
It is designed more for command execution rather than interactive communication.

</td><td>

在半双工通信中, 数据在两个方向流动但不能同时进行. 双方必须轮流说话和听.

如图 08(b) 所示, 用户首先说话, 随后机器人会在响应延迟期间 "思考" 再回复.
机器人的回复仅在用户完成说话后发生, 反之亦然.

这种轮流的方法类似于使用步话机 (一种小型便携式无线电通信设备), 每一方只能在对方完成后才能发声, 效率受限.

半双工是早期声音交互系统的常用模式.
在典型的半双工交互中, 对话中会出现明显的停顿; 用户和系统不能同时发声, 使得对话感觉不够流畅, 就像通过步话机进行通信一样.
例如, 语音助手 Siri 采用唤醒词或按键触发对话, 要求说话者在完成完整句子后才回复.
这些系统通常采用 ASR-LM-TTS 级联结构, 并受到级联延迟和基于文本语言模型的轮流性质的限制.

尽管这种交互方法简单且易于实现, 但在自然对话设置下, 它可能感觉僵硬和分离, 带来明显的延迟.
它主要为命令执行而设计, 而不是交互通信.

</td></tr></table>

#### Full-Duplex: 全双工

<table><tr><td width="50%">

Full-duplex communication allows both parties to send and receive data simultaneously ([LSLM [142]](../../Models/SpokenDialogue/2024.08.05_LSLM.md)).
Figure 08(c) shows the user and robot engaging in overlapping, real-time interaction, where backchannels and interruptions are possible.
This mode enables a natural, two-way conversation, where both the user and robot can speak, respond, and even interrupt each other as needed, much like a phone call.In dialogue systems, full-duplex means that the system and user can speak simultaneously and interrupt each other, making it closer to natural conversation in real life.
Full-duplex large voice models allow the system not only to listen and understand the user while they speak but also to interrupt at appropriate moments or respond with backchannel cues.
Moreover, the system can detect the user’s intent to interrupt and pause itself accordingly, maintaining a smooth flow in the interaction.

</td><td>

全双工通信允许双方同时发送和接收数据 ([LSLM [142]](../../Models/SpokenDialogue/2024.08.05_LSLM.md)).
图 08(c) 展示了用户和机器人在重叠, 实时互动中进行交流, 反向通道和中断都是可以的.
这种模式使得双方可以进行自然的双向对话, 即用户和机器人可以同时说话, 回应, 甚至可以互相打断, 就像打电话一样.

在对话系统中, 全双工意味着系统和用户可以同时说话, 并可以互相打断, 这使得它更接近于现实生活中的自然对话.
全双工的大型声音模型允许系统不仅能在说话时听取并理解用户, 还能在适当的时候打断, 并以反向通道的形式回应.
此外, 系统还可以检测到用户的打断意图, 并相应地暂停自己, 保持对话的流畅.

</td></tr></table>

#### Summary: 小结

<table><tr><td width="50%">

The ultimate goal of a spoken dialogue model is to make the user feel as though they are conversing with a real human friend.
Clearly, full-duplex technology is essential for achieving natural voice dialogue systems, enabling the system to send and receive audio signals simultaneously, thus facilitating real-time interaction.
Unlike text-based models, it doesn’t “cover its ears” while speaking.
Users and intelligent agents can interrupt each other while listening or express their attitude through non-verbal signals, such as interjections or laughter.
The challenges in realizing this lie in ensuring conversational fluidity, seamless turn-taking, and precise timing of interactions.
Developing a full-duplex system that can both generate and receive voice signals in complex interactive scenarios remains a key focus in academic and industrial research.

</td><td>

口语对话模型的最终目标是让用户感觉自己正在和真正的人类朋友进行对话.
很明显, 全双工技术是实现自然语音对话系统的必要条件, 它使得系统能够同时发送和接收音频信号, 这有助于实现实时互动.
与基于文本的模型不同, 在说话时它并不会 "掩耳盗铃".
用户和智能体可以在听取或通过非语言信号 (如感叹或笑声) 表达它们的态度时打断对方.

实现这一目标的挑战在于确保对话流畅, 无缝轮流, 以及对话的精准时机.
开发能够在复杂的交互场景中生成和接收语音信号的全双工系统, 仍然是学术和工业研究的关键关注点.

</td></tr></table>

### 5.2.2·Interaction: 交互

<table><tr><td width="50%">

Now that we understand duplex technology, we can further explore duplex spoken dialogue model.

We start with some concept.

Turn-taking is the core concept in duplex dialogue.
It refers to the process in which speakers take turns speaking in an orderly manner during a conversation, forming a pattern of turn-taking.
Over the past few decades and has been extensively studied across fields such as linguistics, phonetics, and sociology.
Some research ([Raux et al [173]](../../Models/_Full/A_Finite-State_Turn-Taking_Model_for_Spoken_Dialogue_Systems.md); [Sacks et al [180]](../../Models/_Full/A_Simplest_Systematics_for_the_Organization_of_Turn-Taking_for_Conversation.md)) uses a non-deterministic finite-state machine with six states to describe the turn-taking behavior between the system and the user in a spoken dialogue system (SDS).
It outlines all possible states of turn-taking within an SDS, defining the objective of turn-taking as minimizing mutual silence or overlap between interlocutors, thereby improving communication efficiency.
Turn-taking encompasses three fundamental concepts:

- Turn-taking cues ([Duncan et al [53]](../../Models/_Full/Some_Signals_and_Rules_for_Taking_Speaking_Turns_in_Conversations.md); [Duncan et al [54]](../../Models/_Full/On_Signalling_that_It's_Your_Turn_to_Speak.md)).
These include voice, rhythm, breathing, gaze, or gestures.
Agents can use these cues to determine whether to take a turn from the user or to relinquish the turn.

- Turn-end detection or prediction
The distinction between detection ([Hara et al [73]](../../Models/_Full/Turn-Taking_Prediction_Based_on_Detection_of_Transition_Relevance_Place.md); [Lala et al [115]](../../Models/_Full/Attentive_Listening_System_with_Backchanneling_Response_Generation_and_Flexible_Turn-Taking.md)) and prediction ([TurnGPT [55]](../../Models/_Full/TurnGPT.md); [Lala et al [114]](../../Models/_Full/Smooth_Turn-Taking_by_a_Robot_Using_an_Online_Continuous_Model_to_Generate_Turn-Taking_Cues.md)) lies in that detection determines whether the agent should take a turn at the current moment, whereas prediction decides when the turn-taking should occur in the future.

- Overlap
This mainly involves two situations.
When the user and agent’s voices overlap, if the user intends to take the turn from the agent, this behavior is defined as an **interruption** ([Khouzaimi et al [103]](../../Models/_Full/Reinforcement_Learning_for_Turn-Taking_Management_in_Incremental_Spoken_Dialogue_Systems.md); [Marge et al. [146]](../../Models/_Full/Spoken_Language_Interaction_with_Robots__Recommendations_for_Future_Research.md)).
If the user has no intention of taking the turn, this behavior is considered backchannel ([Hara et al [72]](../../Models/_Full/Prediction_of_Turn-Taking_Using_Multitask_Learning_with_Prediction_of_Backchannels_and_Fillers.md)) or a listener response, such as "uh-huh," "right."

Through these concepts, we can better understand turn-taking behavior in duplex dialogues.
In summary, our interactions with voice dialogue systems can be categorized as interruptions, backchannels, and normal turn exchanges.

The earliest full-duplex systems used a simple Voice Activity Detection (VAD) component to model whether the user intended to interrupt.
However, this approach is inadequate for handling backchannel interaction forms, leading to frequent interruptions and introducing considerable delays.

We can briefly categorize the exploration of interactions into cascaded systems and end-to-end systems based on duplex technology.
Regardless of the system type, the critical core idea is that the system must continuously track external information in real-time, analyze it, and determine the model’s operational state accordingly.
An interactive voice system must meet two requirements:
1) The ability to accept external information in real-time at any moment.
2) The ability to respond to this information accurately.

This includes:

- Detecting User Interactions
When the user tries to interject or provide new information, the system can recognize this intent and immediately stop its output to allow the user to speak.
- Backchanneling During User Speech
While the user is speaking, the system can provide brief acknowledgments like "uh-huh" or "I see" to indicate active listening, which encourages the user to continue.
- Quickly Responding After User Completion
When the user finishes speaking, the system can promptly recognize this cue and respond without unnecessary delays, maintaining a smooth conversational flow.
- Handling Pauses in User Speech
When the user briefly pauses, the system can interpret this as a moment of thought rather than an invitation to respond, thus avoiding premature interruptions and preserving the natural flow.
- Interrupting the User When Necessary
In situations where the system detects critical information, it can choose to interrupt the user to provide immediate feedback.
For example, if the user is speaking but the system needs to alert them to an error, it can intervene in real-time to ensure effective communication.

</td><td>

现在我们理解了双工技术, 可以进一步探索双工模式口语对话模型.

我们从一些概念开始.

**轮次交换/轮流 (Turn-Taking)** 是双工对话中的核心概念. 它指的是说话人在对话中按照有序的方式轮流说话的过程, 形成轮次交换的模式.
在过去几十年中, 轮次交换已经在语言学, 语音学, 社会学等领域广泛研究.
- 一些研究 ([Raux et al [173]](../../Models/_Full/A_Finite-State_Turn-Taking_Model_for_Spoken_Dialogue_Systems.md); [Sacks et al [180]](../../Models/_Full/A_Simplest_Systematics_for_the_Organization_of_Turn-Taking_for_Conversation.md)) 使用具有六个状态的非确定性有限状态机来描述口语对话系统 (SDS) 中用户和系统之间的轮次交换行为.
它概述了 SDS 中所有可能的轮次交换状态, 定义了轮次交换的目标, 即在交流双方之间最小化相互静默或重叠, 从而提高交流效率.

轮次交换包含三个基本概念:
- **轮次交换提示 (Turn-Taking Cues)** ([Duncan et al [53]](../../Models/_Full/Some_Signals_and_Rules_for_Taking_Speaking_Turns_in_Conversations.md); [Duncan et al [54]](../../Models/_Full/On_Signalling_that_It's_Your_Turn_to_Speak.md))
  这些包括声音, 节奏, 呼吸, 视线或手势. 智能体可以使用这些提示来判断是否应该从用户那里获得轮次.
- **轮次结束检测或预测 (Turn-End Detection or Prediction)**.
  - **检测** ([Hara et al [73]](../../Models/_Full/Turn-Taking_Prediction_Based_on_Detection_of_Transition_Relevance_Place.md); [Lala et al [115]](../../Models/_Full/Attentive_Listening_System_with_Backchanneling_Response_Generation_and_Flexible_Turn-Taking.md)) 决定智能体是否应在当前时刻获得轮次,
  - **预测** ([TurnGPT [55]](../../Models/_Full/TurnGPT.md); [Lala et al [114]](../../Models/_Full/Smooth_Turn-Taking_by_a_Robot_Using_an_Online_Continuous_Model_to_Generate_Turn-Taking_Cues.md)) 决定轮次交换何时发生.
- **重叠**. 主要涉及两种情况. 当用户和智能体的声音重叠时,
  - **中断 (Interruption)**: 用户打算从智能体获得轮次 ([Khouzaimi et al [103]](../../Models/_Full/Reinforcement_Learning_for_Turn-Taking_Management_in_Incremental_Spoken_Dialogue_Systems.md); [Marge et al. [146]](../../Models/_Full/Spoken_Language_Interaction_with_Robots__Recommendations_for_Future_Research.md)).
  - **反向通道 (Backchannel)/听众反应 (Listener Response)**: 用户没有打算获得轮次, 如 "uh-huh" 或 "right". ([Hara et al [72]](../../Models/_Full/Prediction_of_Turn-Taking_Using_Multitask_Learning_with_Prediction_of_Backchannels_and_Fillers.md))

通过这些概念, 我们可以更好地理解双工对话中的轮次交换行为.
总而言之, 与基于文本的对话系统相比, 我们的与语音对话系统的互动可以分为中断, 反向通道和正常轮次交换.

</td></tr></table>

#### Cascaded Systems: 级联系统

<table><tr><td width="50%">

To enable interactive functionality, cascaded spoken dialogue models typically require explicit modeling of dialogue turns.
As the core, the large language model needs effective context and turn management.
Next, we introduce several representative works on interaction in cascaded systems.

</td><td>

为了实现交互功能, 级联口语对话模型通常要求显式建模对话轮次.
作为核心, 大语音模型需要有效的内容和轮次管理.
接下来, 我们介绍级联系统中的几个代表性工作.

</td></tr></table>

##### Duplex Conversation: 双工对话

<table><tr><td width="50%">

In [Lin et al [130]](../../Models/_Full/Duplex_Conversation__Towards_Human-Like_Interaction_in_Spoken_Dialogue_Systems.md), three core modules are proposed to achieve smooth full-duplex dialogue: user state detection, response signal selection, and interruption detection.
The user state detection module not only focuses on traditional turn-end detection but also identifies whether the user intends to switch turns, continue speaking, or hesitates during their speech.
To achieve this, the system uses a multimodal model, taking audio and text as inputs, and incorporates features such as speech rhythm, pitch, and pauses for more accurate assessment of the user’s state, determining whether to respond immediately or wait longer.
The response signal selection module inserts small backchannel cues (such as "uh-huh" or "right") at appropriate times to simulate natural human conversation.
By analyzing a large volume of real dialogues, this module extracts and trains suitable response signals for various conversation scenarios.
Using multi-label classification, the system selects the optimal response for each dialogue context, significantly reducing user waiting time and enhancing conversation flow.
The interruption detection module flexibly responds to user interruptions.
Unlike traditional rule-based detection methods, this system builds an end-to-end detection model with multimodal input (audio and text) that not only identifies genuine user interruptions but also avoids misinterpreting background noise or unintended voice signals as interruptions.

</td><td>

[Lin et al [130]](../../Models/_Full/Duplex_Conversation__Towards_Human-Like_Interaction_in_Spoken_Dialogue_Systems.md) 提出了三个核心模块, 以实现平滑的全双工对话: 用户状态检测, 响应信号选择, 和中断检测.
- 用户状态检测模块: 不仅关注传统的轮次结束检测, 还可以识别用户是否打算切换轮次, 继续说话或是在说话时犹豫. 为了实现这一点, 系统使用多模态模型, 接收音频和文本作为输入, 并整合诸如语音节奏, 音高和停顿等特征以更精确地评估用户状态, 确定是否立即响应或等待更长时间.
- 响应信号选择模块: 在合适的时机插入小的反向通道提示 (如 "嗯" 或 "对") 以模拟自然人类对话. 通过分析大量真实对话, 该模块提取并训练适合各种对话场景的响应信号. 使用多标签分类, 系统为每个对话上下文选择最优的响应, 显著减少用户等待时间并增强了对话流动.
- 终端检测模块: 灵活响应用户中断. 和传统的基于规则的检测方法不同, 该系统构建了具有多模态输入 (音频和文本) 的端到端检测模型, 既可以识别真正的用户中断, 也避免误解背景噪声或意外语音信号为中断.

</td></tr></table>

##### Outbound Agent System: 外呼智能体系统

<table><tr><td width="50%">

[Jin et al [98]](../../Models/_Full/Duplex_Conversation_in_Outbound_Agent_System.md) proposed a full-duplex dialogue scheme for outbound systems, focusing on the issues of conversational fluidity and timing of interaction in speech dialogue.
This scheme uses semantic analysis to determine whether the user truly intends to interrupt the system and can handle disjointed expressions when users mention named entities.
The core of this system is a full-duplex interaction finite-state machine (FSM), which retrieves text snippets from ASR results every 300 milliseconds to decide whether to interrupt.
Through continuous semantic analysis of user speech, the interruption model identifies meaningful user interruptions and avoids frequent interruptions caused by brief, meaningless responses (like "uh-huh").
The model employs a pre-trained BERT-based text classifier and utilizes streaming input, ensuring that the system can process and analyze user speech in real-time as it is received.
Additionally, the system includes a Discontinuous Expression module to handle user pauses when mentioning named entities.
Specifically, when users hesitate over entities (such as numbers, locations, or company names), VAD may erroneously detect turn-end.

</td><td>

[Jin et al [98]](../../Models/_Full/Duplex_Conversation_in_Outbound_Agent_System.md) 提出了用于外呼系统的全双工对话方案, 着重于语音对话中的对话流畅性和交互时机.
该方案使用语义分析来确定用户是否真的打算中断系统, 并在用户提到命名实体是可以处理杂乱的表达.
该系统的核心是全双工交互有限状态机, 每隔三百毫秒从 ASR 结果中检索文本片段以决定是否中断.
通过对用户语音的连续语义分析, 中断模型可以识别有意义的用户中断并避免由简短无意义响应 (如 "嗯") 引起的频繁中断.
该模型采用基于预训练 BERT 的文本分类器, 并使用流式输入, 确保系统可以实时接收和分析用户语音.
此外, 该系统还包含了不连续表达模块, 用于处理当提到命名实体时的用户停顿.
具体来说, 当用户在提到实体时犹豫 (如数字, 位置或公司名称), VAD 可能错误地检测到轮次结束.

</td></tr></table>

##### Transition: 过渡段落

<table><tr><td width="50%">

The advent of Large Language Models has significantly advanced generative AI development.
Models like ChatGPT demonstrate strong capabilities in semantic understanding and logical reasoning, offering a simplified method to integrate various dialogue components into a unified framework, which may simplify SDS construction.
GPT-4o represents a milestone for dialogue systems, showcasing a nearly human-like conversational voice model.
Its flexible interaction style and interruption mechanisms make human-computer interaction more natural and fluid.
However, as a commercial model, its training data and implementation details remain proprietary, making replication challenging.

</td><td>

大语言模型的出现显著推动了生成式人工智能的发展.
模型如 ChatGPT 展示了在语义理解和逻辑推理方面的强大能力, 提供了一种简化的方法将各种对话组件集成到统一框架中, 这可能简化 SDS 构建.
GPT-4o 代表对话系统的里程碑, 展示了一种与人类十分相似的对话语音模型.
其灵活的交互方式和中断机制使人机交互更加自然和流畅.
然而, 作为商业模型, 其训练数据和实现细节仍然是专有, 使得复制具有挑战性.

</td></tr></table>

##### Full-Duplex LLM: 全双工大语言模型

<table><tr><td width="50%">

[Wang et al [211]](../../Models/_Full/A_Full-Duplex_Speech_Dialogue_Scheme_Based_on_Large_Language_Models.md) proposed a full-duplex spoken dialogue models based on LLMs, enabling simultaneous reception and transmission of voice signals through a perception module, an action module, and a neural finite-state machine (FSM).
The perception module uses a streaming ASR model, capturing and processing user speech in real-time with 640-millisecond intervals per time step, converting it into token inputs for the LLM.
The action module, utilizing a streaming TTS model, instantly converts the LLM-generated text into audio output and can pause or resume playback as needed, ensuring the system can generate audio while receiving user input.
At the core is the neural FSM, allowing the LLM to switch between "speaking" and "listening" states.
Controlled by FSM signals, the system can dynamically decide to continue speaking, listen, or interrupt based on the dialogue context.
Experimental results show that Wang et al.'s full-duplex streaming system reduces response latency by threefold, achieves a response time within 500 milliseconds in over 50\% of dialogues, and handles user interruptions at a rate of 96.7\%, with an interruption accuracy of 54.7\%.

</td><td>

[Wang et al [211]](../../Models/_Full/A_Full-Duplex_Speech_Dialogue_Scheme_Based_on_Large_Language_Models.md) 提出了一个基于大语言模型的全双工口语对话模型, 通过感知模块, 动作模块和神经有限状态机 (FSM) 实现了同时接收和传输语音信号.
- 感知模块: 使用流式 ASR 模型, 实时捕获和处理用户语音, 每隔 640 毫秒的时间步长转化为 LLM 的 Token 输入.
- 动作模块: 使用流式 TTS 模型, 即时将 LLM 生成的文本转换为音频输出, 并根据需要暂停或恢复播放, 确保系统在生成音频时可以接收用户输入.
- 神经有限状态机: 作为核心, 使得 LLM 可以在说话和听取状态之间切换. 由 FSM 信号控制, 系统可以根据对话内容动态地决定是否继续说话, 听取或中断.
- 实验结果表明该全双工流式系统通过三方面减少响应延迟, 能在 50% 以上的对话中实现响应时间在 500 毫秒内, 并以 96.7% 的比率处理用户中断, 而中断准确率为 54.7%.

</td></tr></table>

##### VITA

<table><tr><td width="50%">

VITA is an open-source multimodal large language model which aimed at enhancing multimodal interaction experiences.
VITA can process multiple modalities, such as video, image, text, and audio, and achieves fluid human-computer interaction through a new duplex architecture involving two simultaneously operating models: one for generating responses to user queries, and another for continuously monitoring environmental inputs.
When a new user query is detected, the generation model pauses, and the monitoring model processes the new query and generates an updated response.
This setup enables VITA to support audio interruption, allowing users to ask new questions during system generation, with the system immediately pausing the current response to handle new input.
VITA’s perception abilities are achieved through multimodal alignment and instruction fine-tuning, enabling it to switch automatically between different inputs.
Additionally, VITA employs state tokens to distinguish user input types, such as query audio, background noise, and text input, facilitating wake-free interaction.
VITA's enhanced listening module prevents unnecessary user feedback from interrupting system responses, improving robustness.

</td><td>

VITA 是一个开源的多模态大语言模型, 旨在增强多模态交互体验.
VITA 可以处理多模态输入, 如视频, 图像, 文本, 音频, 并通过涉及两个同时运作的模型的新式双工架构实现流畅的人机交互.
- 一个用于根据用户查询生成响应;
- 另一个用于持续监测环境输入.

当检测到新的用户查询, 生成模型暂停, 监测模型将处理新查询并生成更新的响应.
这一设置使得 VITA 可以支持音频中断, 允许用户在系统生成期间提出新的问题, 系统立即暂停当前响应来处理新的输入.

VITA 的感知能力是通过多模态对齐和指令微调实现的, 使得它能够在不同的输入间自动切换.
此外, VITA 采用状态 Token 来区分用户输入类型, 例如查询音频, 背景噪声, 文本输入等, 有助于无唤醒交互.
VITA 增强的听取模块防止不必要的用户反馈中断用户响应, 提升鲁棒性.

</td></tr></table>

##### [CleanS2S [159]](../../Models/_tmp/CleanS2S.md)

<table><tr><td width="50%">

This model employs a structured pipeline to enable responsive and flexible interactions in a spoken dialogue setting.
Designed to facilitate seamless turn-taking and interruption handling, the model consists of several interconnected modules working in a coordinated sequence to optimize user experience.
Starting with user input, the system uses a Voice Activity Detection (VAD) module to continuously monitor for incoming audio signals.
As soon as a user starts speaking, VAD captures the input and immediately initiates processing by sending the audio data to the Automatic Speech Recognition (ASR) module.
This quick detection and response setup allows the system to react to user input without delay.
Once ASR transcribes the audio into text, the transcription is passed to the Large Language Model (LLM), which generates a relevant response based on the user’s query.
Meanwhile, the model is designed to be interruption-aware.
During response generation, if VAD detects a new user input (indicating an interruption or a follow-up query), the system can promptly adjust its processing flow.
In this case, the LLM temporarily pauses its current task, allowing ASR to transcribe the new input, which the LLM then uses to generate an updated response.
This interruption capability is achieved through the model’s layered processing design, allowing for adaptive turn-taking that feels natural and responsive.
The Text-to-Speech (TTS) module then converts the generated text response into audio, which is transmitted to the user via WebSocket.
To further support interruption handling, TTS breaks down lengthy responses into smaller audio segments that are sent progressively.
This segmentation allows the system to stop audio output instantly if an interruption occurs, switching to the new input without delay.
Each segment is prepared and sent only after a brief VAD check, ensuring that the system is ready to pause and handle new input at any time.
This interconnected processing chain—VAD detecting input, ASR transcribing, LLM generating responses, and TTS outputting segmented audio—creates a duplex interaction framework that balances response generation and user-driven interruptions.
By seamlessly coordinating these components, the model provides a fluid, real-time dialogue experience that adapts to user interactions dynamically.

</td><td>

该模型采用结构化流水线实现在口语对话设置下具有响应性和灵活性的交互.
为了促进无缝轮次切换和中断处理, 模型由数个互连模块组成, 以协调的顺序工作以优化用户体验.

- 从用户输入开始, 系统使用语音活动检测 (VAD) 模块来连续检测传入的音频信号.
一旦用户开始说话, VAD 就会捕获输入并立刻启动处理, 将音频数据发送到自动语音识别 (ASR) 模块进行处理.
快速检测和响应设置使得系统能够无延迟地响应用户输入.
- 当 ASR 将音频转写为文本, 转录结果传给 LLM, 基于用户查询生成相关响应.
- 同时, 模型被设计为可感知中断. 在响应生成过程中, 若 VAD 检测到新的用户输入 (表示中断或跟随查询), 系统可以迅速调整其处理流程. 在这种情况下, LLM 暂时停止当前任务, 允许 ASR 转录新的输入, 然后 LLM 使用它来生成更新的响应.
  中断能力是通过模型的分层处理设计实现的, 允许自适应轮次交换, 使得交互感觉自然而流畅.
- 然后 TTS 模块将生成的文本响应转换为音频, 并通过 WebSocket 传输给用户.
  为了进一步支持中断处理, TTS 将长响应分解为较小的音频片段, 并逐步发送.
  这种分段方式允许系统在中断发生时立即停止音频输出, 无延迟地切换到新的输入.
  每个音频分段准备好后, 仅在短暂地 VAD 检查后发送, 确保用户随时中断并处理新的输入.

这种互连处理链: VAD 检测输入 + ASR 转写 + LLM 生成响应 + TTS 输出分段音频, 构成了一个双工交互框架, 能够平衡响应生成和用户驱动的中断.
通过无缝地协调这些组件, 模型提供了流畅实时的对话体验, 能够根据用户交互动态调整.

</td></tr></table>

#### End-to-End Systems: 端到端系统

<table><tr><td width="50%">

In contrast, end-to-end spoken dialogue models do not require explicit modeling of dialogue turns; instead, they learn interaction modeling directly from training data.
Next, we introduce several representative works on interaction in end-to-end systems.

</td><td>

与级联系统不同, 端到端口语对话模型不需要显式建模对话轮次, 它们直接从训练数据学习交互建模.
接下来, 我们介绍数个端到端系统在交互方面的代表性工作.

</td></tr></table>

##### dGSLM

<table><tr><td width="50%">

In end-to-end systems, the introduction of the dGSLM model marks a significant milestone in full-duplex technology development.
Within the dGSLM framework, duplex technology is effectively implemented.
This model demonstrates how to capture complex interactions within dialogues directly from raw audio data through generative spoken dialogue modeling, without relying on text.
The core innovation of dGSLM is the dual-tower Transformer architecture, called the Dialogue Transformer Language Model (DLM), which uses a cross-attention mechanism to enable the system to process two parallel audio channels simultaneously.
Through this architecture, the model not only independently generates speech for each channel but also shares information between channels using cross-attention, effectively modeling silences and interaction events.
It leverages the HuBERT encoder and HiFi-GAN decoder, combined with the dual-tower DLM, and is trained on 2,000 hours of dual-channel telephone conversation audio (Fisher dataset), where each speaker in a conversation is allocated an independent audio track.
The dGSLM model transforms the audio on both channels into discrete tokens using HuBERT, and the DLM model autoregressively predicts the next audio token and its duration.
Finally, the [HiFi-GAN [108]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md) decoder reconstructs the audio for both channels.
This approach differs significantly from traditional text-dependent spoken dialogue models, with a particular emphasis on modeling turn-taking and backchanneling capabilities.
This capability gives dGSLM a notable advantage in duplex voice interaction, better mimicking the natural dynamics of human conversation.
Through its duplex model design, dGSLM represents an essential step forward in interactive capabilities and provides a foundation for further advancements.

</td><td>

在端到端系统中, dGSLM 模型的出现是全双工技术发展的一个重要里程碑.
在 dGSLM 框架中, 全双工技术被有效实现.

模型展示了如何通过生成式口语对话建模而无需文本来直接从原始音频数据捕获对话中的复杂互动.
dGSLM 的核心创新是双塔 Transformer 架构, 称为**对话 Transformer 语言模型 (DLM)**, 使用交叉注意力机制来使得系统同时处理两个并行的音频通道.
通过这一架构, 模型不仅为每个通道独立生成语音还使用交叉注意力来共享通道间的信息, 有效地建模静默和交互事件.

它使用了 HuBERT 编码器和 HiFi-GAN 解码器, 和双塔 DLM 组合, 然后在两千小时的双通道电话对话音频 (Fisher 数据集) 上训练, 对话中的每个说话人被分配到单独的音轨上.
dGSLM 将通道上的音频使用 HuBERT 转换为离散 Token, 然后 DLM 自回归地预测下一个音频 Token 和时长.
最后, [HiFi-GAN [108]](../../Models/Vocoder/2020.10.12_HiFi-GAN.md) 解码器重构了两个通道上的音频.

这种方法与传统的依赖文本的口语对话模型有很大不同, 特别强调了建模对话轮次和反向通道能力.
这种能力给 dGSLM 在双工声音交互方面带来了显著优势, 能够更好地模拟人类对话的自然动态.
通过其双工模型设计, dGSLM 代表向交互能力迈出了重要的一步, 为进一步发展奠定了基础.

</td></tr></table>

##### Moshi

<table><tr><td width="50%">

As a novel full-duplex architecture, Moshi incorporates a rich array of design concepts.
Unlike dGSLM, Moshi does not abandon the language model’s ability in text dialogue.
Moshi’s architecture is based on the Helium language model and Mimi neural audio codec, both trained from scratch.
Helium, as a large pre-trained text language model, provides strong reasoning capabilities, while Mimi handles audio signal encoding and decoding.
To achieve real-time interaction, Moshi is designed as a multi-stream architecture, simultaneously processing "user" and "moshi" audio streams without explicitly modeling speaker turns.
Moshi also introduces the "Inner Monologue" method within the "moshi" audio stream, a process that jointly models text and audio tokens during training and inference.
This approach allows the model to fully utilize textual knowledge while maintaining speech-to-speech system characteristics, significantly enhancing generation quality.
Mimi, a neural audio codec integrating semantic and acoustic information through residual vector quantization and knowledge distillation, captures high-quality user input audio and Moshi’s output voice efficiently.
To jointly model Moshi and user audio streams alongside Moshi’s text tokens, Depth Transformer with streaming inference capabilities is employed.
The Mimi encoder and decoder combine convolutional and Transformer layers, with causal convolutions, allowing for streaming operation.
Moshi is pre-trained on unsupervised audio data to handle speech scenarios and then fine-tuned on the Fisher dataset to address overlapping speech and interruptions.
Finally, the system is further optimized on a custom instruction-tuning dataset, ensuring robust performance across various interactive scenarios.
Experimental results show that Moshi excels in speech modeling and spoken QA tasks, especially in latency, achieving a theoretical latency of 160 milliseconds and 200 milliseconds in practice, significantly lower than the typical 230 milliseconds in natural conversation, enhancing real-time interaction and conversation flow.

</td><td>

作为新式全双工架构, Moshi 整合了丰富的设计概念.
和 dGSLM 不同, Moshi 并未放弃语言模型在文本对话方面的能力.
Moshi 的架构基于 Helium 语言模型和 Mimi 神经音频编解码器, 两者都从零开始训练.
Helium 作为预训练的大型文本语言模型, 提供了强力的推理能力, 而 Mimi 处理音频信号的编码和解码.

为了实现实时交互, Moshi 被设计为多流架构, 同时处理用户和 Moshi 音频流, 而不显式建模说话人轮次.
Moshi 在 Moshi 音频流上还引入了 "Inner Monologue (内心独白)" 方法, 一种在训练和推理时联合建模文本和音频 Token 的过程.
该方法允许模型充分利用文本知识, 同时保持语音到语音的系统特性, 显著增强生成质量.

Mimi 是一个神经音频编解码器, 通过残差向量量化和知识蒸馏融合了语义和声学信息, 有效地捕获高质量的用户输入音频和 Moshi 的输出声音.
为了联合建模 Moshi 和用户音频流以及 Moshi 文本 Token, 采用了具备流式推理能力的 Depth Transformer.
Mimi 编码器和解码器将卷积和 Transformer 层组合, 采用因果卷积, 实现流式操作.

Moshi 在无监督音频数据上预训练以处理语音场景, 然后在 Fisher 数据集上微调以处理重叠语音和中断.
最后, 系统进一步在定制的指令微调数据集上优化, 确保在各种交互场景下保持鲁棒性能.

实验结果表明 Moshi 在语音建模和口语 QA 任务表现卓越, 特别是在延迟上, 理论延迟为 160 毫秒, 实际延迟为 200 毫秒, 显著低于自然对话的通常延迟 230 毫秒, 增强了实时交互和对话流动.

</td></tr></table>

##### Parrot

<table><tr><td width="50%">

[Parrot [148]](../../Models/SpeechLM/Parrot.md) model incorporates multiple features specifically designed to enhance interaction in spoken dialogue.
It uses a dual-channel audio setup, where each channel represents a different speaker.
This configuration allows Parrot to manage both sides of a conversation independently, facilitating real-time turn-taking.
By distinguishing between the user’s input and the system’s response on separate channels, the model can listen and respond in parallel, creating a more natural conversational flow.
To handle simultaneous speaker inputs effectively, Parrot employs a "next-token-pair prediction" mechanism, allowing it to predict tokens for both channels in a coordinated sequence.
This approach helps the model manage conversational dynamics such as overlapping speech and smooth transitions between turns, adjusting response timing based on the user’s input.
During inference, Parrot supports streaming input, enabling continuous processing of user audio on one channel while generating responses on the other.
This streaming capability allows the model to respond to live spoken input in real-time, handling turn-taking, pauses, and interruptions dynamically.
Unlike cascaded systems that rely on intermediate text conversions, Parrot processes audio directly, reducing latency and allowing immediate responses to spoken input.
These interaction-focused design choices make Parrot highly responsive, enabling it to manage turn-taking naturally, respond to interruptions, and handle overlapping speech.

</td><td>

[Parrot [148]](../../Models/SpeechLM/Parrot.md) 模型整合了特别设计的多个特性以增强口语对话中的交互性.
它使用了双通道音频设置, 每个通道表示不同的说话人.
这种配置允许 Parrot 单独管理对话双方, 实现实时轮次交换.
通过区分单独通道上用户输入和系统响应, 模型可以并行地听取和响应, 创造更自然的对话流.
为了有效地处理同时的发言人输入, Parrot 采用了 "下一个 Token 对预测" 机制, 允许它在协调序列中预测两个通道的 Token.
这种方法帮助模型管理对话动态, 如重叠语音和平滑的轮次切换, 基于用户输入调整响应时间.
在推理时, Parrot 支持流式输入, 确保用户音频在一个通道上的连续处理, 而另一个通道生成响应.
这种流式能力使得模型能够实时响应实况口语输入, 动态地处理轮次交换, 暂停和中断.
和级联系统依赖中间文本对话不同, Parrot 直接处理音频, 减少延迟并能对口语输入立即响应.
这些着重于交互的设计选择使得 Parrot 具有高度响应性, 能够自然地管理轮次交换, 应对中断, 并处理重叠语音.

#TODO: 信息量不大的段落.

</td></tr></table>

##### Mini-Omni2

<table><tr><td width="50%">

Mini-Omni2 is an open-source multimodal large language model aimed at simulating the multimodal capabilities of GPT-4o in vision, hearing, and text, supporting real-time full-duplex interaction.
Mini-Omni2 combines visual and audio encoders with a language model to enable simultaneous input and output of images, audio, and text.
The model incorporates an interrupt mechanism based on instruction design for more flexible user interactions.
This system uses a delayed parallel generation algorithm, allowing the model to generate text and audio responses simultaneously, greatly improving conversational real-time capabilities and response speed.
To achieve full-duplex interaction, Mini-Omni2 introduces an interrupt mechanism based on a limited instruction approach, trained on a specially constructed dataset with specific irq (interrupt) and n-irq (non-interrupt) state markers for model optimization.
For training Mini-Omni2’s interruption functionality, the researchers used noisy speech data synthesized with specific command phrases (such as "Stop Omni") in various voices and tones to simulate scenarios where users might issue interrupt commands.
The dataset also includes background noises, such as environmental sounds, music, and other dialogues, enhancing the model’s robustness in complex environments.
During training, Mini-Omni2 controls output flow through irq and n-irq state markers, generating these markers in real-time to determine whether to continue output.
In this way, the model can immediately halt generation based on user instructions and switch to "listening" mode in real-time dialogue.
The training data consists of long audio streams from which the model extracts and encodes user commands like "Stop Omni."
Researchers inserted interrupt commands at various time points, marking data after the insertion point as irq (interrupt) and data before as n-irq (non-interrupt).
This labeling method ensures that the model learns to accurately identify interrupt commands in complex audio inputs and respond appropriately.

</td><td>

Mini-Omni2 是一个开源的多模态大语言模型, 旨在模拟 GPT-4o 在视觉听觉和文本上的多模态能力并支持实时全双工交互.

Mini-Omni2 将视觉和音频编码器和语言模型组合并确保图像音频文本的同时输入和输出.
模型整合了基于指令设计的中断机制, 以便更灵活的用户交互.
该系统使用延迟并行生成算法, 允许模型同时生成文本和音频响应, 极大提升了对话实时能力和响应速度.
为了实现全双工交互, Mini-Omni2 引入基于有限指令方法的中断机制, 在特别构建的数据集上训练, 该数据集包含特定的 irq (中断) 和 n-irq (非中断) 状态标记, 以优化模型.

为了训练 Mini-Omni2 的中断功能, 研究人员使用了合成的带有特定命令短语 (如 "停止 Omni") 的噪声语音数据, 在各种声音和音调的场景中模拟用户可能发出中断命令的场景.
数据集还包括背景噪声, 如环境音, 音乐和其他对话, 增强模型在复杂环境中的鲁棒性.

在训练时, Mini-Omni2 通过 irq 和 n-irq 状态标记控制输出流, 以实时生成这些标记以决定是否继续输出.
通过这种方式, 模型可以在实时对话中基于用户指令立即中断生成并切换到 "监听" 模式.
训练数据由长音频流组成, 模型从中提取并编码用户命令, 如 "停止 Omni".
研究人员在不同时间点插入中断命令, 将插入点后数据标记为 irq (中断), 插入点前数据标记为 n-irq (非中断).
这种标记方法确保模型学习在复杂音频输入中准确识别中断命令并相应地作出响应.

</td></tr></table>

##### SyncLLM

<table><tr><td width="50%">

SyncLLM achieves full-duplex dialogue and interruption capabilities through multi-stream interleaving and chunk processing.
SyncLLM divides the conversation's audio stream into fixed-sized chunks, each corresponding to a specific time interval.
The model alternates between generating user and system speech segments within each time step (chunk), ensuring real-time system responses while processing user speech input.
To maintain temporal synchronization with the user, SyncLLM predicts the user’s speech at each time step before generating each system chunk, using it as context to infer the system’s next response.
This mechanism enables the system to keep pace with the conversation even with network latency.
The chunk method allows SyncLLM to handle both user and system audio streams simultaneously, supporting complex dialogue features like speech overlap, interruption, and real-time feedback.
Additionally, by using de-duplicated speech token sequences and periodic synchronization markers, the model efficiently performs chunk-level real-time inference, making conversation more fluid and natural.

</td><td>

SyncLLM 通过多流交错和分块处理来实现全双工对话和中断能力.
SyncLLM 将对话音频流分割成固定大小的块, 每块对应特定的时间间隔.
模型在每个时间步 (块) 内交替生成用户和系统语音段, 确保实时系统响应的同时处理用户语音输入.
为了保持和用户的时序同步, SyncLLM 在每个时间步时在生成每个系统块之前预测用户语音, 并作为内容用于推理系统的下一个响应.
这种机制使得系统能够在网络延迟下保持对话的节奏.
分块方法允许 SyncLLM 同时处理用户和系统音频流, 支持复杂对话特性如语音重叠, 中断, 实时反馈.
此外, 通过使用去重的语音 Token 序列和周期同步标记, 模型有效地执行块级实时推理, 使对话更加流畅自然.

</td></tr></table>

##### OmniFlatten

<table><tr><td width="50%">

Similar to SyncLLM, the OmniFlatten model achieves full-duplex and interruption functionality primarily through multi-stream data processing and progressive training.
To enable full-duplex dialogue, the model adopts a multi-stream architecture that interleaves the user’s speech stream with the assistant’s speech and text streams into a single sequence for training, simplifying multimodal modeling and enhancing real-time capability.
The model first aligns the text language model with modality through multitask supervised fine-tuning, enabling it to understand and generate both speech and text, ensuring basic capability for handling speech and text simultaneously.
Through a progressive training process, OmniFlatten attains full-duplex capability in three stages: initial training for half-duplex dialogue, then removing the user’s text stream to support real-time prediction with multi-stream data, and finally removing the assistant’s text stream to enable pure speech stream generation.
These steps reduce reliance on text and decrease latency, allowing the system to generate voice responses while receiving user speech input.
By using a block-by-block generation strategy, OmniFlatten divides the input and output speech sequences into fixed-size blocks, processing each segment in turn.
This effectively implements streaming processing, ensuring low latency and high responsiveness in full-duplex dialogue, thereby providing a more natural response to user interruptions.

</td><td>

类似于 SyncLLM, OmniFlatten 模型通过多流数据处理和渐进训练实现全双工和中断功能.
为了实现全双工对话, 模型采用多流架构, 将用户语音流, 助手语音和文本流交错到单个序列中用于训练, 简化多模态建模并增强实时能力.
模型首先通过多任务监督微调来对齐文本语言模型和模态, 使其能够理解和生成语音和文本, 确保基础能力能够同时处理语音和文本.
通过渐进训练过程, OmniFlatten 在三个阶段实现全双工能力:
- 初始训练实现半双工对话,
- 移除用户文本流, 用多流数据支持实时预测,
- 移除助手文本流, 实现纯语音流生成.

这些步骤减少了对文本的依赖并降低延迟, 允许系统生成语音响应的同时接收用户语音输入.
通过使用逐块生成策略, OmniFlatten 将输入和输出语音序列划分为固定大小的块, 逐个处理每个片段.
这有效地实现了流式处理, 确保全双工对话中的低延迟和高响应, 从而为用户中断提供更自然的响应.

</td></tr></table>

##### Freeze-Omni

<table><tr><td width="50%">

To support duplex dialogue, [Freeze-Omni [213]](../../Models/SpokenDialogue/2024.11.01_Freeze-Omni.md) uses a chunk-level state prediction mechanism for natural turn-taking.
When the user begins speaking, a voice activity detection module identifies the audio input, prompting the model to process the audio chunk by chunk.
After processing each chunk, the model's classification layer predicts the conversation state to determine the next action.
There are three possible states: State 0, where the model continues listening for more input, assuming the user hasn’t completed their turn; State 1, where the model interrupts to provide an immediate response if a quick acknowledgment or feedback is needed; and State 2, where the model has completed processing the current user input and is ready to generate and output a response, thus transitioning smoothly into the response phase without further listening.
This chunk-wise state prediction enables the model to decide effectively when to respond and when to continue listening, enhancing its ability to handle natural conversational cues and support interactive dialogue.

</td><td>

为了支持双工对话, [Freeze-Omni [213]](../../Models/SpokenDialogue/2024.11.01_Freeze-Omni.md) 使用块级状态预测机制来实现自然轮次切换.
当用户开始说话时, 语音活动检测模块识别音频输入, 提示模型按块处理音频.
在处理每个块后, 模型的分类层预测对话状态, 以确定下一步动作.
在三个可能的状态:
- 状态 0: 模型继续监听更多输入, 假设用户还未完成他们的轮次;
- 状态 1: 模型中断以提供即时响应, 如果需要快速确认或反馈;
- 状态 2: 模型完成处理当前用户输入, 准备生成并输出响应, 因此无需再继续监听, 转变到响应阶段.

这种块级状态预测使得模型能有效地决定何时响应, 何时继续监听, 增强其处理自然对话线索和支持交互式对话的能力.

</td></tr></table>

### 5.2.3·Discussions about Streaming and Interaction: 流式和交互的讨论

<table><tr><td width="50%">

Significant progress has been made in dialogues models, particularly in real-time interaction and semantic understanding, with notable achievements in streaming processing and full-duplex interaction.
Current systems exhibit strong technical capabilities in reducing response latency, enhancing interruption handling, and improving the naturalness of conversation.
However, existing spoken dialogues models still lack a unified system that can handle all forms of interaction seamlessly.
Future research could explore new frameworks to better manage both user interruptions and the system’s ability to interrupt users, making interactions more natural.
Additionally, standardized benchmarks for evaluating interaction capabilities remain underdeveloped.
A unified evaluation benchmark would provide a consistent method for assessing and comparing the performance of different models, thereby advancing the development of more intelligent and responsive interaction systems.

</td><td>

对话模型取得了显著的进步, 尤其是在实时交互和语义理解, 在流式处理和全双工交互方面取得了重要成果.
现有的系统在减少响应延迟, 增强中断处理和提升对话自然度方面表现出强大的技术能力.
然而, 现有的口语对话模型仍然缺乏统一系统以无缝处理所有形式的交互.
未来研究可以探索新的框架来更好地管理用户中断和系统终端用户的能力, 使交互更自然.
此外, 评估交互能力的标准化基准仍未开发.
统一的评估基准可以为不同模型提供一致的方法以评估和比较性能, 从而促进更智能和响应的交互系统的发展.

</td></tr></table>
