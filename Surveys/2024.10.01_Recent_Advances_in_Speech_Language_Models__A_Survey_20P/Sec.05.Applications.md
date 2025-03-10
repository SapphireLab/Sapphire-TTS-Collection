# 5.Downstream Applications: 下游应用

<table><tr><td width="50%">

Unlike traditional speech systems like ASR and TTS, which usually focus on specific tasks, SpeechLMs function as generative foundation models.
They can handle a diverse array of speech-only, text-only, and multi-modal tasks by following various instructions.
In this section, we explore the primary downstream applications of SpeechLMs.
The tasks discussed here primarily consist of traditional speech-related tasks, along with some that are unique to SpeechLMs.
In contrast to TextLMs, which generate text containing only semantic information, SpeechLMs can model both semantic and paralinguistic information---such as pitch and timbre---making them more powerful models.
We thereby categorize the downstream applications of SpeechLMs into three main classes: semantic-related applications, speaker-related applications, and paralinguistic applications.

</td><td>

和传统的语音系统如 ASR 和 TTS 通常关注特定任务不同, 语音语言模型是作为生成基础模型而存在的.
它们可以遵循各种指令来处理仅语音, 仅文本和多模态任务.
本节探讨了语音语言模型的主要下游应用.
讨论的任务主要是传统语音相关任务, 还有一些语音语言模型特有的任务.
与文本语言模型生成仅有语义信息的文本不同, 语音语言模型可以建模语义和副语言信息, 如音高和音色, 因此更为强大.
我们将语音语言模型的下游应用分为三个主要类别: 语义相关应用, 说话人相关应用, 和副语言应用.

</td></tr></table>

![](Images/Tab.04.png)

## 5.1.Semantic-Related Applications: 语义相关应用

<table><tr><td width="50%">

Semantic-related applications encompass key tasks that facilitate meaningful interactions between humans and machines.
These applications require SpeechLMs to understand the semantic meaning of the input and generate responses that are not only contextually relevant but also logically coherent.
The primary Semantic-related applications of SpeechLMs are as follows.

</td><td>

语义相关的应用涵盖了促进人机之间有意义互动的关键任务.
这些应用需要语音语言模型理解输入的语义含义, 并生成不仅上下文相关还要逻辑连贯的响应.
语音语言模型的主要语义相关应用如下.

</td></tr></table>

### Spoken Dialogue: 语音对话

<table><tr><td width="50%">

Spoken dialogue is the most natural application of SpeechLMs.
Spoken dialogue systems are designed to facilitate natural conversations between humans and machines in spoken format.
They can engage users in interactive exchanges, understanding and generating responses based on the context of the conversation.
Unlike TextLMs, SpeechLMs are able to perform conversations with humans directly in speech, which is a more natural way of communication.
Note that SpeechLMs can not only perform speech-only dialogues but also perform cross-modal dialogues, such as taking texts as input and responding in speech format.

</td><td>

语音对话时语音语言模型的最自然的应用.
语音对话系统旨在促进人机之间以口头形式进行自然对话.
它们可以参与互动交流, 基于对话上下文进行理解和生成响应.
不像文本语言模型, 语音语言模型能够直接以语音形式进行对话, 这更加自然.
需要注意的是, 语音语言模型不仅可以进行纯语音对话, 还可以进行跨模态对话, 如输入文本并以语音格式进行响应.

</td></tr></table>

### Speech Translation: 语音翻译

<table><tr><td width="50%">

Speech translation (ST) is the process of converting spoken language from one language to another.
Similar to Spoken dialogue, SpeechLMs can perform ST in both single-modal and cross-modal settings.
Specifically, the input and output of the ST task can be either in text or speech format.

</td><td>

语音翻译是将口语语言从一种语言转换为另一种语言的过程.
与语音对话类似, 语音语言模型可以进行单模态和跨模态的语音翻译.
具体来说, 语音翻译任务的输入和输出可以是文本或语音格式.

</td></tr></table>

### Automated Speech Recognition: 自动语音识别

<table><tr><td width="50%">

Automatic speech recognition (ASR) enables systems to convert spoken language into text.
The input of ASR is a speech waveform, and the system outputs the transcription in textual form.
For SpeechLMs, the input would be a combination of the speech waveform and the instruction to tell the model to perform ASR on the given speech.

</td><td>

自动语音识别使系统能够将口语语言转换为文本.
ASR 的输入是语音波形, 系统输出的文本形式的转录.
对于语音语言模型, 输入可以是语音波形和指令的组合, 告诉模型以给定的语音进行 ASR.

</td></tr></table>

### Keyword Spotting: 关键词识别

<table><tr><td width="50%">

Keyword spotting can be considered a special type of ASR, where its primary objective is to identify specific words or phrases within continuous speech.
While traditional ASR systems aim to transcribe entire spoken utterances into text, keyword spotting focuses specifically on identifying and extracting predefined keywords or phrases within continuous speech.
The primary application of keyword spotting is to build voice-activated assistants in smart home devices.
Those devices are activated when the specific keywords are triggered.
Therefore, although SpeechLMs are capable of spotting and understanding more than just a couple of words, keyword spotting can be used to efficiently trigger SpeechLMs to respond to user inputs.

</td><td>

关键词检测可以被视为一种特殊的自动语音识别, 其主要目标是识别连续语音中的特定词或短语.
传统的 ASR 系统旨在将整个口语话语转换为文本, 而关键词检测则专注于识别并提取连续语音中的预定义关键字或短语.
关键词检测的主要应用是构建智能家居设备的语音助手.
这些设备在特定关键字被触发时被激活.
因此, 尽管语音语言模型能够识别和理解比起几句话还多的词, 但关键词检测可以有效地触发语音语言模型响应用户输入.

</td></tr></table>

### Text-to-Speech Synthesis: 文本转语音合成

<table><tr><td width="50%">

Text-to-speech synthesis (TTS) enables systems to synthesize written text into spoken language.
In contrast to ASR, TTS takes text as input and outputs the converted speech waveform.
Similarly, the input of the SpeechLMs would be a combination of the text to synthesize and the instruction, and the output is the synthesized speech.

</td><td>

文本转语音合成 (TTS) 使系统能够合成文本为语音.
与 ASR 不同, TTS 接受文本作为输入, 输出转换后的语音波形.
类似地, 语音语言模型的输入可以是合成文本和指令的组合, 输出是合成的语音.

</td></tr></table>

### Intent Classification: 意图分类

<table><tr><td width="50%">

Intent classification is a critical task that identifies the underlying intention behind a user's input speech.
The AI system can then perform certain actions based on the identified user intent (e.g., book a flight).
Intent classification is particularly important in applications such as virtual assistants, customer service bots, and interactive voice response systems.
To perform Intent Classification, it is more natural for SpeechLMs to take speech inputs and classify the results in text since it is easier to parse and classify the intent classification result in text than speech.

</td><td>

意图分类是识别用户输入语音的潜在意图的关键任务.
AI 系统可以根据识别的用户意图执行某些操作 (如预订航班).
意图分类在虚拟助手, 客户服务机器人, 和交互式语音响应系统等应用中尤为重要.
为了进行意图分类, 语音语言模型更倾向于接受语音输入, 并以文本形式分类结果, 因为解析和分类文本形式的意图分类结果比在语音上更容易.

</td></tr></table>

### Slot Filling: 词槽填充

<table><tr><td width="50%">

Slot filling is an important task in spoken language understanding that involves identifying and extracting specific pieces of information from user inputs into predefined classes, such as intents, entities, and parameters that are essential for completing a task.
For example, slot filling extracts the phrase `I want to fly from New York to San Francisco on June 5th.` into distinct slots like `departure city` (New York), `destination city` (San Francisco), and `date` (June 5th).
Similar to Intent Classification, it is more natural for SpeechLMs to take speech inputs and extract the pieces in texts.

</td><td>

词槽填充是口语理解中的重要任务, 涉及从用户输入识别和提取特定信息片段到预定义的类别, 如意图, 实体和完成任务所需的参数.
例如, 词槽填充可以将 `我想从纽约飞往旧金山的日期是6月5日.` 识别为 `出发城市` (纽约), `目的城市` (旧金山), `日期` (6月5日).
与意图分类类似, 语音语言模型更倾向于接受语音输入, 提取文本形式的片段.

</td></tr></table>

### Query by Example Spoken Term Detection: 查询示例语音术语检测

<table><tr><td width="50%">

Another spoken term detection task is query by example spoken term detection (QbE-STD), which allows users to identify specific spoken terms or phrases within a larger audio stream by providing an example of the desired term.
Unlike traditional keyword spotting methods that rely on predefined lists of keywords, QbE-STD leverages the flexibility of example-based querying, enabling users to specify their search terms through audio samples.

</td><td>

另一个口语术语检测任务是基于示例的口语术语检测 (QbE-STD), 允许用户通过提供所需术语的示例, 在更大的音频流中识别特定口语术语或短语.
与依赖预定义关键词列表的传统关键词发现方法不同, QbE-STD 利用基于示例的查询的灵活性, 允许用户通过音频示例指定搜索词.

</td></tr></table>

## 5.2.Speaker-Related Applications: 说话人相关应用

<table><tr><td width="50%">

Speaker-related applications refer to the tasks that involve the processing of information related to speaker identity.
It could involve classification tasks such as identifying, verifying, and distinguishing individual speakers based on their unique vocal characteristics, as well as generation tasks such as maintaining or modifying the timbre of a given speech.
While we acknowledge that voice characteristics can be considered paralinguistic information, we believe that speaker-related applications are unique because they enable SpeechLMs to function in complex scenarios such as participating in multi-speaker conversations.
In this section, we survey common speaker-related applications of SpeechLMs.

</td><td>

与说话人相关的应用指的是涉及处理与说话人身份相关信息的任务.
这可能包括分类任务, 如基于说话人独特的语音特征进行识别, 验证和区分个体说话人, 以及生成任务, 如维持或修改给定语音的音色.
尽管我们承认语音特征可以被视为副语言信息, 但我们认为与说话人相关的应用是独特的, 因为它们使语音语言模型能够在复杂场景中发挥作用, 例如参与多方对话.
在本节中, 我们将调研语音语言模型的常见说话人相关应用.

</td></tr></table>

### Speaker Identification: 说话人识别

<table><tr><td width="50%">

Speaker identification is the process of recognizing a person's identity based on their voice characteristics.
It is a multi-class classification of a given speech as input.
SpeechLMs can perform this task by taking an input speech and outputting the classification result in text or speech format.
Moreover, SpeechLMs can also identify different speakers implicitly.
Specifically, it can chat with multiple speakers at the same time, distinguishing the words from different speakers and responding to each speaker appropriately.

</td><td>

说话人识别是根据说话人的语音特征识别其身份的过程.
它是对给定语音进行多类分类的任务.
语音语言模型可以通过接受输入语音并以文本或语音格式输出分类结果来执行此任务.
此外, 语音语言模型还可以隐式地识别不同说话人.
具体来说, 它可以与多个说话人进行聊天, 区分不同说话人的词汇, 并相应地回应每个说话人.

</td></tr></table>

### Speaker Verification: 说话人验证

<table><tr><td width="50%">

Speaker verification involves determining whether the speakers of a pair of speeches match with each other.
Unlike speaker identification, which is a multi-class classification process, speaker verification is a binary classification process.

</td><td>

说话人验证涉及确定一对语音的说话人是否匹配.
与多类分类过程的说话人识别不同, 说话人验证是二分类过程.

</td></tr></table>

### Speaker Diarization: 说话人分割

<table><tr><td width="50%">

Speaker diarization is the process of partitioning an audio stream into segments according to the identity of the speakers.
It predicts "who is speaking when" for each timestamp ([SUPERB (2021)](../../Evaluations/2021.05.03_SUPERB.md)).
A natural way to integrate speaker diarization into SpeechLMs is to have the model generate the transcript of each audio segment along with the identification of the speaker.

</td><td>

说话人分割是将音频流根据说话人的身份划分为不同段落的过程.
它预测每个时间戳的 "谁在说话" ([SUPERB (2021)](../../Evaluations/2021.05.03_SUPERB.md)).
将说话人分割集成到语音语言模型的自然方式是, 模型生成每个音频段的转录, 并识别说话人的身份.

</td></tr></table>

### Voice-Conditioned Speech Generation: 声学条件下的语音生成

<table><tr><td width="50%">

Voice-conditioned speech generation involves synthesizing speech based on the vocal characteristics of a specific speaker.
This could involve voice cloning and voice conversion.
Voice cloning utilizes a sample of the speaker's voice as a reference, enabling the model to reproduce the speaker's timbre when generating speech from input text.
Voice conversion, on the other hand, modifies an existing speech signal to sound like it was produced by a different speaker while retaining the original content.
Additionally, instead of giving the target vocal characteristics, SpeechLMs should also be able to adapt their output timbre based on various speech or text instructions.

</td><td>

基于声音条件的语音生成涉及根据特定说话人的语音特征合成语音.
这包括语音克隆和语音转换.
语音克隆利用说话人的声音样本作为参考, 使模型能够在输入文本生成语音时复现说话人的音色.
另一方面, 语音转换修改现有的语音信号, 使其听起来像是由不同说话人产生的, 同时保留原始内容.
此外, 与给定目标声学特征不同, 语音语言模型应该也能够根据各种语音或文本指令调整输出音色.

</td></tr></table>

## 5.3.Paralinguistic Applications: 副语言应用

<table><tr><td width="50%">

Paralinguistics refers to the non-verbal elements of communication that accompany spoken language.
It encompasses various vocal attributes that convey meaning beyond the actual words spoken.
These elements can significantly influence how messages are interpreted and understood.
The key elements of paralinguistics include pitch, timbre, column, rate of speech, pauses, etc.
Since combining elements in paralinguistics in different ways can result in a speech with different emotions, we include emotion-related tasks as paralinguistic applications as well.

</td><td>

副语言指的是交流中伴随口语的非语言元素.
它包括传达超出实际说话内容的多种语音属性.
这些元素可以显著影响信息的解读和理解.
副语言的关键元素包括音高, 音色, 音量, 语速, 停顿等.
由于以不同方式组合副语言学中的元素可以产生带有不同情感的语音, 因此我们也将与情感相关的任务视为副语言应用.

</td></tr></table>

### Emotion Recognition: 情感识别

<table><tr><td width="50%">

Emotion recognition task involves identifying and classifying the emotion carried by a given speech into predefined classes.
Similar to speaker identification, SpeechLMs are capable of not only directly performing this task but also implicitly recognizing users' emotions through their speech queries and responding accordingly.

</td><td>

情感识别任务涉及识别并将给定语音所携带的情感分类到预定义的类别中.
与说话人识别类似, 语音语言模型不仅能够直接执行此任务, 还能够通过用户的语音查询隐式识别用户的情感, 并相应地做出回应.

</td></tr></table>

### Speech Separation: 语音分离

<table><tr><td width="50%">

Speech separation refers to the process of isolating individual speech signals from a mixture of sounds, such as when multiple speakers are talking simultaneously.
When separating the input speech, SpeechLMs can not only output the contents of each person in speech but also in text format (i.e., transcriptions).

</td><td>

语音分离指的是从混合声音中分离出单独的语音信号的过程, 例如当多个说话人同时说话时.
在分离输入语音时, 语音语言模型不仅可以输出每个说话人的语音内容, 还可以以文本格式输出 (即转录文本).

</td></tr></table>

### Paralinguistics-Enhanced Generation: 副语言增强生成

<table><tr><td width="50%">

Paralinguistics-enhanced generation refers to the process of instructing SpeechLMs to produce speech that exhibits specific paralinguistic characteristics.
Users can define these characteristics in their prompts, allowing the model to generate speech that aligns with their specifications.
Examples of paralinguistics-enhanced generation include synthesizing speech with a specific style, speaking at a fast pace, and even singing.
This capability distinguishes SpeechLMs from TextLMs and facilitates a more engaging and interactive form of communication with the AI models.

</td><td>

副语言增强生成指的是知道语音语言模型生成具有特定副语言特征的语音的过程.
用户可以在提示中定义这些特征, 使模型生成符合其规格的语音.
副语言增强生成的例子包括合成具有特定风格的语音, 以快速语速说话, 甚至唱歌.
这一能力使得语音语言模型区别于文本语言模型, 并促进了与人工智能模型更引人入胜和互动形式的交流.

</td></tr></table>
