# 6·Challenges and Future Directions: 挑战与未来方向

In this section, we elaborate on current challenges for fully controllable TTS and discuss promising future directions.

## A·Challenges: 挑战

Controllable TTS aims to synthesize speech while allowing precise control over speech characteristics such as pitch, duration, energy, prosody, speaking style, and emotion.
While significant progress has been made, achieving truly controllable TTS remains a complex task due to the multifaceted nature of human speech and the technical challenges in modeling and synthesizing it.
In this section, we delve into the primary challenges and analyze their underlying reasons.

### Controllable Granularity: 控制粒度

A critical challenge in controllable TTS is determining what aspects of speech should be controlled and how to control speech characteristics at a specific granularity.
Different applications require varying levels of control granularity.
For instance, audiobook narration may need sentence-level control of emotion, while conversational AI like ChatGPT may require word or phoneme-level control over prosody.
Moreover, the emotion, prosody, and other characteristics of human speech are often intricately intertwined and can manifest across varying levels of granularity.
Additionally, achieving fine-grained control requires high-resolution annotations and sophisticated models capable of handling subtle variations without compromising synthesis quality.

Although some LLM-based TTS methods such as VoxInstruct~\cite{zhou2024voxinstruct} can control various aspects of speech through attribute descriptions, determining the appropriate level of granularity for control and devising methods to achieve precise control at a \emph{specific granularity} or to enable \emph{multiscale and fine-grained control} remains a significant challenge.

### Feature Extraction & Representation: 特征提取与表示

Achieving fully controllable TTS needs good feature disentanglement.
Accurately extracting meaningful and disentangled speech features like pitch contours, energy patterns, emotion variation, and prosodic elements from training data is difficult.
The reason is that speech features are interdependent and context-sensitive, making it hard to isolate specific attributes for control.
For example, altering pitch often affects prosody, emotion, and naturalness to some extent.
To tackle this, several methods~\cite{an2022disentangling,wang2023generalizable,an2021effective} utilize pre-trained models for different speech recognition tasks (\eg pitch, energy, and duration prediction, gender classification, age estimation, and speaker verification) to supervise feature extraction.
For example, NaturalSpeech3~\cite{tan2024naturalspeech} factorizes speech into separate feature subspaces to capture different speech attributes.

However, these methods are limited to coarse or high-level feature disentanglement, leaving a significant gap in \emph{fully disentangled control}.
On the other hand, selecting \emph{suitable representations} (\eg continuous variables like mel-spectrograms or latent embeddings like tokens) for controllable attributes is non-trivial because representations must be both interpretable for humans and expressive enough for TTS models.
For example, transformer-based models are good at processing discrete tokens, while GAN and Diffusion-based models excel in modeling continuous representations.

### Scarcity of Datasets: 数据集稀缺

High-quality, diverse, and appropriately annotated datasets are essential for training controllable TTS systems.
However, such datasets are scarce and difficult to construct.
To achieve controllable TTS, training data must encompass a wide range of styles, emotions, accents, and prosodic variations to enable versatile control because limited diversity in datasets can restrict the model's ability to generalize across unseen styles or emotions.
Although there are some TTS datasets, such as LibriTTS~\cite{zen2019libritts}, Gigaspeech~\cite{chen2021gigaspeech}, and TextrolSpeech~\cite{ji2024textrolspeech}, their diversity is still not enough for fully controllable TTS due to the lack of corpus of \emph{diverse content} such as comedies, thrillers, cartoons, etc.
Constructing large-scale datasets with rich diversity is also expensive and time-consuming.

Another obstacle is that creating datasets with fine-grained, attribute-specific annotations is labor-intensive and costly.
Besides, manual annotation of speech attributes requires expert knowledge and is prone to inconsistencies and errors, particularly for subjective qualities like emotion.
Currently, most datasets provide only coarse labels, such as gender, age, or a limited range of emotions.
While some datasets, such as SpeechCraft~\cite{jin2024speechcraft}, include natural language descriptions of speech attributes, no existing dataset offers \emph{fine-grained variations and annotations} within the speech of the same speakers.
Available datasets for controllable TTS are summarized in Table~\ref{tab:sec6_datasets_controllable}.

### Generalization Ability: 泛化能力

The ability of a TTS system to generalize effectively is crucial for producing natural, high-quality speech across a wide range of conditions, such as unseen speakers, languages, or topics.
However, achieving robust generalization remains a significant challenge for modern TTS methods due to various factors.

Zero-shot controllable TTS~\cite{wang2024maskgct,cooper2020zero} aims to synthesize speech for unseen speakers with various speech customization such as emotion using minimal reference audio, which can offer flexibility for personalized voice generation.
However, it faces significant challenges, including capturing unique speaker characteristics from limited data, accurately reproducing prosody and style, and disentangling speaker identity from other audio attributes like emotion or noise.

Multilingual generalization ~\cite{casanova2024xtts,cho2022sane} in TTS refers to the ability to synthesize natural and intelligible speech across multiple languages, including those not seen during training.
This capability is essential for applications like cross-lingual communication, multilingual virtual assistants, and speech synthesis for low-resource languages~\cite{magueresse2020low}.
Multilingual generalization still faces many challenges such as linguistic diversity and mismatch and the scarcity of data.
Cross-lingual speaker generalization is another hurdle, as preserving speaker identity across languages can lead to artifacts.

Domain adaptation ~\cite{farahani2021brief} in TTS refers to tailoring a pre-trained TTS model to generate speech for a specific domain or context, such as medical terminology and conversational speech.
One challenge is that many specialized domains lack sufficient high-quality annotated data for fine-tuning.
Besides, adapting prosody, intonation, and speaking style to match domain-specific requirements such as comic dialogue is complex.
Failing to capture domain-specific nuances can make speech sound unnatural or inconsistent with the target context.

### Efficiency: 效率

Efficiency in controllable TTS systems is a critical requirement for practical applications, as these models aim to offer fine-grained control over various speech attributes such as prosody, emotion, style, and speaker identity.
However, achieving such control often comes at the cost of increased computational complexity, larger model sizes, and longer inference times, creating significant challenges.

High latency is a major issue, as existing controllable TTS models~\cite{wang2024maskgct,guo2023prompttts,leng2023prompttts2,zhou2024voxinstruct} often necessitate autoregressive processes to synthesize speech.
The inference time of these models ranges from several to tens of seconds.
This can be especially problematic for real-time applications like live broadcasting or interactive systems.
Additionally, the challenge of balancing granularity and efficiency arises, as finer controls demand higher-resolution data and more precise models, leading to increased resource requirements and \emph{inefficient training and inference}.

Another major obstacle lies in the trade-off between model complexity and performance.
State-of-the-art controllable TTS systems often rely on large neural networks such as LLMs with billions of parameters, which provide superior naturalness and expressiveness but demand significant computational resources.
Simplifying these architectures can lead to quality degradation, including artifacts, unnatural prosody, or limited expressiveness.
Therefore, designing \emph{light-weight} controllable TTS models is significantly tricky.

## B·Future Directions: 未来方向

In this survey, we conduct a comprehensive investigation and analysis of existing TTS methods, particularly on controllable TTS technologies.~
While these methods show great potential in real-world applications, there are still some limitations that need to be addressed.~
Based on our observations, we outline several promising future directions as follows:~

### Fine-Grained Speech Synthesis by Natural Language Description: 基于自然语言描述的细粒度语音合成

Using natural language description to synthesize human speech with fine-grained control over various audio attributes is currently underexplored.
Most of the existing works can only control a fixed number of attributes of the synthesized speech.
Although a few works show great control of emotion, timbres, pitch, gender, and styles, \eg VoxInstruct~\cite{zhou2024voxinstruct} and CosyVoice~\cite{du2024cosyvoice}, they can frequently synthesize unwanted speech clips.
Users need to synthesize multiple times to get satisfactory speech.~

### Fine-Grained Speech Editing by Natural Language Description: 基于自然语言描述的细粒度语音编辑

Speech or audio editing has been studied for a long time.
However, existing methods usually train conditional models and adjust a fixed number of conditional inputs to modify the attributes of synthesized speech, thus lacking fine-grained manipulations~\cite{tae2021editts,tan2021editspeech}.
Therefore, how to learn disentangled speech representations for speech attributes while supporting editing by using natural language description is worthy of investigation.~

### Expressive Multi-Modal Speech Synthesis: 表达性多模态语音合成

Synthesizing speech from multi-modal data such as texts, images, and videos is an appealing research topic due to its various applications in the industry such as storytelling, filming, and gaming.
Although there are several related works on this task~\cite{li2001speech,goto2020face2speech,rong2024seeing,lu2022visualtts}, few of them can fully extract useful information from multi-modal data.
Particularly, synthesizing engaging speech and expressive voiceover for complex visual content sees great opportunities in the future.

### Natural and Emotional Conversational Tts: 自然和情感对话式语音合成

Speech conversational TTS have come out for several decades but remained as cascaded systems for a long time and cannot generate natural and emotional speech.
These systems are not context-aware, making the synthesized speech sound robotic.
With the advent of LLMs, existing TTS technologies were directly introduced by simply synthesizing speech from the text generated by LLMs~\cite{fang2024llama}.
However, context-aware conversational TTS with rich emotion and good naturalness has not been well studied.~

### Zero-Shot Long Speech Synthesis with Emotion Consistency: 情感一致的零样本长语音合成

Zero-shot TTS emerged in recent years to achieve voice cloning and speech style imitation without fine-tuning, making them more practical in real scenarios~\cite{wang2024maskgct,chen2024f5,du2024cosyvoice}.
However, synthesizing long speech with rich emotion and style variation in a zero-shot setting remains challenging due to the lack of rich speech information in short reference audio clips.
Addressing this issue will make a big step towards fully controllable zero-shot TTS.~

### Efficient TTS by Natural Language Description: 自然语言描述的高效文本转语音

Synthesizing speech with natural language description usually involves training large language encoders and bridge nets between the two modalities which can bring about much more computation overhead compared to previous TTS methods.
The inference time is also relatively slow, \eg existing methods usually take tens of seconds to synthesize a short speech audio clip of less than 10 seconds~\cite{du2024cosyvoice,shimizu2024promptttspp}.
Therefore, efficient text and speech modeling and interaction is critical for natural language description-based TTS systems.~
