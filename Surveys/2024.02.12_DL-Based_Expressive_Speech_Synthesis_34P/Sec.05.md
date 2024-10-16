
## 5.表现性语音合成的主要挑战

In this section, we list and explain the most important challenges that face expressive TTS models and the main solutions that have been proposed in the literature to overcome these challenges. We then provide a summary of papers addressing each challenge in [Table 5]().

本节列出并展示表现性语音合成模型面临的最重要挑战, 以及在现有文献中提出的克服这些挑战的主要解决方案. 在表格五种提供了解决每个挑战的文献的总览.

|参考文献|信息泄露|缺少参考音频|韵律可控性|未知风格/说话人|
|---|:-:|:-:|:-:|:-:|
||√|√|√|√|
||√|√||√|
|[097](#K.Lee2021)|√||√|√|
|[NaturalSpeech2](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md);[VALL-E](../../Models/Speech_LLM/2023.01.05_VALL-E.md)|√|||√|
|[102](#K.Zhang2022)|√||||
|||√|√||
|[InstructTTS](../../Models/_tmp/2023.01.31_InstructTTS.md) [047](#C.Lu2021) [111](#D.Tan2020)|√|√|||
|[019](#S.Jo2023)|√||√||
||||√||
|||√|||

- [5.1.Irrelevant Information Leakage](Sec.05.01_Irrelevant_Information_Leakage.md)
- [5.2.Inference without Reference Audio](Sec.05.02_Inference_without_Reference_Audio.md)
- [5.3.Prosody Controllability](Sec.05.03_Prosody_Controllability.md)
- [5.4.Unseen Speakers & Styles](Sec.05.04_Unseen_Speakers_&_Styles.md)