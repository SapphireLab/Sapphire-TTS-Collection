# 5·Datasets and Evaluation: 数据集与评估

## A·Datasets: 数据集

<table><tr><td width="50%">

Achieving fully controllable TTS requires large-scale datasets with rich diversity and fine-grained annotations.
In this subsection, we categorize speech datasets into four types according to the labels they provide, i.e., attribute tags such as age and gender, description, environment, and dialogues.

</td></tr></table>

### Tag-Based Speech Datasets: 基于标记的语音数据集

<table><tr><td width="50%">

Tag-based datasets ([Libri-Light [255]](../../Datasets/2019.12.17_Libri-Light.md); [AISHELL-3 [256]](../../Datasets/2020.10.22_AISHELL-3.md); [ESD [257]](../../Datasets/2020.10.28_ESD.md); [GigaSpeech](../../Datasets/2021.06.13_GigaSpeech.md); [WenetSpeech [259]](../../Datasets/2021.10.07_WenetSpeech.md); [DailyTalk [260]](../../Datasets/DailyTalk.md); [MSceneSpeech [261]](../../Datasets/2024.07.19_MSceneSpeech.md)) are specialized collections of speech data that include metadata tags, such as pitch, emotion, and energy, age, gender, and so on, to enhance the expressiveness and control of TTS systems.
These datasets not only provide audio-text pairs but also offer additional labels that guide the model in generating diverse and context-aware speech outputs.

</td></tr></table>

### Description-Based Speech Datasets: 基于描述的语音数据集

<table><tr><td width="50%">

Description-based TTS datasets ([PromptTTS [101]](../../Models/Acoustic/2022.11.22_PromptTTS.md); [TextrolSpeech](../../Datasets/2023.08.28_TextrolSpeech.md); [VoiceLDM [231]](../../Datasets/VoiceLDM_Data.md); [ControlSpeech [106]](../../Models/SpeechLM/2024.06.03_ControlSpeech.md); [SpeechCraft](../../Datasets/2024.08.24_SpeechCraft.md)) contain speech data paired with detailed textual descriptions of the speech attributes or characteristics, such as emotion, speed, intonation, and style.
These datasets enable training TTS models that interpret descriptive prompts to generate expressive and context-aware speech.

</td></tr></table>

### Speech Environment Datasets: 语音环境数据集

<table><tr><td width="50%">

Speech environment datasets ([VoiceLDM [231]](../../Datasets/VoiceLDM_Data.md); [Fifth CHiME [268]](../../Datasets/2018.03.28_Fifth_CHiME.md)) are collections of speech data annotated with environmental labels, such as park, stadium, office, or street, capturing the acoustic characteristics of various real-world settings.
These datasets are crucial for training models to handle diverse acoustic scenarios, improving their robustness and adaptability.

</td></tr></table>

### Dialogue Speech Datasets: 对话语音数据集

<table><tr><td width="50%">

Dialogue speech datasets ([DailyTalk [260]](../../Datasets/DailyTalk.md)；[Taskmaster-1 [254]](../../Datasets/2019.09.01_Taskmaster-1.md)) capture conversational speech between two or more people, focusing on the natural flow, turn-taking, and context of human dialogue.
These datasets are crucial for training systems in applications like chatbots, virtual assistants, and interactive storytelling.

We summarize open-source speech datasets for controllable TTS in Table.05.
Labels: Pitch, Energy, Speed, Age, Gender, Emotion, Emphasis, Accent, Top., Description, Environment, Dialogue.

</td></tr></table>

|Dataset|Hours|#Speakers|Pitch|Energy|Speed|Age|Gender|Emotion|Emphasis|Accent|Top.|Description|Environment|Dialogue|Language|Release Time|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Taskmaster-1 [254]](../../Datasets/2019.09.01_Taskmaster-1.md)| / | /|  |  |  |  |  |  |  |  |  |  |  | √ | en|2019.09 |
|[Libri-Light [255]](../../Datasets/2019.12.17_Libri-Light.md)| 60,000 | 9,722 | | | | | | | | | √ | | | | en | 2019.12 |
|[AISHELL-3 [256]](../../Datasets/2020.10.22_AISHELL-3.md)| 85 | 218 | | | | √ | √ | | | √ | | | | | zh | 2020.10 |
|[ESD [257]](../../Datasets/2020.10.28_ESD.md)|29|10|  |  |  | |  | √ |  |  |  |  |  |  |en,zh |2021.05 |
|[GigaSpeech [258]](../../Datasets/2021.06.13_GigaSpeech.md)| 10,000 | / | |  |  |  |  |  |  |  | √ |  |  |  | en | 2021.06 |
|[WenetSpeech [259]](../../Datasets/2021.10.07_WenetSpeech.md)| 10,000 | / |  |  |  |  |  |  |  | |√ | | | | zh | 2021.07 |
|[PromptSpeech [101]](../../Datasets/PromptSpeech.md)| / | / | √ | √ | √ | |  | √ |  |  |  | √ |  |  | en | 2022.11 |
|[DailyTalk [260]](../../Datasets/DailyTalk.md)| 20 | 2|  |  |  |  |  | √ |  |  | √ |  |  | √ | en|2023.05 |
|[TextrolSpeech [247]](../../Datasets/2023.08.28_TextrolSpeech.md)| 330 | 1,324 | √ | √ | √ |  | √ | √ |  |  |  | √ |  |  | en  | 2023.08|
|[VoiceLDM [231]](../../Datasets/VoiceLDM_Data.md)| /| /| √ |  | |  | √ | √ |  |  |  | √ | √ |  |en | 2023.09|
|[VccmDataset [106]](../../Datasets/2024.06.03_VccmDataset.md)| 330| 1,324| √ | √ | √ |  | √ | √ |  |  |  | √ |  |  | en|2024.06|
|[MSceneSpeech [261]](../../Datasets/2024.07.19_MSceneSpeech.md)| 13 | 13 |  |  |  |  | |  |  |  | √ |  |  | | zh | 2024.07 |
|[SpeechCraft [262]](../../Datasets/2024.08.24_SpeechCraft.md)| 2,391 | 3,200 | √ | √ | √ | √ | √ | √ | √ | | √ | √ | | | en,zh | 2024.08 |

## B·Evaluation: 评估

<table><tr><td width="50%">

The performance of controllable TTS often requires objective and subjective evaluation.
We introduce common evaluation metrics in this subsection.

</td></tr></table>

### Objective Evaluation Metrics: 客观评估指标

<table><tr><td width="50%">

Objective metrics offer automated and reproducible evaluations.
[Mel Cepstral Distortion (MCD) [263]](../../Evaluations/MCD.md) measures the spectral distance between synthesized and reference speech, reflecting how closely the generated audio matches the target in terms of acoustic features.
For intelligibility, the [Word Error Rate (WER) [264]](../../Evaluations/WER.md) is used, comparing transcriptions of synthesized speech to the input text via automated speech recognition.
[Perceptual Evaluation of Speech Quality (PESQ) [265]](../../Evaluations/PESQ.md) is another objective metric designed to evaluate speech quality by comparing degraded audio with a clean reference.
It is widely used in telecommunications and speech synthesis, PESQ models human auditory perception, producing a score (typically 1–4.5) that reflects intelligibility and distortion under various conditions, including noise or compression.

</td></tr></table>

### Subjective Evaluation Metrics: 主观评估指标

<table><tr><td width="50%">

The [Mean Opinion Score (MOS) [266]](../../Evaluations/MOS.md) is the most commonly used subjective metric.
In MOS evaluations, listeners rate the naturalness of synthesized speech on a scale (e.g., 1 to 5), where higher scores indicate better quality.
MOS captures human perception effectively but requires substantial resources for large-scale evaluations.
[Comparison Mean Opinion Score (CMOS) [267]](../../Evaluations/CMOS.md) further evaluates relative quality differences between two TTS audio samples.
Participants listen to paired samples and rate their preference on a scale (e.g., -3 to +3, where negative values favor the first sample).
CMOS is used to measure subtle improvements in TTS systems, complementing absolute MOS ratings.

</td><td>

</td></tr>
<tr><td colspan="2">

Table.06 summarizes common evaluation metrics for TTS.

|Metric | Type | Eval Target | GT Required |
|---|----|----|----|
|[MCD [263]](../../Evaluations/MCD.md)| Objective | Acoustic similarity | √ |
|[PESQ [264]](../../Evaluations/PESQ.md)| Objective | Perceptual quality | √ |
|[WER [265]](../../Evaluations/WER.md)| Objective | Intelligibility | √ |
|[MOS [266]](../../Evaluations/MOS.md)| Subjective | Preference | |
|[CMOS [267]](../../Evaluations/CMOS.md)| Subjective | Preference |

</td></tr></table>