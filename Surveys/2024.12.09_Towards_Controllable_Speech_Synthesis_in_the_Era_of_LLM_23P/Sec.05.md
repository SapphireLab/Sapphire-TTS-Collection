# 5·Datasets and Evaluation: 数据集与评估

## A·Datasets: 数据集

Achieving fully controllable TTS requires large-scale datasets with rich diversity and fine-grained annotations.
In this subsection, we categorize speech datasets into four types according to the labels they provide, \ie attribute tags such as age and gender, description, environment, and dialogues.

### Tag-Based Speech Datasets: 基于标记的语音数据集

Tag-based datasets~\cite{kahn2020librilight,shi2020aishell,zhou2022emotional,chen2021gigaspeech,zhang2022wenetspeech,lee2023dailytalk,yang2024mscenespeech} are specialized collections of speech data that include metadata tags, such as pitch, emotion, and energy, age, gender, and so on, to enhance the expressiveness and control of TTS systems.
These datasets not only provide audio-text pairs but also offer additional labels that guide the model in generating diverse and context-aware speech outputs.

### Description-Based Speech Datasets: 基于描述的语音数据集

Description-based TTS datasets~\cite{guo2023prompttts,ji2024textrolspeech,lee2024voiceldm,ji2024controlspeech,jin2024speechcraft} contain speech data paired with detailed textual descriptions of the speech attributes or characteristics, such as emotion, speed, intonation, and style.
These datasets enable training TTS models that interpret descriptive prompts to generate expressive and context-aware speech.

### Speech Environment Datasets: 语音环境数据集

Speech environment datasets~\cite{lee2024voiceldm,barker2018fifthCHiME} are collections of speech data annotated with environmental labels, such as park, stadium, office, or street, capturing the acoustic characteristics of various real-world settings.
These datasets are crucial for training models to handle diverse acoustic scenarios, improving their robustness and adaptability.

### Dialogue Speech Datasets: 对话语音数据集

Dialogue speech datasets~\cite{lee2023dailytalk,byrne2019taskmaster} capture conversational speech between two or more people, focusing on the natural flow, turn-taking, and context of human dialogue.
These datasets are crucial for training systems in applications like chatbots, virtual assistants, and interactive storytelling.

We summarize open-source speech datasets for controllable TTS in Table~\ref{tab:sec6_datasets_controllable}.

## B·Evaluation: 评估

The performance of controllable TTS often requires objective and subjective evaluation.
We introduce common evaluation metrics in this subsection.

### Objective Evaluation Metrics: 客观评估指标

Objective metrics offer automated and reproducible evaluations.
Mel Cepstral Distortion (MCD)~\cite{kominek2008synthesizer} measures the spectral distance between synthesized and reference speech, reflecting how closely the generated audio matches the target in terms of acoustic features.
For intelligibility, the Word Error Rate (WER)~\cite{enwiki2024wer} is used, comparing transcriptions of synthesized speech to the input text via automated speech recognition.
Perceptual Evaluation of Speech Quality (PESQ)~\cite{Rix2001PESQ} is another objective metric designed to evaluate speech quality by comparing degraded audio with a clean reference.
It is widely used in telecommunications and speech synthesis, PESQ models human auditory perception, producing a score (typically 1–4.5) that reflects intelligibility and distortion under various conditions, including noise or compression.

### Subjective Evaluation Metrics: 主观评估指标

The Mean Opinion Score (MOS)~\cite{enwiki2024mos} is the most commonly used subjective metric.
In MOS evaluations, listeners rate the naturalness of synthesized speech on a scale (\eg 1 to 5), where higher scores indicate better quality.
MOS captures human perception effectively but requires substantial resources for large-scale evaluations.
Comparison Mean Opinion Score (CMOS)~\cite{loizou2011speech} further evaluates relative quality differences between two TTS audio samples.
Participants listen to paired samples and rate their preference on a scale (\eg -3 to +3, where negative values favor the first sample).
CMOS is used to measure subtle improvements in TTS systems, complementing absolute MOS ratings.

Table~\ref{tab:sec6_eval_metrics} summarizes common evaluation metrics for TTS.