# 6.Evaluations: 评估

Similar to TextLMs, SpeechLMs have a wide range of capabilities, making it challenging to compare different SpeechLMs.
Consequently, it's essential to evaluate SpeechLMs from various perspectives to determine their effectiveness.
In this section, we review the commonly used methods and benchmarks for evaluating SpeechLMs.
We categorize these evaluation methods into automatic and human assessments, each containing distinct evaluation aspects.

## 6.1.Automatic (Objective) Evaluation

Automatic evaluation methods are essential for providing quick and consistent assessments of SpeechLMs.
These methods typically rely on quantitative metrics that can be computed without human intervention.
Below, we outline some of the most commonly used automatic evaluation techniques.

### Representation Evaluation.

Representation (embedding) is a crucial component in SpeechLMs (and TextLMs).
It refers to how input data, such as speech or text, is transformed into a format that the model can understand and process.
Effective representation lays a solid foundation for models to understand lexical, syntax, and contextual information, which are vital for generating coherent and contextually relevant outputs.

In the context of SpeechLMs, representation evaluation focuses on how well the model encodes speech features into meaningful vectors.
[GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md) uses \textit{between-speaker ABX score} to measure the embedding similarity.
It quantifies how well-separated the phonetic categories are.
Specifically, It works by comparing three sound samples: two from the same category (A) and one from a different category (B).
The test measures how often the system correctly identifies that two sounds from category A are more similar to each other than one sound from A is to a sound from B.
Another way of evaluating representations is through speech resynthesis ([GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md)).
Specifically, an input speech is encoded into tokens and then synthesized back to speech.
Then, word error rate (WER) or character error rate (CER) can be computed on the ASR results of the input and resynthesized speech.
This measures the information loss caused by discretizing the input speech into speech tokens, thereby evaluating the robustness of the latent representations.

### Linguistic Evaluation.

Linguistics, including lexical, syntactic, and semantic evaluation methods, assess the model’s ability to generate and understand the rules for constructing words, sentences, and meaningful contents.
These evaluations focus on the correctness and appropriateness of word choices, the grammatical structure of the outputs, and the coherence and relevance of the generated content.
In terms of benchmark datasets, sWUGGY \cite{speechbenchmark2021} assesses at the lexical level by determining if the model can distinguish a real word from a (real, non-real) word pair.
sBLIMP \cite{speechbenchmark2021} evaluates at the syntactic level by determining if the model can identify the grammatically correct sentence from a (grammatical, ungrammatical) sentence pair.
Spoken StoryCloze \cite{textuallypretrainSLM} evaluates semantic comprehension by assessing the model's capability to select the genuine ending of a story from a pair of ending choices.
All the evaluation is conducted by comparing the model's negative log-likelihood of the data pair.

### Paralinguistic Evaluation.

In contrast to linguistic evaluation, paralinguistic evaluation focuses on the non-verbal aspects of communication that accompany speech.
Some works choose to utilize paralinguistic tokens alongside semantic tokens to enhance the paralinguistic abilities of SpeechLMs \cite{prosodyawareSLM,[SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md)}, so one way is to evaluate the paralinguistic tokens.
pGSLM \cite{prosodyawareSLM} measures the correctness, consistency, and expressiveness of the prosodic tokens.
Correctness evaluates the model's ability to generate accurate prosodic profiles by calculating the minimal mean absolute error (min-MAE) of the prosodic tokens from 20 generated samples against the prosodic tokens from the reference, consistency is assessed through the Pearson correlation between the mean values of the prompt prosodic and its generated continuation prosodic tokens, and expressiveness is measured by the standard deviation of the generated prosody token values, with the expectation that it matches the variability of the ground truth.
We note that the same metrics can also be applied to other paralinguistic tokens.
Instead of evaluating from the token level, [SpiRit-LM (2024)](../../Models/Speech_LLM/2024.02.08_SpiRit-LM.md) propose to measure on the perceptual level.
They introduced a speech-text sentiment preservation benchmark (STSP), which requires the model to generate a text or speech sequence of tokens that preserves the sentiment of the prompt.
A sentiment classifier is used to assess the sentiment in the generated speech.
It should be noted that although they only apply the preservation approach on sentiment, this idea can be generalized to other paralinguistic features, such as timbre or prosody.

### Generation Quality and Diversity.

Quality and diversity are two crucial aspects of model generation.
Typically, there is a trade-off between these dimensions when sampling model responses at different temperatures, so [GSLM (2021)](../../Models/Speech_LLM/2021.02.01_GSLM.md) suggests using the Area Under the Curve (AUC) with various temperature values.
Specifically, AUC on perplexity and VERT are employed to assess these factors, where VERT represents the geometric mean of the ratio of k-grams in the generated speech that appears at least once.
Additionally, the ChatGPT score can be utilized to evaluate the quality of the generated speech.
In this process, the generated speech is transcribed using state-of-the-art ASR models and then sent to ChatGPT for quality (and diversity) assessment.

### Downstream Evaluation.

Downstream evaluation refers to evaluating the ability of SpeechLMs to perform specific tasks, such as ASR, TTS, Speaker Identification, etc.
The evaluation can be performed on pre-trained models by adding few-shot example(s) at the start of the prompt or on the instruction-tuned models by directly instructing them to do so.
[SUPERB (2021)](../../Evaluations/2021.05.03_SUPERB.md) is a benchmark containing a wide range of downstream tasks that can be performed by SpeechLMs.

## 6.2.Human (Subjective) Evaluation.

Human evaluation plays a crucial role in assessing the performance of SpeechLMs, as ultimately, speech is designed to be heard and perceived by humans.
This type of evaluation relies on human judgment to assess the quality of the outputs generated by SpeechLMs.
Below, we outline several commonly used human evaluation methods.

### Mean Opinion Score.

Mean opinion score (MOS) is a widely used metric in the field of speech evaluation that quantifies the perceived quality of speech output as judged by human listeners.
Typically, a group of evaluators listens to a series of audio samples generated by the SpeechLM and rates each sample on a predefined scale, often from 1 (poor quality) to 5 (excellent quality).

MOS is calculated by averaging the scores given by all evaluators for each audio sample, providing a single score that reflects the overall quality as perceived by humans.
Variations of MOS focus on different aspects of speech quality, including MMOS, PMOS, and SMOS \cite{prosodyawareSLM,speechgpt-gen}.
They evaluate the aspects of naturalness, prosody, and timbre similarity of the given speech, respectively.

Typically, evaluating naturalness or timbre similarity involves collecting human opinions.
However, this process can be complicated due to the challenges of recruiting participants and gathering their evaluations.
As a result, researchers often turn to machine-based evaluations.
They commonly employ neural network models specifically trained for these tasks.
For instance, a naturalness prediction model \cite{nisqanaturalness} can assess the naturalness of generated outputs, while a speaker identification model can evaluate timbre similarity.
