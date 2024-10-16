# Sec.08: Discussion

This systematic review of ETTS models within the last 5 years has shown a wide variety of methods and techniques that have been applied in this field, particularly DL-based approaches.
However, current ETTS models are still far away from achieving their goal of generating human-like speech in terms of expressiveness, variability, and control flexibility.
The main contribution of this review to the literature is to provide a full picture of the efforts that have been conducted in this field for newcomers or beginner researchers, helping them to identify their roadmap within this area.

On the other hand, we hope that the provided information and summaries in this review including methods taxonomy, modeling challenges, datasets and evaluation metrics can be of good support and guidance for researchers in this area to compare and identify state-of-the-art models on one side, and to spot gaps yet to be filled on the other side.
Our focus in this review was to identify the main methods and structures applied in literature for ETTS besides challenges and problems that they encounter.
Nevertheless, papers covered here can be further investigated to analyze models’ performance and compare their results utilizing the same datasets and evaluation metrics.

Additionally, based on our investigation in this review, we would like in this discussion to highlight some research gaps within this research area that need to be considered in future work.

- **Terminology Identification**: During the course of this work, we observed a lack of clear definitions for main terminologies used in this research area, such as “expressive,” “emotion,” “prosody,” and “style”.
Early studies, as in [78, 80, 82], often used the terms “emotion” and “style” interchangeably, encompassing data with different emotions (happy, sad, etc.) or a blend of emotion and style (e.g., happy call center, sad technical support).
Furthermore, the term “expressive speech” is used in a general sense to describe speech that is natural-sounding and resembles human speech overall, as in studies [35, 40, 54, 76, 94, 97, 114, 119, 134, 135].
However, it is also utilized in other studies to describe speech with different labels for emotions [26, 28, 49, 64, 91], styles (newscaster, talk-show, call-center, storytelling.) [45, 106, 115, 118], or combinations of emotions and styles [19, 59, 68, 78, 79, 84].
On the other hand, a single style itself can encompass speech featuring multiple emotions and variable prosody attributes, as exemplified by the Blizzard2013 dataset [171], which includes data in a storytelling style.
Many studies [50, 74, 93, 94, 121, 122] employ the Blizzard2013 dataset to train their TTS models and generate expressive speech.
The resulting speech from these models in this case exhibits varying prosody and conveys different emotions.
In general, the existing literature lacks a distinct differentiation among these terminologies and their associated variations, which can lead to confusion among researchers and complicated comparisons between models.
Therefore, it is highly recommended to conduct further investigation to establish clear and comprehensive definitions for these terms.
Specifically, each term needs to be accurately delineated, specifying its respective types, attributes, and speech features.
- **Controllable Expressive Speech**: Providing control over the expressiveness of synthesized speech can be considered an advanced step in this area of research.
As we have discussed in Section 5.3 several recent studies have addressed different aspects to provide more control over expressivity in synthesized speech.
The aspects covered in these studies include selection and adjusting different prosody-related features at coarse levels (utterance, paragraph, sentence, etc.) as well as fine-grained levels (word, syllables, phonemes, etc.).
However, the proposed controlling techniques with their achieved results are still considered small steps in this important research area of expressive speech synthesis, and more efforts are needed and expected in this direction in the near future.
In fact, bridging this research gap is a crucial step towards the goal of speech synthesis research to produce human-like speech.
- **Evaluation Metrics**: Despite the existence of several metrics applied in the literature to evaluate the performance of ETTS models, no general and standard metrics have been identified to facilitate the comparison process among different approaches.
Furthermore, since the evaluation of expressive speech is more sophisticated and challenging, there is still a high demand for more accurate metrics capable of capturing various aspects of expressiveness in speech.
Additionally, with the increased attention on building controllable ETTS models, the need arises for efficient evaluation metrics for controllability related aspects.
- **Datasets**: As discussed in [Section 6](Sec.06.md), availability of inclusive, high quality and large size expressive dataset is crucial for achieving efficient ETTS models.
However, building a comprehensive emotional speech dataset that encompasses a wide range of emotions, styles, speakers, and languages with high quality remains a formidable objective in the expressive speech synthesis field.
The challenges extend beyond the issues mentioned in Section 6 and encompass aspects such as time and cost.
Language modeling-based approaches could be the future of the field, overcoming these challenges, but they are still in the early stages, and further research in this direction is necessary.