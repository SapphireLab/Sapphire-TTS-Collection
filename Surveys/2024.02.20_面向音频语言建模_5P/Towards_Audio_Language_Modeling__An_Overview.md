# Towards Audio Language Modeling: An Overview

## Abstract

> Neural audio codec models were first introduced to compress audio for efficient data transmission. 
> The encoder converts the audio into codec codes, which are then transmitted.
> The receiver then uses the codec decoder to reconstruct the audio using the received codes.

> Language modeling has proven to be highly successful in the field of Natural Language Processing (NLP). 
> Audio data encompasses not only textual content but also rich information about speaker timbre, emotion, and general audio, offering deeper possibilities for language model applications.
> Researchers, especially those in large companies with significant computational resources, recently leverage the potential of neural codecs [1]–[8] as suitable tokenizers for converting continuous audio into discrete codes, which can be employed to develop audio language models (LMs) [9]–[20].
> The current codec-based language models and codec models are summarized in Figure 1. 
> These findings promptly garnered the community’s attention, sparking a fervor for developing codecs tailored to audio language modeling. 
> Numerous high performance neural audio codec models and audio LMs have been developed.

> An ideal codec should maintain content while preserving paralinguistic and speaker-related information. 
> Similarly, a universal audio language model should be able to generalize across various audio types, such as speech, music, and general audio, covering a wide range of applications. 
> The arms race in developing codecs and audio LMs is still ongoing.

> Given the significant advancements in codecs and audio language models over the past three years as shown in Figure 1, there has yet to be a comprehensive review comparing them and providing inspiration to the community. In this study, we aim to fill this research gap by thoroughly reviewing and comparing various existing neural codec models and audio codec-based language models. Firstly, we specifically conduct an in-depth analysis of six representative open-source neural codec models to cover their training methodologies, implementation settings, and training data. 
> Secondly, we expand our analysis to include eleven diverse codec-based language models, examining how they utilize the codecs and the tasks to which they can be applied. 
> Through this comprehensive review, we aim to offer the community insights into the diverse methodologies and potential directions in the field of neural codecs and codec-based language modeling.