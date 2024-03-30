# 第 04 章 Text Analyses (文本分析)

> Through text analyses, we can transform input text into linguistic features, which contain rich information about pronunciation and prosody that can ease speech synthesis.
> Text analyses consist of several components, as shown in [Fig.4.1]():
> 1. text processing, which processes raw text from documents, normalizes the text from a written form into a spoken form, and conducts some linguistic analyses;
> 2. phonetic analysis, which converts text into phonetic symbols, including polyphone disambiguation and grapheme-to-phoneme conversion;
> 3. prosodic analysis, which analyzes some prosodic features such as pitch, duration, loudness, stress, and pauses.
>
> In this chapter, we first introduce these components in [Sec 4.1](#Sec4.1), [4.2](#Sec4.2), and [4.3](#Sec4.3), and then discuss the development of text analysis in TTS in [Sec 4.4](#Sec4.4).

## 4.1.Text Processing (文本处理) <a id='Sec4.1'></a>

> Text processing is to extract, simplify, normalize, and analyze text to make it suitable for phonetic analysis and prosodic analysis.
> The processes include
> 1. document structure detection, which helps to locate the document structure of a sentence that is useful for later processing;
> 2. text normalization, which converts the text from a nonorthographic form into an orthographic form to ease phonetic conversion;
> 3. linguistic analysis, which processes the syntactic and semantic features that are helpful for later phonetic and prosodic analyses.

### 4.1.1.Document Structure Detection (文档结构检测)

> The input text of TTS systems is usually not single sentences by itself, but from documents (e.g., book, article, e-mail, web page, conversation, dialog) that can provide context for these sentences.
> Different structures, such as chapter/section headers, lists, paragraphs, sentences, e-mails, and dialogue turns can provide different indications for intonational contours, pitch assignments, and prosodic styles.
> A typical task for document structure detection is sentence breaking since a knowledge of the sentence unit is important for correct pronunciation and prosodic breaking.

### 4.1.2.Text Normalization (文本标准化)

> Non-standard words usually contain a lot of nonorthographic forms or semiotic classes [71], such as
> 1. abbreviations and acronyms (e.g., TTS for Text to Speech, OPEC for the Organization of Petroleum Exporting Countries);
> 2. number formats(e.g., phone number `716-123-4568`, date `03/15/2022`, time `2:18 pm`, money and　currency `$32`, account numbers `6217-9062`, ordinal numbers `1st`, cardinal numbers `3728`);
> 3. scientific formula (e.g., mathematical formula $\dfrac{\sqrt{x}}{y}$, chemical formula H<sub>2</sub>O);
> 4. Web and Internet address (e.g., `https://www.microsoft.com/`);
> 5. special symbols (e.g., emotion `:-)` means smiley).
>
> To make the text suitable for subsequent phonetic conversion and easy to pronounce for TTS systems, text normalization is leveraged to convert text from the nonorthographic form (written form) into the orthographic form (speakable form).
> For example, the date `Jan. 24, 1989` is normalized into `January twenty-fourth nineteen eighty-nine`.
> Early works on text normalization are rule-based [71], and then neural networks are leveraged to model text normalization as a sequence-to-sequence task where the source and target sequences are non-standard words and spoken-form words respectively [23–25].
> Some works [72] propose to combine the advantages of both rule-based and neural-based models to further improve the performance of text normalization.

### 4.1.3.Linguistic Analysis (语言学分析)

> Linguistic analysis is to extract structural and semantic information from sentences through syntactic and semantic parsing.
> Linguistic analysis has several usages in TTS:
> 1. it can provide additional grammar information to help determine the pronunciation of a word in different senses or abstraction inflections (e.g.,`read` is pronounced as `/ri:d/` in present tense and `/red/` in past tense);
> 2. it can provide additional information to differentiate sentences that are the same after text normalization;
> 3. it can provide useful information to determine the prosodic structure that influences the duration and pitch contour (e.g., the syntactic type of a sentence, yes/no question, and wh-question have different duration and pitch contours although both are marked with a question symbol `/?/`).
>
> Some basic syntactic and semantic parsing include sentence type detection, word/phrase/sentence segmentation, part of speech (POS) tagging, word sense disambiguation, and homograph disambiguation.
> We introduce these syntactic and semantic parsing tasks in the following paragraphs.

#### Sentence Breaking and Type Detection

> Sentence breaking from raw documents is important for correct pronunciation and prosodic breaking and can be implemented in some rules based on punctuation.
> Different sentence types, such as declarative sentence, yes/no question, and wh-question, have different prosodic lines in synthesized speech.
> We can use some rules to detect the type of a sentence by checking some keywords (e.g., `please`, `what/how`, `is`, `isn’t`) and punctuations (e.g., `/./`, `/!/`, `/?/`).

#### Word/Phrase Segmentation

> For languages like Chinese, Thai, and Japanese, word segmentation [73–75] is necessary to detect the word boundary from raw text, which is important to ensure the accuracy for POS tagging, phonetic and prosodic analysis.
> The segmentations of phrases such as noun phrases and clauses are important for prosodic analysis, especially in determining the pitch, duration, and pause of phrases to make a sentence more intelligible and natural.
> Word segmentation and phrase parsing are well supported by popular parsing toolkits, such as Jieba and Stanford CoreNLP.

#### Part-of-Speech Tagging

> The part-of-speech (POS) of each word, such as noun, verb, and preposition,is important for the phonetic and prosodic analysis in TTS.
> Several works have investigated POS tagging in speech synthesis [74, 76–79].

#### Homograph and Word Sense Disambiguation

> Homographs represent words that have the same written form (spelling) but have different meanings.
> For example, `bear` is either a noun that represents a kind of animal or a verb that is similar to `tolerate`.
> Thus, `bear` is regarded as a homograph.
> A similar concept is word sense, which means one of the meanings of a word.
> Disambiguating homographs or word senses is helpful to understand the semantics and syntax of a sentence better, which is helpful for later prosodic analysis.

## 4.2.Phonetic Analysis (语音分析) <a id='Sec4.2'></a>

> Phonetic analysis can ease the speech synthesis process by providing information about how a word should be pronounced.
> It involves the study of the pronunciation of a word and the conversion from its grapheme sequence (lexical orthographic symbols) into phoneme sequence, with possible diacritic information (e.g., stress placement) for precise pronunciation.
> Phonetic analysis includes two tasks: poly-phone disambiguation to determine different pronunciations of the same word in different word contexts, and grapheme-to-phoneme conversion to generate the phoneme sequence of a word.

#### 4.2.1.Polyphone Disambiguation (多音字歧义消除)

> A polyphone refers to a word that can be pronounced in two or more different ways, where each way represents a different word sense.
> Many languages have polyphones.
> For example, `resume` in English can be pronounced as `/ri'zju:m'/` (a verb, means to go on or continue after interruption) or `/'rezjumei/` (a noun, means curriculum vitae), `奇` in Chinese can be pronounced as `jī` (means odd or odd number) or `qí` (means strange).
> Polyphone disambiguation is to decide the appropriate pronunciation based on the context of this word/character [26, 27, 80–83].
> Note that polyphones are different from homographs since a polyphone has multiple different pronunciations while a homograph has multiple different meanings but not necessarily have multiple pronunciations.

#### 4.2.2.Grapheme-to-Phoneme Conversion (字符到音素转换)

> After polyphone disambiguation, we further conduct grapheme-to-phoneme conversion to transform characters (graphemes) into pronunciations (phonemes)(e.g., the word `speech` is converted into `s p iy ch`), which can greatly ease speech synthesis.
> For alphabetic languages with simple and clear relationships between graphemes and phonemes (phonetic languages, e.g., Spanish), grapheme-to-phoneme conversion should be easy and can be well processed by handcrafted rules.
> For alphabetic languages with complicated relationships between graphemes and phonemes (non-phonetic language, e.g., English), handcrafted rules cannot cover all words.
> Thus, grapheme-to-phoneme conversion models are usually developed to generate the pronunciations of out-of-vocabulary words [26, 27, 80–83].
> For some languages like Arabic and Hebrew, the vowel information is not available in written text and needed to be determined/predicted from the text (it can be also regarded as a polyphone disambiguation task).
> For non-alphabetic languages (e.g., Chinese), a manually collected grapheme-to-phoneme lexicon is usually leveraged for conversion, which can cover nearly all the characters.

## 4.3.Prosodic Analysis (韵律分析) <a id='Sec4.3'></a>

> To make the synthesized speech sound natural, we need to conduct a prosodic analysis of the sentences properly.
> The prosodic analysis involves the analysis of prosody information such as rhythm, stress, and intonation of speech, which correspond to the variations in phoneme duration, loudness, and pitch, and play an important perceptual role in human speech communication.
> In the following subsections, we introduce how to analyze the pause, stress, and intonation information, as well as the pitch, duration, and loudness of the speech.

#### 4.3.1.Pause, Stress, and Intonation (停顿, 重音与语调)

> Prosody analysis relies on tagging systems to label each kind of prosody, such as pause, stress, and intonation.
> Different languages have different prosody tagging systems and tools [84–88].
> For English, ToBI (tones and break indices) [84, 85] is a popular tagging system, which describes the tags for tones (e.g., pitch accents,phrase accents, and boundary tones) and break (how strong the break is between words).
> For example, in this sentence `Mary went to the store ?`, `Mary` and `store` can be emphasized, and this sentence is raising tone.
> A lot of works [29–31, 89] investigate different models and features to predict the prosody tags based on ToBI.
> For Chinese speech synthesis, the typical prosody boundary labels consist of the prosodic word (PW), the prosodic phrase (PPH), and the intonational phrase(IPH), which can construct a three-layer hierarchical prosody tree [90–92].
> Some works [90–95] investigate different model structures such as conditional random field [96], RNN [97], and self-attention [98] for prosody prediction in Chinese.


#### 4.3.2.Pitch, Duration, and Loudness (音高, 时长与响度)

> Pause, stress, and intonation are fine-grained features to determine the prosody of a speech.
> Alternatively, we can also use pitch, duration, and loudness features to determine the prosody of speech.
> The differences between pause/stress/intonation and pitch/duration/loudness may be that pauses/stresses/intonations are more comprehensible by listeners or users, while pitch/duration/loudness is more like basis features and capture the characteristics of speech prosody in a data-driven way.
> Any kind of prosody such as pause/stress/intonation can be obtained by varying pitch/duration/loudness.
> Due to the fundamental and basic representations brought by pitch/duration/loudness, several neural TTS models [47, 48, 99] leverage them as the prosody features (variance information) to improve the quality of synthesized speech.


## 4.4.Text Analysis from a Historic Perspective (历史视角) <a id='Sec4.4'></a>

> In this section, we briefly overview the development progress of text analysis in TTS, from conventional statistical parametric speech synthesis to neural speech synthesis.

#### 4.4.1.Text Analysis in SPSS (统计参数语音合成中的文本分析)

> In statistical parametric speech synthesis (SPSS), text analysis is used to extract a sequence of linguistic feature vectors [15], which are taken as input to the later part of the TTS pipeline, e.g., acoustic models in SPSS or neural vocoders [1].
> Therefore, text analysis nearly consists of all the modules in text processing(e.g., text normalization [23, 72], word segmentation [73], part-of-speech (POS)tagging [76], phonetic analysis (e.g., grapheme-to-phoneme conversion [26]), and prosodic analysis (e.g., prosody prediction [90])).
> Usually, we can construct linguistic features by aggregating the results of text analysis from different levels including phoneme, syllable, word, phrase, and sentence levels [15].
> - **Phoneme level**: the phonetic symbols of the previous before the previous, the previous, the current, the next, or the next after the next; the forward or backward distance of the current phoneme within the syllable.
> - **Syllable level**: whether the previous, the current, or the next syllable is stressed; the number of phonemes contained in the previous, the current, or the next syllable; the forward or the backward distance of the current syllable within the word or phrase; the number of the stressed syllables before or after the current syllable within the phrase; the distance from the current syllable to the forward or backward most nearest stressed syllable; the vowel phonetics of the current syllable.
> - **Word level**: the part of speech (POS) of the previous, the current, or the next word; the number of syllables of the previous, the current, or the next word; the forward or backward position of the current word in the phrase; the forward or backward content word of the current word within the phrase; the distance from the current word to the forward or backward nearest content word; the POS of the previous, the current or the next word.
> - **Phrase level**: the number of syllables of the previous, the current, or the next phrase; the number of words of the previous, the current, or the next phrase; the forward or backward position of the current phrase in the sentence; the prosodic annotation of the current phrase.
> - **Sentence level**: The number of syllables, words, or phrases in the current sentence.


#### 4.4.2.Text Analysis in Neural TTS (神经语言合成中的文本分析)

> In neural TTS, due to the large modeling capacity of neural-based models, the character or phoneme sequences are directly taken as input for synthesis.
> In this way, only text normalization is needed to convert raw text with a non-standard format to a standard word format if the character is taken as input, and grapheme-to-phoneme conversion is further needed to get phonemes from standard word format if phonemes are taken as input.
> Thus, the text analysis module is largely simplified.

> Although text analysis seems to receive less attention in neural TTS compared to SPSS, it has been incorporated into neural TTS in various ways:
> **1. Neural network-based text analysis module**.
> Char2Wav [65] and DeepVoice 1/2 [35, 36]implement the character-to-linguistic feature conversion into its pipeline, purely based on neural networks.
> Pan et al. [33] and Zhang et al. [34] designed unified models to cover all the tasks in text analysis in a multi-task paradigm and achieve good results.
> **2. Prosody prediction**.
> Prosody is critical for the naturalness of speech synthesis.
> Although neural TTS models simplify the text analysis module, some features for prosody prediction are incorporated into the text encoder, such as the prediction of pitch ([FastSpeech2 (2020)](../../Papers/2020.06_FastSpeech2.md)), duration ([FastSpeech (2019)](../../Papers/2019.05_FastSpeech.md)), phrase break [100], breath or filled pauses [101], or prosodic style features [102] are built on top of the text (character or phoneme) encoder in TTS models.
> **3. Reference encoders**.
> Some works [103–105] use reference encoders to learn the prosody representations from reference speech.
> **4. Text pre-training**.
> Some works learn good text representations with implicit prosody information through self-supervised pre-training [70, 106–108].
> **5. Incorporating syntax information through dedicated modeling methods** such as graph networks [109].
