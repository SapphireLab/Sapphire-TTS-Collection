# %%
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import pipeline

# %% 文本情感分类
classifier = pipeline("sentiment-analysis")
result = classifier("I love you")
print(result)
result = classifier("I hate you")
print(result)
# Results:
# [{'label': 'POSITIVE', 'score': 0.9998656511306763}]
# [{'label': 'NEGATIVE', 'score': 0.9991129040718079}]
# %% 阅读理解
question_answerer = pipeline("question-answering")
context =r"""
Extractive Question Answering is the task of extracting an answer from a text given a question.
An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.
If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/PyTorch/question-answering/run_squad.py script.
"""

result = question_answerer(question="What is extractive question answering?", context=context)
print(result)
result = question_answerer(question="What is a good example of a question answering dataset?", context=context)
print(result)
# Results:
# {'score': 0.6149139404296875, 'start': 34, 'end': 95, 'answer': 'the task of extracting an answer from a text given a question'}
# {'score': 0.5172930955886841, 'start': 147, 'end': 160, 'answer': 'SQuAD dataset'}
# %% 完形填空
unmasker = pipeline("fill-mask")
sentence = "HuggingFace is creating a <mask> that the community uses to solve NLP tasks."
result = unmasker(sentence)
print(result)
# Results:
# Some weights of the model checkpoint at distilbert/distilroberta-base were not used when initializing RobertaForMaskedLM: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
# - This IS expected if you are initializing RobertaForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing RobertaForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# [{'score': 0.17927460372447968, 'token': 3944, 'token_str': ' tool', 'sequence': 'HuggingFace is creating a tool that the community uses to solve NLP tasks.'}, {'score': 0.11349380016326904, 'token': 7208, 'token_str': ' framework', 'sequence': 'HuggingFace is creating a framework that the community uses to solve NLP tasks.'}, {'score': 0.052435602992773056, 'token': 5560, 'token_str': ' library', 'sequence': 'HuggingFace is creating a library that the community uses to solve NLP tasks.'}, {'score': 0.03493543714284897, 'token': 8503, 'token_str': ' database', 'sequence': 'HuggingFace is creating a database that the community uses to solve NLP tasks.'}, {'score': 0.02860250696539879, 'token': 17715, 'token_str': ' prototype', 'sequence': 'HuggingFace is creating a prototype that the community uses to solve NLP tasks.'}

# %% 文本生成
text_generator = pipeline("text-generation")
context = "As far as I am concerned, I will"
result = text_generator(context, max_length=50, do_sample=False)
print(result)
# Results:
# Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# [{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
# %% 命名实体识别
ner_pipe = pipeline("ner")
sequence = \
"""
Hugging Face Inc. is a company based in New York City. 
Its headquarters are in DUMBO,
therefore very close to the Manhattan Bridge which is visible from the window.
"""
for entity in ner_pipe(sequence):
    print(entity)
# Results:
# {'entity': 'I-ORG', 'score': 0.99957865, 'index': 1, 'word': 'Hu', 'start': 1, 'end': 3}
# {'entity': 'I-ORG', 'score': 0.9909764, 'index': 2, 'word': '##gging', 'start': 3, 'end': 8}
# {'entity': 'I-ORG', 'score': 0.9982224, 'index': 3, 'word': 'Face', 'start': 9, 'end': 13}
# {'entity': 'I-ORG', 'score': 0.9994879, 'index': 4, 'word': 'Inc', 'start': 14, 'end': 17}
# {'entity': 'I-LOC', 'score': 0.9994344, 'index': 11, 'word': 'New', 'start': 41, 'end': 44}
# {'entity': 'I-LOC', 'score': 0.99931955, 'index': 12, 'word': 'York', 'start': 45, 'end': 49}
# {'entity': 'I-LOC', 'score': 0.9993794, 'index': 13, 'word': 'City', 'start': 50, 'end': 54}
# {'entity': 'I-LOC', 'score': 0.98625815, 'index': 19, 'word': 'D', 'start': 81, 'end': 82}
# {'entity': 'I-LOC', 'score': 0.951427, 'index': 20, 'word': '##UM', 'start': 82, 'end': 84}
# {'entity': 'I-LOC', 'score': 0.9336589, 'index': 21, 'word': '##BO', 'start': 84, 'end': 86}
# {'entity': 'I-LOC', 'score': 0.9761654, 'index': 28, 'word': 'Manhattan', 'start': 116, 'end': 125}
# {'entity': 'I-LOC', 'score': 0.9914629, 'index': 29, 'word': 'Bridge', 'start': 126, 'end': 132}
# %% 文本摘要
summarizer = pipeline("summarization")
article = \
"""
New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. 
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, <NAME>, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective <NAME>, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth marriage was finalized after an administrative dispute with her second husband, who was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison. 
Her next court appearance is scheduled for May 18.
"""
result = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(result)
# Results:
# [{'summary_text': ' Liana Barrientos, 39, pleaded not guilty to two criminal counts of "offering a false instrument for filing in the first degree" She is believed to still be married to four men, and at one time, she was married to eight men at once .'}]
# %% 文本翻译
translator = pipeline("translation_en_to_de")
sentence = "Hugging Face is a technology company based in New York and Paris."
result = translator(sentence, max_length=40)
print(result)
# Results:
# [{'translation_text': 'Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.'}]
# %% 中译英
# !pip install sentencepiece
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

translator = pipeline("translation", model=model, tokenizer=tokenizer)
sentence = "我叫萨拉, 我住在伦敦."
result = translator(sentence, max_length=20)
print(result)
# Results:
# [{'translation_text': 'My name is Sarah, and I live in London.'}]
# %% 英译中
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

translator = pipeline("translation", model=model, tokenizer=tokenizer)
sentence = "My name is Sarah and I live in London."
result = translator(sentence, max_length=20)
print(result)
# Results:
# [{'translation_text': '我叫莎拉,我住在伦敦'}]
# %%