# %% 代码清单 1-1. 使用 Embedding 的简单示例
from torch import nn
import torch
import jieba
import numpy as np

# %%
raw_text = """越努力就越幸运"""

words = list(jieba.cut(raw_text))
print(f"Jieba 分词结果：{words=}")
# %%
word_to_idx = {i: word for i, word in enumerate(set(words))}

embeds = nn.Embedding(num_embeddings=4, embedding_dim=3)
print(f"Embedding 权重：{embeds.weight=}")
# %%
keys = word_to_idx.keys()
keys_list = list(keys)
tensor_value = torch.LongTensor(keys_list)
output = embeds(tensor_value)
print(f"Embedding 输出：{output=}")
# %%