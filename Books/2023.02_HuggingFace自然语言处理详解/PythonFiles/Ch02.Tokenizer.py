# %%
from transformers import BertTokenizer

# %% 2.3.1.Loading the tokenizer
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

# %% 2.3.2.Prepare the input sentences
sentences = [
    '你站在桥上看风景',
    '看风景的人在楼上看你',
    '明月装饰了你的窗子',
    '你装饰了别人的梦',
]

# %% 2.3.3.Using encode()
out = tokenizer.encode(
    text               = sentences[0],
    text_pair          = sentences[1],
    truncation         = True,
    padding            = 'max_length',
    add_special_tokens = True,
    max_length         = 25,
    return_tensors     = None,
)

print(out)
print(tokenizer.decode(out))

# %% 2.3.4.Using encode_plus()
out = tokenizer.encode_plus(
    text                       = sentences[0],
    text_pair                  = sentences[1],
    truncation                 = True,
    padding                    = 'max_length',
    add_special_tokens         = True,
    max_length                 = 25,
    return_tensors             = None,
    return_token_type_ids      = True,
    return_attention_mask      = True,
    return_special_tokens_mask = True,
    return_length              = True,
)

for k, v in out.items():
    print(k, ':',  v)
print(tokenizer.decode(out['input_ids']))

# %% 2.3.5.Using batch_encode_plus()
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs   = [(sentences[0], sentences[1]), (sentences[2], sentences[3])],
    truncation                 = True,
    padding                    = 'max_length',
    add_special_tokens         = True,
    max_length                 = 25,
    return_tensors             = None,
    return_token_type_ids      = True,
    return_attention_mask      = True,
    return_special_tokens_mask = True,
    return_length              = True,
    # return_offsets_mapping     = True
)
for k, v in out.items():
    print(k, ':',  v)
print(tokenizer.decode(out['input_ids'][0]))

# %% 2.3.6.Modify the vocab
vocab = tokenizer.get_vocab()
print(f"{type(vocab)=}, {len(vocab)=}, {'明月' in vocab=}")

tokenizer.add_tokens(new_tokens=['明月', '装饰', '窗子'])
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

out = tokenizer.encode(
    text               = sentences[2] + '[EOS]', 
    text_pair          = None,
    truncation         = True,
    padding            = 'max_length',
    add_special_tokens = True,
    max_length         = 10,
    return_tensors     = None,
)

print(out)
print(tokenizer.decode(out))

# %%
