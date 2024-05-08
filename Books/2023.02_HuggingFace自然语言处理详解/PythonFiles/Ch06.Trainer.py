# %%
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from evaluate import load
# %% 加载分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
tokenize_result = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs = ["明月装饰了你的窗子", "你装饰了别人的梦"],
    truncation               = True)
print("分词器试运行".center(50, "="))
print(tokenize_result)

# %% 加载数据集
print("加载数据集".center(50, "="))
dataset_path = r"D:\Speech\_Datasets\Seamew_ChnSentiCorp"
dataset = load_dataset(dataset_path)
print(dataset)
print("打乱并选取数据集".center(50, "="))
dataset['train'] = dataset['train'].shuffle().select(range(2000))
dataset['test']  = dataset['test'].shuffle().select(range(100))
print(dataset)

# %% 编码数据集
print("编码数据集".center(50, "="))
def encode(data):
    return tokenizer.batch_encode_plus(data['text'], truncation=True)
dataset = dataset.map(encode, batched=True, batch_size=1000, num_proc=1, remove_columns=['text'])
print(dataset)

# %% 过滤长数据
print("过滤长数据".center(50, "="))
def remove_too_long(data):
    return [(len(i)) <= 512 for i in data['input_ids']]
dataset = dataset.filter(remove_too_long, batched=True, batch_size=1000, num_proc=1)
print(dataset)

# %% 模型定义
print("加载模型".center(50, "="))
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=2)
print(f"{sum([p.numel() for p in model.parameters()]) / 10000}" + "万参数")

# %% 运算测试
print("模型推理测试".center(50, "="))
data = {
    'input_ids': torch.ones(4, 10, dtype=torch.long),
    'token_type_ids': torch.ones(4, 10, dtype=torch.long),
    'attention_mask': torch.ones(4, 10, dtype=torch.long),
    'labels': torch.ones(4, dtype=torch.long)
}
out = model(**data)
print(f"{out['loss']=}, {out['logits'].shape=}")

# %% 评价函数
print("加载评价函数".center(50, "="))
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)
print("评价函数试运行".center(50, "="))
eval_pred = EvalPrediction(
    predictions=np.array([[0,1],[2,3],[4,5],[6,7]]),
    label_ids=np.array([1,1,0,1])
)
print(compute_metrics(eval_pred))

# %% 定义整理函数
print("定义整理函数".center(50, "="))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("数据整理试运行".center(50, "="))
data = dataset['train'][:5]
for i in data['input_ids']:
    print(len(i))
data = data_collator(data)
for k, v in data.items():
    print(k, v.shape)
print(tokenizer.decode(data['input_ids'][0]))

# %% 定义超参数
print("定义超参数".center(50, "="))
output_dir = r"./results"
os.makedirs(output_dir, exist_ok=True)
args = TrainingArguments(
    output_dir                  = output_dir,
    evaluation_strategy         = 'steps',
    eval_steps                  = 30,
    save_strategy               = 'steps',
    save_steps                  = 30,
    num_train_epochs            = 1,
    learning_rate               = 1e-4,
    weight_decay                = 1e-2,
    per_device_eval_batch_size  = 16,
    per_device_train_batch_size = 16,
    no_cuda                     = False,
    save_safetensors            = False
)

# %% 定义训练器
print("定义训练器".center(50, "="))
trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = dataset['train'],
    eval_dataset    = dataset['test'],
    compute_metrics = compute_metrics,
    data_collator   = data_collator
)

print("训练前评估".center(50, "="))
print(trainer.evaluate())
print("开始训练".center(50, "="))
trainer.train()

# trainer.train(
#     resume_from_checkpoint=os.path.join(output_dir, "checkpoint-90"),
# )

print("训练后评估".center(50, "="))
print(trainer.evaluate())

# %% 保存模型
trainer.save_model(output_dir=output_dir)

# %% 推理测试
print("推理测试".center(50, "="))
import torch
from safetensors.torch import load_file

if os.path.exists(os.path.join(output_dir, "model.safetensors")):
    model.load_state_dict(load_file(os.path.join(output_dir, "model.safetensors")))
elif os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
    model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
else:
    raise FileNotFoundError("模型文件不存在")
model.eval()
for i, data in enumerate(trainer.get_eval_dataloader()):
    break
out = model(**data)
out = out['logits'].argmax(axis=-1)
for i in range(8):
    print(tokenizer.decode(data['input_ids'][i], skip_special_tokens=True))
    print('label=', data['labels'][i].item())
    print('pred=', out[i].item())

# %%
