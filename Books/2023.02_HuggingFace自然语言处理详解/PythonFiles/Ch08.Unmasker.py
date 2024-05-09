# %% 导入所需模块
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_scheduler

dataset_path = "D:\Speech\_Datasets\Seamew_ChnSentiCorp"
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% 加载分词器
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
print("加载分词器".center(50, "="))
print(tokenizer)

# %% 定义数据集
Dataset = load_dataset(dataset_path)
print("加载数据集".center(50, "="))
print(Dataset)

# %% 编码数据集
def encode(data):
    out = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = data["text"],
        truncation               = True,
        padding                  = "max_length",
        max_length               = 30,
        return_length            = True)
    return out

Encoded_Dataset = Dataset.map(
    function       = encode,
    batched        = True,
    batch_size     = 1000,
    num_proc       = 1,
    remove_columns = ["text", "label"],
)

print("编码数据集".center(50, "="))
print(Encoded_Dataset)

# %% 过滤数据集
def filter_dataset(data):
    return [i >= 30 for i in data["length"]]

Filtered_Dataset = Encoded_Dataset.filter(
    function   = filter_dataset,
    batched    = True,
    batch_size = 1000,
    num_proc   = 1,
)

print("过滤数据集".center(50, "="))
print(Filtered_Dataset)

# %% 定义整理函数
def collate_fn(data):
    input_ids = [i['input_ids'] for i in data]
    attention_mask = [i['attention_mask'] for i in data]
    token_type_ids = [i['token_type_ids'] for i in data]

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)

    # 将第十五位替换为 MASK
    labels = input_ids[:,15].reshape(-1).clone()
    input_ids[:,15] = tokenizer.get_vocab()[tokenizer.mask_token]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels

print("定义整理函数".center(50, "="))

tmp_data = [{
    "input_ids": [
        101, 2769, 3221, 3791, 6427, 1159, 2110, 5442, 117, 2110, 749, 8409,
        702, 6440, 3198, 4638, 1159, 5277, 4408, 119, 1728, 711, 2769, 3221,
        5439, 2399, 782, 117, 3791, 102,
    ],
    "token_type_ids": [0] * 30,
    "attention_mask": [1] * 30,
    },
    {
    "input_ids": [
        101, 679, 7231, 8024, 2376, 3301, 1351, 6848, 4638, 8024, 3301, 1351,
        3683, 6772, 4007, 2692, 8024, 2218, 3221, 100, 2970, 1366, 2208, 749,
        8024, 5445, 684, 1059, 3221, 102,
    ],
    "token_type_ids": [0] * 30,
    "attention_mask": [1] * 30,
    },
]

print("试运行".center(50, "="))
input_ids, attention_mask, token_type_ids, labels = collate_fn(tmp_data)
print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(labels[0]))
print(f"{input_ids.shape=}")
print(f"{attention_mask.shape=}")
print(f"{token_type_ids.shape=}")
print(f"{labels=}")

# %% 定义数据加载器
loader = torch.utils.data.DataLoader(
    dataset     = Filtered_Dataset['train'],
    batch_size  = 16,
    collate_fn  = collate_fn,
    shuffle     = True,
    drop_last   = True,
)
print("定义数据加载器".center(50, "="))
print(f"{len(loader)=}")
print("查看数据样例".center(50, "="))
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break
print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(labels[0]))
print(f"{input_ids.shape=}")
print(f"{attention_mask.shape=}")
print(f"{token_type_ids.shape=}")
print(f"{labels=}")

# %% 加载预训练模型
pretrained = BertModel.from_pretrained("bert-base-chinese")
print("加载预训练模型".center(50, "="))
print("模型参数:", sum(p.numel() for p in pretrained.parameters()))
for name, param in pretrained.named_parameters():
    param.requires_grad_(False)
print("模型可训练参数:", sum(p.numel() for p in pretrained.parameters() if p.requires_grad))

print("试运行".center(50, "="))
pretrained.to(device)
out = pretrained(
    input_ids      = input_ids,
    attention_mask = attention_mask,
    token_type_ids = token_type_ids,
)
print(f"{out.last_hidden_state.shape=}")

# %% 定义下游模型
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(768, out_features=tokenizer.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(tokenizer.vocab_size))
        self.decoder.bias = self.bias
        self.Dropout = torch.nn.Dropout(0.5)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )
        out = self.Dropout(out.last_hidden_state[:,15])
        out = self.decoder(out)
        return out
    
model = Model().to(device)
print("定义下游模型".center(50, "="))
print("模型可训练参数:", sum(p.numel() for p in model.parameters() if p.requires_grad))

print("试运行".center(50, "="))
out = model(input_ids, attention_mask, token_type_ids)
print(f"{out.shape=}")

# %% 训练
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.0)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = get_scheduler(
        name               = "linear",
        num_warmup_steps   = 0,
        num_training_steps = len(loader) * 5,
        optimizer          = optimizer,
    )
    model.train()
    for epoch in range(5):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 50 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f"{epoch=} {i=:3d} loss={loss.item():6.4e} accuracy={accuracy*100:>4.2f}% {lr=:6.4e}")

print("训练".center(50, "="))
train()

# %% 测试
def test():
    test_loader = torch.utils.data.DataLoader(
        dataset     = Filtered_Dataset['test'],
        batch_size  = 32,
        collate_fn  = collate_fn,
        shuffle     = True,
        drop_last   = True)
    model.eval()
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    accuracy = correct / total
    print(f"accuracy={accuracy*100:4.2f}%")

print("测试".center(50, "="))
test()
# %%