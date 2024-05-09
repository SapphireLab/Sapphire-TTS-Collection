# %% 导入所需模块
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_scheduler
import random

dataset_path = "D:\Speech\_Datasets\Seamew_ChnSentiCorp"
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% 加载分词器
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
print("加载分词器".center(50, "="))
print(tokenizer)

# %% 定义数据集
class PairDataset(Dataset):

    def __init__(self, split):
        super().__init__()
        dataset = load_dataset(dataset_path, split=split)
        def cut(data):
            return len(data['text']) > 40
        self.dataset = dataset.filter(cut)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text = self.dataset[index]['text']
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = random.randint(0, 1)
        if label == 1:
            j = random.randint(0, len(self.dataset)-1)
            sentence2 = self.dataset[j]['text'][20:40]
        return sentence1, sentence2, label

print("定义数据集".center(50, "="))
train_dataset = PairDataset('train')
print("查看数据集样例".center(50, "="))
s1, s2, label = train_dataset[7]
print(f"数据集长度: {len(train_dataset)}")
print(f"句子 1: {s1}")
print(f"句子 2: {s2}")
print(f"标签  : {label}")

# %% 定义整理函数
def collate_fn(data):
    sentences = [i[:2] for i in data]
    labels = [i[2] for i in data]
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = sentences,
        truncation               = True,
        padding                  = "max_length",
        max_length               = 45,
        return_tensors           = "pt",
        return_length            = True,
        add_special_tokens       = True)
    
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels

print("定义整理函数".center(50, "="))

tmp_data = [
    ("酒店还是非常的不错，我预定的是套件，服务", "非常好，随叫随到，结账非常快。", 0),
    ("外观很漂亮，性价比感觉还不错，功能简", "单，适合出差携带。蓝牙摄像头都有了。", 0),
    ("《穆斯林的葬礼》我已闻名已久，只是一直没", "怎能享受4星的服务，连空调都不能用的。", 1),
]

print("试运行".center(50, "="))
input_ids, attention_mask, token_type_ids, labels = collate_fn(tmp_data)
print(tokenizer.decode(input_ids[0]))
print(f"{input_ids.shape=}")
print(f"{attention_mask.shape=}")
print(f"{token_type_ids.shape=}")
print(f"{labels=}")

# %% 定义数据加载器
loader = DataLoader(
    dataset     = train_dataset,
    batch_size  = 8,
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
        self.fc = torch.nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )
        out = self.fc(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)
        return out
    
model = Model().to(device)
print("定义下游模型".center(50, "="))
print("模型可训练参数:", sum(p.numel() for p in model.parameters() if p.requires_grad))

print("试运行".center(50, "="))
out = model(input_ids, attention_mask, token_type_ids)
print(f"{out.shape=}")

# %% 训练
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = get_scheduler(
        name               = "linear",
        num_warmup_steps   = 0,
        num_training_steps = len(loader),
        optimizer          = optimizer,
    )
    model.train()

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 20 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Step={i:3d} loss={loss.item():6.4e} accuracy={accuracy*100:>6.2f}% {lr=:6.4e}")

print("训练".center(50, "="))
train()

# %% 测试
def test():
    test_loader = torch.utils.data.DataLoader(
        dataset     = PairDataset('test'),
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