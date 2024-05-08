# %% 加载模块
from transformers import BertTokenizer, BertModel, AdamW
from transformers.optimization import get_scheduler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# %% 定义计算设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("使用设备：", device)

# %% 定义数据集
class ClassificationDataset(Dataset):

    def __init__(self, split):
        data_dir = 'D:\Speech\_Datasets\Seamew_ChnSentiCorp'
        self.dataset = load_dataset(data_dir)[split]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text  = self.dataset[index]['text']
        label = self.dataset[index]['label']
        return text, label

print("加载数据集".center(50, "="))
train_dataset = ClassificationDataset(split='train')
print("训练集大小：", len(train_dataset))
print("训练样本示例：", train_dataset[20])
test_dataset = ClassificationDataset(split='test')
print("测试集大小：", len(test_dataset))
print("测试样本示例：", test_dataset[20])

# %% 加载分词器
print("加载分词器".center(50, "="))
tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer)
# %% 定义数据整理函数
print("定义数据整理函数".center(50, "="))
def collate_fn(data):
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = sentences,
        truncation               = True,
        padding                  = 'max_length',
        return_length            = True,
        return_tensors           = 'pt'
    )
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels

print("模拟数据整理运行".center(50, "="))
data = [
    ('你站在桥上看风景', 1),
    ('看风景的人在楼上看你', 0),
    ('明月装饰了你的窗子', 1),
    ('你装饰了别人的梦', 0),
]
input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
print("input_ids 大小：", input_ids.size())
print("attention_mask 大小：", attention_mask.size())
print("token_type_ids 大小：", token_type_ids.size())
print("labels：", labels)

# %% 定义数据加载器
loader = DataLoader(
    dataset    = train_dataset,
    batch_size = 16,
    collate_fn = collate_fn,
    shuffle    = True,
    drop_last  = True
)
print("数据加载器：", len(loader))

print("查看数据示例".center(50, "="))
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break
print("input_ids 大小：", input_ids.size())
print("attention_mask 大小：", attention_mask.size())
print("token_type_ids 大小：", token_type_ids.size())
print("labels：", labels)

# %% 定义模型
print("加载预训练模型".center(50, "="))
pretrained = BertModel.from_pretrained('bert-base-chinese')
pretrained.to(device)
print("模型参数", sum(p.numel() for p in pretrained.parameters()) / 10000, "万")

for param in pretrained.parameters():
    param.requires_grad = False
print("模型运行".center(50, "="))
out = pretrained(
    input_ids   = input_ids,
    attention_mask = attention_mask,
    token_type_ids = token_type_ids
)
print(out.last_hidden_state.shape)

# %% 定义下游任务
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(
                input_ids   = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out

print("定义模型".center(50, "="))
model = Model()
model.to(device)
print("模型参数", sum(p.numel() for p in model.parameters()))

print("模型运行".center(50, "="))
out = model(
    input_ids   = input_ids,
    attention_mask = attention_mask,
    token_type_ids = token_type_ids)
print("模型输出尺寸:", out.shape)

# %% 训练
def train():
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(
        name               = 'linear',
        num_warmup_steps   = 0,
        num_training_steps = len(loader),
        optimizer          = optimizer)
    model.train()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(
            input_ids   = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f"第{i}步，损失：{loss.item():.4f}, 准确率：{accuracy:.4f}, 学习率：{lr:.6f}")

train()

# %% 测试
def test():
    loader_test = DataLoader(
        dataset    = test_dataset,
        batch_size = 32,
        collate_fn = collate_fn,
        shuffle    = True,
        drop_last  = True)
    model.eval()
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        # if i == 5:
        #     break
        print(i)
        with torch.no_grad():
            out = model(
                input_ids   = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
    accuracy = correct / total
    print(f"测试集准确率：{accuracy:.4f}")

test()
# %%
