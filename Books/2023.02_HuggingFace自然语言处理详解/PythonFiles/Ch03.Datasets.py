# %%
import os

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from datasets import load_dataset, load_from_disk

# %% 1.加载与保存
local_path = "D:/Speech/_Datasets/Seamew_ChnSentiCorp"
dataset1 = load_dataset(local_path)
print(dataset1)

# dataset2 = load_dataset('glue', name='sst2', split='train')
# print(dataset2)
# dataset2.save_to_disk("./Datasets/glue_sst2_train")
# dataset2 = load_from_disk("./Datasets/glue_sst2_train")
# print(dataset2)

# %% 取数据子集
train_dataset = dataset1['train']
print(train_dataset)
# %% 查看内容
for i in [12, 17, 20, 26, 56]:
    print(train_dataset[i])
# %% 排序
print('排序前'.center(50, '-'))
print(train_dataset['label'][:10])
sorted_dataset = train_dataset.sort('label')
print('排序后'.center(50, '-'))
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])
# %% 打乱
print('打乱数据'.center(50, '-'))
shuffled_dataset = sorted_dataset.shuffle()
print(shuffled_dataset['label'][:10])
# %% 选择
selected_dataset = train_dataset.select([0,10,20,30,40,50])
print(selected_dataset)
# %% 过滤
def f(data):
    return data['text'].startswith('非常不错')

filtered_dataset = train_dataset.filter(f)
print(filtered_dataset)
# %% 拆分数据集
split_dataset= train_dataset.train_test_split(test_size=0.1)
print(split_dataset)
# %% 分桶
bucket_dataset = train_dataset.shard(num_shards=4, index=0)
print(bucket_dataset)
# %% 重命名列
renamed_dataset = train_dataset.rename_column('text', 'text_rename')
print(renamed_dataset)
# %% 删除列
removed_dataset = train_dataset.remove_columns('text')
print(removed_dataset)
# %% 映射
def f(data):
    data['text'] = 'My sentence:' + data["text"]
    return data
mapped_dataset = train_dataset.map(f)
print(train_dataset['text'][20])
print(mapped_dataset['text'][20])
# %% 批处理加速
def f(data):
    text = data['text']
    text = ['My sentence:' + i for i in text]
    data['text'] = text
    return data

mapped_dataset2 = train_dataset.map(
    function   = f,
    batched    = True,
    batch_size = 1000,
    num_proc   = 1)
print(mapped_dataset2['text'][20])
# %% 设置数据格式
train_dataset.set_format(
    type               = 'numpy',
    columns            = ['label'],
    output_all_columns = True
)
print(train_dataset[20])
# %% 保存为 CSV 文件
train_dataset.to_csv(path_or_buf='./Datasets/train.csv')
csv_dataset = load_dataset(path='csv', data_files='./Datasets/train.csv', split='train')
print(csv_dataset[20])
# %% 保存为 Json 文件
train_dataset.to_json(path_or_buf='./Datasets/train.json')
json_dataset = load_dataset(path='json', data_files='./Datasets/train.json', split='train')
print(json_dataset[20])
# %%
