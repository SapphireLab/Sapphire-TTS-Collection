# F5-TTS 训练教程

## 准备数据集

示例的数据处理脚本为 Emilia 和 Wenetspeech4TTS 设计，你可以在 `src/f5_tts/model/dataset.py` 中定义自己的 Dataset 类进行私人定制.

### 1. 预训练模型使用的数据集

首先下载对应的数据集, 并在脚本中填入路径.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py
```

### 2. 使用 metadata.csv 创建自定义数据集

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py
```

可以遵循 [讨论 #57](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029) 评论中提供的指导.
内容复述如下:

#### 1.创建数据集

准确 3~12 s 时长的音频文件 (效果不确定) 和它们的转录文本.
相应结构为:
```
data/
├── metadata.csv
├── wavs/
    ├── audio1.wav
    ├── audio2.wav
```

metadata.csv 内容示例 (`<relative_path_to_wav>|<transcript>`):

```
audio_file|text
wavs/audio_0001.wav|Yo! Hello? Hello?
wavs/audio_0002.wav|Hi, how are you doing today? I want to go shopping and buy me some lemons.
```

#### 2.调用脚本准备数据集

分词对英文数据集和拼音有效, 没有其他的分词, 需要自己修改.

```bash
python "src\f5_tts\train\datasets\prepare_csv_wavs.py" <path_to_your_dataset> <F5-TTS_repo_data_path>/<dataset_name>_pinyin
```

#### 3.调整训练文件中的超参数

将 f5_tts 数据文件夹设置数据集名称

```python
dataset_name       = "your_dataset"
max_samples        = 2 #最大样本数
learning_rate      = 5e-06 #学习率
epochs             = 10 #线性衰减
num_warmup_updates = 20 #预热步数
last_per_steps     = 500 # 保存模型的步数
```

#### 4.执行训练

## 训练与微调

数据集准备完成后, 可以开始训练过程

### 1. 预训练模型的训练脚本

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config
accelerate launch src/f5_tts/train/train.py
```

### 2. 微调实践

- 微调讨论: [#57](https://github.com/SWivid/F5-TTS/discussions/57).
- Gradio UI 训练/微调 (`src/f5_tts/train/finetune_gradio.py`): [#143](https://github.com/SWivid/F5-TTS/discussions/143)

### 3. Wandb 记录

`wandb/` 文件夹会在训练/微调时的路径下自动创建.
默认情况下, 训练脚本不会使用 logging (假设你没有手动使用 `wandb login` 进行登录).
为了启用 wandb logging, 你可以选择以下两种方式:
1. 使用命令 `wandb login` 手动登录 wandb 账号: 了解更多 [这里](https://docs.wandb.ai/ref/cli/wandb-login)
2. 设置环境变量自动登录, 从[此处](https://wandb.ai/site/)获取 API KEY, 并按照如下方式设置环境变量:
   - Mac & Linux: `export WANDB_API_KEY=<YOUR WANDB API KEY>`
   - Windows: `set WANDB_API_KEY=<YOUR WANDB API KEY>`

此外, 如果你无法访问 Wandb 并希望离线记录指标, 你可以设置环境变量如下:
```
export WANDB_MODE=offline
```
