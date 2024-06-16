# Vocos 源码解读

- 项目创建: 2023.06.01
- 项目更新: 2023.11.21
- 项目作者: [Hubert Siuzdak](../../Authors/Hubert_Siuzdak.md); 
- 笔记更新: 2024.06.16

## 项目结构

- [ ] `vocos/`
  - [x] ~~`.github/`~~
     - [x] ~~`workflows/`~~
       - [x] ~~`pypi-release.yml`~~
  - [ ] `configs/`
    - [x] `vocos-encodec.yaml`
    - [ ] `vocos-imdct.yaml`
    - [ ] `vocos-resnet.yaml`
    - [ ] `vocos.yaml`
  - [ ] `metrics/`
    - [ ] `UTMOS.py`
    - [ ] `periodicity.py`
  - [ ] `notebooks/`
    - [ ] `Bark+Vocos.ipynb`
  - [ ] `vocos/`
    - [x] `__init__.py` (version=0.1.0, `from pretrained import Vocos`)
    - [ ] `dataset.py`
    - [ ] `discriminator.py`
    - [ ] `experiment.py`
    - [ ] `feature_extractor.py`
    - [ ] `heads.py`
    - [ ] `helpers.py`
    - [ ] `loss.py`
    - [ ] `models.py`
    - [ ] `modules.py`
    - [ ] `pretrained.py`
    - [ ] `spectral_ops.py`
  - [x] ~~`.gitignore`~~
  - [x] ~~`LICENSE`~~ (MIT License)
  - [x] `README.md`
  - [x] `requirements-train.txt`
  - [x] `requirements.txt`
  - [ ] `setup.py`
  - [x] `train.py`

### 预训练模型

- [ ] `charactr/`
  - [ ] `vocos-mel-24khz/`: <https://hf-mirror.com/charactr/vocos-mel-24khz/tree/main>
    - [ ] `.gitattributes`
    - [ ] `README.md` 
    - [ ] `config.yaml`
    - [ ] `pytorch_model.bin` (54.4 MB)
  - [ ] `vocos-encoder-24khz/`: <https://hf-mirror.com/charactr/vocos-encodec-24khz/tree/main>
    - [ ] `.gitattributes`
    - [ ] `README.md` 
    - [ ] `config.yaml`
    - [ ] `pytorch_model.bin` (40.4 MB)

## 环境配置

运行环境：`requirements.txt`
- torch
- torchaudio
- numpy
- scipy
- einops
- pyyaml
- huggingface_hub
- encodec=0.1.1

训练环境: `requirements-train.txt`
- pytorch_lightning==1.8.6
- jsonargparse['signature']
- transformers
- matplotlib
- torchcrepe
- pesq
- fairseq

## 基本用法

- 安装: `pip install vocos`
- 如需训练: `pip install vocos[train]`

### 从梅尔频谱重构音频:

```python
import torch
from vocos import Vocos

vocos = Vocos.from_pretrained('charactr/vocos-mel-24khz')

mel = torch.randn(1, 100, 256) # B, C, T
audio = vocos.decode(mel)
```

读取音频文件后，先将其采样率转换为 24 kHz, 然后使用 Vocos 进行重构.

```python
import torchaudio

y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
y_hat = vocos(y)
```

### 从 EnCodec Token 重构音频

需要提供嵌入对应的 `bandwidth_id`, 可选值: `[1.5,3.0,6.0,12.0]`

```python
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))  # 8 codeboooks, 200 frames
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])  # 6 kbps

audio = vocos.decode(features, bandwidth_id=bandwidth_id)
```

读取音频文件后, 先将其采样率转换为 24 kHz, 然后使用 EnCodec 提取并量化得到特征, 然后使用 Vocos 进行重构.

```python
y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

### 与 Bark 文本转语音模型结合

`notebook/Bark+Vocos.ipynb`

### 训练模型

准备训练和验证的音频文件列表:
```python
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

填写配置文件, 如 `configs/vocos.yaml` 然后开始训练:


```python
python train.py -c configs/vocos.yaml
```

具体代码:
```
from pytorch_lightning.cli import LightningCLI

cli = LightningCLI(run=False)
cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
```

自定义训练过程则请查阅 PyTorch Lightning 的文档.

## 引用

```
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```