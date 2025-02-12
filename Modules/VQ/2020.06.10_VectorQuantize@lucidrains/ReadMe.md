# Vector Quantization (VQ)

- 项目创建: 2020.06.10
- 项目更新: 2025.02.26
- 项目作者: Phil Wang@lucidrains
- 笔记更新: 2025.02.27

## 项目结构

版本: 1.21.9

<table><tr><td width="50%">

- [x] ~~`.github/workflows`~~
  - [x] ~~`build.yml`~~
  - [x] ~~`python-publish.yml`~~
  - [x] ~~`test.yml`~~

</td><td>

- [x] Github 工作流文件夹
  - [x] Rye 项目管理工具-构建工作流 (Push 时触发)
  - [x] PyPI 发布工作流
  - [x] Rye 项目管理工具-PyTest 测试工作流 (Push/PR 时触发, tests 文件夹)

</td></tr>
<tr><td>

- [ ] `examples/`
  - [ ] `autoencoder_fsq.py`
  - [ ] `autoencoder_lfq.py`
  - [ ] `autoencoder_sim_vq.py`
  - [ ] `autoencoder.py`

</td></tr>
<tr><td>

- [ ] `images/`
  - [ ] `fsq.png`
  - [ ] `lfq.png`
  - [ ] `simvq.png`
  - [ ] `vq.png`

</td></tr>
<tr><td>

- [ ] `tests/`
  - [ ] `test_latent_quantization.py`
  - [ ] `test_lfq.py`
  - [ ] `test_readme.py`

</td></tr>
<tr><td>

- [ ] `vector_quantize_pytorch.py`
  - [x] `__init__.py`
  - [ ] `finite_scalar_quantization.py` -> `FSQ`
  - [ ] `latent_quantization.py` -> `LatentQuantizer`
  - [ ] `lookup_free_quantization.py` -> `LFQ`
  - [ ] `random_projection_quantizer.py` -> `RandomProjectionQuantizer`
  - [ ] `residual_fsq.py` -> `ResidualFSQ`, `GroupedResidualFSQ`
  - [ ] `residual_lfq.py` -> `ResidualLFQ`, `GroupedResidualLFQ`
  - [ ] `residual_sim_vq.py` -> `ResidualSimVQ`
  - [ ] `residual_vq.py` -> `ResidualVQ`, `GroupedResidualVQ`
  - [ ] `sim_vq.py` -> `SimVQ`
  - [ ] `utils.py`
  - [ ] `vector_quantize_pytorch.py` -> `VectorQuantizer`

</td><td>

- [x] 向量量化文件夹
  - [x] 初始化: 从其他文件导入对应模块.

</td></tr>
<tr><td>

- [x] `.gitignore`
- [x] `LICENSE`
- [ ] `pyproject.toml`
- [ ] `README.md`
- [ ] `ruff.toml`

</td><td>

- [x] 忽略文件
- [x] **MIT** 开源许可
- [x] 项目环境配置文件
  - Python >= 3.9
  - PyTorch >= 2.0
  - einops >= 0.8.0
  - einx >= 0.3.0
  - tqdm 可选
  - torchvision 可选
  - hatchling 构建
  - Rye 配置
    - RuFF >= 0.4.2
    - pytest >= 8.2.0
    - pytest-cov >= 5.0.0
- [x] RuFF 代码检查配置文件

</td></tr></table>

## 基本用法

<table><tr><td width="50%">

本项目是根据 DeepMind 的 TensorFlow 实现版本改写的矢量量化库, 能以包的方式方便调用.
采用指数移动平均用于更新字典.

VQ 被 DeepMind 和 OpenAI 成功用于高质量图像生成 (VQ-VAE-2) 和音乐生成 (Jukebox).

- 安装: `pip install vector-quantize-pytorch`

</td></tr>
<tr><td>

### VQ

```python
import torch
from vector_quantize_pytorch import VectorQuantizer

vq = VectorQuantizer(
  dim               = 256,
  codebook_size     = 512, # 码本尺寸
  decay             = 0.8, # 指数移动平均衰减率, 越低表示字典更新越快
  commitment_weight = 1.   # Commitment Loss 权重
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = vq(x)
#(1,1024,256), (1,1024), (1)
```

</td></tr>
<tr><td>

### Residual VQ

[SoundStream](../../../Models/SpeechCodec/2021.07.07_SoundStream.md) 提出使用多个向量量化器以递归地量化波形的残差.
可以使用 `ResidualVQ` 类和一个额外的初始参数进行调用.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim               = 256,
    codebook_size     = 1024,
    num_quantizers    = 8,    # 指定量化器的数量
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = vq(x)
#(1,1024,256), (1,1024,8), (1,8)
```

如果需要横跨整个量化层的所有编码, 只需要传递 `return_all_codes=True`:


```python
quantized, indices, commit_loss, all_codes = vq(x, return_all_codes=True)
#(1,1024,256), (1,1024,8), (1,8), (8,1,1024,256)
```

另外, **RQ-VAE** 使用 `ResidualVQ` 用于构造 **RQ-VAE**, 使用更紧凑的编码以生成高分辨率图像.
**RQ-VAE** 做了两处修改:
1. 所有量化器共享码本 Codebook;
2. 随机采样编码而不是选择最近匹配.

可以通过两个额外的关键词参数来启用这两个特性.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim                     = 256,
    codebook_size           = 1024,
    num_quantizers          = 8,
    stochastic_sample_codes = True, # 是否随机采样编码
    sample_codebook_temp    = 0.1,  # 随机采样编码的温度, 0 等价于非随机
    shared_codebook         = True, # 所有量化器是否共享码本
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = residual_vq(x)
#(1,1024,256), (1,1024,8), (1,8)
```

</td></tr>
<tr><td>

### Grouped Residual VQ

[HiFi-Codec](../../../Models/SpeechCodec/2023.05.04_HiFi-Codec.md) 进一步提出在特征维度分组使用 `ResidualVQ`, 使用非常少的码本就能获得和 **EnCodec** 等价的结果.
可以使用 `GroupedResidualVQ` 调用

```python
import torch
from vector_quantize_pytorch import GroupedResidualVQ

residual_vq = GroupedResidualVQ(
    dim                     = 256,
    codebook_size           = 1024,
    num_quantizers          = 8,
    groups                  = 2,    # 分组数量
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = residual_vq(x)
#(1,1024,256), (2,1,1024,8), (2,1,8)
```

</td></tr>
<tr><td>

### 初始化技巧

[SoundStream](../../../Models/SpeechCodec/2021.07.07_SoundStream.md) 提出码本应该使用第一个批次的 K 均值中心来初始化.
可以设置 `kmeans_init=True` 来启动这一特性, 这对 `VectorQuantizer` 和 `ResidualVQ` 都适用.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim            = 256,
    codebook_size  = 145,
    num_quantizers = 4,
    kmeans_init    = True, # 是否使用 K 均值初始化
    kmeans_iters   = 10,   # K 均值迭代数
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = residual_vq(x)
#(1,1024,256), (1,1024,4), (1,4)?
```

</td></tr>
<tr><td>

### 梯度计算

**VQ-VAEs** 通常使用**直通估计器 (Straight-Through Estimator, STE)** 来训练.
在反向传播的过程中, 梯度在 VQ 层周围流动而不是通过它.

2024 年 10 月的论文 "**Restructuring Vector Quantization with the Rotation Trick**" 提出通过 VQ 层转换梯度, 使得输入向量和量化输出之间的相对角度和大小被编码到梯度中.
用户可以使用 `rotation_trick=True/False` 来启用或禁用 `VectorQuantizer` 类的这一特性.

```python
from vector_quantize_pytorch import VectorQuantizer

vq = VectorQuantizer(
  dim               = 256,
  codebook_size     = 256,
  rotation_trick    = True, # True 使用旋转技巧/False 使用直通估计器
)
```

</td></tr>
<tr><td>

### 提高码本使用率

本项目会包含来自各种文献中的数种技术来处理使用向量量化器时常见问题: 无效码本元素 ("Dead" Codebook Entries).

### 更小的码本维度

论文 "**Vector-Quantized Image Modeling with Improved VQGAN** (Improved ViT-VQGAN)" 提出使用更小维度的码本.
编码器的值在量化之前被投影到低维空间，然后在量化之后再投影回高维空间.
用户可以通过设置 `codebook_dim` 超参数来实现这一点。

```python
import torch
from vector_quantize_pytorch import VectorQuantizer

vq = VectorQuantizer(
    dim               = 256,
    codebook_size     = 256,
    codebook_dim      = 16,   # 论文提出设置为 32 或 8 来提升码本使用率
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = vq(x)
#(1,1024,256), (1,1024), (1,)
```

### 余弦相似度

论文 "**Vector-Quantized Image Modeling with Improved VQGAN** (Improved ViT-VQGAN)" 还提出对码本编码和编码后的向量进行 L2 规范化, 简化为使用余弦相似度作为距离.
他们声称将向量限制在球面上能够提升码本使用率和下游重构.
用户可以使用 `use_cosine_sim=True` 来启用.

```python
import torch
from vector_quantize_pytorch import VectorQuantizer

vq = VectorQuantizer(
    dim               = 256,
    codebook_size     = 256,
    use_cosine_sim    = True,
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = vq(x)
#(1,1024,256), (1,1024), (1,)
```

### 移除陈旧编码

[SoundStream](../../../Models/SpeechCodec/2021.07.07_SoundStream.md) 使用了一个方案: 若某个编码的命中次数低于预设的阈值, 使用当前批次的随机选择的向量来替换.
用户可以使用 `threshold_ema_dead_code` 关键字来设置.

```python
import torch
from vector_quantize_pytorch import VectorQuantizer

vq = VectorQuantizer(
  dim                     = 256,
  codebook_size           = 512,
  threshold_ema_dead_code = 2,   # 应主动替换那些有指数移动平均聚类大小小于 2 的编码
)

x = torch.randn(1,1024,256)
quantized, indices, commit_loss = vq(x)
#(1,1024,256), (1,1024), (1,)
```

### 正交正则损失

"**Exploration into Translation-Equivariant Image Quantization**" 提出当在图像上使用向量量化, 强制码本是正交的可以使得离散编码具有平移等变性, 从而在下游的文本到图像生成任务中带来显著的提升.
用户可以使用 `orthogonal_reg_weight` 设置为大于零的值来启用这一特性.
这种情况下, 正交正则化将被添加到模块输出的辅助损失.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    accept_image_fmap = True,                   # True 表示传递图像特征图
    orthogonal_reg_weight = 10,                 # 论文中推荐设置为 10
    orthogonal_reg_max_codes = 128,             # 限制内存用量, 从码本随机采样用于正交正则化损失
    orthogonal_reg_active_codes_only = False    # 非常大的码本设为 True, 仅计算每批次激活的码本的损失
)

img_fmap = torch.randn(1, 256, 32, 32)
quantized, indices, loss = vq(img_fmap)
# (1, 256, 32, 32), (1, 32, 32), (1,)

# 损失现在包含了带权重的正交正则化损失
```

### 多头 VQ

许多文献提出使用多头方法 (每个特征有多个编码) 的离散潜在表示的变体.
本仓库决定提供一个变体: 同一个码本用于矢量量化输入维度 `head` 次.

用户可以使用更有效的方法 (memcodes) 来实现, 来自 [NWT 论文 [Github]](https://github.com/lucidrains/nwt-pytorch).


```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
    heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
    separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
    codebook_size = 8196,
    accept_image_fmap = True
)

img_fmap = torch.randn(1, 256, 32, 32)
quantized, indices, loss = vq(img_fmap)
# (1, 256, 32, 32), (1, 32, 32, 8), (1,)
```

### 随机映射量化

<a href="https://arxiv.org/abs/2202.01855">This paper</a> first proposed to use a random projection quantizer for masked speech modeling, where signals are projected with a randomly initialized matrix and then matched with a random initialized codebook. One therefore does not need to learn the quantizer. This technique was used by Google's <a href="https://ai.googleblog.com/2023/03/universal-speech-model-usm-state-of-art.html">Universal Speech Model</a> to achieve SOTA for speech-to-text modeling.

USM further proposes to use multiple codebook, and the masked speech modeling with a multi-softmax objective. You can do this easily by setting `num_codebooks` to be greater than 1

"Self-supervised Learning with Random-projection Quantizer for Speech Recognition" 首次提出使用随机映射量化器用于掩膜语音建模, 其中信号使用随机初始化矩阵量化器进行映射, 然后和随机初始化码本匹配.
因此不需要学习量化器.
这一技术被 Google 的 Universal Speech Model 采用, 取得了语音到文本建模的 SOTA.

USM


```python
import torch
from vector_quantize_pytorch import RandomProjectionQuantizer

quantizer = RandomProjectionQuantizer(
    dim = 512,               # input dimensions
    num_codebooks = 16,      # in USM, they used up to 16 for 5% gain
    codebook_dim = 256,      # codebook dimension
    codebook_size = 1024     # codebook size
)

x = torch.randn(1, 1024, 512)
indices = quantizer(x)

# (1, 1024, 16)
```

This repository should also automatically synchronizing the codebooks in a multi-process setting. If somehow it isn't, please open an issue. You can override whether to synchronize codebooks or not by setting `sync_codebook = True | False`

### Sim VQ

ICLR 2025 论文提出冻结码本, 编码由线性映射隐式生成的方案.
作者生成这种设置可以减少码本坍缩并更容易收敛.
本仓库发现和旋转技巧结合以及将线性映射扩展为小的一层 MLP 可以获得更好的结果.



### Finite Scalar Quantization

<img src="./images/fsq.png" width="500px"></img>

|                  | VQ | FSQ |
|------------------|----|-----|
| Quantization     | argmin_c \|\| z-c \|\| | round(f(z)) |
| Gradients        | Straight Through Estimation (STE) | STE |
| Auxiliary Losses | Commitment, codebook, entropy loss, ... | N/A |
| Tricks           | EMA on codebook, codebook splitting, projections, ...| N/A |
| Parameters       | Codebook | N/A |

[This](https://arxiv.org/abs/2309.15505) work out of Google Deepmind aims to vastly simplify the way vector quantization is done for generative modeling, removing the need for commitment losses, EMA updating of the codebook, as well as tackle the issues with codebook collapse or insufficient utilization. They simply round each scalar into discrete levels with straight through gradients; the codes become uniform points in a hypercube.

Thanks goes out to [@sekstini](https://github.com/sekstini) for porting over this implementation in record time!

```python
import torch
from vector_quantize_pytorch import FSQ

quantizer = FSQ(
    levels = [8, 5, 5, 5]
)

x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
xhat, indices = quantizer(x)

# (1, 1024, 4), (1, 1024)

assert torch.all(xhat == quantizer.indices_to_codes(indices))
```

An improvised Residual FSQ, for an attempt to improve audio encoding.

Credit goes to [@sekstini](https://github.com/sekstini) for originally incepting the idea [here](https://github.com/lucidrains/vector-quantize-pytorch/pull/74#issuecomment-1742048597)

```python
import torch
from vector_quantize_pytorch import ResidualFSQ

residual_fsq = ResidualFSQ(
    dim = 256,
    levels = [8, 5, 5, 3],
    num_quantizers = 8
)

x = torch.randn(1, 1024, 256)

residual_fsq.eval()

quantized, indices = residual_fsq(x)

# (1, 1024, 256), (1, 1024, 8)

quantized_out = residual_fsq.get_output_from_indices(indices)

# (1, 1024, 256)

assert torch.all(quantized == quantized_out)
```

### Lookup Free Quantization

<img src="./images/lfq.png" width="450px"></img>

The research team behind <a href="https://arxiv.org/abs/2212.05199">MagViT</a> has released new SOTA results for generative video modeling. A core change between v1 and v2 include a new type of quantization, look-up free quantization (LFQ), which eliminates the codebook and embedding lookup entirely.

This paper presents a simple LFQ quantizer of using independent binary latents. Other implementations of LFQ exist. However, the team shows that MAGVIT-v2 with LFQ significantly improves on the ImageNet benchmark. The differences between LFQ and 2-level FSQ includes entropy regularizations as well as maintained commitment loss.

Developing a more advanced method of LFQ quantization without codebook-lookup could revolutionize generative modeling.

You can use it simply as follows. Will be dogfooded at <a href="https://github.com/lucidrains/magvit2-pytorch">MagViT2 pytorch port</a>

```python
import torch
from vector_quantize_pytorch import LFQ

# you can specify either dim or codebook_size
# if both specified, will be validated against each other

quantizer = LFQ(
    codebook_size = 65536,      # codebook size, must be a power of 2
    dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
    entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
    diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
)

image_feats = torch.randn(1, 16, 32, 32)

quantized, indices, entropy_aux_loss = quantizer(image_feats, inv_temperature=100.)  # you may want to experiment with temperature

# (1, 16, 32, 32), (1, 32, 32), ()

assert (quantized == quantizer.indices_to_codes(indices)).all()
```

You can also pass in video features as `(batch, feat, time, height, width)` or sequences as `(batch, seq, feat)`

```python
import torch
from vector_quantize_pytorch import LFQ

quantizer = LFQ(
    codebook_size = 65536,
    dim = 16,
    entropy_loss_weight = 0.1,
    diversity_gamma = 1.
)

seq = torch.randn(1, 32, 16)
quantized, *_ = quantizer(seq)

assert seq.shape == quantized.shape

video_feats = torch.randn(1, 16, 10, 32, 32)
quantized, *_ = quantizer(video_feats)

assert video_feats.shape == quantized.shape
```

Or support multiple codebooks

```python
import torch
from vector_quantize_pytorch import LFQ

quantizer = LFQ(
    codebook_size = 4096,
    dim = 16,
    num_codebooks = 4  # 4 codebooks, total codebook dimension is log2(4096) * 4
)

image_feats = torch.randn(1, 16, 32, 32)

quantized, indices, entropy_aux_loss = quantizer(image_feats)

# (1, 16, 32, 32), (1, 32, 32, 4), ()

assert image_feats.shape == quantized.shape
assert (quantized == quantizer.indices_to_codes(indices)).all()
```

An improvised Residual LFQ, to see if it can lead to an improvement for audio compression.

```python
import torch
from vector_quantize_pytorch import ResidualLFQ

residual_lfq = ResidualLFQ(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 8
)

x = torch.randn(1, 1024, 256)

residual_lfq.eval()

quantized, indices, commit_loss = residual_lfq(x)

# (1, 1024, 256), (1, 1024, 8), (8)

quantized_out = residual_lfq.get_output_from_indices(indices)

# (1, 1024, 256)

assert torch.all(quantized == quantized_out)
```

### Latent Quantization

Disentanglement is essential for representation learning as it promotes interpretability, generalization, improved learning, and robustness. It aligns with the goal of capturing meaningful and independent features of the data, facilitating more effective use of learned representations across various applications. For better disentanglement, the challenge is to disentangle underlying variations in a dataset without explicit ground truth information. This work introduces a key inductive bias aimed at encoding and decoding within an organized latent space. The strategy incorporated encompasses discretizing the latent space by assigning discrete code vectors through the utilization of an individual learnable scalar codebook for each dimension. This methodology enables their models to surpass robust prior methods effectively.

Be aware they had to use a very high weight decay for the results in this paper.

```python
import torch
from vector_quantize_pytorch import LatentQuantize

# you can specify either dim or codebook_size
# if both specified, will be validated against each other

quantizer = LatentQuantize(
    levels = [5, 5, 8],      # number of levels per codebook dimension
    dim = 16,                   # input dim
    commitment_loss_weight=0.1,
    quantization_loss_weight=0.1,
)

image_feats = torch.randn(1, 16, 32, 32)

quantized, indices, loss = quantizer(image_feats)

# (1, 16, 32, 32), (1, 32, 32), ()

assert image_feats.shape == quantized.shape
assert (quantized == quantizer.indices_to_codes(indices)).all()
```

You can also pass in video features as `(batch, feat, time, height, width)` or sequences as `(batch, seq, feat)`

```python

import torch
from vector_quantize_pytorch import LatentQuantize

quantizer = LatentQuantize(
    levels = [5, 5, 8],
    dim = 16,
    commitment_loss_weight=0.1,
    quantization_loss_weight=0.1,
)

seq = torch.randn(1, 32, 16)
quantized, *_ = quantizer(seq)

# (1, 32, 16)

video_feats = torch.randn(1, 16, 10, 32, 32)
quantized, *_ = quantizer(video_feats)

# (1, 16, 10, 32, 32)

```

Or support multiple codebooks

```python
import torch
from vector_quantize_pytorch import LatentQuantize

model = LatentQuantize(
    levels = [4, 8, 16],
    dim = 9,
    num_codebooks = 3
)

input_tensor = torch.randn(2, 3, dim)
output_tensor, indices, loss = model(input_tensor)

# (2, 3, 9), (2, 3, 3), ()

assert output_tensor.shape == input_tensor.shape
assert indices.shape == (2, 3, num_codebooks)
assert loss.item() >= 0
```

## 引用