# GPT-SoVITS

整个项目的基本结构:

- [ ] GPT_SoVITS 文件夹
  - [ ] prepare_datasets 文件夹对应创建数据集；
  - [ ] text 文件夹对应文本前端；
  - [ ] feature_extractor 音频特征提取部分;
  - [ ] AR 文件夹 对应 GPT 部分;
  - [ ] module 文件夹 对应 SoVITS 部分;
- [ ] tools 文件夹
  - [ ] ASR 部分
  - [ ] 降噪部分
  - [ ] 语言部分
  - [ ] 音源分离
  - [ ] 校对部分

注: 忽略 Onnx 部分.

## GPT 部分

- [ ] AR 文件夹 (基于 pytorch.lightning)
  - [ ] data 文件夹
    - [ ] dataset.py 数据集
    - [ ] data_module.py 数据模块
    - [ ] bucket_sampler.py 分桶采样
  - [ ] models 文件夹
    - [ ] t2s_lightning_module.py 
    - [ ] t2s_model.py 
    - [ ] utils.py 
  - [ ] modules 文件夹
    - [ ] activation.py 激活函数
    - [ ] embedding.py 嵌入
    - [ ] lr_scheduler.py 学习率衰减
    - [ ] optim.py 优化
    - [ ] patched_mha_with_cache.py 多头注意力机制
    - [ ] scaling.py 缩放
    - [ ] transformer.py Transformer
  - [ ] text_processing 文件夹
    - [ ] phonemizer.py 音素化
    - [ ] symbols.py 符号
  - [ ] utils 文件夹
    - [ ] initializer.py 初始化
    - [ ] io.py 输入输出
- [ ] s1_train.py 使用 lightning Traniner 训练 

---

## SoVITS 部分

- [ ] module 文件夹 (基于 torch)
  - [ ] attentions.py 注意力机制
  - [ ] commons.py 基本函数
  - [ ] core_vq.py 矢量量化核心部分
  - [ ] data_utils.py 数据集加载
  - [ ] losses.py 损失函数
  - [ ] mel_processing.py 梅尔处理部分
  - [ ] models.py 模型
  - [ ] modules.py 模块
  - [ ] mrte_model.py MRTE 模块
  - [ ] quantize.py 量化
  - [ ] transforms.py Transforms 函数 (时长预测部分)
- [ ] s2_train.py 训练使用 DDP 并行训练

---

这部分以 VITS 为基础, 进行了一些修改.
没有使用随机时长预测器和时长预测器, 增加了残差矢量量化器.

`SynthesizerTrn` 类中包含:
- 文本编码器 TextEncoder;
- 生成器 Generator;
- 后验编码器 PosteriorEncoder;
- 流模型 (残差耦合层) ResidualCouplingBlock;
- 参考编码器 MelStyleEncoder;
- 残差矢量量化器 ResidualVector;

**基本的前向传播过程为**:

SSL; SPEC, SPEC_LEN, TEXT, TEXT_LEN;

对 `y` 应用掩膜, 然后由参考编码器输出 `ge`;
对 `ssl` 应用 `ssl_proj` 后然后经过残差矢量量化器得到量化结果 `quantized`;

量化结果 `quantized`, 文本 `text`, `ge` 经过文本编码器得到 `x`, `m_p`, `logs_p`;

`y` 和 `ge` 经过后验编码器得到 `z`, `m_q`, `logs_q`;
- `z` 经过切片得到 `z_slice` 和 `ids_slice`, 然后经过生成器得到结果 `o`;
- `z` 经过流模型得到 `z_p`

输出: 切片解码 `o`; 量化的 commit 损失; z 切片索引; 掩膜; (z, z_p, m_p, logs_p, m_q, logs_q); 量化结果.

**推理过程**:

对 `y` 进行掩膜, 然后由参考编码器输出 `ge`;
对 `ssl` 应用 `ssl_proj` 后然后经过残差矢量量化器得到量化结果 `quantized`;

量化结果 `quantized`, 文本 `text`, `ge` 经过文本编码器得到 `x`, `m_p`, `logs_p`;
用 `m_p` 和 `logs_p` 采样出 `z_p`, `z_p` 经过流模型得到 `z`;
`z` 经过解码得到 `o`.

---

文本编码器 TextEncoder: 
- 对 `y` 掩膜后, 使用 `ssl_proj` 进行维度变换, 然后 `encoder_ssl` 编码器输出 `y`;
- 对 `text` 掩膜后, 使用 `text_embedding` 进行嵌入, 然后 `encoder_text` 编码器输出 `text`;
- 将 `y` 和 `text` 使用 MRTE 模块进行融合输出 `y`;
- `y` 经过第二个编码器 `encoder2` (注意力机制编码器) 输出 `y`;
- 对 `y` 进行映射 `proj`, 对半划分得到均值和对数方差.

相比 VITS 的文本编码器, 增加了 MRTE 模块引入 `y` 的信息. 
(注: 有量化部分的冗余代码)

生成器 Generator (HiFi-GAN):
- conv_pre + ResBlock1/ResBlock2 + conv_post

后验编码器 PosteriorEncoder:
- 一维卷积 + WaveNet + 一维卷积 + 对半划分得到均值和对数方差

流模型 ResidualCouplingBlock:
- 遍历 ResidualCouplingLayer (残差耦合层) 
- 耦合层: Conv1d → WaveNet → Conv1d → 对半划分出不变的 x0 和需要变的 x1 → x1 = m + x1 sigma, 和不变的部分拼接.

Meta-StyleSpeech (https://github.com/KevinMIN95/StyleSpeech/blob/main/models/StyleSpeech.py)
梅尔风格编码器 MelStyleEncoder:
- MISH: `x * torch.tanh(F.softplus(x))`
- spectral (LinearNorm + Mish + Dropout + LinearNorm + Mish + Dropout)
- temporal (Conv1dGLU + Conv1dGLU)
- self_attn (MultiHeadAttention)
- fc (LinearNorm)
- 时域平均池化

正向过程: x → spectral → temporal → self_attn → fc → 时域平均池化 → w

残差矢量量化器 ResidualVectorQuantizer:
- 基于矢量量化 VQ 增加残差机制.

训练用的判别器为 MultiPeriodDiscriminator

---

### 损失函数的计算

首先 `SSL 特征`+`频谱`+`频谱长度`+`文本`+`文本长度` 输入到`生成器 SynthesizerTrn` 得到`分段预测波形` `y_hat`, 量化的 commit_loss 为 `SSL_KL`, 分段索引

将`真实波形`用分段索引进行划分得到`分段真实波形 y`
和 `分段预测波形 y_hat` 一同输入到 `判别器 Discriminator` 得到 `y_d_hat_r` 和 `y_d_hat_g`, 然后计算`判别器损失` 用于更新判别器;

将真实波形对应的 `频谱` 转化为`目标梅尔频谱`, 然后用分段索引切片得到`分段目标梅尔频谱 y_mel`.
将 `分段预测波形` 转化为 `预测梅尔频谱 y_hat_mel`.
将 `分段真实波形` 和 `分段预测波形` 输入到判别器中, 得到 `y_d_hat_r` 和 `y_d_hat_g`, 以及对应的 `fmap_r` 和 `fmap_g`.

- Loss_Mel = L1(y_hat_mel, y_mel)
- Loss_KL = kl_loss(z_p, logs_q, mp, logs_p) ?
- Loss_FM = feature_loss(fmap_r, fmap_g)
- loss_gen = generator_loss(y_d_hat_g)
- loss_commit = kl_ssl 

用于优化生成器.

---

