# Vocos + Encodec

数据部分: 
- 定义: vocos.dataset.VocosDataModule
- 训练参数: 文件路径; 采样率: 24000; 样本数: 24,000; 批量大小: 16; 线程数 0;
- 验证参数: 文件路径; 采样率: 24000; 样本数: 24,000; 批量大小: 16; 线程数 8;

模型部分:
- 定义: vocos.experiment.VocosEncodecExp
- 初始参数:
  - 采样率: 24000
  - 学习率: 5e-4
  - 损失权重: Mel 45, MRD 1.0
  - 预热步骤: 0
  - 预训练 Mel 步: 0 (初次迭代为 GAN 损失)
  - 评估: UTMOS + PESQ + Periodicty
  - 特征提取: vocos.feature_extractors.EncodecFeatures
  - 骨干网络: vocos.models.VocosBackbone
  - 头部网络: vocos.heads.ISTFTHead

训练部分:
- 日志: pytorch_lightning.loggers.TensorBoardLogger -> logs/
- 回调:
  - 学习率监控
  - 模型总结
  - 模型检查: 验证损失 + 存储权重
  - vocos.helpers.GradNormCallback
- 迭代: 2000000
- 设备: GPU 0
- 策略: DDP
- 记录频率: 100

## VocosDataModule

## VocosExp → VocosEncodecExp

### VocosExp(LightningModule)

- 特征提取器 + 骨干网络 + 头部网络

前向传播时:
- feature_extractor -> backbone -> head

单步训练:
- 训练判别器
  注: 优化器索引为 0
  调用 `self.forward` 重构音频, 得到 `audio_hat`;
  调用 `self.multiperioddisc` 计算判别器损失, 得到 `real_score_mp`, `gen_score_mp` -> `disc_loss()` -> `loss_mp`, `loss_mp_real`;
  调用 `self.multiresddisc` 计算判别器损失, 得到 `real_score_mrd`, `gen_score_mrd` -> `disc_loss()` -> `loss_mrd`, `loss_mrd_real`;
  `loss = loss_mp/len(loss_mp_real) + mrd_loss_coeff*loss_mrd/len(loss_mrd_real)`
- 训练生成器
  注: 优化器索引为 1
  调用 `self.forward` 重构音频, 得到 `audio_hat`;
  - 若 `train_discriminator`:
    - 调用 `self.multiperioddisc` 计算判别器损失, 得到 `gen_score_mp`, `fmap_rs_mp`, `fmap_gs_mp`;
    - 调用 `self.multiresddisc` 计算判别器损失, 得到 `gen_score_mrd`, `fmap_rs_mrd`, `fmap_gs_mrd`;
    - `gen_score_mp` -> `gen_loss()` -> `loss_gen_mp/=len(list_loss_gen_mp)`;
    - `gen_score_mrd` -> `gen_loss()` -> `loss_gen_mrd/=len(list_loss_gen_mrd)`;
    - `fmap_rs_mp`, `fmap_gs_mp` -> `feat_matching_loss()` -> `loss_fm_mp`;
    - `fmap_rs_mrd`, `fmap_gs_mrd` -> `feat_matching_loss()` -> `loss_fm_mrd`;
  - 否则: `loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd = 0`;
  - `audio_hat` -> `melspec_loss` -> `mel_loss`;
  - `loss = loss_gen_mp + mrd_loss_coeff*loss_gen_mrd + loss_fm_mp + mrf_loss_coeff*loss_fm_mrd+mel_loss_coeff*mel_loss`;
- 返回损失值

单步验证: #TODO
- 从 Metric 的 `calculate_periodicity_metrics` 计算 `periodicity_loss`, `pitch_loss`, `f1_score`
- 使用 utmos_model.score 计算 utmos 得分;
- 计算 pesq_score
- 计算 mel_loss

核心部分是:
- 判别器: `MultiPeriodDiscriminator`, `MultiResolutionDiscriminator`;
- 损失函数: `DiscriminatorLoss`, `GeneratorLoss`, `FeatureMatchingLoss`, `MelSpecReconstructionLoss`

### VocosEncodecExp(VocosExp)

注: VocosEncodecExp: 继承 VocosExp, 条件判别器为 MultiPeriodDiscriminator 和 MultiResolutionDiscriminator.
对单步训练, 单步验证, 结束验证加入 bandwidth_id 参数.

## FeatureExtractor → EncodecFeatures

FeatureExtractor 仅占位. 
对给定的音频波形提取特征, 特征为 (B,C,L), B 为 batch size, C 为特征维度, L 为序列长度.

EncodecFeatures 继承自 FeatureExtractor, 实现了 Encodec 特征提取器.

- 根据字符串选择 EnCodec 模型, 创建 `EncodecModel.encodec_model_{24/48}khz`, 设置参数不可训练.
- `self.num_q`: 根据 `bandwidth` 从 `quantizer` 获取量化器数.
- `self.codebook_weights`: 遍历 `self.encodec.quantizer.vq.layers[: self.num_q]` 的每个 `vq.codebook` 进行拼接.

前向传播时:
- 判断输入参数是否有 `bandwidth_id`, 没有则为 None. 注: EnCodec 强制需求此参数. 从 `self.bandwidths=[1.5,3.0,6.0,12.0]` 中取出对应 `bandwidth_id` 的值.
- 切换 `self.encodec` 为 `eval` 模式
- 设置目标带宽 `self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])`
- 调用 `self.get_encodec_codes`:
  - 输入音频 `audio`
  - `unsqueeze(1)`
  - 调用 `self.encodec.encoder(audio)` 输出 `emb`
  - 调用 `self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)` 输出 `codes`
- 定义 `offsets = torch.arange(0, self.encodec.quantizer.bins * len(codec), self.encodec.quantizer.bins, device)`
- `embedding_idxs = codes + offsets.view(-1,1,1)`
- 调用 `F.embedding(embedding_idxs, self.codebook_weight).sum(dim=0)` 输出特征
- 转置 `(1,2)`

核心部分是:
- encodec.EncodecModel (第三方包)
- 对 VQ 码本的处理 #TODO

## Backbone → VocosBackbone

Backbone 仅占位.

VocosBackbone 建立在 ConvNeXt 块上, 额外的条件信息使用 Adaptive Layer Normalization 加入.

前向传播时:
- 判断输入参数是否有 `bandwidth_id`, 没有则为 None.
- 调用 `self.embed [nn.Conv1d(input_channels, dim, kernal_size=7, padding=3)]`, 进行卷积.
- 若有条件信息 `self.adanorm`, 断言 `bandwidth_id` 非 None, 转置 `(1,2)` 调用 `self.norm [AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)]`
- 否则, 转置 `(1,2)` 调用 `self.norm [LayerNorm(dim, eps=1e-6)]`
- 将结果转置 `(1,2)`;
- 遍历 `self.convnext` 的每个卷积块, 调用 `conv_block [ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value, adanorm_num_embeddings)]`;
- 将结果转置 `(1,2)`
- 调用 `self.final_layer_norm [nn.LayerNorm(dim, eps=1e-6)]`

核心部分是:
- modules.ConvNeXtBlock
- modules.AdaLayerNorm

### ConvNeXtBlock

该模块修改自 `https://github.com/facebookresearch/ConvNeXt` 以适配一维音频信号.

ConvNeXtBlock 继承自 nn.Module, 实现了 ConvNeXt 块.

前向传播时:
- 输入参数: x, cond_embedding_id
- 定义残差连接 residual
- 调用深度可分离卷积 `self.dwconv [nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)]`
- 转置 `(1,2)`: B,C,T -> B,T,C
- 若有条件信息 `self.adanorm`, 调用 `self.norm [AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)]`
- 否则, 调用 `self.norm [LayerNorm(dim, eps=1e-6)]`
- 调用 1×1 卷积, `self.pwconv [nn.Linear(dim, intermediate_dim)]`
- 激活函数 `self.act [nn.GELU()]`
- 调用 1×1 卷积, `self.pwconv2 [nn.Linear(intermediate_dim, dim)]`
- 若 `layer_scale_init_value>0`, 创建可训练参数 `self.gamma`, 和结果相乘
- 转置 `(1,2)`: B,T,C -> B,C,T
- 加上残差连接: x + residual
- 输出: x

### AdaLayerNorm

AdaLayerNorm 继承自 nn.Module, 实现了 Adaptive Layer Normalization.
每个 `num_embeddings` 类都有可训练嵌入.

前向传播时:
- 输入参数: x, cond_embedding_id
- 对条件信息调用 `self.scale [nn.Embedding(num_embeddings, embedding_dim)]`, 初始为 1, 得到 `scale`;
- 对条件信息调用 `self.shift [nn.Embedding(num_embeddings, embedding_dim)]`, 初始为 0, 得到 `shift`;
- 对输入 `x` 调用 `layer_norm(x, (self.dim, ), eps=self.eps)`
- 输出: `x * scale + shift`

## FourierHead → ISTFTHead

FourierHead 仅占位: 傅里叶逆变换模块, 输入 (B,L,H), L 为序列长度, H 为模型维度, 用于重构形为 (B,T) 的时域音频信号.

ISTFTHead 继承自 FourierHead, 用于预测 STFT 复系数.

前向传播时:
- 输入参数: x: (B,L,M)
- 调用 `self.out [nn.Linear(dim, out_dim)]`, 并转置 `(1,2)`, 输出 x;
- `x.chunk(2, dim=1)` 输出 mag, p;
- 对过大的 `magnitudes` 进行裁剪 `torch.clip`;
- `cos(p)->x; sin(p)->y`;
- `S = mag*(x+1j*y)`
- `audio = self.isft(S)`

核心部分是:
- spectral_ops.ISTFT

### ISTFT

因为 `torch.istft` 仅支持 center 填充方式, 会导致边缘处 NOLA (非零重叠相加) 检查失败.
具体而言, 在声码器的背景下, 关注的是类似卷积神经网络的 same 填充.
由于会修建填充样本, 因此 NOLA 约束得以满足.

前向传播时:
- center 时, 直接调用 `torch.istft(spec, n_fft, hop_length, win_length, window, center=True)`;
- same 时, 定义 `pad = (win_length - hop_length)//2`
- 频谱的维度为 3, 即 B, N, T
- 逆实数快速傅里叶变换: `ifft = torch.fft.irfft(spec, n_fft, dim=1, norm='backward')`
- `ifft *= self.window[None, :, None]`
- 计算输出信号的总长度: `output_size=(T-1)*self.hop_length+self.win_length`
- 调用 `F.fold()` 将 `ifft` 重叠相加以重建信号
- 计算窗口函数的平方, 并扩展维度以匹配后续操作
- 计算窗口函数的平方在时间上的分布, 窗口包络? 
- 断言窗口包络的所有值都大于 1e-11 避免归一化时除 0
- 将重建的信号除以窗口包络, 进行归一化处理, 确保信号能量在整个时间轴上保持一致.

#TODO: 完善理论