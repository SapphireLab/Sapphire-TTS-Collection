# Encodec

[论文笔记](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md)
[官方代码](https://github.com/facebookresearch/encodec) 

## 提交记录

初次提交: 2022.10.25
最近提交: 2023.06.20

## 框架

整体框架图
![](../../Models/_Images/2022.10.24_Encodec_FIG01.png)

代码结构树
```
└── Encodec
    ├── modules
    │   ├── __init__.py
    │   ├── conv.py
    │   ├── lstm.py
    │   ├── norm.py
    │   ├── seanet.py
    │   └── transformer.py
    ├── quantization
    │   ├── __init__.py
    │   ├── ac.py
    │   ├── core_vq.py
    │   └── vq.py
    ├── __init__.py
    ├── __main__.py
    ├── balancer.py
    ├── binary.py
    ├── compress.py
    ├── disturb.py
    ├── model.py
    ├── msstftd.py
    ├── py.typed
    └── utils.py
```

下面从 `__main__.py` 文件开始分析:

该文件创建了命令行解析对象 `EnCodec`, 参数有
- `input`: 输入文件路径, torchaudio 支持的数据类型;
- `output`: 输出文件路径;
- `bandwidth`: 带宽, 默认为 `6`, 可选值 `1.5, 3, 6, 12, 24`;
- `hq`: 在 48k Hz 音频上使用 HQ stereo model;
- `lm`: 使用语言模型以降低模型尺寸, 速度减慢 5 倍;
- `force`: 覆盖输出文件;
- `decompress_suffix`: 若未指定输出路径, 则添加解压后缀
- `rescale`: 自动缩放输出结果避免裁剪.

如果输入的文件路径后缀为 `.ecdc`:
- 从 `compress.py` 文件调用 `decompress` 进行解压操作; 

- 否则, 进行压缩操作.
  - 参数为 `hq` 时模型名称为 `encodec_48khz`, 否则为 `encodec_24khz`.
  - 从 `compress.py` 的 `MODELS` 字典中按键取值, 加载模型, 然后设置带宽 `bandwidth`.
  - 用 torchaudio 加载输入文件, 
  - 从 `utils.py` 文件调用 `convert_audio` 将输入转换为模型的输入尺寸,
  - 从 `compress.py` 文件调用 `compress` 函数进行压缩操作;
  - 如果指定的输出格式不为 `.ecdc` 而是 `.wav` 则进行解压后再检查裁剪, 并从 `utils.py` 文件调用 `save_audio` 保存到指定路径.

---

上述过程的关键部分是 `compress.py` 文件:

该文件定义了 
- `MODELS` 字典: 字典的值表示加载模型的方法, 来源于 `model.py` 中的 `EncodecModel` 类.
- `compress()` 函数: 调用 `compress_to_file(model, wav, file_object, use_lm)`
- `decompress()` 函数: 调用 `decompress_from_file(file_object, device)`
- `compress_to_file()` 函数: 使用给定模型将波形压缩到文件对象
  - `model`: `EncodecModel` 类实例;
  - `wav`: `Tensor` 输入波形;
  - `file_object`: 输出文件对象;
  - `use_lm`: 为 True 时使用预训练的语言模型使用熵编码进一步压缩流. 将会减速, 尺寸缩小 20%~30%.
  - 具体过程:
    - lm = model.get_lm_model() if use_lm == True else None
    - frames = model.encode(wav[None])
    - 创建元数据 metadata = {模型名称, 音频长度, 码本数量 `frames[0][0].shape[1]`, use_lm}
    - 遍历 frames, 每个元素为 (frame, scale):
      - _, K, T = frame.shape
      - 若 use_lm, 则创建 coder = ArithmeticCoder(fo) 和相关初值如 input_ = (1,K,1) 的全零张量, states=None, offset=0.
      - 遍历 T:
        - 若 use_lm, 则调用 probas, states, offset = lm(input_, states, offset), input_ = 1 + `frame[:,:,t:t+1]`
        - 遍历 `frame[0,:,t].tolist()`, 每个元素为 k, v:
          - 若 use_lm, 则调用 `q_cdf=build_stable_quantized_cdf(probas[0,:,k,0], coder.total_range_bits, check=False)` 然后 coder.push(value, q_cdf)
          - 否则: packer.push(value)
      - 若 use_lm, 则 coder.flush()
      - 否则, packer.flush()
- `decompress_from_file()` 函数: 从文件对象中解压, 并返回解压后的波形.
  首先读取元数据, 然后创建 lm, frames, segment_length, segment_stride
  遍历 offset 从 0 到音频长度, 以 segment_stride 为步长:
    - 当前片段长度 = min(音频长度-offset, segment_length)
    - frame_length = 取整(当前片段长度*model.frame_rate/model.sample_rate)
    - 若 model.normalize: 获得 scale 值
    - 若 use_lm: 创建 decoder = ArithmeticDecoder(fo) 和相关初值
    - 初始化帧
    - 遍历帧长度:
      - 若 use_lm: 调用 probas, states, offset = lm(input_, states, offset)
      - 遍历码本数量:
        - 若 use_lm: 调用 q_cdf = build_stable_quantized_cdf(probas[0,:,k,0], decoder.total_range_bits, check=False) 然后 decoder.pull(q_cdf) 得到 code
        - 否则: 调用 packer.pull() 得到 code
        - 若 code 为 None, 则 EOF 错误, 否则将 code 添加到 code_list.
      - 将 code_list 张量化后赋值给 `frame[0,:,t]`
      - 若 use_lm, 则 input_ = 1 + `frame[:,:,t:t+1]`
    - frames.append((frame, scale))
  - 调用 model.decode(frames) 得到解压后的波形.
  - 返回 `wav[0,:,:audio_length]`, model.sample_rate

---

上述过程的关键部分是 
- `model.py` 文件: `EncodecModel` 类
- `quantization/ac.py` 文件: `ArithmeticCoder` 和 `ArithmeticDecoder` 类, `build_stable_quantized_cdf` 函数.

`model.py` 文件中定义了
- 语言模型 LMModel 类: 用于估计每个码本元素的概率. 在给定时间步上并行预测所有码本.
  - emb 列表: n_q 个 card+1 × dim 的 Embedding
  - transformer: 采用 **StreamingTransformerEncoder**
  - linears: n_q 个线性层, 从 dim 映射到 card (codebook cardinality)
  - 运行逻辑:
    - 输入 indices, 来自前一个时间步. indices 应该是 1+实际的codebook索引, 尺寸为 B, n_q, T
    - states 表示流式解码
    - offset 表示当前时间步的偏移量
    
    遍历 n_q 个码本, 即从 self.emb[k] 取出 indices[:,k] 然后求和.
    将结果输入到 transformer, 得到输出, 状态, 偏移量
    然后遍历 n_q 个线性层, 将输出传入进行计算, 并按照第二个轴堆叠, 然后 permute 成 (0,3,1,2) 得到 logits
    对 logits 按第二个轴进行 softmax, 得到概率分布

---

`modules/transformer.py` 文件中定义了
- StreamingTransformerEncoder 类: 
  - self.max_period
  - self.past_context
  - self.norm_in = nn.LayerNorm 或 nn.Identity
  - self.layers = StreamingTransformerEncoderLayer(dim, num_head, hidden_dim)
  - self.forward(x, states, offset)
    - B, T, C = x.shape
    - positions = torch.arange(T).view(1,-1,1) + offset
    - pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)
    - x = self.norm_in(x)
    - x = x + pos_emb
    - 遍历 states 和 self.layers:
      - x, new_layer_state = layer(x, layer_state, self.past_context)
      - new_layer_state = cat([layer_state, new_layer_state])
      - new_state.append(new_layer_state[:, -self.past_context:, :])
    - 返回 x, new_state, offset + T
- StreamingTransformerEncoderLayer 类:
  继承 TransformerEncoderLayer
  - 重载 self._sa_block(): 
    - x.shape 为 _, T, _
    - x_past.shape 为 _, H, _
    - queries = x
    - keys = [x_past, x]
    - values = keys
    - queries_pos = arange(H, T+H).view(-1,1)
    - keys_pos = arange(T+H).view(1,-1)
    - delta = queries_pos - keys_pos
    - valid_access = (delta >= 0) & (delta <= past_context)
    - x = self.self_attn(q, k, v, attn_mask)
    - x = self.dropout1(x)

---

`model.py` 文件中定义了

- EncodecModel: **Encodec**模型, 直接处理原始波形.
  - 编码器: SEANetEncoder (`modules/seanet.py`)
  - 量化器: ResidualVectorQuantizer (`quantization/vq.py`)
  - 解码器: SEANetDecoder (`modules/seanet.py`)
  - 性质: segment_length
  - 性质: segment_stride
  - 编码: encode()
    - 给定张量 x, 返回一个包含 x 离散编码的帧列表, 每一帧都是一个元组 (codebook, scale), 前者 [B, K, T], K 为 codebook 数
    _, 通道, 长度 = x.shape
    通道数需要限制为 1~2;
    定义encoded_frames = []
    遍历 offset 从 0 到 length, 步长为 stride, frame = x[:,:, offset:offset+segment_length]
    encoded_frames.append(_encode_frame(frame))
  - 编码单帧: _encode_frame()
    长度为 x 的最后一维度, 时长 = 长度/采样率
    调用 encoder, 得到 emb, 然后调用量化器的encode 得到编码, 然后转置, 得到 B, K, T
  - 解码: decode()
    遍历帧编码列表, 依次调用 _decode_frame() 解码, 得到帧列表, 调用 `_linear_overlap_add()`
  - 解码单帧: _decode_frame()
    调用量化器的 decode 得到 emb, 然后调用 decoder 得到输出
  - 正向运行: forward()
    - 调用 `encode()` 得到 frame, 然后调用 `decode(frame)[:,:,:x.shape[-1]]`
  - 设置目标带宽: set_target_bandwidth()
  - 获取语言模型: get_lm_model()
    - lm = LMModel(量化器.n_q, 量化器.bin, 层数=5, 维度=200, past_context=3.5*帧率)
    - 加载预训练的权重, 将 lm 转化为 eval() 模式.
  - (静态) 获取模型: _get_model()
    - 创建 encoder, decoder, quantizer, model = EncodecModel(encoder, decoder, quantizer)
  - (静态) 获取预训练: _get_pretrained(): 从 URL 获取.
  - encodec_model_24khz(): 加载 24kHz 时的模型, 调用 _get_model(), 转换为 eval() 模式
  - encodec_model_48khz(): 加载 48kHz 时的模型, 调用 _get_model(), 转换为 eval() 模式

---

上述模型的关键部分在 `modules/seanet.py` 和 `quantization/vq.py` 文件.

`modules/seanet.py` 文件中定义了
- SEANet Resnet Block:
  SEANet 的残差块: shortcut(x) + block(x)
  - block 部分添加了 
    - [激活函数， SConv1d(输入通道=dim, 输出通道=hidden, 卷积核大小=3, dilation=1, norm, causal, pad_mode)]
    - [激活函数， SConv1d(输入通道=hidden, 输出通道=dim, 卷积核大小=1, dilation=1, norm, causal, pad_mode)]
  - shortcut 部分, 如果 true_skip 则使用 nn.Identity(), 否则 SConv1d(输入通道=dim, 输出通道=dim, 卷积核大小=1, dilation=1, norm, causal, pad_mode)
- SEANet Encoder:
  第一层: SConv1d(输入通道, 输出通道=mult*n_filter, 卷积核大小=7, 其他参数)
  遍历 ratios: [8,5,4,2],
    遍历 残差层数 j=1, model 增加 SEANet Resnet Block(mult*n_filter, [3,1], dilations=[2**j, 1])
    model 增加 SConv1d(`mult*n_filter`, `mult*n_filter*2`, 卷积核大小=ratio*2, 其他参数)
    mult *= 2
  如果有 lstm, 则model 添加 SLSTM(mult*n_filters, num_layers=lstm)
  最后一层: SConv1d(mult*n_filter, 128, 卷积核大小=7, 其他参数)
- SEANet Decoder:
  第一层: SConv1d(128, 输出通道=mult*n_filter, 卷积核大小=7, 其他参数)
  LSTM 层: SLSTM(mult*n_filter, num_layers=lstm)
  遍历 ratios: [8,5,4,2],
    添加转置卷积 SConvTransposed(`mult*n_filter`, `mult*n_filter//2`, 卷积核大小=ratio*2, 其他参数)
    遍历残差层数 j=1, model 增加 SEANet Resnet Block(`mult*n_filter//2`, [3,1], dilations=[2**j, 1])
    mult //= 2
  最后一层: SConv1d(n_filter, 输出通道, 卷积核大小=7, 其他参数)
  
上述代码依赖于 `modules/conv.py` 和 `modules/lstm.py` 文件.

`modules/conv.py` 文件中定义了
- SConv1d:
  conv = NormConv1d(输入通道, 输出通道, 卷积核大小, 步长, 其他参数)
- SConv1dTransposed:
  convtr = NormConvTranspose1d(输入通道, 输出通道, 卷积核大小, 步长, 其他参数)
- NormConv1d:
  conv = nn.Conv1d(输入通道, 输出通道, 卷积核大小, 步长, 填充, 其他参数)
  应用权重归一化.
  应用层归一化, 时间组归一化

`modules/lstm.py` 文件中定义了
- SLSTM:
  skip = bool
  lstm = nn.LSTM(dim, dim, 层数=2)
  运行: x.permute(2,0,1), lstm(x) -> y -> y.permute(1,2,0)

上述代码依赖于 nn.Conv, nn.LSTM, spectral_norm, weight_norm nn.GroupNorm 等

---

`quantization/vq.py` 文件中定义了
- Residual Vector Quantizer
  - self.vq = ResidualVectorQuantization(dim, codebook_size, num_quantizers, decay, kmeans_init, kmeans_iters, threshold_ema_dead_core)
  - self.forward(x, 帧率, 带宽):
    - bw_per_q = self.get_bandwidth_per_quantizer(帧率)
    - n_q = self.get_num_quanitizers_for_bandwidth(帧率, 带宽)
    - quantized, codes, commit_loss = self.vq(x, n_q)
    - bw = n_q * bw_per_q
    - opt = QuantizedResult(quantized, codes, bw, penalty=mean(commit_loss))
  - self.encode(): 调用 self.vq.encode()
  - self.decode(): 调用 self.vq.decode()

上述代码依赖于 `quantization/core_vq.py` 文件.

`quantization/core_vq.py` 文件中定义了
- ResidualVectorQuantization:
  - self.layers: num_quantizers 个 VectorQuantization
  - self.forward(x, n_q):
    res = x
    遍历 self.layers:
    - quantized, indices, loss = layer(res)
    - res = res - quantized
    - quantized_out = quantized_out + quantized
    - 堆叠输出 indices, loss
  - self.encode():
    遍历 self.layers:
    - indices = layer.encode(res)
    - quantized = layer.decode(indices)
    - res = res - quantized
    - 堆叠输出 indices
  - self.decode():
    遍历 q_indices:
    - layer = self.layers[i]
    - quantized = layer.decode(q_indices[i])
    - quantized_out = quantized_out + quantized
- VectorQuantization:
  - self.project_in = 线性层(dim, codebook_dim) 若 codebook_dim != dim, 否则恒等映射
  - self.project_out = 线性层(codebook_dim, dim) 若 codebook_dim != dim, 否则恒等映射
  - self.eps = eps
  - self.commitment_w = weight
  - self.codebook = EuclideanCodebook(codebook_dim, codebook_size, kmeans_init, kmeans_iters, decay, epsilon, threshold_ema_dead_core)
  - 性质 self.codebook = self.codebook.embed
  - `self.encode()`: x = rearrange(x, "b d n -> b n d"), x = self.project_in(x), 调用 self.codebook.encode(x)
  - `self.decode()`: x = self.codebook.decode(embed), x = self.project_out(x), rearrange(x, "b n d -> b d n")
  - self.forward():
    - rearrange() -> project_in() -> x-> codebook() -> quantize, embed_ind -> project_out() -> rearrange()
    - 训练时 loss += MES(quantize, x) * weight 否则为 0
    - 返回 quantize, embed_ind, loss
- Euclidean Codebook (Codebook with Euclidean distance):
  - self.forward(x):
    - x = self.preprocess(x)
    - self.init_embed_(x)
    - embed_ind = self.quantize(x)
    - embed_onehot = one_hot(embed_ind, self.codebook_size)
    - embed_ind = self.postprocess_emb(embed_ind)
    - quantize = self.dequantize(embed_ind)
    - if training: self.expire_codes_(x)
  - self.encode(): x = self.preprocess(x) -> self.quantize(x) -> self.postprocess_emb() -> embed_ind
  - self.decode(): embed_ind -> self.dequantize() -> quantize
  - self.preprocess(): rearrange()
  - self.quantize(): embed = self.embed.T, dist = -(x^2-2x emb + emd^2), dist.max(dim=-1) -> embed_ind
  - self.postprocess_emb(): embed_ind.view(*shape[:-1])
  - self.dequantize(): F.embedding(embed_ind, self.embed)
  - self.init_embed_(): embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)

## 相关依赖

- Python: 3.8.0+

## 参考文献

- SEANet
