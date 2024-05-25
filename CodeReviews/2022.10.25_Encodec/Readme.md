# Encodec

[论文笔记](../../Models/Speech_Neural_Codec/2022.10.24_EnCodec.md)
[官方代码](https://github.com/facebookresearch/encodec) 

## 提交记录

初次提交: 2022.10.25
最近提交: 2023.06.20

## 框架

整体框架图
![](../../Models/_Images/2022.10_Encodec_FIG01.png)

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
- LMModel: 语言模型, 用于估计每个码本元素的概率. 在给定时间步上并行预测所有码本.
- EncodecModel: **Encodec**模型, 直接处理原始波形.
  - 编码器: SEANetEncoder (`modules/seanet.py`)
  - 量化器: ResidualVectorQuantizer
  - 解码器: SEANetDecoder (`modules/seanet.py`)



## 相关依赖

- Python: 3.8.0+

## 参考文献

- SEANet
