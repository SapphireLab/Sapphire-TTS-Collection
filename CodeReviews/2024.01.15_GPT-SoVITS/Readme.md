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
  - [ ] transformer.py Transformer 模块
- [ ] s2train.py 训练使用 DDP 并行训练

