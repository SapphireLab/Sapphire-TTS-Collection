# F5-TTS 项目

## 基本信息

- 标题: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- 链接: [HuggingFace](https://huggingface.co/SWivid/F5-TTS) | [ModelScope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN) | [WiseModel](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN) | [GitHub](https://github.com/SWivid/F5-TTS) | [ArXiv](https://arxiv.org/abs/2410.06885) | [PaperNote](../../../Models/Diffusion/2024.10.09_F5-TTS.md)
- 开源: [代码] MIT License; [模型] CC-BY-NC License (数据集 Emilia 限制)
- 更新: 2024.11.18
- 笔记: 2024.11.18

### 主要内容
- **F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.
- **E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).
- **Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### 更新日志

- 2024.11.11 v0.1.1 版本
- 2024.10.23 PR#228: 项目结构重新组织, 将 `model/` 重构为 `src/f5_tts/model/`.
- 2024.10.08 F5-TTS & E2-TTS 基础模型发布在 HuggingFace, ModelScope, WiseModel.

### 致谢列表

| 项目 | 贡献者 | 说明 |
| --- | --- | --- |
| [E2-TTS](../../../Models/Diffusion/2024.06.26_E2_TTS.md) | | 简单高效的杰出工作 |
| [Emilia](../../../Datasets/2024.07.07_Emilia.md), [WenetSpeech4TTS](../../../Datasets/2024.06.09_WenetSpeech4TTS.md) | | 数据集 |
| | [lucidrains](https://github.com/lucidrains) <br> [bfs](https://github.com/bfs18) | CFM 架构初始化|
| [StableDiffusion3](../../../Models/Diffusion/2024.03.05_StableDiffusion3.md)<br>[HF Diffusers](https://github.com/huggingface/diffusers) | | DiT 和 MMDiT 代码结构 |
| [TorchDiffEQ](https://github.com/rtqichen/torchdiffeq) | | ODE 求解器 |
| [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) | | 声码器 |
| [FunASR](https://github.com/modelscope/FunASR)<br>[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)<br>[UniSpeech](https://github.com/microsoft/UniSpeech) | | 评估工具 |
| [CTC-Forced-Aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | | CTC 强制对齐工具 |
| [realmrfakename](https://x.com/realmrfakename) | | HF Space 示例项目 |
| [F5-TTS-MLX](https://github.com/lucasnewman/f5-tts-mlx/tree/main) | [Lucas Newman](https://github.com/lucasnewman) | MLX 版本 |
| [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) | [DakeQQ](https://github.com/DakeQQ) | ONNX Runtime 版本 |

### 引用信息

```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```


## 文件结构

```
F5-TTS
├─ [x] .github                                  Github 工作流配置
│  ├─ [ ] ISSUE_TEMPLATE
│  │  ├─ [ ] bug_report.yml
│  │  ├─ [ ] config.yml
│  │  ├─ [ ] feature_request.yml
│  │  ├─ [ ] help_wanted.yml
│  │  └─ [ ] question.yml
│  └─ [x] workflows
│     ├─ [x] pre-commit.yaml:                   调用 pre-commit-config.yaml
│     ├─ [x] publish-docker-image.yaml:         发布 Docker 镜像
│     └─ [x] sync-hf.yaml:                      同步到 HuggingFace Space
├─ [x] .gitignore
├─ [x] .gitmodules:                             引入 BigVGAN 仓库
├─ [x] .pre-commit-config.yaml:                 运行 Ruff 代码风格检查, 格式化 + YAML 校验
├─ [x] Dockerfile:                              Docker 镜像构建文件
├─ [x] LICENSE:                                 MIT License
├─ [x] pyproject.toml:                          环境配置文件
├─ [x] README.md:                               项目说明
├─ [x] ruff.toml:                               行宽 120, python=3.10, 忽略私有变量, import 单行等.
├─ [x] ckpts                                    预训练模型
│  ├─ [x] README.md                             模型文件结构说明
│  ├─ [x] E2TTS_Base
│  │  └─ [x] model_1200000.pt:                  1.35GB, 预训练模型 (pt 或 safetensors 格式)
│  ├─ [x] F5TTS_Base
│  │   └─ [x] model_1200000.pt:                 1.35GB, 预训练模型 (pt 或 safetensors 格式)
│  └─ [x] F5TTS_Base_bigvgan
│     └─ [x] model_1250000.pt:                  1.35GB, 预训练模型 (pt)
├─ [x] data
│  ├─ [x] Emilia_ZH_EN_pinyin
│  │  └─ [x] vocab.txt:                         2545 行
│  └─ [x] librispeech_pc_test_clean_cross_sentence.lst: 1127 行
└─ [x] src
   ├─ [x] f5_tts
   │  ├─ [x] api.py
   │  ├─ [ ] socket_server.py
   │  ├─ [ ] eval
   │  │  ├─ [ ] ecapa_tdnn.py
   │  │  ├─ [ ] eval_infer_batch.py
   │  │  ├─ [ ] eval_infer_batch.sh
   │  │  ├─ [ ] eval_librispeech_test_clean.py
   │  │  ├─ [ ] eval_seedtts_testset.py
   │  │  ├─ [x] README.md:                      评估说明 Eval.md
   │  │  └─ [ ] utils_eval.py
   │  ├─ [ ] infer
   │  │  ├─ [x] examples                        推理示例
   │  │  │  ├─ [x] basic                        基础推理示例
   │  │  │  │  ├─ [x] basic.toml:               推理的基础配置 (参考音频为 `basic_ref_en.wav`)
   │  │  │  │  ├─ [x] basic_ref_en.wav:         对应 "Some call me nature, others call me mother nature."
   │  │  │  │  └─ [x] basic_ref_zh.wav
   │  │  │  ├─ [x] multi                        多风格/多人推理示例
   │  │  │  │  ├─ [x] country.flac              角色: 乡下老鼠参考音频
   │  │  │  │  ├─ [x] main.flac                 角色: 主线旁白
   │  │  │  │  ├─ [x] story.toml:               参考音频为 main.flac (另外有 town.flac, country.flac 指定角色音色), 用于合成 story.txt 内的文本
   │  │  │  │  ├─ [x] story.txt:                待合成的故事文本
   │  │  │  │  └─ [x] town.flac                 角色: 城镇老鼠参考音频
   │  │  │  └─ [x] vocab.txt:                   2545 行 (重复文件?)
   │  │  ├─ [ ] infer_cli.py
   │  │  ├─ [ ] infer_gradio.py
   │  │  ├─ [x] README.md:                      推理说明 Infer.md
   │  │  ├─ [ ] SHARED.md:                      共享模型说明
   │  │  ├─ [ ] speech_edit.py
   │  │  └─ [ ] utils_infer.py
   │  ├─ [ ] model
   │  │  ├─ [ ] backbones
   │  │  │  ├─ [ ] dit.py
   │  │  │  ├─ [ ] mmdit.py
   │  │  │  ├─ [ ] README.md
   │  │  │  └─ [ ] unett.py
   │  │  ├─ [ ] cfm.py
   │  │  ├─ [ ] dataset.py
   │  │  ├─ [ ] modules.py
   │  │  ├─ [x] trainer.py                      使用 Accelerate 训练 CFM
   │  │  ├─ [ ] utils.py
   │  │  └─ [x] __init__.py:                    引入三种 backbone, CFM 和 trainer
   │  ├─ [ ] scripts
   │  │  ├─ [ ] count_max_epoch.py
   │  │  └─ [ ] count_params_gflops.py
   │  └─ [ ] train
   │     ├─ [ ] datasets
   │     │  ├─ [ ] prepare_csv_wavs.py
   │     │  ├─ [ ] prepare_emilia.py
   │     │  └─ [ ] prepare_wenetspeech4tts.py
   │     ├─ [ ] finetune_cli.py
   │     ├─ [ ] finetune_gradio.py
   │     ├─ [x] README.md:                      训练说明 Train.md
   │     └─ [ ] train.py
   └─ [ ] third_party
      └─ [ ] BigVGAN:                           https://github.com/NVIDIA/BigVGAN/tree/7d2b454564a6c7d014227f635b7423881f14bdac
```

<details>
<summary>环境配置</summary>

Python = 3.10

常用库
- [x] torch==2.3.0+cu118
- [x] torchaudio==2.3.0+cu118
- [x] [matplotlib](https://github.com/matplotlib/matplotlib): 绘图
- [x] [numpy](https://github.com/numpy/numpy)==1.22.0<2.x: 数值计算
- [x] [torchdiffeq](https://github.com/rtqichen/torchdiffeq): Pytorch 框架下高精度的 ODE 求解器, 实现了自适应步长算法
- [x] [tqdm](https://github.com/tqdm/tqdm)>=4.65.0
- [x] [wandb](https://github.com/wandb/wandb): 日志记录
- [x] [tomli](https://github.com/hukkin/tomli): 解析 TOML 文件

HuggingFace 库
- [x] [accelerate](https://github.com/huggingface/accelerate)>=0.33.0: 抽象多卡训练代码
- [x] [datasets](https://github.com/huggingface/datasets)
- [x] [transformers](https://github.com/huggingface/transformers)
- [x] [safetensors](https://github.com/huggingface/safetensors): 格式处理

- [x] [transformers_stream_generator](https://github.com/LowinLi/transformers-stream-generator): 基于 HuggingFace Transformers 的流式生成器
- [x] [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes): 量化与优化库

Einstein 求和
- [x] [einops](https://github.com/arogozhnikov/einops)>=0.8.0: 处理张量维度
- [x] [einx](https://github.com/fferflo/einx)>=0.3.0: 处理基础张量运算

Lucidrains 复现
- [x] [x_transformers](https://github.com/lucidrains/x-transformers)>=1.31.14 @lucidrains 含各种实验特性的 Transformer
- [x] [ema_pytorch](https://github.com/lucidrains/ema-pytorch)>=0.5.2: 指数移动平均

用户交互部分
- [x] [cached_path](https://github.com/allenai/cached_path): 处理文件访问
- [x] [click](https://github.com/pallets/click): 命令行美化
- [x] [gradio](https://github.com/gradio-app/gradio): WebUI 交互

中文处理
- [x] [jieba](https://github.com/fxsjy/jieba): 中文分词
- [x] [pypinyin](https://github.com/mozillazg/python-pinyin): 中文拼音转换
- [x] [zhconv](https://github.com/gumblex/zhconv): 基于 MediaWiki 词汇表的最大正向匹配简繁转换
- [x] [zhon](https://github.com/tsroten/zhon): 字符串中查找 CJK, 拼音字节, 词语, 句子等

音频处理
- [x] [librosa](https://github.com/librosa/librosa): 音频信号分析
- [x] [pydub](https://github.com/jiaaro/pydub): 音频文件处理
- [x] [soundfile](https://github.com/bastibe/python-soundfile): 读写音频文件

语音识别
- [x] [modelscope](https://github.com/modelscope/modelscope): 模型库
- [x] [faster_whisper](https://github.com/SYSTRAN/faster-whisper): 语音识别
- [x] [funasr](https://github.com/modelscope/FunASR): 语音识别
- [x] [jiwer](https://github.com/jitsi/jiwer): 自动评估语音识别系统, 含词错误率, 匹配错误率, 词信息丢失, 词信息保留, 字符错误率等指标 (最小编辑距离使用 RapidFuzz, 底层基于 C++)

声码器
- [x] [vocos](https://github.com/gemelo-ai/vocos): 声码器

</details>

## 使用方法

### 安装

1. 以 pip 包的形式安装 (仅推理): `pip install git+https://github.com/SWivid/F5-TTS.git`
2. 本地配置 (训练, 微调, 推理): `git clone https://github.com/SWivid/F5-TTS.git` + `pip install -e .`
3. Docker 镜像: `docker build -t f5-tts:v1 .` + `docker pull ghcr.io/swivid/f5-tts:main`

注意: `BigVGAN` 在 `git clone` 后需要在 `bigvgan.py` 开头添加路径 `sys.path.append(os.path.dirname(os.path.abspath(__file__)))`

### [推理](Infer.md)

### [训练 (Gradio APP)](Train.md)

### [评估](Eval.md)

### 开发

在提交 PR 之前, 请使用 `pre-commit` 来确保代码质量 (将会运行代码检查工具和格式化)
注意: 一些模型组件对 E722 进行了额外处理, 以适应张量表示.

```bash
pip install pre-commit
pre-commit install

pre-commit run --all-files
```

## 模型核心

### Backbone

- `unett.py`: Flat U-Net Transformer
- `dit.py`
- `mmdit.py`: Stable Diffusion 3 架构

### CFM

### Trainer

- 从模型文件夹中导入 CFM,
- 从模型文件夹的数据集文件中导入 DynamicBatchSampler, collate_fn
- 从 utils 文件中导入 default, exists

整个 Trainer 包含:
- 初始化函数
- is_main 判断函数
- save_checkpoint 保存函数
- load_checkpoint 加载函数
- train 训练函数

初始化函数需要了解:
- Accelerator 的用法;
- wandb/tensorboard 记录的用法;
- model 为 CFM
- 若 is_main 函数结果 (is_main_process) 为真, 则创建 EMA 模型
- 训练参数配置: epochs, num_warmup_updates, save_per_updates, last_per_steps, checkpoint_path, batch_size, batch_size_type, max_samples, grad_accumulation_steps, max_grad_norm
- vocoder_name (mel_spec_type := vocos)
- noise_scheduler
- duration_predictor
- 优化器: 量化 bnb_optimizer (AdamW8bit)/AdamW

保存函数略过, 加载函数做了一些处理:
- 若有 model_last 则加载, 否则对文件名按数字排序
- 将检查点加载到 CPU
- 确保能反向传播的补丁: 删除掉 ema_model 梅尔频谱 mel_stft 的 mel_scale.fb 和 spectrogram.window ?
- 主进程加载 EMA 模型参数
- 如果检查点内含有 step, 同样删除 model 梅尔频谱的两个键; 加载模型和优化器, 如果调度器存在, 则加载调度器
- 如果检查点内不含有 step, 则从 EMA 模型中加载模型参数, step 从 0 开始

核心的训练函数:
- 如果需要记录样本, 则加载声码器, 并设置好目标采样率;
- 如果随机种子存在, 则进行设置;
- 若 batch_size_type 为样本点, 创建 DataLoader (以 batch_size 和打乱实现),
- 若 batch_size_type 为帧, 则调用 SequentialSampler 然后用 DynamicBatchSampler 进行二次创建 batch_sampler, 然后创建 DataLoader;
- 使用多 GPU 时, 考虑固定的热身步数, 并准备好学习率调度器, 以设置学习率.
- 加载模型并读取可能的训练步数, 并继续训练.
- 核心部分如下:
```python
global_step = start_step
for epoch in range(start_epoch, epochs):

      for batch in train_dataloader:
            with accelerator.accumulate(model):
                  text = batch['text']
                  mel_spec = batch["mel"].permute(0, 2, 1)
                  mel_lengths = batch["mel_length"]

                  if duration_predictor:
                        dur_loss = duration_predictor(mel_length, lens=batch["durations"])

                  loss, cond, pred = model(mel_spec, text, mel_lengths, noise_scheduler)
                  accelerator.backward(loss)

                  accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                  optimizer.step()
                  scheduler.step()
                  optimizer.zero_grad()

            if is_main():
                  ema_model.update()

            global_step += 1
```

- 后面是在循环中打印信息, 记录信息, 保存模型, 评估时通过声码器解码 mel 得到参考音频, 模型预测 mel 然后同样解码得到预测音频, 并保存.

**总结: Trainer 的核心是 CFM 模型的前向过程, 其他的代码逻辑主要是 Accelerator 的使用**.