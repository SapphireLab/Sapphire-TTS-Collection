# F5-TTS 项目

## 基本信息

- 标题: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- 链接: [HuggingFace](https://huggingface.co/SWivid/F5-TTS) | [ModelScope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN) | [GitHub](https://github.com/SWivid/F5-TTS) | [ArXiv](https://arxiv.org/abs/2410.06885)
- 开源: MIT License
- 更新: 2024.11.02
- 笔记: 2024.11.02

## 更新备注

- 2024.10.23 PR#228: 项目结构重新组织, 将 `model/` 重构为 `src/f5_tts/model/`.

## 文件结构

<details>
<summary>项目结构</summary>

- [x] .github/workflows/
  - [x] `pre-commit.yaml`: 调用 `pre-commit-config.yaml`.
  - [x] `publish-docker-image.yaml`: 发布 Docker 镜像.
  - [x] `sync-hf.yaml`: 同步到 HuggingFace Space
- [x] ckpts/
  - [x] `README.md`
  - [x] E2TTS_Base/`model_1200000.pt` 1.35 GB (pt或safetensors格式)
  - [x] F5TTS_Base/`model_1200000.pt` 1.35 GB (pt或safetensors格式)
- [x] data/
  - [x] Emilia_ZH_EN_pinyin/`vocab.txt`: 2545 行
  - [x] `librispeech_pc_test_clean_cross_sentence.lst`: 1127 行
- [x] src/
  - [x] f5_tts/
    - [ ] eval/
      - [ ] `README.md`
      - [ ] `ecapa_tdnn.py`
      - [ ] `eval_infer_batch.py`
      - [ ] `eval_infer_batch.sh`
      - [ ] `eval_librispeech_test_clean.py`
      - [ ] `eval_seedtts_testset.py`
      - [ ] `utils_eval.py`
    - [ ] infer/
      - [x] examples/
        - [x] basis/
          - [x] `basic.toml`: 推理的基础配置 (参考音频为 `basic_ref_en.wav`)
          - [x] `basic_ref_en.wav`: 对应 "Some call me nature, others call me mother nature."
          - [x] `basic_ref_cn.wav`
        - [x] multi/
          - [x] `country.flac`
          - [x] `main.flac`
          - [x] `story.toml`: 参考音频为 `main.flac` (另外有 `town.flac`, `country.flac` 子配置), 用于合成 `story.txt` 内的文本.
          - [x] `story.txt`
          - [x] `town.flac`
        - [x] `vocab.txt`: 2545 行 (重复文件?)
      - [ ] `README.md`
      - [ ] `infer_cli.py`
      - [ ] `infer_gradio.py`
      - [ ] `speech_edit.py`
      - [ ] `utils_infer.py`
    - [ ] model/
      - [ ] backbones/
        - [x] `README.md`
        - [ ] `dit.py`
        - [ ] `mmdit.py`
        - [ ] `unett.py`
      - [x] `__init__.py`: 引入三种 backbone, CFM 和 trainer
      - [ ] `cfm.py`
      - [ ] `dataset.py`
      - [ ] `modules.py`
      - [ ] `trainer.py`
      - [ ] `utils.py`
    - [ ] scripts/
      - [ ] `count_max_epoch.py`
      - [ ] `count_params_gflops.py`
    - [ ] train/
      - [ ] dataset/
        - [ ] `prepare_csv_wavs.py`
        - [ ] `prepare_emilia.py`
        - [ ] `prepare_wenetspeech4tts.py`
      - [ ] `README.md`
      - [ ] `finetune_cli.py`
      - [ ] `finetune_gradio.py`
      - [ ] `train.py`
    - [ ] `api.py`
  - [ ] third_party/
    - [ ] BigVGAN: [Github](https://github.com/NVIDIA/BigVGAN/tree/7d2b454564a6c7d014227f635b7423881f14bdac)
- [x] `.gitignore`
- [x] `.gitmodules`: 引入 BigVGAN 仓库
- [x] `.pre-commit-config.yaml`: 运行 Ruff 代码风格检查, 格式化 + YAML 校验.
- [ ] `Dockerfile`
- [x] `LICENSE`: MIT License
- [ ] `README.md`
- [x] `pyproject.toml`: 环境配置文件
- [x] `ruff.toml`: 行宽 120, python=3.10, 忽略私有变量, import 单行等.

</details>

<details>
<summary>环境配置</summary>

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
- [ ] [modelscope]: 模型库
- [x] [faster_whisper](https://github.com/SYSTRAN/faster-whisper): 语音识别
- [x] [funasr](https://github.com/modelscope/FunASR): 语音识别
- [x] [jiwer](https://github.com/jitsi/jiwer): 自动评估语音识别系统, 含词错误率, 匹配错误率, 词信息丢失, 词信息保留, 字符错误率等指标 (最小编辑距离使用 RapidFuzz, 底层基于 C++)

声码器
- [x] [vocos](https://github.com/gemelo-ai/vocos): 声码器

</details>

## 用法教程

## 模型核心

### Backbone

- `unett.py`: Flat U-Net Transformer
- `dit.py`
- `mmdit.py`: Stable Diffusion 3 架构

### CFM

### Trainer
