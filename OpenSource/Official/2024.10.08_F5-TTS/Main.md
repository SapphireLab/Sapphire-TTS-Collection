# F5-TTS 项目

## 基本信息

- 标题: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- 链接: [HuggingFace](https://huggingface.co/SWivid/F5-TTS) | [GitHub](https://github.com/SWivid/F5-TTS) | [ArXiv](https://arxiv.org/abs/2410.06885)
- 开源: MIT License
- 更新: 2024.10.15
- 笔记: 2024.10.15


## 文件结构

<details>
<summary>项目结构</summary>

- [x] ckpts/
  - [x] `README.md`
  - [x] E2TTS_Base/`model_1200000.pt` 1.35 GB (pt或safetensors格式)
  - [x] F5TTS_Base/`model_1200000.pt` 1.35 GB (pt或safetensors格式)
- [x] data/
  - [x] Emilia_ZH_EN_pinyin/`vocab.txt`
  - [x] `librispeech_pc_test_clean_cross_sentence.lst`
- [ ] model/
  - [ ] backbones/
    - [x] `README.md`
    - [ ] `dit.py`
    - [ ] `mmdit.py`
    - [ ] `unett.py`
  - [x] `__init__.py`: 引入三种 backbone, CFM 和 trainer
  - [ ] `cfm.py`
  - [ ] `dataset.py`
  - [ ] `ecapa_tdnn.py`
  - [ ] `modules.py`
  - [ ] `trainer.py`
  - [ ] `utils.py`
- [ ] scripts/
  - [ ] `count_max_epoch.py`
  - [ ] `count_params_gflops.py`
  - [ ] `eval_infer_batch.py`
  - [ ] `eval_infer_batch.sh`
  - [ ] `eval_librispeech_test_clean.py`
  - [ ] `eval_seedtts_testset.py`
  - [ ] `prepare_emilia.py`
  - [ ] `prepare_wenetspeech4tts.py`
- [x] tests/ref_audio/
  - [x] `test_en_1_ref_short.wav`
  - [x] `test_zh_1_ref_short.wav`
- [x] `.gitignore`
- [x] `LICENSE` -> MIT License
- [ ] `README.md`
- [ ] `gradio_app.py`
- [ ] `inference-cli.py`
- [ ] `inference-cli.toml`
- [x] `requirements.txt`
- [ ] `speech_edit.py`
- [ ] `train.py`

</details>

<details>
<summary>环境配置</summary>

- [ ] accelerate>=0.33.0
- [ ] cached_path
- [ ] click
- [ ] datasets
- [ ] einops>=0.8.0
- [ ] einx>=0.3.0
- [ ] ema_pytorch>=0.5.2
- [x] faster_whisper
- [x] funasr
- [ ] gradio
- [x] jieba
- [ ] jiwer
- [x] librosa
- [x] matplotlib
- [x] numpy==1.22.0<2.x
- [ ] pydub
- [ ] pypinyin
- [ ] safetensors
- [ ] soundfile
- [x] torch==2.3.0+cu118
- [x] torchaudio==2.3.0+cu118
- [x] torchdiffeq: Pytorch 框架下高精度的 ODE 求解器, 实现了自适应步长算法
- [x] tqdm>=4.65.0
- [ ] transformers
- [x] vocos 声码器
- [ ] wandb
- [ ] x_transformers>=1.31.14
- [ ] zhconv
- [ ] zhon

</details>

## 用法教程

## 模型核心

### Backbone

- `unett.py`: Flat U-Net Transformer
- `dit.py`
- `mmdit.py`: Stable Diffusion 3 架构

### CFM

### Trainer
