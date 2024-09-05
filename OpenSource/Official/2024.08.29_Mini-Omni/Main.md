# Mini-Omni 项目

## 基本信息

标题: Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming
链接: [HuggingFace](https://huggingface.co/gpt-omni/mini-omni) | [Github](https://github.com/gpt-omni/mini-omni) | [ArXiv](https://arxiv.org/abs/2408.16725)
开源: MIT License

**Mini-Omni** is an open-source multi-model large language model that can hear, talk while thinking. Featuring real-time end-to-end speech input and streaming audio output conversational capabilities.

**Mini-Omni** 是一个开源多模态大语言模型, 能够在思考时进行听说. 特点是实时端到端语音输入和流式音频输出对话能力.

特性:
- 实时语音到语音对话能力, 无需额外的 ASR 或 TTS 模型;
- 思考时发言, 即同时生成文本和音频;
- 流式音频输出能力;
- 音频转文本和音频转音频的批量推理进一步增强性能.

<details>
<summary>展开项目结构</summary>

- [x] data/
  - [x] figures/frameworkv3.jpg
  - [x] samples/output12345.wav
  - [x] demo_gradio.mov
  - [x] demo_streamlit.mov
- [ ] litgpt/
  - [ ] generate/
    - [x] `__init__.py` 空文件
    - [ ] `base.py`
  - [ ] `__init__.py`
  - [ ] `config.py`
  - [ ] `model.py`
  - [ ] `tokenizer.py`
  - [ ] `utils.py`
- [ ] utils/
  - [ ] assets/
    - [ ] silero_vad.onnx
  - [ ] `snac_utils.py`
  - [ ] `vad.py`
- [ ] webui/
  - [ ] `omni_gradio.py`
  - [ ] `omni_streamlit.py`
- [x] .gitignore
- [x] LICENSE
- [x] README.md
- [ ] inference.py
- [ ] requirements.txt
- [ ] server.py

</details>

--- 

通过 `requirements.txt` 文件可知依赖的其他项目:

| 项目 | 版本 | 用途 |
| --- | --- | --- |
| PyTorch/TorchVision/TorchAudio | 2.3.1 |
| SoundFile | 0.12.1 | 音频处理 |
| Librosa | 0.10.2post1 | 音频处理 |
| PyDub | 0.25.1 | 音频处理 |
| Gradio | 4.42.0 | 用于创建交互式界面 `webui/omni_gradio.py`|
| Streamlit | 1.37.1 | 用于创建交互式界面 `webui/omni_streamlit.py`|
| Fire | | 用于命令行接口 |
| Flask | 3.0.3 | 用于构建服务端 `server.py`|
| OpenAI-Whisper |
| Tokenizers | 0.19.1 |
| OnnxRuntime | 1.19.0 |
| [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | v0.4.3 (非最新) | |
| [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac/) | v1.2.0 (非最新) | |

---

以 `inference.py` 为入口进行分析.

整个文件有 667 行. 具有 14 个函数和一个类, 列出如下:

| 函数名 | 入参 | 功能 |
| --- | --- | --- |
| get_input_ids_TA() | Text, Text Tokenizer | 
| get_input_ids_TT() | Text, Text_Tokenizer |
| get_input_ids_whisper() | Mel_Spec, Length, WhisperModel |
| get_input_ids_whisper_ATBatch() | Mel_Spec, Length, WhisperModel |
| load_audio() | Path | 使用 whisper.load_audio() 并返回对数梅尔频谱 |
| A1_A2() | fabric, Audio Feature, Input_IDs, Length, Model, Text Tokenizer, Step, SNAC model, Output Dir | 1. 调用 `generate_AA()`<br> 2. 调用 SNAC 模型解码 + Text Tokenizer 解码 | 
| A1_A2_batch() | fabric, Audio Feature, Input_IDs, Length, Model, Text Tokenizer, Step, SNAC model, Output Dir | 1. 调用 `generate_TA_batch()` <br> 2. 调用 SNAC 模型解码 + Text Tokenizer 解码 |
| A1_T1() | fabric, Audio Feature, Input_IDs, Length, Model, Text Tokenizer | 1. 调用 `generate_ASR()` <br> 2. 调用 Text Tokenizer 解码 |
| A1_T2() | fabric, Audio Feature, Input_IDs, Length, Model, Text Tokenizer | 1. 调用 `generate_AT()` <br> 2. 调用 Text Tokenizer 解码 |
| T1_A2() | fabric, Input_IDs, Model, Text Tokenizer, Step, SNAC model, Output Dir | 1. 调用 `generate_TA()` <br> 2. 调用 SNAC 模型解码 + Text Tokenizer 解码 |
| T1_T2() | fabric, Input_IDs, Model, Text Tokenizer, Step | 1. 调用 `generate_TT()` <br> 2. 调用 Text Tokenizer 解码 |
| load_model() | Checkpoint Path | 加载 SNAC_24KHz 模型, Whipser-Small 模型, Tokenizer, Lightning.Fabric, GPT |
| download_model() | Checkpoint Path | 从 HuggingFace 的 GPT-Omni/Mini-Omni 仓库下载模型 |
| test_infer() | | 测试以上所有函数的功能. 包括 A1A2, ASR, T1A2, AA-Batch, T1T2, AT|
| OmniInference 类 | Checkpoint Path | 调用 `load_model()`, 创建 `run_AT_batch_stream()` 函数 |

该文件的函数核心来源于 `litgpt/generate/base.py` 文件下的 `generate_XX()` 函数.
模型有:
- OpenAI 的 Whisper;
- SNAC 用于重构音频;
- LitGPT 的 GPT 模型和 Tokenizer 为主干.

---

## LitGPT-Modified

模型主体是基于 LitGPT 进行修改的. 
- `config.py` 文件相同.
- `tokenizer.py`
- `model.py`
- `utils.py` 有一些格式上的变化.

