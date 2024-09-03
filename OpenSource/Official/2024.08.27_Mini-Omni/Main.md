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
    - [ ] `__init__.py`
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
- [ ] webui
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