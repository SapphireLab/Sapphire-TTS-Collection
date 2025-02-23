# Sapphire-TTS-Collection

本仓库收集/整理/学习语音合成技术相关资料.
(可能含有部分 NLP 和其他语音相关技术)
仍在初步更新中

[腾讯文档汇总表](https://docs.qq.com/sheet/DZGpLVG5nTFZxb3NQ?nlc=1&tab=ss_t310ak&viewId=voTHIY)

## 推荐链接

- [Speech.Zone](https://speech.zone)
- [语音之家](https://www.speechhome.com)
- 课题组 [X-LANCE@SJTU](https://x-lance.sjtu.edu.cn)
- 课题组 [ASLP@NPU](http://www.npu-aslp.org)
- 课题组 [Speech@NTU](https://www.youtube.com/@HungyiLeeNTU)

## 同类项目

- [Awesome Audio Plaza](https://github.com/metame-ai/awesome-audio-plaza) by Metame AI ![Star](https://img.shields.io/github/stars/metame-ai/awesome-audio-plaza?style=social)
- [Speech Trident](https://github.com/ga642381/speech-trident) by 李宏毅 Team ![Star](https://img.shields.io/github/stars/ga642381/speech-trident)
- [WavChat](https://github.com/jishengpeng/WavChat) by 浙江大学 (赵洲 Team) & 微软 & 阿里巴巴 & 腾讯优图实验室 ![Star](https://img.shields.io/github/stars/jishengpeng/WavChat.svg?style=social)
- [Neural Codec & Speech Language Models](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models) by LqNoob ![Star](https://img.shields.io/github/stars/LqNoob/Neural-Codec-and-Speech-Language-Models?style=social)
- [Awesome MLLM](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) by VITA Team ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models?style=social)
- [Awesome Large Speech Model](https://github.com/huangcanan/Awesome-Large-Speech-Model) by Huang Can'an ![Star](https://img.shields.io/github/stars/huangcanan/Awesome-Large-Speech-Model?style=social)
- [VoiceBench](https://github.com/MatthewCYM/VoiceBench) by 新加坡国立大学 ![Star](https://img.shields.io/github/stars/MatthewCYM/VoiceBench)
- [Awesome Controllable Speech Synthesis](https://github.com/imxtx/awesome-controllabe-speech-synthesis) by 香港科技大学 (广州) ![Star](https://img.shields.io/github/stars/imxtx/awesome-controllabe-speech-synthesis?style=social)
- [Awesome-Multimodal-Next-Token-Prediction](https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction) by Liang Chen (Leader) ![Star](https://img.shields.io/github/stars/LMM101/Awesome-Multimodal-Next-Token-Prediction)

## 开源项目·Text-to-Speech

|时间|名称|仓库|演示|论文|
|---|---|:-:|---|---|
|2024.12.13|**CosyVoice2**<br>@阿里巴巴|[Github](https://github.com/FunAudioLLM/CosyVoice)<br>![Star](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice?style=social)<br>[ModelScope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)||[ArXiv](https://arxiv.org/abs/2412.10117)<br>[PaperNote](Models/SpeechLM/ST2S/2024.12.13_CosyVoice2.md)|
|2024.11.04|**OuteTTS**<br>@OuteAI|[Github](https://github.com/edwko/OuteTTS/)<br>![Star](https://img.shields.io/github/stars/edwko/OuteTTS?style=social)<br>[HuggingFace](https://huggingface.co/OuteAI/OuteTTS-0.1-350M)<br>[HF Mirror](https://hf-mirror.com/OuteAI/OuteTTS-0.1-350M)|-|[Blog](https://www.outeai.com/blog/outetts-0.1-350m)|
|2024.10.20|**MaskGCT**<br>@香港中文大学 (深圳)<br>&广州趣玩网络科技|[Github (Amphion)](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)<br>![Star](https://img.shields.io/github/stars/open-mmlab/Amphion)<br>[HuggingFace](https://huggingface.co/amphion/MaskGCT/tree/main)<br>[HF Mirror](https://hf-mirror.com/amphion/MaskGCT/tree/main)|[Github.IO](https://maskgct.github.io)<br>[趣玩科技](https://voice.funnycp.com)|[ArXiv](https://arxiv.org/abs/2409.00750)<br>[PaperNote](Models/SpeechLM/ST2S/2024.09.01_MaskGCT.md)<br>CodeReview|
|2024.10.08|**F5-TTS**<br>@上海交通大学&剑桥大学<br>&吉利汽车研究院(宁波)|[Github](https://github.com/SWivid/F5-TTS)<br>![Star](https://img.shields.io/github/stars/SWivid/F5-TTS?style=social)<br>[HuggingFace](https://huggingface.co/SWivid/F5-TTS)<br>[HF Mirror](https://hf-mirror.com/SWivid/F5-TTS)|[Github.IO](https://swivid.github.io/F5-TTS/)<br>[HF Space](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)|[ArXiv](https://arxiv.org/abs/2410.06885)<br>[PaperNote](Models/Diffusion/2024.10.09_F5-TTS.md)<br>[CodeReview](OpenSource/Official/2024.10.08_F5-TTS/Main.md)|
|2024.07.03|**CosyVoice**<br>@阿里巴巴语音实验室|[Github](https://github.com/FunAudioLLM/CosyVoice)<br>![Star](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice?style=social)<br>[ModelScope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)||[ArXiv](https://arxiv.org/abs/2407.05407)<br>[PaperNote](Models/SpeechLM/ST2S/2024.07.07_CosyVoice.md)|
|2024.05.28|**ChatTTS**<br>@2noise 团队|[Github](https://github.com/2noise/ChatTTS)<br>![Star](https://img.shields.io/github/stars/2noise/ChatTTS?style=social)<br>[HuggingFace](https://huggingface.co/2Noise/ChatTTS)<br>[HF Mirror](https://hf-mirrors.com/2Noise/ChatTTS)|[Video](https://www.bilibili.com/video/BV1zn4y1o7iV)|||
|2024.02.20|**MeloTTS**<br>@MIT&MyShell.AI&清华大学|[Github](https://github.com/myshell-ai/MeloTTS/)<br>![Star](https://img.shields.io/github/stars/myshell-ai/MeloTTS)<br>[HuggingFace](https://huggingface.co/myshell-ai)<br>[HF Mirror](https://hf-mirror.com/myshell-ai)|||
|2024.02.13|**Parler-TTS**<br>@HuggingFace (Reproduce)|[Github](https://github.com/huggingface/parler-tts/)<br>![Star](https://img.shields.io/github/stars/huggingface/parler-tts)<br>[HuggingFace](https://huggingface.co/parler-tts/)<br>[HF Mirror](https://hf-mirror.com/parler-tts/)|[HF Space](https://huggingface.co/spaces/parler-tts/parler_tts)<br>[Demo Page](https://www.text-description-to-speech.com)|[ArXiv](https://arxiv.org/abs/2402.01912)|
|2024.01.15 (v1)<br>2024.08.07 (v2)|**GPT-SoVITS**<br>@RVC-Boss (花儿不哭)|[Github](https://github.com/RVC-Boss/GPT-SoVITS)<br>![Star](https://img.shields.io/github/stars/RVC-Boss/GPT-SoVITS?style=social)|[Video](https://www.bilibili.com/video/BV12g4y1m7Uw/)|
|2023.11.29<br>2024.04.09 (v2)|**OpenVoice**<br>@MIT&MyShell.AI&清华大学|[Github](https://github.com/myshell-ai/OpenVoice)<br>![Star](https://img.shields.io/github/stars/myshell-ai/openvoice.svg?style=social&label=Star)||[ArXiv](https://arxiv.org/abs/2312.01479)|
|2023.10.10<br>2024.05.10 (v1.1)<br>2024.07.02 (v1.2)<br>2024.09.10 (v1.4)<br>2024.11.02 Paper|**Fish-Speech**<br>@FishAudio |[Github](https://github.com/fishaudio/fish-speech)<br>![Star](https://img.shields.io/github/stars/fishaudio/fish-speech?style=social)<br>[HuggingFace](https://huggingface.co/fishaudio/fish-speech-1)|[Video](https://www.bilibili.com/video/BV1mQ4y1E7qD/) <br>[Video ver1.1](https://www.bilibili.com/video/BV1zJ4m1K7cj/)<br>[Video ver1.2](https://www.bilibili.com/video/BV1wz421B71D/)<br>[Video ver1.4](https://www.bilibili.com/video/BV1pu46eVEk7)<br>[Online](https://fs.firefly.matce.cn)|[ArXiv](https://arxiv.org/abs/2411.01156)|
|2023.09.06|**Matcha-TTS**<br>@瑞典皇家理工学院|[Github](https://github.com/shivammehta25/Matcha-TTS)<br>![Star](https://img.shields.io/github/stars/shivammehta25/Matcha-TTS?style=social) | | [ArXiv](https://arxiv.org/abs/2309.03199)<br>[PaperNote](Models/Diffusion/2023.09.06_Matcha-TTS.md) |
|2023.07.21<br>2024.07.12|**BERT-VITS2**<br>@FishAudio|[Github](https://github.com/fishaudio/Bert-VITS2)<br>![Star](https://img.shields.io/github/stars/fishaudio/Bert-VITS2?style=social)||
|2023.04.13|**MassTTS**<br>@2noise 团队|[Github](https://github.com/anyvoiceai/MassTTS)<br>![stars](https://img.shields.io/github/stars/anyvoiceai/MassTTS?style=social)|[Video](https://www.bilibili.com/video/BV1w24y1c7z9)|
|2022.01.28|**TorToise-TTS**<br>@neonbjb|[Github](https://github.com/neonbjb/tortoise-tts)<br>![Star](https://img.shields.io/github/stars/neonbjb/tortoise-tts?style=social)<br>|[HF Space](https://huggingface.co/spaces/Manmay/tortoise-tts)<br>[Demo Page](http://nonint.com/static/tortoise_v2_examples.html)|[ArXiv](https://arxiv.org/abs/2305.07243)<br>[PaperNote](Models/Diffusion/2023.05.12_TorToise-TTS.md)|
|2020.05.20|**XTTS v2**<br>@Coqui.AI|[Github](https://github.com/coqui-ai/TTS)<br>![Star](https://img.shields.io/github/stars/coqui-ai/TTS?style=social) ||[ArXiv](https://arxiv.org/abs/2406.04904)<br>[PaperNote](Models/SpeechLM/2024.06.07_XTTS.md)|
|2017.11.14|**PaddleSpeech**<br>@百度飞桨|[Github](https://github.com/PaddlePaddle/PaddleSpeech)<br>![Star](https://img.shields.io/github/stars/PaddlePaddle/PaddleSpeech?style=social)|

## 开源项目·Singing Voice Synthesis

- [DiffSinger Website](https://DiffSinger.com)

## 开源项目·Speech Interaction (语音交互)

- 2024.08.07 [Moshi [Github]](https://github.com/kyutai-labs/moshi) ![Star](https://img.shields.io/github/stars/kyutai-labs/moshi?style=social)
  - 开发团队: 法国 Kyutai 实验室
  - 技术报告:
    - 2024.09.17 发布 [ArXiv:2410.00037](https://arxiv.org/abs/2410.00037);
    - 2024.10.02 更新 v2 版本;
  - 仓库创建: 2024.08.07
  - 最近更新: 2025.03.03
  - 开源程度: 权重 + 推理 ([会发布一些训练代码但无具体时间, 不会发布预训练数据集](https://github.com/kyutai-labs/moshi/blob/main/FAQ.md))
  - 开源内容:
    - [Moshi v0.1 [HF]](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)
      - 含 BF16/INT8/INT4 版本.
      - Speech Codec (**Mimi**): WavLM 通过余弦相似度蒸馏语义到第一个码本 Token, Split RVQ 重建由对抗损失训练.
      - **Moshiko** (Moshi 男性合成声音微调)
      - **Moshika** (Moshi 女性合成声音微调)
  - 效果示例:
    - [在线网站](https://moshi.chat/)

- 2024.08.10 [VITA [Github]](https://github.com/VITA-MLLM/VITA) ![Star](https://img.shields.io/github/stars/VITA-MLLM/VITA?style=social)
  - 开发团队: VITA Team (腾讯优图实验室 & 南京大学 & 厦门大学 & 中科院自动化所)
  - 技术报告:
    - 2024.08.09 VITA [ArXiv:2408.05211](https://arxiv.org/abs/2408.05211)
      - 2024.09.10 更新 v2 版本;
    - 2025.01.03 VITA 1.5 [ArXiv:2501.01957](https://arxiv.org/abs/2501.01957)
      - 2025.01.16 更新 v2 版本;
      - 2025.01.21 更新 v3 版本.
  - 仓库创建: 2024.08.10
  - 最近更新: 2025.02.13
  - 开源程度: 权重 + 推理 + 训练
  - 开源内容:
      - 2024.09.06 发布 [VITA [HF]](https://huggingface.co/VITA-MLLM/VITA)
        - Visual Encoder (**InternViT-300M-448px**) + Visual Connector
        - Audio Encoder (4 CNN + 24 Transformer Blocks ~341M) + Audio Connector
        - VITA (**Mixtral-8x7B-v0.1**)
        - External TTS (**TencentCloud API**)
      - 2024.12.20 发布 [VITA 1.5 [HF]](https://huggingface.co/VITA-MLLM/VITA-1.5)
        - Vision Encoder (**InternViT-300M**) + Vision Adapter (2 **MLP**)
        - Speech Encoder (类似 **Freeze-Omni** 采用 Conv + 24 Transformer Blocks ~350M) + Speech Adapter (Conv 2x Downsample)
        - VITA 1.5 (**Qwen2-7B**): 输入多模态, 输出文本 Token
        - NAR Speech Decoder: 输入文本 Token, 输出语音 Token 初始分布;
        - AR Speech Decoder: 输入 NAR 信息, 输出语音 Token;
        - Codec Decoder (**TiCodec**): 输入语音 Token, 输出语音.
  - 效果示例:
    - [Github.IO](https://vita-home.github.io)
    - [YouTube](https://youtu.be/tyi6SVFT5mM)
    - [ModelScope Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)
    - 特性: VITA 采用两个模型 Monitor 和 Generation 实现交互; 三阶段训练: 双语种指令微调, 视觉音频模态对齐, 多模态指令微调.
    - 特性: VITA 1.5 时延从 4 s 降低到 1.5 s; 三阶段训练: 视觉语言训练, 音频输入微调, 音频输出微调.

- 2024.08.29 [Mini-Omni [Github]](https://github.com/gpt-omni/mini-omni) ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni?style=social)
  - 开发团队: InspirAI & 清华大学
  - 技术报告:
    - 2024.08.29 发布 [ArXiv:2408.16725](https://arxiv.org/abs/2408.16725)
    - 2024.08.30 更新 v2;
    - 2024.11.05 更新 v3;
  - 仓库创建: 2024.08.29
  - 最近更新: 2024.11.05
  - 开源程度: 权重 + 推理 (训练不会开源, 但基于 [litgpt](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py) 进行的修改)
  - 开源内容:
    - [Mini-Omni [HF]](https://huggingface.co/gpt-omni/mini-omni) #TODO
    - [VoiceAssistant-400K [HF]](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K)
  - 效果示例:
    - [HF Spaces](https://huggingface.co/spaces/gradio/omni-mini)
    - 特性: 仅在英语上训练, 由 Whisper 支持可理解多语种, 但仅能输出英语; 开源版本不支持 TTS Adapter.

- 2024.09.10 [LLaMA-Omni [Github]](https://github.com/ictnlp/LLaMA-Omni) ![Star](https://img.shields.io/github/stars/ictnlp/LLaMA-Omni?style=social)
  - 开发团队: [中国科学院 & 中国科学院大学 ICT@NLP 研究组](https://nlp.ict.ac.cn)
  - 技术报告:
    - 2024.09.10 发布 [ArXiv:2409.06666](https://arxiv.org/abs/2409.06666);
    - 2025.03.01 更新 v2 版本 **ICLR2025**.
  - 仓库创建: 2024.09.10
  - 最近更新: 2024.11.14
  - 开源程度: 权重 + 推理
  - 开源内容:
    - [LLaMA-3.1-8B-Omni [HF]](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni)
      - LLaMA-3.1-8B-Instruct
      - Whisper-Large-V3
      - Unit-Based HiFi-GAN Vocoder ([FairSeq ver.](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md))
  - 效果示例：
    - [Replicate 平台](https://replicate.com/ictnlp/llama-omni)
    - 特性: 时延 226 ms, 四卡训练少于三天, 同时生成文本和语音模态.
  - 相关代码:
    - 训练复现: https://github.com/wntg/LLaMA-Omni

- 2024.09.30 [Baichuan-Omni [Github]](https://github.com/westlake-baichuan-mllm/bc-omni) ![Star](https://img.shields.io/github/stars/westlake-baichuan-mllm/bc-omni?style=social)
  - 开发团队: 百川智能 & 西湖大学 & 浙江大学
  - 技术报告:
    - 2024.10.11 发布 [ArXiv:2410.08565](https://arxiv.org/abs/2410.08565);
    - 2024.11.02 更新 v2 版本, 改名 Ocean-Omni;
    - 2024.11.05 更新 v3 版本;
    - 2024.12.27 更新 v4 版本, 改名 Baichuan-Omni.
  - 仓库创建: 2024.09.30
  - 最近更新: 2025.01.27
  - 开源程度: 未开源
  - 注: 后续版本发布, 本项目应已归档. **2025.01.25 发布 Baichuan-Omni-1.5 版本**

- 2024.10.16 [Mini-Omni2 [Github]](https://github.com/gpt-omni/mini-omni2) ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni2?style=social)
  - 开发团队: 启元世界 & 清华大学
  - 技术报告:
    - 2024.10.15 发布 [ArXiv:2410.11190](https://arxiv.org/abs/2410.11190);
  - 仓库创建: 2024.10.16
  - 最近更新: 2025.01.16
  - 开源程度: #TODO
  - 开源内容:
    - [Mini-Omni2 [HF]](https://huggingface.co/gpt-omni/mini-omni2)
  - 效果示例:
    - 特性: 仅在英语上训练, 由 Whisper 支持可理解多语种, 但仅能输出英语;

- 2024.10.24 [GLM-4-Voice [Github]](https://github.com/THUDM/GLM-4-Voice) ![Star](https://img.shields.io/github/stars/THUDM/GLM-4-Voice?style=social)
  - 开发团队: 智谱 AI & 清华大学
  - 技术报告:
    - 2024.12.03 发布 [ArXiv:2412.02612](https://arxiv.org/abs/2412.02612);
  - 仓库创建: 2024.10.24
  - 最近更新: 2024.12.05
  - 开源程度: 权重 + 推理
  - 开源内容:
    - [GLW-4-Voice-9B [HF]](https://huggingface.co/THUDM/glm-4-voice-9b)
      - [GLM-4-Voice-Tokenizer [HF]](https://huggingface.co/THUDM/glm-4-voice-tokenizer)
        - Whisper Encoder + VQ, 在 ASR 数据上监督训练;
      - GLM-4-Voice-9B: 在 GLM-4-9B 基础上进行语音模态的预训练和对齐.
      - [GLM-4-Voice-Decoder [HF]](https://huggingface.co/THUDM/glm-4-voice-decoder)
        - CosyVoice Flow-Matching, 输入语音 Token
  - 效果示例:
    - [ModelScope](https://modelscope.cn/studios/ZhipuAI/GLM-4-Voice-Demo)

- 2024.11.04 [Freeze-Omni [Github]](https://github.com/VITA-MLLM/Freeze-Omni) ![Star](https://img.shields.io/github/stars/VITA-MLLM/Freeze-Omni?style=social)
  - 开发团队: VITA Team & 腾讯优图实验室 & ASLP(NPU) & 南京大学
  - 技术报告:
    - 2024.11.01 发布 [ArXiv:2411.00774](https://arxiv.org/abs/2411.00774);
  - 仓库创建: 2024.11.04
  - 最近更新:
  - 开源程度:
  - 开源内容: #TODO
  - 效果示例:
    - [Github.IO](https://freeze-omni.github.io)

- 2025.01.23 [Baichuan-Omni-1.5 [Github]](https://github.com/baichuan-inc/Baichuan-Omni-1.5) ![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan-Omni-1.5?style=social)
  - 开发团队:
  - 技术报告:
    - 2025.01.26 发布 [ArXiv:2501.15368](https://arxiv.org/abs/2501.15368)
  - 仓库创建: 2025.01.23
  - 最近更新: 2025.02.08
  - 开源程度: 权重 + 推理
  - 开源内容:
    - [Baichuan-Omni-1.5-Base [HF]](https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5-Base): 未 SFT;
      - Visual Encoder (**NaViT** + Qwen2-VL-7B Weight)
      - Baichuan-Audio Tokenizer (8 RVQ)
      - Text Tokenizer
      - **Qwen2.5-7B LLM**: 输入多模态 Token, 输出文本和语音模态.
      - **HiFi-GAN Vocoder (CosyVoice2 ver.)**
    - [Baichuan-Omni-1.5 [HF]](https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5): 全模态指令训练;
    - [OpenMM-Medical [HF]](https://huggingface.co/datasets/baichuan-inc/OpenMM_Medical): 医学理解基准, 从公开医学图像数据集收集得到 88,996 张图像;
    - [OpenAudio-Bench [HF]](https://huggingface.co/datasets/baichuan-inc/openAudioBench): 音频基准, 含 4 个公开评测集 (LLaMA Question, Web QA, TriviaQA, AlpacaEval) + 自建语音逻辑评测集 2701 条.

- 2025.02.24 [Baichuan-Audio [Github]](https://github.com/baichuan-inc/Baichuan-Audio) ![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan-Audio?style=social)
  - 开发团队:
  - 技术报告:
    - 2025.02.24 发布 [ArXiv:2502.17239](https://arxiv.org/abs/2502.17239)
  - 仓库创建: 2025.02.24
  - 最近更新: 2025.02.28
  - 开源程度: 权重 + 推理
  - 开源内容:
    - [Baichuan-Audio-Base [HF]](https://huggingface.co/baichuan-inc/Baichuan-Audio-Base): 未 SFT;
      - **Baichuan-Audio Tokenizer (Whisper Large + 8 RVQ)**: 输入语音, 输出语音 Token, 由梅尔频谱重构和预训练 LLM 进行声学+语义监督训练;
      - **Qwen2.5-7B Audio LLM**: 输入离散音频 Token, 输出模态交错 Token, 根据特殊 Token 可切换输出模态;
      - **Flow-Matching Audio Decoder**: 输入语音 Token, 输出梅尔频谱图, 在 24KHz 音频训练;
      - **HiFi-GAN Vocoder (CosyVoice2 ver.)**: 输入梅尔频谱图, 输出语音;
    - [Baichuan-Audio-Instruct [HF]](https://huggingface.co/baichuan-inc/Baichuan-Audio-Instruct)
    - [OpenAudio-Bench [HF]](https://huggingface.co/datasets/baichuan-inc/openAudioBench): 音频基准.
