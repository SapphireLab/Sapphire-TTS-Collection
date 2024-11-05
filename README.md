# Sapphire-TTS-Collection

本仓库收集/整理/学习语音合成技术相关资料.
(可能含有部分 NLP 和其他语音相关技术)
仍在初步更新中

[模型总结表 Models.CSV](Models.csv)

## 推荐链接

- [Speech.Zone](https://speech.zone)
- [语音之家](https://www.speechhome.com)
- 课题组 [X-LANCE@SJTU](https://x-lance.sjtu.edu.cn)
- 课题组 [ASLP@NPU](http://www.npu-aslp.org)
- 课题组 [Speech@NTU](https://www.youtube.com/@HungyiLeeNTU)

## 同类项目

- [Awesome Audio Plaza](https://github.com/metame-ai/awesome-audio-plaza) by Metame AI ![Star](https://img.shields.io/github/stars/metame-ai/awesome-audio-plaza?style=social)
- [Speech Trident](https://github.com/ga642381/speech-trident) by 李宏毅 Team ![Star](https://img.shields.io/github/stars/ga642381/speech-trident)
- [Awesome MLLM](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) by VITA Team ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models?style=social)
- [Awesome Large Speech Model](https://github.com/huangcanan/Awesome-Large-Speech-Model) by Huang Can'an ![Star](https://img.shields.io/github/stars/huangcanan/Awesome-Large-Speech-Model?style=social)

## 开源项目·Text-to-Speech

|时间|名称|仓库|演示|论文|
|---|---|:-:|---|---|
|2024.10.08|**F5-TTS**<br>@上海交通大学&剑桥大学<br>&吉利汽车研究院(宁波)|[Github](https://github.com/SWivid/F5-TTS)<br>![Star](https://img.shields.io/github/stars/SWivid/F5-TTS?style=social)<br>[HuggingFace](https://huggingface.co/SWivid/F5-TTS)<br>[HF Mirror](https://hf-mirror.com/SWivid/F5-TTS)|[Demo Page](https://swivid.github.io/F5-TTS/)<br>[HuggingFace Space](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)|[ArXiv](https://arxiv.org/abs/2410.06885)<br>[PaperNote](Models/Diffusion/2024.10.09_F5-TTS.md)<br>[CodeReview](OpenSource/Official/2024.10.08_F5-TTS/Main.md)|
|2024.07.03|**CosyVoice**<br>@阿里巴巴语音实验室|[Github](https://github.com/FunAudioLLM/CosyVoice)<br>![Star](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice?style=social)<br>[ModelScope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)||[ArXiv](https://arxiv.org/abs/2407.05407)<br>[PaperNote](Models/SpeechLM/2024.07.07_CosyVoice.md)|
|2024.05.28|**ChatTTS**<br>@2noise 团队|[Github](https://github.com/2noise/ChatTTS)<br>![Star](https://img.shields.io/github/stars/2noise/ChatTTS?style=social)<br>[HuggingFace](https://huggingface.co/2Noise/ChatTTS)<br>[HF Mirror](https://hf-mirrors.com/2Noise/ChatTTS)|[Video](https://www.bilibili.com/video/BV1zn4y1o7iV)|||
|2024.02.20|**MeloTTS**<br>@MIT&MyShell.AI&清华大学|[Github](https://github.com/myshell-ai/MeloTTS/)<br>![Star](https://img.shields.io/github/stars/myshell-ai/MeloTTS)<br>[HuggingFace](https://huggingface.co/myshell-ai)<br>[HF Mirror](https://hf-mirror.com/myshell-ai)|||
|2024.02.13|**Parler-TTS**<br>@HuggingFace (Reproduce)|[Github](https://github.com/huggingface/parler-tts/)<br>![Star](https://img.shields.io/github/stars/huggingface/parler-tts)<br>[HuggingFace](https://huggingface.co/parler-tts/)<br>[HF Mirror](https://hf-mirror.com/parler-tts/)|[Space](https://huggingface.co/spaces/parler-tts/parler_tts)<br>[Page](https://www.text-description-to-speech.com)|[ArXiv](https://arxiv.org/abs/2402.01912)|
|2024.01.15 (v1)<br>2024.08.07 (v2)|**GPT-SoVITS**<br>@RVC-Boss (花儿不哭)|[Github](https://github.com/RVC-Boss/GPT-SoVITS)<br>![Star](https://img.shields.io/github/stars/RVC-Boss/GPT-SoVITS?style=social)|[Video](https://www.bilibili.com/video/BV12g4y1m7Uw/)|
|2023.11.29<br>2024.04.09 (v2)|**OpenVoice**<br>@MIT&MyShell.AI&清华大学|[Github](https://github.com/myshell-ai/OpenVoice)<br>![Star](https://img.shields.io/github/stars/myshell-ai/openvoice.svg?style=social&label=Star)||[ArXiv](https://arxiv.org/abs/2312.01479)|
|2023.10.10<br>2024.05.10 (v1.1)<br>2024.07.02 (v1.2)<br>2024.09.10 (v1.4)<br>2024.11.02 Paper|**Fish-Speech**<br>@FishAudio |[Github](https://github.com/fishaudio/fish-speech)<br>![Star](https://img.shields.io/github/stars/fishaudio/fish-speech?style=social)<br>[HuggingFace](https://huggingface.co/fishaudio/fish-speech-1)|[Video](https://www.bilibili.com/video/BV1mQ4y1E7qD/) <br>[Video ver1.1](https://www.bilibili.com/video/BV1zJ4m1K7cj/)<br>[Video ver1.2](https://www.bilibili.com/video/BV1wz421B71D/)<br>[Video ver1.4](https://www.bilibili.com/video/BV1pu46eVEk7)<br>[Online](https://fs.firefly.matce.cn)|[ArXiv](https://arxiv.org/abs/2411.01156)|
|2023.09.06|**Matcha-TTS**<br>@瑞典皇家理工学院|[Github](https://github.com/shivammehta25/Matcha-TTS)<br>![Star](https://img.shields.io/github/stars/shivammehta25/Matcha-TTS?style=social) | | [ArXiv](https://arxiv.org/abs/2309.03199)<br>[PaperNote](Models/Diffusion/2023.09.06_Matcha-TTS.md) |
|2023.07.21<br>2024.07.12|**BERT-VITS2**<br>@FishAudio|[Github](https://github.com/fishaudio/Bert-VITS2)<br>![Star](https://img.shields.io/github/stars/fishaudio/Bert-VITS2?style=social)||
|2023.04.13|**MassTTS**<br>@2noise 团队|[Github](https://github.com/anyvoiceai/MassTTS)<br>![stars](https://img.shields.io/github/stars/anyvoiceai/MassTTS?style=social)|[Video](https://www.bilibili.com/video/BV1w24y1c7z9)|
|2022.01.28|**TorToise-TTS**<br>@neonbjb|[Github](https://github.com/neonbjb/tortoise-tts)<br>![Star](https://img.shields.io/github/stars/neonbjb/tortoise-tts?style=social)<br>|[Space](https://huggingface.co/spaces/Manmay/tortoise-tts)<br>[Demo](http://nonint.com/static/tortoise_v2_examples.html)|[ArXiv](https://arxiv.org/abs/2305.07243)<br>[PaperNote](Models/Diffusion/2023.05.12_TorToise-TTS.md)|
|2020.05.20|**XTTS v2**<br>@Coqui.AI|[Github](https://github.com/coqui-ai/TTS)<br>![Star](https://img.shields.io/github/stars/coqui-ai/TTS?style=social) ||[ArXiv](https://arxiv.org/abs/2406.04904)<br>[PaperNote](Models/SpeechLM/2024.06.07_XTTS.md)|
|2017.11.14|**PaddleSpeech**<br>@百度飞桨|[Github](https://github.com/PaddlePaddle/PaddleSpeech)<br>![Star](https://img.shields.io/github/stars/PaddlePaddle/PaddleSpeech?style=social)|

## 开源项目·Speech Interaction

按仓库创建时间排序

|时间|名称|仓库|演示|论文|
|---|---|:-:|---|---|
|2023.11.07 Code<br>2023.11.14 Paper|**Qwen-Audio**<br>@阿里巴巴|[Github](https://github.com/QwenLM/Qwen-Audio)|[Demo](https://qwen-audio.github.io/Qwen-Audio/)|[ArXiv](https://arxiv.org/abs/2311.07919)|
|2024.05.14 Claim<br>2024.07.30 Release<br>2024.10.25 Paper|**GPT-4o** System Card<br>@OpenAI||[ChatGPT](https://chatgpt.com/)|[ArXiv](https://arxiv.org/abs/2410.21276)|
|2024.07.02 Code<br>2024.10.20 Paper|**Ichigo**<br>@HomebrewResearch|[Github](https://github.com/homebrewltd/ichigo)|[Demo](https://ichigo.homebrew.ltd)|[ArXiv](https://arxiv.org/abs/2410.15316)|
|2024.07.15 Paper<br>2024.07.16 Code|**Qwen2-Audio**<br>@阿里巴巴千问团队|[Github](https://github.com/QwenLM/Qwen2-Audio/)<br>![Star](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio?style=social)<br>[HuggingFace](https://huggingface.co/Qwen/)<br>[ModelScope](https://modelscope.cn/models/qwen/)|[Space](https://modelscope.cn/studios/qwen/Qwen2-Audio-Instruct-Demo)<br>[Studio](https://modelscope.cn/studios/qwen/Qwen2-Audio-Instruct-Demo)<br>[Blog](https://qwenlm.github.io/blog/qwen2-audio)|[ArXiv](https://arxiv.org/abs/2407.10759)|
|2024.08.07 Code<br>2024.09.17 Paper|**Moshi**<br>@法国 Kyutai 实验室|[Github](https://github.com/kyutai-labs/moshi)<br>![Star](https://img.shields.io/github/stars/kyutai-labs/moshi)<br>[HuggingFace](https://huggingface.co/kyutai)<br>[HF Mirror](https://hf-mirror.com/kyutai)|[Demo](https://moshi.chat/)|[ArXiv](https://arxiv.org/abs/2410.00037)|
|2024.08.09 Paper<br>2024.08.10 Code|**VITA**<br>@VITA Team<br>(腾讯优图实验室&南京大学<br>厦门大学&中科院自动化所)|[Github](https://github.com/VITA-MLLM/VITA)<br>![Star](https://img.shields.io/github/stars/VITA-MLLM/VITA)<br>[HuggingFace](https://huggingface.co/spaces/VITA-MLLM)<br>[HF Mirror](https://hf-mirror.com/VITA-MLLM)|[Page](https://vita-home.github.io/)|[ArXiv](https://arxiv.org/abs/2408.05211)|
|2024.08.29 Paper<br>2024.08.29 Code|**Mini-Omni**<br>@清华大学&(启元世界?)|[Github](https://github.com/gpt-omni/mini-omni)<br>![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni)<br>[HuggingFace](https://huggingface.co/gpt-omni/mini-omni)<br>[HF Mirror](https://hf-mirror.com/gpt-omni/mini-omni)|[Space](https://huggingface.co/spaces/gradio/omni-mini)|[ArXiv](https://arxiv.org/abs/2408.16725)|
|2024.09.10 Paper<br>2024.09.10 Code|**LLaMA-Omni**<br>@中国科学院&中国科学院大学|[Github](https://github.com/ictnlp/LLaMA-Omni)<br>![Star](https://img.shields.io/github/stars/ictnlp/LLaMA-Omni?style=social)<br>[HuggingFace](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni)<br>[HF Mirror](https://hf-mirror.com/ICTNLP/Llama-3.1-8B-Omni)<br>[ModelScope](https://modelscope.cn/models/ICTNLP/Llama-3.1-8B-Omni)|[Demo](https://replicate.com/ictnlp/llama-omni)|[ArXiv](https://arxiv.org/abs/2409.06666)|
|2024.09.24 Code|**WestLake-Omni**<br>@西湖心辰|[Github](https://github.com/xinchen-ai/Westlake-Omni)<br>![Star](https://img.shields.io/github/stars/xinchen-ai/Westlake-Omni)<br>[HuggingFace](https://huggingface.co/xinchen-ai/Westlake-Omni)<br>[HF Mirror](https://hf-mirror.com/xinchen-ai/Westlake-Omni)||ArXiv|
|2024.09.30 Code<br>2024.10.11 Paper|**Baichuan-Omni**<br>@百川智能&西湖大学&浙江大学|[Github](https://github.com/westlake-baichuan-mllm/bc-omni)<br>![Star](https://img.shields.io/github/stars/westlake-baichuan-mllm/bc-omni)<br>HuggingFace<br>HF Mirror||[ArXiv](https://arxiv.org/abs/2410.08565)|
|2024.10.15 Paper<br>2024.10.16 Code|**Mini-Omni2**<br>@启元世界&清华大学|[Github](https://github.com/gpt-omni/mini-omni2)<br>![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni2)<br>[HuggingFace](https://huggingface.co/gpt-omni/mini-omni2)<br>[HF Mirror](https://hf-mirror.com/gpt-omni/mini-omni2)||[ArXiv](https://arxiv.org/abs/2410.11190)|
|2024.10.24 Code|**GLM-4-Voice**<br>@智谱 AI|[Github](https://github.com/THUDM/GLM-4-Voice)<br>![Star](https://img.shields.io/github/stars/THUDM/GLM-4-Voice)<br>[HuggingFace](https://huggingface.co/THUDM)<br>[HF Mirror](https://hf-mirror.com/THUDM)<br>[ModelScope](https://modelscope.cn/models/ZhipuAI/)||ArXiv|
|2024.11.01 Paper<br>2024.11.04 Code|**Freeze-Omni**<br>@VITA Team<br>(腾讯优图实验室&ASLP(NPU)&南京大学)|[Github](https://github.com/VITA-MLLM/Freeze-Omni)<br>![Star](https://img.shields.io/github/stars/VITA-MLLM/Freeze-Omni) | [Demo](https://freeze-omni.github.io) | [ArXiv](https://arxiv.org/abs/2411.00774) |
|2024.11.03 Code|**Hertz-Dev**<br>@Standard Intelligence|[Github](https://github.com/Standard-Intelligence/hertz-dev)<br>![Star](https://img.shields.io/github/stars/Standard-Intelligence/hertz-dev?style=social)<br>[Checkpoints](https://ckpt.si.inc/hertz-dev/index.txt)|[Blog](https://si.inc/hertz-dev/)|