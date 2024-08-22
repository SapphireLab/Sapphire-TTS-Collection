# Amphion 

本页面概括项目内容及进展.

## 更新日志

- 2023.11.28 [PR002](https://github.com/open-mmlab/Amphion/pull/2) **Alpha 版本发布**:
  - TTS 文本转语音: 
    - [FastSpeech2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md)
    - [VITS](../../Models/E2E/2021.06.11_VITS.md)
  - SVC 歌声转换:
    - NeurIPS 2023
  - TTA 文本转音频:
    - NeurIPS 2023 基于 LDM 的 TTS, 也是 [AUDIT] 的 TTA 生成部分的官方实现. #TODO
  - Vocoder 声码器:
    - 复现基于 GAN 的声码器: [MelGAN](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md); [HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md); [NSF-HiFiGAN](../../Models/TTS3_Vocoder/NSF-HiFiGAN.md); [BigVGAN](../../Models/TTS3_Vocoder/2022.06.09_BigVGAN.md); [APNet](../../Models/TTS3_Vocoder/2023.05.13_APNet.md);
    - 开放 Multi-Scale Constant-Q Transfrom Discriminator 官方代码;
    - 开放两个声码器权重: `Amphion Speech HiFi-GAN` 和 `Amphion Singing BigVGAN`;
  - 评估:
    - 支持十六项客观度量指标: F0 建模, 能量建模, 可理解性, 频谱失真, 说话人相似度;
  - 数据集:
    - 支持十五个学术数据集.
- 2023.12.02 PR004 新增模型 [VALL-E](../../Models/Speech_LLM/2023.01.05_VALL-E.md) 复现代码;
- 2023.12.18 PR035 新增模型 [NaturalSpeech2](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md) 复现代码;
- 2023.12.18 PR039 **v0.1 版本发布**;
- 2023.12.21 PR056 新增模型 [DiffWave](../../Models/TTS3_Vocoder/2020.09.21_DiffWave.md) 声码器复现代码;
- 2024.01.11 PR097 支持基于 [WavLM](../../Models/Speech_Representaion/2021.10.26_WavLM.md) 的说话人相似性度量;
- 2024.02.23 PR141 新增工具 [SingVisio](../../Models/Singing_Voice/2024.02.20_SingVisio.md) 构建代码;
- 2024.03.12 PR152 新增模型 [FACodec](../../Models/Speech_Neural_Codec/2024.03.05_FACodec.md) 构建代码;
- 2024.06.22 PR220 发布模型 [VALL-E](../../Models/Speech_LLM/2023.01.05_VALL-E.md) 新版本;
- 2024.07.09 PR227 新增数据集 [Emilia](../../Datasets/2024.07.07_Emilia.md) 构建代码;
- 2024.07.15 PR231 新增模型 [JETS](../../Models/E2E/2022.07.01_JETS.md) 复现代码;
- 2024.08.22 PR265 开放数据集 [Emilia](../../Datasets/2024.07.07_Emilia.md);
