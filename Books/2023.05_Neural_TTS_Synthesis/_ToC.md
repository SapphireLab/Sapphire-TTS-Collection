# Neural Text-to-Speech Synthesis

## 目录

|Parts|Sections|
|---|---|
||[01.Introduction](Ch01.Introduction.md)|
|**Part I** Preliminary <br> 第一部分 预备知识|[02.Basics of Spoken Language Processing](Ch02.Basics_of_Spoken_Language_Processing.md)<br>[03.Basics of Deep Learning](Ch03.Basics_of_DL.md)|
|**Part II** Key Components in TTS <br> 第二部分 文本转语音的关键组成部分|[04.Text Analyses](Ch04.Text_Analyses.md)<br>[05.Acoustic Models](Ch05.Acoustic_Models.md)<br>[06.Vocoders](Ch06.Vocoders.md)|<br>[07.Fully End-to-End TTS](Ch07.Fully_End-to-End_TTS.md)|
|**Part III** Advanced Topics in TTS <br>第三部分 文本转语音的前沿主题|[08.Expressive and Controllable TTS](Ch08.Expressive&Controllable_TTS.md) <br> [09.Robust TTS](Ch09.Robust_TTS.md) <br> [10.Model-Efficient TTS](Ch10.Model-Efficient_TTS.md)<br>[11.Data-Efficient TTS](Ch11.Data-Efficient_TTS.md)<br>[12.Beyond Text-to-Speech Synthesis](Ch12.Beyond_TTS_Synthesis.md)
|**Part IV** Summary and Outlook<br>第四部分 总结与展望|[13.Summary and Outlook](Ch13.Summary.md)|

## A.语音合成相关资源

## B.语音合成模型列表

### End-to-End

|模型|类型|数据流|发表|时间|
|:-:|:-:|:-:|---|:-:|
|[Char2Wav](../../Models/E2E/2017.02.18_Char2Wav.md)|E2E|ch→(AR)→ceps→(AR)→wav|ICLR17 WS|2017.02|
|[ClariNet](../../Models/E2E/2018.07.19_ClariNet.md)|E2E|ch/ph→(AR)→wav|ICLR19|2018.07|
|[FastSpeech 2s](../../Models/E2E/2020.06.08_FastSpeech2s.md)|E2E|ph→(FF)→wav|ICLR21|2020.06|
|[EATS](../../Models/E2E/2020.06.05_EATS.md)|E2E|ch/ph→(FF)→wav|ICLR21|2020.06|
|Wave-Tacotron [114]|E2E|ch/ph→(AR)→wav|ICASSP21|2020.11|
|EfficientTTS-Wav [116]|E2E|ch→(FF)→wav|ICML21|2020.12|
|[VITS](../../Models/E2E/2021.06.11_VITS.md)|E2E|ph→(FF)→wav|ICML21|2021.06|
|[WaveGrad 2](../../Models/TTS3_Vocoder/2021.06.17_WaveGrad2.md)|E2E|ph→(FF)→wav|IS21|2021.06|
|[NaturalSpeech](../../Models/E2E/2022.05.09_NaturalSpeech.md)|E2E|ph→(FF)→wav|arXiv22|2022.05|

### Acoustic Models

|模型|类型|数据流|发表|时间|
|:-:|:-:|:-:|---|:-:|
|[Tacotron](../../Models/TTS2_Acoustic/2017.03.29_Tacotron.md)|AM|ch/ph→(AR)→linS→wav|IS17|2017.03|
|[VoiceLoop](../../Models/TTS2_Acoustic/2017.07.20_VoiceLoop.md)|AM|ph→ceps→wav|ICLR18|2017.07|
|[Deep voice 3](../../Models/TTS2_Acoustic/2017.10.20_DeepVoice3.md)|AM|ch/ph→(AR)→melS→(AR)→wav|ICLR18|2017.10|
|DCTTS [53]|AM|ch→(AR)→melS→wav|ICASSP18|2017.10|
|[Tacotron 2](../../Models/TTS2_Acoustic/2017.12.16_Tacotron2.md)|AM|ch/ph→(AR)→melS→(AR)→wav|ICASSP18|2017.12|
|DV3-Clone [58]|AM|ch/ph→(AR)→linS→wav|NeurIPS18|2018.02|
|GST-Tacotron [59]|AM|ph→(AR)→melS→wav|ICML18|2018.03|
|Ref-Tacotron [60]|AM|ph→(AR)→melS→wav|ICML18|2018.03|
|VAE-Loop [62]|AM|ph→ceps→wav|IS18|2018.04|
|SV-Tacotron [63]|AM|ch/ph→(AR)→melS→(AR)→wav|NeurIPS18|2018.06|
|ForwardAtt [65]|AM|ph→(AR)→linS→wav|ICASSP18|2018.07|
|[TransformerTTS](../../Models/TTS2_Acoustic/2018.09.19_Transformer_TTS.md) [67]|AM|ph→(AR)→melS→(AR)→wav|AAAI19|2018.09|
|GMVAE-Tacotron [69]|AM|ph→(AR)→melS→(AR)→wav|ICLR19|2018.10|
|VAE-TTS [74]|AM|ph→(AR)→melS→(AR)→wav|ICASSP19|2018.12|
|TTS-Stylization [75]|AM|ch→(AR)→melS→wav|ICLR19|2018.12|
|GAN exposure [77]|AM|ph→(AR)→melS→(AR)→wav|IS19|2019.04|
|Almost unsup [79]|AM|ph→(AR)→melS→wav|ICML19|2019.05|
|[FastSpeech](../../Models/TTS2_Acoustic/2019.05.22_FastSpeech.md)|AM|ph→(FF)→melS→(FF)→wav|NeurIPS19|2019.05|
|[ParaNet](../../Models/TTS2_Acoustic/2019.05.21_ParaNet.md)|AM|ph→(FF)→melS→(FF)→wav|ICML20|2019.05|
|MelNet [82]|AM|ch→(AR)→melS→wav|arXiv19|2019.06|
|StepwiseMA [83]|AM|ph→(AR)→melS→(AR)→wav|IS19|2019.06|
|DurIAN [85]|AM|ph→(AR)→melS→(AR)→wav|IS20|2019.09|
|DCA-Tacotron [88]|AM|ph→(AR)→melS→(AR)→wav|ICASSP20|2019.10|
|[AlignTTS](../../Models/TTS2_Acoustic/2020.03.04_AlignTTS.md)|AM|ch/ph→(FF)→melS→(FF)→wav|ICASSP20|2020.03|
|RobuTrans [92]|AM|ph→(AR)→melS→(AR)→wav|AAAI20|2020.04|
|[Flow-TTS](../../Models/TTS2_Acoustic/2020.05.04_Flow-TTS.md)|AM|ch/ph→(FF)→melS→(FF)→wav|ICASSP20|2020.05|
|[Flowtron](../../Models/TTS2_Acoustic/2020.05.12_Flowtron.md)|AM|ph→(AR)→melS→(FF)→wav|ICLR21|2020.05|
|[Glow-TTS](../../Models/TTS2_Acoustic/2020.05.22_Glow-TTS.md)|AM|ph→(FF)→melS→(FF)→wav|NeurIPS20|2020.05|
|[JDI-T](../../Models/TTS2_Acoustic/2020.05.15_JDI-T.md)|AM|ph→(FF)→melS→(FF)→wav|IS20|2020.05|
|TalkNet [97]|AM|ch→(FF)→melS→(FF)→wav|IS21|2020.05|
|MultiSpeech [99]|AM|ph→(AR)→melS→(FF)→wav|IS20|2020.06|
|[FastSpeech 2](../../Models/TTS2_Acoustic/2020.06.08_FastSpeech2.md)|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2020.06|
|[FastPitch](../../Models/TTS2_Acoustic/2020.06.11_FastPitch.md)|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2020.06|
|LRSpeech [104]|AM|ch→(AR)→melS→(FF)→wav|KDD20|2020.08|
|[SpeedySpeech](../../Models/TTS2_Acoustic/2020.08.09_SpeedySpeech.md) [105]|AM|ph→(FF)→melS→(FF)→wav|IS20|2020.08|
|NonAtt tacotron [111]|AM|ph→(AR)→melS→(AR)→wav|arXiv20|2020.10|
|Para. tacotron [112]|AM|ph→(FF)→melS→(AR)→wav|ICASSP21|2020.10|
|DeviceTTS [113]|AM|ph→(AR)→Ceps→wav|arXiv20|2020.10|
|DenoiSpeech [115]|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2020.12|
|EfficientTTS [116]|AM|ch→(FF)→melS→(FF)→wav|ICML21|2020.12|
|Multi-SpectroGAN [117]|AM|ph→(AR)→melS→(FF)→wav|AAAI21|2020.12|
|[LightSpeech](../../Models/TTS2_Acoustic/2021.02.08_LightSpeech.md)|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2021.02|
|Para. Tacotron 2 [119]|AM|ph→(FF)→melS→(AR)→wav|IS21|2021.03|
|[AdaSpeech](../../Models/TTS2_Acoustic/2021.03.01_AdaSpeech.md)|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2021.03|
|[BVAE-TTS](../../Models/TTS2_Acoustic/2021.01.13_BVAE-TTS.md)|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2021.03|
|PnG BERT [122]|AM|ph→(AR)→melS→(AR)→wav|IS21|2021.03|
|Fast DCTTS [123]|AM|ch→(AR)→melS→(FF)→wav|ICASSP21|2021.04|
|[AdaSpeech 2](../../Models/TTS2_Acoustic/2021.04.20_AdaSpeech2.md)|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2021.04|
|TalkNet 2 [125]|AM|ch→(FF)→melS→(FF)→wav|arXiv21|2021.04|
|Diff-TTS [127]|AM|ph→(FF)→melS→(FF)→wav|IS21|2021.04|
|[Grad-TTS](../../Models/TTS2_Acoustic/2021.05.13_Grad-TTS.md)|AM|ph→(FF)→melS→(FF)→wav|ICML21|2021.05|
|[AdaSpeech 3](../../Models/TTS2_Acoustic/2021.07.06_AdaSpeech3.md)|AM|ph→(FF)→melS→(FF)→wav|IS21|2021.06|
|PriorGrad-AM [132]|AM|ph→(FF)→melS→(FF)→wav|ICLR22|2021.06|
|Meta-StyleSpeech [133]|AM|ph→(FF)→melS→(FF)→wav|ICML21|2021.06|

### Vocoders

|模型|类型|数据流|发表|时间|
|:-:|:-:|:-:|---|:-:|
|[WaveNet](../../Models/TTS3_Vocoder/2016.09.12_WaveNet.md)|Voc|ling→(AR)→wav|arXiv16|2016.09|
|[SampleRNN](../../Models/TTS3_Vocoder/2016.12.22_SampleRNN.md)|Voc|∅→(AR)→wav|ICLR17|2016.12|
|[Para. WaveNet](../../Models/TTS3_Vocoder/2017.11.28_Parallel_WaveNet.md)|Voc|ling→(FF)→wav|ICML18|2017.11|
|[WaveGAN](../../Models/TTS3_Vocoder/2018.02.12_WaveGAN.md)|Voc|.∅FF−→wav|ICLR19|2018.02|
|[WaveRNN](../../Models/TTS3_Vocoder/2018.02.23_WaveRNN.md)|Voc|ling→(AR)→wav|ICML18|2018.02|
|FFTNet [61]|Voc|ceps→(AR)→wav|ICASSP18|2018.04|
|MCNN [66]|Voc|linS→(FF)→wav|SPL18|2018.08|
|SEA-TTS [68]|Voc|ling→(AR)→wav|ICLR19|2018.09|
|[LPCNet](../../Models/TTS3_Vocoder/2018.10.28_LPCNet.md)|Voc|ceps→(AR)→wav|ICASSP19|2018.10|
|[WaveGlow](../../Models/TTS3_Vocoder/2018.10.31_WaveGlow.md)|Voc|melS→(FF)→wav|ICASSP19|2018.10|
|[FloWaveNet](../../Models/TTS3_Vocoder/2018.11.06_FloWaveNet.md)|Voc|melS→(FF)→wav|ICML19|2018.11|
|Univ. WaveRNN [73]|Voc|melS→(AR)→wav|IS19|2018.11|
|AdVoc [76]|Voc|melS→(FF)→linS→wav|IS19|2019.04|
|GELP [78]|Voc|melS→(FF)→wav|IS19|2019.04|
|WaveVAE [81]|Voc|melS→(FF)→wav|ICML20|2019.05|
|[GAN-TTS](../../Models/TTS3_Vocoder/2019.09.25_GAN-TTS.md)|Voc|ling→(FF)→wav|ICLR20|2019.09|
|MB WaveRNN [85]|Voc|melS→(AR)→wav|IS20|2019.09|
|[MelGAN](../../Models/TTS3_Vocoder/2019.10.08_MelGAN.md)|Voc|melS→(FF)→wav|NeurIPS19|2019.10|
|[Para. WaveGAN](../../Models/TTS3_Vocoder/2019.10.25_Parallel_WaveGAN.md)|Voc|melS→(FF)→wav|ICASSP20|2019.10|
|[WaveFlow](../../Models/TTS3_Vocoder/2019.12.03_WaveFlow.md)|Voc|melS→(AR)→wav|ICML20|2019.12|
|SqueezeWave [90]|Voc|melS→(FF)→wav|arXiv20|2020.01|
|[MB MelGAN](../../Models/TTS3_Vocoder/2020.05.11_Multi-Band_MelGAN.md)|Voc|melS→(FF)→wav|SLT21|2020.05|
|[VocGAN](../../Models/TTS3_Vocoder/2020.07.30_VocGAN.md)|Voc|melS→(FF)→wav|IS20|2020.07|
|GED [106]|Voc|ling→(FF)→wav|NeurIPS20|2020.08|
|SC-WaveRNN [107]|Voc|melS→(AR)→wav|IS20|2020.08|
|[WaveGrad](../../Models/TTS3_Vocoder/2020.09.02_WaveGrad.md)|Voc|melS→(FF)→wav|ICLR21|2020.09|
|[DiffWave](../../Models/TTS3_Vocoder/2020.09.21_DiffWave.md)|Voc|melS→(FF)→wav|ICLR21|2020.09|
|[HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md)|Voc|melS→(FF)→wav|NeurIPS20|2020.10|
|Fre-GAN [129]|Voc|melS→(FF)→wav|IS21|2021.06|
|PriorGrad-Voc[132]|Voc|melS→(FF)→wav|ICLR22|2021.06|
|InferGrad [135]|Voc|melS→(FF)→wav|ICASSP22|2022.02|
|SpecGrad [136]|Voc|melS→(FF)→wav|IS22|2022.03|

### Acoustic Models + Vocoders

|模型|类型|数据流|发表|时间|
|:-:|:-:|:-:|---|:-:|
|[Deep Voice](../../Models/TTS0_System/2017.02.25_DeepVoice.md)|AM+Voc|ch→ ph→ling→(AR)→wav|ICML17|2017.02|
|[Deep Voice 2](../../Models/TTS0_System/2017.05.24_DeepVoice2.md)|AM+Voc|ch→ph→(FF)→ling→(AR)→wav|NIPS17|2017.05|
|DV2-Tacotron [50]|AM+Voc|ch→(AR)→linS→(AR)→wav|NIPS17|2017.05|
|Triple M [126]|AM+Voc|ch→(AR)→melS→(AR)→wav|IS21|2021.04|