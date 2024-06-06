# Neural Text-to-Speech Synthesis

## 目录

|Parts|Sections|
|---|---|
||01.Introduction|
|**Part I** Preliminary <br> 第一部分 预备知识|02.Basics of Spoken Language Processing<br>03.Basics of Deep Learning|
|**Part II** Key Components in TTS <br> 第二部分 文本转语音的关键组成部分|[04.Text Analyses](Ch04.Text_Analyses.md)<br>[05.Acoustic Models](Ch05.Acoustic_Models.md)<br>[06.Vocoders](Ch06.Vocoders.md)|<br>[07.Fully End-to-End TTS](Ch07.Fully_End-to-End_TTS.md)|
|**Part III** Advanced Topics in TTS <br>第三部分 文本转语音的前沿主题|[08.Expressive and Controllable TTS](Ch08.Expressive&Controllable_TTS.md)<br>09.Robust TTS<br>10.Model-Efficient TTS<br>11.Data-Efficient TTS<br>12.Beyond Text-to-Speech Synthesis
|**Part IV** Summary and Outlook<br>第四部分 总结与展望|13.Summary and Outlook|

## A.语音合成相关资源

## B.语音合成模型列表

|模型|类型|数据流|发表|时间|
|:-:|:-:|:-:|---|:-:|
|WaveNet [45]|Voc|ling→(AR)→wav|arXiv16|2016.09|
|SampleRNN [46]|Voc|∅→(AR)→wav|ICLR17|2016.12|
|Deep voice [47]|AM+Voc|ch→ ph→ling→(AR)→wav|ICML17|2017.02|
|Char2Wav [48]|E2E|ch→(AR)→ceps→(AR)→wav|ICLR17 WS|2017.02|
|[Tacotron](../../Models/TTS2_Acoustic/2017.03.29_Tacotron.md) [49]|AM|ch/ph→(AR)→linS→wav|IS17|2017.03|
|Deep voice 2 [50]|AM+Voc|ch→ph→(FF)→ling→(AR)→wav|NIPS17|2017.05|
|DV2-Tacotron [50]|AM+Voc|ch→(AR)→linS→(AR)→wav|NIPS17|2017.05|
|VoiceLoop [51]|AM|ph→ceps→wav|ICLR18|2017.07|
|Deep voice 3 [52]|AM|ch/ph→(AR)→melS→(AR)→wav|ICLR18|2017.10|
|DCTTS [53]|AM|ch→(AR)→melS→wav|ICASSP18|2017.10|
|Par.WaveNet [54]|Voc|ling→(FF)→wav|ICML18|2017.11|
|[Tacotron 2](../../Models/TTS2_Acoustic/2017.12.16_Tacotron2.md) [55]|AM|ch/ph→(AR)→melS→(AR)→wav|ICASSP18|2017.12|
|WaveGAN [56]|Voc|.∅FF−→wav|ICLR19|2018.02|
|WaveRNN [57]|Voc|ling→(AR)→wav|ICML18|2018.02|
|DV3-Clone [58]|AM|ch/ph→(AR)→linS→wav|NeurIPS18|2018.02|
|GST-Tacotron [59]|AM|ph→(AR)→melS→wav|ICML18|2018.03|
|Ref-Tacotron [60]|AM|ph→(AR)→melS→wav|ICML18|2018.03|
|FFTNet [61]|Voc|ceps→(AR)→wav|ICASSP18|2018.04|
|VAE-Loop [62]|AM|ph→ceps→wav|IS18|2018.04|
|SV-Tacotron [63]|AM|ch/ph→(AR)→melS→(AR)→wav|NeurIPS18|2018.06|
|ClariNet [64]|E2E|ch/ph→(AR)→wav|ICLR19|2018.07|
|ForwardAtt [65]|AM|ph→(AR)→linS→wav|ICASSP18|2018.07|
|MCNN [66]|Voc|linS→(FF)→wav|SPL18|2018.08|
|[TransformerTTS](../../Models/TTS2_Acoustic/2018.09.19_Transformer_TTS.md) [67]|AM|ph→(AR)→melS→(AR)→wav|AAAI19|2018.09|
|SEA-TTS [68]|Voc|ling→(AR)→wav|ICLR19|2018.09|
|GMVAE-Tacotron [69]|AM|ph→(AR)→melS→(AR)→wav|ICLR19|2018.10|
|LPCNet [70]|Voc|ceps→(AR)→wav|ICASSP19|2018.10|
|WaveGlow [71]|Voc|melS→(FF)→wav|ICASSP19|2018.10|
|FloWaveNet [72]|Voc|melS→(FF)→wav|ICML19|2018.11|
|Univ. WaveRNN [73]|Voc|melS→(AR)→wav|IS19|2018.11|
|VAE-TTS [74]|AM|ph→(AR)→melS→(AR)→wav|ICASSP19|2018.12|
|TTS-Stylization [75]|AM|ch→(AR)→melS→wav|ICLR19|2018.12|
|AdVoc [76]|Voc|melS→(FF)→linS→wav|IS19|2019.04|
|GAN exposure [77]|AM|ph→(AR)→melS→(AR)→wav|IS19|2019.04|
|GELP [78]|Voc|melS→(FF)→wav|IS19|2019.04|
|Almost unsup [79]|AM|ph→(AR)→melS→wav|ICML19|2019.05|
|FastSpeech [80]|AM|ph→(FF)→melS→(FF)→wav|NeurIPS19|2019.05|
|ParaNet [81]|AM|ph→(FF)→melS→(FF)→wav|ICML20|2019.05|
|WaveVAE [81]|Voc|melS→(FF)→wav|ICML20|2019.05|
|MelNet [82]|AM|ch→(AR)→melS→wav|arXiv19|2019.06|
|StepwiseMA [83]|AM|ph→(AR)→melS→(AR)→wav|IS19|2019.06|
|GAN-TTS [84]|Voc|ling→(FF)→wav|ICLR20|2019.09|
|DurIAN [85]|AM|ph→(AR)→melS→(AR)→wav|IS20|2019.09|
|MB WaveRNN [85]|Voc|melS→(AR)→wav|IS20|2019.09|
|MelGAN [86]|Voc|melS→(FF)→wav|NeurIPS19|2019.10|
|Para. WaveGAN [87]|Voc|melS→(FF)→wav|ICASSP20|2019.10|
|DCA-Tacotron [88]|AM|ph→(AR)→melS→(AR)→wav|ICASSP20|2019.10|
|WaveFlow [89]|Voc|melS→(AR)→wav|ICML20|2019.12|
|SqueezeWave [90]|Voc|melS→(FF)→wav|arXiv20|2020.01|
|AlignTTS [91]|AM|ch/ph→(FF)→melS→(FF)→wav|ICASSP20|2020.03|
|RobuTrans [92]|AM|ph→(AR)→melS→(AR)→wav|AAAI20|2020.04|
|Flow-TTS [93]|AM|ch/ph→(FF)→melS→(FF)→wav|ICASSP20|2020.05|
|Flowtron [94]|AM|ph→(AR)→melS→(FF)→wav|ICLR21|2020.05|
|Glow-TTS [95]|AM|ph→(FF)→melS→(FF)→wav|NeurIPS20|2020.05|
|JDI-T [96]|AM|ph→(FF)→melS→(FF)→wav|IS20|2020.05|
|TalkNet [97]|AM|ch→(FF)→melS→(FF)→wav|IS21|2020.05|
|MB MelGAN [98]|Voc|melS→(FF)→wav|SLT21|2020.05|
|MultiSpeech [99]|AM|ph→(AR)→melS→(FF)→wav|IS20|2020.06|
|FastSpeech 2 [100]|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2020.06|
|FastSpeech 2s [100]|E2E|ph→(FF)→wav|ICLR21|2020.06|
|EATS [101]|E2E|ch/ph→(FF)→wav|ICLR21|2020.06|
|FastPitch [102]|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2020.06|
|VocGAN [103]|Voc|melS→(FF)→wav|IS20|2020.07|
|LRSpeech [104]|AM|ch→(AR)→melS→(FF)→wav|KDD20|2020.08|
|[SpeedySpeech](../../Models/TTS2_Acoustic/2020.08.09_SpeedySpeech.md) [105]|AM|ph→(FF)→melS→(FF)→wav|IS20|2020.08|
|GED [106]|Voc|ling→(FF)→wav|NeurIPS20|2020.08|
|SC-WaveRNN [107]|Voc|melS→(AR)→wav|IS20|2020.08|
|WaveGrad [108]|Voc|melS→(FF)→wav|ICLR21|2020.09|
|DiffWave [109]|Voc|melS→(FF)→wav|ICLR21|2020.09|
|[HiFi-GAN](../../Models/TTS3_Vocoder/2020.10.12_HiFi-GAN.md) [110]|Voc|melS→(FF)→wav|NeurIPS20|2020.10|
|NonAtt tacotron [111]|AM|ph→(AR)→melS→(AR)→wav|arXiv20|2020.10|
|Para. tacotron [112]|AM|ph→(FF)→melS→(AR)→wav|ICASSP21|2020.10|
|DeviceTTS [113]|AM|ph→(AR)→Ceps→wav|arXiv20|2020.10|
|Wave-Tacotron [114]|E2E|ch/ph→(AR)→wav|ICASSP21|2020.11|
|DenoiSpeech [115]|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2020.12|
|EfficientTTS [116]|AM|ch→(FF)→melS→(FF)→wav|ICML21|2020.12|
|EfficientTTS-Wav [116]|E2E|ch→(FF)→wav|ICML21|2020.12|
|Multi-SpectroGAN [117]|AM|ph→(AR)→melS→(FF)→wav|AAAI21|2020.12|
|LightSpeech [118]|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2021.02|
|Para. Tacotron 2 [119]|AM|ph→(FF)→melS→(AR)→wav|IS21|2021.03|
|AdaSpeech [120]|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2021.03|
|BVAE-TTS [121]|AM|ph→(FF)→melS→(FF)→wav|ICLR21|2021.03|
|PnG BERT [122]|AM|ph→(AR)→melS→(AR)→wav|IS21|2021.03|
|Fast DCTTS [123]|AM|ch→(AR)→melS→(FF)→wav|ICASSP21|2021.04|
|AdaSpeech 2 [124]|AM|ph→(FF)→melS→(FF)→wav|ICASSP21|2021.04|
|TalkNet 2 [125]|AM|ch→(FF)→melS→(FF)→wav|arXiv21|2021.04|
|Triple M [126]|AM+Voc|ch→(AR)→melS→(AR)→wav|IS21|2021.04|
|Diff-TTS [127]|AM|ph→(FF)→melS→(FF)→wav|IS21|2021.04|
|Grad-TTS [128]|AM|ph→(FF)→melS→(FF)→wav|ICML21|2021.05|
|Fre-GAN [129]|Voc|melS→(FF)→wav|IS21|2021.06|
|[VITS](../../Models/E2E/2021.06.11_VITS.md) [130]|E2E|ph→(FF)→wav|ICML21|2021.06|
|AdaSpeech 3 [131]|AM|ph→(FF)→melS→(FF)→wav|IS21|2021.06|
|PriorGrad-AM [132]|AM|ph→(FF)→melS→(FF)→wav|ICLR22|2021.06|
|PriorGrad-Voc[132]|Voc|melS→(FF)→wav|ICLR22|2021.06|
|Meta-StyleSpeech [133]|AM|ph→(FF)→melS→(FF)→wav|ICML21|2021.06|
|WaveGrad 2 [134]|E2E|ph→(FF)→wav|IS21|2021.06|
|InferGrad [135]|Voc|melS→(FF)→wav|ICASSP22|2022.02|
|SpecGrad [136]|Voc|melS→(FF)→wav|IS22|2022.03|
|[NaturalSpeech](../../Models/E2E/2022.05.09_NaturalSpeech.md) [137]|E2E|ph→(FF)→wav|arXiv22|2022.05|