# F5-TTS 评估教程

对应文件: `src/f5_tts/eval/README.md`

## 环境准备

安装评估依赖的包:
```bash
pip install -e .[eval]
```
注: 目前包含
- [x] [faster_whisper](https://github.com/SYSTRAN/faster-whisper): 语音识别
- [x] [funasr](https://github.com/modelscope/FunASR): 语音识别
- [x] [jiwer](https://github.com/jitsi/jiwer): 自动评估语音识别系统, 含词错误率, 匹配错误率, 词信息丢失, 词信息保留, 字符错误率等指标 (最小编辑距离使用 RapidFuzz, 底层基于 C++)
- [x] [modelscope](https://github.com/modelscope/modelscope): 模型库
- [x] [zhconv](https://github.com/gumblex/zhconv): 基于 MediaWiki 词汇表的最大正向匹配简繁转换
- [x] [zhon](https://github.com/tsroten/zhon): 字符串中查找 CJK, 拼音字节, 词语, 句子等

## 生成样本

### 准备测试集

1. *Seed-TTS testset*: 从 [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) 下载.
2. *LibriSpeech test-clean*: 从 [OpenSLR](http://www.openslr.org/12/) 下载.
3. 解压下载的压缩包然后放置到 `data/` 文件夹.
4. 更新 `src/f5_tts/eval/eval_infer_batch.py` 里的 *LibriSpeech test-clean* 路径.
5. 我们过滤好的 LibriSpeech-PC 4 到 10 秒子集: `data/librispeech_pc_test_clean_cross_sentence.lst`

### 批量推理

执行如下命令来批量推理:
```bash
# batch inference for evaluations
accelerate config  # if not set before
bash src/f5_tts/eval/eval_infer_batch.sh
```

---
补充信息:
`accelerate config` 命令用来配置环境;
`eval_infer_batch.sh` 脚本用来批量推理;

```bash
# e.g. F5-TTS, 16 NFE
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_zh" -nfe 16
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "seedtts_test_en" -nfe 16
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Base" -t "ls_pc_test_clean" -nfe 16

# e.g. Vanilla E2 TTS, 32 NFE
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_zh" -o "midpoint" -ss 0
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "seedtts_test_en" -o "midpoint" -ss 0
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "E2TTS_Base" -t "ls_pc_test_clean" -o "midpoint" -ss 0
```

调用 `eval_infer_batch.py` 脚本: -s 表示 seed, -n 表示 expname, -t 表示 testset, -o 表示 ODE 方法, -ss 表示是否启用 SwaySampling.

---

## 客观评估

### 下载评估模型的权重

1. 中文识别模型: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. 英文识别模型: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM 模型: [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

然后更新对应脚本中的模型检查点路径.

更新你的批量推理结果的路径, 然后进行 WER / SIM 评估.

```bash
# Evaluation for Seed-TTS test set
python src/f5_tts/eval/eval_seedtts_testset.py

# Evaluation for LibriSpeech-PC test-clean (cross-sentence)
python src/f5_tts/eval/eval_librispeech_test_clean.py
```