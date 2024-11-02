# * Speech Recognition with Wav2Vec2.0
# * Authors: Moto Hira
# * Original Code: https://github.com/pytorch/audio/commits/main/examples/tutorials/speech_recognition_pipeline_tutorial.py
# * This tutorial shows how to perform speech recognition using pre-trained models from wav2vec2.0.
# * wav2vec2.0: https://arxiv.org/abs/2006.11477
# * Note: "../../../../Models/Speech_Representation/2020.06.20_Wav2Vec2.0.md"

# %% Overview (概览)

"""
# The process of Speech Recognition looks like the following:
# 1. Extract the acoustic features from audio waveform;
# 2. Estimate the class of the acoustic features frame-by-frame;
# 3. Generate hypothesis from the sequence of the class probabilities;

# TorchAudio provides easy access to the pre-trained weights and associated information, such as the expected sample rate and class labels.
# They are bundled together and available under `torchaudio.pipelines` module.

# * 语音识别的过程如下:
# * 1. 提取音频波形的声学特征;
# * 2. 逐帧估计声学特征的类;
# * 3. 根据类概率序列生成假设;
# * TorchAudio 提供了易于访问的预训练权重和相关信息，例如期望的采样率和类标签.
# * 它们被捆绑在一起，并可在 `torchaudio.pipelines` 模块中获得.
"""
# %% Preparation (准备)

import torch
import torchaudio

print(f"{torch.__version__=}")
print(f"{torchaudio.__version__=}")

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

# %% Create a Pipeline (创建流水线)

"""
# First, we will create a Wav2Vec2 model that performs the feature extraction and the classification.
# There are two types of Wav2Vec2 pre-trained weights available in torchaudio.
# The ones fine-tuned for ASR task, and the ones not fine-tuned.
# Wav2Vec2 (and HuBERT) models are trained in self-supervised manner.
# They are firstly trained with audio only for representation learning, then fine-tuned for a specific task with additional labels.
# The pre-trained weights without fine-tuning can be fine-tuned for other downstream tasks as well, but this tutorial does not cover that.
# We will use `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` here.
# There are multiple pre-trained models available in `torchaudio.pipelines`.
# Please check the documentation for the detail of how they are trained.
# The bundle object provides the interface to instantiate model and other information.
# Sampling rate and the class labels are found as follow.

# * 首先, 我们将创建一个 Wav2Vec2 模型, 用于执行特征提取和分类.
# * TorchAudio 中有两种类型的 Wav2Vec2 预训练权重可用.
# * 一种是用于 ASR 任务的微调权重, 另一种是没有微调的权重.
# * Wav2Vec2 (和 HuBERT) 模型是以自监督的方式训练的.
# * 它们首先用音频进行训练以进行表示学习, 然后用额外的标签进行特定任务的微调.
# * 预训练权重没有微调的版本也可以用于其他下游任务, 但本教程不涉及这些.

# * 这里我们将使用 `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.
# * `torchaudio.pipelines` 模块中有多个预训练模型可供选择.
# * 请参考文档了解它们的训练细节.
# * 捆绑对象提供了模型的实例化和其他信息的接口.
# * 采样率和类标签都可以从捆绑对象中获得.
"""

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print(f"{bundle.sample_rate=}")
print(f"{bundle.get_labels()=}")

# Model can be constructed as following.
# This process will automatically fetch the pre-trained weights and load it into the model.
# * 模型可以按照以下方式构造.
# * 这个过程将自动获取预训练权重并加载到模型中.
# * 注: 该模型大小为 360 MB.

model = bundle.get_model().to(device)
print(f"{model.__class__=}")

# %% Loading Data (加载数据)

"""
# We will use the speech data from VOiCES dataset, which is licensed under Creative Commos BY 4.0.
# VOiCES: https://iqtlabs.github.io/voices/

# To load data, we use `torchaudio.load()`.
# If the sampling rate is different from what the pipeline expects, then we can use `torchaudio.functional.resample()` for resampling.

# Note:
# `torchaudio.functional.resample()` works on CUDA tensors as well.
# When performing resampling multiple times on the same set of sample rates, using `torchaudio.transforms.Resample()` might improve the performance.

# * 我们将使用 VOiCES 数据集中的语音数据, 该数据集遵循 Creative Commons BY 4.0 许可.
# * VOiCES: https://iqtlabs.github.io/voices/

# * 加载数据使用 `torchaudio.load()`.
# * 如果采样率与流水线期望的不同, 则可以使用 `torchaudio.functional.resample()` 进行重采样.

# * 注意:
# * `torchaudio.functional.resample()` 也可以在 CUDA 张量上工作.
# * 在相同的采样率集合上进行多次重采样时, 使用 `torchaudio.transforms.Resample()` 可能会提高性能.
"""

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate!= bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# %% Extracting Acoustic Features (提取声学特征)

"""
# The next step is to extract acoustic features from the audio.

# Note:
# Wav2Vec2 models fine-tuned for ASR task can perform feature extraction and classification with one step,
# but for the sake of the tutorial, we also show how to perform feature extraction here.

# * 下一步是从音频中提取声学特征.

# * 注意:
# * 用于 ASR 任务的 Wav2Vec2 模型可以一步完成特征提取和分类, 但为了教程的完整性, 我们也展示了如何进行特征提取.
"""

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

"""
# The returned features is a list of tensors.
# Each tensor is the output of a transformer layer.
# * 返回的特征是一个张量列表.
# * 每个张量是一个 transformer 层的输出.
"""

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from Transformer layer {i+1}")
    ax[i].set_xlabel("Feature Dimension")
    ax[i].set_ylabel("Frame (Time-Axis)")
fig.tight_layout()

# %% Feature Classification (特征分类)

"""
# Once the acoustic features are extracted, the next step is to classify them into a set of categories.
# Wav2Vec2 model provides method to perform the feature extraction and classification in one step.
# * 声学特征提取和分类的下一步是将它们分类到一组类别中.
# * Wav2Vec2 模型提供了一种方法, 可以一步完成特征提取和分类.
"""

with torch.inference_mode():
    emission, _ = model(waveform)

"""
# The output is in the form of logits. It is not in the form of probability.
# Let’s visualize this.
# * 输出的形式是 logits. 它不是概率形式.
# * 让我们可视化一下.
"""

plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())

"""
# We can see that there are strong indications to certain labels across the time line.
# * 我们可以看到时间线上存在着某些标签的强烈指示.
"""

# %% Generating Transcripts (生成转录)

"""
# From the sequence of label probabilities, now we want to generate transcripts. The process to generate hypotheses is often called “decoding”.
# Decoding is more elaborate than simple classification because decoding at certain time step can be affected by surrounding observations.
#
# For example, take a word like night and knight.
# Even if their prior probability distribution are different (in typical conversations, night would occur way more often than knight),
# to accurately generate transcripts with knight, such as a knight with a sword,
# the decoding process has to postpone the final decision until it sees enough context.
#
# There are many decoding techniques proposed, and they require external resources, such as word dictionary and language models.
# In this tutorial, for the sake of simplicity, we will perform greedy decoding which does not depend on such external components,
# and simply pick up the best hypothesis at each time step.
# Therefore, the context information are not used, and only one transcript can be generated.
# We start by defining greedy decoding algorithm.

# * 从标签概率序列中, 现在我们想要生成转录. 生成假设的过程通常称为 "解码".
# * 解码比简单分类更复杂, 因为在某一时间步解码受到周围观察的影响.
# *
# * 例如, 考虑词汇 night 和 knight.
# * 即使它们的先验概率分布不同 (在典型的对话中, night 出现的频率要远远高于 knight),
# * 要准确地生成带有剑的 knight 等转录, 解码过程必须推迟最终决策, 直到它看到足够的上下文.
# *
# * 现在有许多解码技术, 它们通常需要外部组件, 例如词典和语言模型.
# * 在本教程中, 为了简单起见, 我们将执行贪婪解码, 它不依赖于这些外部组件, 而是简单地在每个时间步选择最佳假设.
# * 因此, 上下文信息不会被使用, 只能生成一个转录.
# * 我们首先从定义贪婪解码算法开始.
"""

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
        - emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
        - str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

"""
# Now create the decoder object and decode the transcript.
# * 现在创建解码器对象并解码转录.
"""

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

"""
# Let’s check the result and listen again to the audio.
# * 我们检查一下结果并再次听一下音频.
"""

print(transcript)
IPython.display.Audio(SPEECH_FILE)

"""
# The ASR model is fine-tuned using a loss function called Connectionist Temporal Classification (CTC).
# The detail of CTC loss is explained at https://distill.pub/2017/ctc/.
# In CTC a blank token (ϵ) is a special token which represents a repetition of the previous symbol.
# In decoding, these are simply ignored.
# * 解码器使用连接时序分类 (CTC) 损失函数进行微调.
# * CTC 损失函数的详细介绍请参考 https://distill.pub/2017/ctc/.
# * 在解码过程中, ϵ 符号 (空白符号) 是一种特殊符号, 表示上一个符号的重复.
# * 这些符号在解码过程中会被忽略.
"""

# %% Conclusion (结论)

"""
# In this tutorial, we looked at how to use Wav2Vec2ASRBundle to perform acoustic feature extraction and speech recognition.
# Constructing a model and getting the emission is as short as two lines.
# * 本教程介绍了如何使用 Wav2Vec2ASRBundle 执行声学特征提取和语音识别.
# * 构造模型和获取输出的过程只需两行代码.
"""

model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
model = model.to(device)
emission, _ = model(waveform)

# %% Finished (结束)