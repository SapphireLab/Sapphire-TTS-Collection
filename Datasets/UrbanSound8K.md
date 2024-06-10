# UrbanSound8K

UrbanSound8K 是目前应用较为广泛的, 用于自动城市环境声分类研究的公共数据集, 涵盖十个不同城市环境的 8,732 个音频.
数据主要来自城市区域, 道路, 高速公路, 公园, 居民区, 轻轨站, 地铁站, 公共汽车, 火车.

数据集中的每个音频文件都被分类为十个不同的可能性之一, 这些类别包括:
- 空调
- 汽车喇叭
- 儿童玩耍
- 狗叫声
- 钻头声
- 发动机声
- 枪声
- 敲击声
- 街道音乐
- 城市噪声

每个音频片段都持续了 4 秒钟, 采用 44.1 kHz 的采样率进行立体声录音, 从而确保音频的高品质.
这些音频片段已经经过了精细的过滤和精确的处理, 以确保音频的强度准确无误.
同时, 这个处理过程考虑了不同录制设备和各种噪声环境对城市环境测量的影响.

该数据集的目的是为声音分类和识别算法的开发和测试提供了一个具有挑战性和多样性的环境, 并且可以应用于安全监控, 城市规划和交通管理等领域.

注意, 虽然 UrbanSound8K 数据集对音频的分类是十种, 但是与语音命令数据集 SpeechCommands 的唤醒词分类不同, 其数据的具体分类夹杂在单个文件夹中, 即 `fold1` 文件夹中有 10 种不同地数据分类.

具体每个音频对应的类别, 可以通过 `metadata` 文件夹下的 `UrbanSound8K.csv` 文件获取.

第一列是音频名称
最后一列为音频分类
其他的还有所属文件夹和类的 ID.

下载文件:

```python
import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()
```

读取方法:

```python
import csv

datafolder = "../UrbanSound8K/audio/"
UrbanSound8K_name_list = "../UrbanSound8K/metadata/UrbanSound8K.csv"

class_list = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

class_wav_dict = {}
with open(UrbanSound8K_name_list, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        class_wav_dict[row[0]] = (row[-1])
```