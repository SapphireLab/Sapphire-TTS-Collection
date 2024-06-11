# 细水长 Flow 系列

## NICE · 流模型的基本概念与实现

时间: 2018-08-11
链接: [原文](https://spaces.ac.cn/archives/5776)

动机: 机器之心的报道 [下一个 GAN? OpenAI 提出可逆生成模型 Glow](https://www.jiqizhixin.com/articles/2018-07-10-6)

本文主要是 [NICE: Non-Linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) 的介绍和实现.
这篇文章也是 Glow 这个模型的基础文章之一, 可以说它就是 Glow 的奠基石.

总所周知, 当前主流的生成模型包括 VAE 和 GAN, 但事实上除了这两个之外, 还有基于 Flow 的模型.
事实上, Flow 的历史和 VAE, GAN 一样悠久, 但却鲜为人知.
在笔者看来, 大概原因是 Flow 找不到像 GAN 一样的诸如 "造假者-鉴别者" 的直观解释, 因为 Flow 整体偏数学化, 加上早期效果没有特别好但计算量又特别大, 所以很难让人提起兴趣.
不过现在看来, OpenAI 的这个好得让人惊叹的, 基于 Flow 的 Glow 模型, 估计会让更多的人投入到 Flow 模型的改进中.

生成模型的本质就是希望用一个我们知道的概率模型来拟合所给定的数据样本, 即得到一个带有参数 $\theta$ 的分布 $q_{\theta}(x)$.
然而神经网络只是万能函数逼近器, 而不是万能分布拟合器, 即原则上能够拟合任意函数, 但却不能随意拟合一个概率分布, 因为概率分布具有非负和归一化的要求.
这样一来, 能够直接写出来的只有离散型分布或连续型高斯分布.

当然从最严格的角度来看, 图像应该是一个离散的分布, 因为它是由有限个像素组成的, 而每个像素的取值也是离散有限的, 因此可以通过离散分布来描述.
这个思路的成果就是 PixelRNN 这一类模型, 称之为自回归.
其特点是无法并行, 所以计算量特别大.

所以我们更希望用连续分布来描述图像.
当然图像只是一个场景, 其他场景下也有很多连续型的数据, 所以连续型的分布的研究是很有必要的.

那么问题是对于连续型的, 也就只能写出高斯分布了, 而且很多时候为了方便处理, 只能写出各个分量独立的高斯分布, 这显然只是众多连续分布中极小的一部分, 显然是不够用的.
为了解决这个困境, 我们通过积分来创造更多的分布:

$$
    q(x) =\int q(z)q(x|z)\text{d}z. \tag{01}
$$

这里的 $q(z)$ 一般是标准高斯分布, 而 $q_{\theta}(x|z)$ 可以选择任意的条件高斯分布或狄拉克分布.
这样的积分形式可以形成很多复杂的分布.
理论上讲, 它能拟合任意分布.

有了分布形式, 需要求出参数 $\theta$, 一般使用最大似然.
假设真实数据分布为 $\tilde{p}(x)$, 那么需要最大化目标:

$$
    \mathbb{E}_{x\sim\tilde{p}(x)} [\log q(x)]\tag{02}
$$

然而 $q(x)$ 是积分形式的, 能不能计算出来很难说.

于是就出现了各种方法, VAE 和 GAN 在不同方向上避开了这个困难.
- VAE 没有直接优化目标 (02), 而是优化一个更强的上界, 这使得它只能是一个近似模型, 无法达到良好的生成效果.
- GAN 通过交替训练的方式绕开了这个困难, 保留了模型的精确性, 所以才能有较好的生成效果. 但也有自己的缺点.

### 基本思想

Flow 模型选择了一条硬路: 直接把积分算出来.

据来说, Flow 选择 $q(x|z)$ 为狄拉克分布 $\delta(x-g(z))$, 且 $g(z)$ 必须是可逆的, 即

$$
    x = g(z) \lrArr z = f(x)\tag{03}
$$

要从理论上实现可逆, 那么要求 $z$ 和 $x$ 的维度一致.
假设 $f$ $g$ 的形式已知, 那么通过方程 (01) 计算 $q(x)$ 相当于对 $q(z)$ 做一个积分变换 $z=f(x)$.

如标准正态分布
$$
    q(z) = \dfrac{1}{(2\pi)^{D/2}}\exp(-\dfrac{\|z\|^2}{2}),\tag{04}
$$

进行变换 $z=f(x)$.
此时概率密度函数的变量替换不是简单地将 $z$ 替换为 $f(x)$, 还多出了一个雅可比行列式的绝对值, 即

$$
    q(x) = \dfrac{1}{(2\pi)^{D/2}} \exp(-\dfrac{\|f(x)\|^2}{2})\left|\text{det}[\dfrac{\partial f}{\partial x}]\right|\tag{05}
$$

> #TODO 补充证明

这样对 $f$ 就有了两点要求:
1. 可逆, 并且易于求逆函数 ($g: z\to x$ 就是所需的生成模型);
2. 对应的雅可比行列式容易计算.

此时的优化目标为

$$
\begin{aligned}
    \log q(x) 
    &= \log \dfrac{1}{(2\pi)^{D/2}} \exp(-\dfrac{\|f(x)\|^2}{2})\left|\text{det}[\dfrac{\partial f}{\partial x}]\right|\\
    &= \log\dfrac{1}{(2\pi)^{D/2}} + \log \exp(-\dfrac{\|f(x)\|^2}{2}) + \log \left|\text{det}[\dfrac{\partial f}{\partial x}]\right|\\
    &= -\dfrac{D}{2}\log 2\pi - \dfrac{1}{2}\|f(x)\|^2 + \log \left|\text{det}[\dfrac{\partial f}{\partial x}]\right|
\end{aligned}\tag{06}
$$

这个优化目标是可以求解的.
并且由于 $f$ 容易求逆, 因此一旦训练完成, 就可以随机采样一个 $z$, 然后通过 $f$ 的逆函数 $g$ 来生成一个样本 $f^{-1}(z)=g(z)$, 这就得到了生成模型.

### 分块耦合

下面详细介绍 Flow 模型是如何针对难点来解决问题的.

相对而言, 行列式的计算要比函数求逆困难.
因此我们从要求 2 出发.

由于三角矩阵的行列式最容易计算, 因为三角矩阵的行列式等于对角元的乘积.
所以我们要办法使得变换 $f$ 的雅可比矩阵是三角矩阵.

NICE 的做法很精巧, 它将 $D$ 维度的 $x$ 分为两个部分 $x_1$ 和 $x_2$, 然后进行下述变换:

$$
    h_1 = x_1,\quad h_2 = x_2+m(x_1), \tag{07}
$$

其中 $x_1$ 和 $x_2$ 是 $x$ 的某种划分, $m$ 是 $x_1$ 的任意函数.
也就是将 $x$ 分为两部分, 然后按照上述公式进行变换, 得到新的变量 $h$, 这部分我们称为**加性耦合层 (Additive Coupling)**.

不失一般性, 可以将 $x$ 各个维度进行重排, 使得 $x_1=x_{1:d}$ 为前 $d$ 个元素, $x_2=x_{d+1:D}$ 为 $d+1\sim D$ 个元素.

不难看出, 这个变换的雅可比矩阵 $[\dfrac{\partial h}{\partial x}]$ 是一个三角矩阵, 而且对角线全部为 1, 用分块矩阵表示为

$$
    [\dfrac{\partial h}{\partial x}] = \begin{bmatrix}
        I_{1:d} & O \\
        \dfrac{\partial m}{\partial x_1} & I_{d+1:D}
    \end{bmatrix}\tag{08}
$$

这样得到的变换 $f$ 对应的雅可比行列式为 1, 其对数为 0, 就解决了要求 2.

同时公式 (07) 也是可逆的, 逆变换为

$$
    x_1 = h_1, x_2 = h_2-m(h_1), \tag{09}
$$

上面的变换让人十分惊喜: 可逆, 且逆变换也很简单, 并没有增加额外的计算量.
但是可以留意到公式 (07) 定义的变换的第一部分是平凡的 (恒等变换), 因此单个变换并不能获得很强的非线性, 所以需要多个简单变换的复合, 以达到强非线性, 增强拟合能力.

$$
    x=h^{(0)}\lrarr h^{(1)}\lrarr h^{(2)}\lrarr \cdots \lrarr h^{(n-1)}\lrarr h^{(n)}=z\tag{10}
$$

其中每个变换都是加性耦合层.
这就好比流水一般, 积少成多, 细水长流, 所以这样的一个流程称为一个流 (Flow).

根据链式法则:

$$
    [\dfrac{\partial z}{\partial x}] = [\dfrac{\partial h^{(n)}}{\partial h^{(0)}}]=\cdots=[\dfrac{\partial h^{(n)}}{\partial h^{(n-1)}}][\dfrac{\partial h^{(n-1)}}{\partial h^{(n-2)}}]\cdots[\dfrac{\partial h^{(1)}}{\partial h^{(0)}}].\tag{11}
$$

因为矩阵乘积的行列式等于矩阵行列式的乘积, 而每一层都是加性耦合层, 所以结果

$$
    \text{det}[\dfrac{\partial z}{\partial x}] = 1
$$

注: 考虑到后续介绍的交错, 行列式可能为 -1, 但绝对值依然为 1.

### 交错前进

可以发现如果耦合的顺序一直不变, 即

$$
\begin{aligned}
    &x_1 \to h_1^{(1)} \to h_1^{(2)} \to \cdots \to h_1^{(n-1)} \to z_1;\\
    &x_2+m_1(x_1) \to h_2^{(1)} + m_2(h_1^{(1)})\to \cdots \to h_2^{(n-1)} + m_n(h_1^{(n-1)}) \to z_2.
\end{aligned}\tag{12}
$$

那么最后 $z_1=x_1$, 第一部分依然是平凡的.

![](Images/2018.08.11.Fig.01.png)

简单的耦合使得其中一部分仍然保持恒等, 信息没有充分混合.

那么为了得到不平凡的变换, 可以考虑在每次进行加性耦合前, 打乱或者反转输入的各个维度的顺序, 或者简单地直接交换这两部分的位置, 使得信息可以充分混合, 以达到更强的非线性.

![](Images/2018.08.11.Fig.02.png)

### 尺度变换

Flow 是基于可逆变换的, 所以当模型训练完成后, 同时得到一个生成模型和编码模型.
但正因为是可逆变换, 随机变量 $z$ 和输入样本 $x$ 具有相同的维度 $D$.
当指定 $z$ 服从高斯分布时, 它是遍布整个 $D$ 维空间的, 而虽然 $x$ 具有 $D$ 维大小, 但却未必是遍布整个 $D$ 维空间.
例如 MNIST 图像虽然有 784 个像素, 但有些像素不管在训练集还是在测试集都一直保持为 0, 这说明它远远没有 784 维那么大.

所以, Flow 这种基于可逆变换的模型, 天生就存在比较严重的维度浪费问题: 输入数据不是 D 维流形却编码为一个 D 维流形.

为了解决这个情况, NICE 引入了尺度变换层, 它对最后编码出来的每个维度的特征都做了尺度变换, 即 $z=s\otimes h^{(n)}$, 其中 $s=(s_1,\cdots s_D)$ 是一个需要优化的非负的参数向量, 这个向量能识别该维度的重要程度, 值越小表明越重要, 从而起到压缩流形的作用.

注意这个尺度变换层的雅可比行列式不再是 1 了, 可以计算出它的雅可比矩阵为对角阵:
$$
    [\dfrac{\partial z}{\partial x}] = \text{diag}(s)\tag{14}
$$

所以行列式为 $\prod_{i} s_i$.

根据公式 (06), 有对数似然:

$$
\begin{aligned}
    \log q(x) 
    &\sim - \dfrac{1}{2}\|s\otimes f(x)\|^2 + \log \left|\text{diag}(s)\right|\\
    &= -\dfrac{1}{2}\|s\otimes f(x)\|^2 + \sum_{i} \log s_i\\
\end{aligned}\tag{15}
$$

为什么这个尺度变换能够识别维度的重要程度呢?
其实这个尺度变换可以用一种更加清晰的方式描述:

开始的时候设 $z$ 的先验分布为标准正态分布, 即各个方差都为 1.
事实上, 可以将先验分布的方差作为训练参数, 训练后得到的方差大小不一.
而方差越小说明该维度的弥散程度越低, 如果方差为 0, 则该维度恒为均值 0, 该维度的分布也就坍缩为一个点, 从而流形少了一个维度.

不同于方程 (04), 写出带有方差的正态分布:

$$
    q(z) = \dfrac{1}{(2\pi)^{D/2}\prod_{i=1}^D \sigma_i} \exp(-\dfrac{1}{2}\sum_{i=1}^{D}\dfrac{z_i^2}{\sigma_i^2})\tag{16}
$$

将流形 $z=f(x)$ 代入, 然后取对数, 得到

$$
    \log q(x) \sim - \dfrac{1}{2}\sum_{i=1}^{D}\dfrac{f_i^2(x)}{\sigma_i^2} - \sum_{i=1}^{D}\log \sigma_i,\tag{17}
$$

对比一下方程 (15) 和方程 (17), 就有 $s_i=1/\sigma_i$.

所以尺度变换层等价于将先验分布的方差作为训练参数, 如果方差足够小就可以认为该维度所表示的流形坍缩为一个点, 从而总体流形的维度减一, 暗含了降维的可能.

### 特征解耦

将先验分布定义为各个分量相互独立的高斯分布, 除了采样方便还有什么好处呢?

在 Flow 模型中, $f^{-1}=g$ 是生成模型用于随机生成样本, 那么 $f$ 本身就是编码器.
但是不同于普通神经网络中的自编码器强迫低维重建高维来提取有效信息的做法, Flow 模型是完全可逆的, 那么就不存在信息损失的问题, 那么这个编码器还有什么价值呢.

这就涉及到"什么是好的特征"的问题了. 一个好的特征, 理想情况下各个维度之间应该是相互独立的, 实现了特征的解耦, 使得每个维度都有自己独立的含义.

由于各个分量的独立性, 有理由说当我们用 $f$ 对原始特征进行编码时, 输出的编码特征 $z$ 的各个维度是解耦的.
NICE 的全称 Non-Linear Independent Components Estimation, 非线性独立成分估计, 就是这个含义.

反过来, 由于 $z$ 每个维度的独立性, 理论上控制改变单个维度时, 就可以看出生成图像是如何随着该维度的改变而改变, 从而发现该维度的含义.

类似地, 也可以对两幅图像的编码进行插值, 得到过渡自然的生成样本, 这些在后面发展起来的 Glow 模型中体现得很充分.


### 实验复现

Keras + MNIST 数据集复现.

总结一下 NICE 模型, 它是 Flow 模型的一种, 由多个加性耦合层组成, 每个加性耦合层在耦合前需要反转输入的维度, 使得信息充分混合, 最后一层增加一个尺度缩放曾, 最后的损失函数是公式 (15) 的相反数.

加性耦合层需要将输入分为两部分, NICE 采用交错分区, 即下标为偶数的为第一部分, 下标为奇数的为第二部分, 而每个 $m(x)$ 则简单地使用多层全连接 (5 隐藏层×1000 神经元 + ReLU 激活).
NICE 中一共耦合了四个加性耦合层.

输入数据将 0~255 的图像像素压缩为 0~1, 然后给输入加上服从 `[-0.01,0]` 上均匀分布的噪声.
噪声的加入能有效地防止过拟合, 提高生成的图片质量.
也可以看成缓解维度浪费问题的一个措施, 因为 MINST 图像没办法充满 784 维, 但算上噪声维度就增加了.

虽然从损失函数看选择各种噪声的效果应该都差不多, 但选择这个噪声区间而不是 `[0,0.01]` 或 `[-0.05,0.05]` 的原因是加入噪声后理论上生成的图片也会含有噪声, 这不是我们希望的, 而加入负噪声, 会让最终生成的图片的像素稍微偏向负区间, 然后裁剪就能去掉一部分噪声, 这是针对 MNIST 的一个小技巧.

代码链接为: https://github.com/bojone/flow/blob/master/nice.py

<details>

<summary>完整代码</summary>

```python
#! -*- coding: utf-8 -*-
# Keras implement of NICE (Non-linear Independent Components Estimation)
# https://arxiv.org/abs/1410.8516

from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import imageio

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

class Shuffle(Layer):
    """打乱层，提供两种方式打乱输入维度
    一种是直接反转，一种是随机打乱，默认是直接反转维度
    """
    def __init__(self, idxs=None, mode='reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs
        self.mode = mode
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)
        inputs = K.transpose(inputs)
        outputs = K.gather(inputs, self.idxs)
        outputs = K.transpose(outputs)
        return outputs
    def inverse(self):
        v_dim = len(self.idxs)
        _ = sorted(zip(range(v_dim), self.idxs), key=lambda s: s[1])
        reverse_idxs = [i[0] for i in _]
        return Shuffle(reverse_idxs)


class SplitVector(Layer):
    """将输入分区为两部分，交错分区
    """
    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        inputs = K.reshape(inputs, (-1, v_dim//2, 2))
        return [inputs[:,:,0], inputs[:,:,1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
    def inverse(self):
        layer = ConcatVector()
        return layer


class ConcatVector(Layer):
    """将分区的两部分重新合并
    """
    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        inputs = [K.expand_dims(i, 2) for i in inputs]
        inputs = K.concatenate(inputs, 2)
        return K.reshape(inputs, (-1, np.prod(K.int_shape(inputs)[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer


class AddCouple(Layer):
    """加性耦合层
    """
    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)
    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1] # 逆为加
        else:
            return [part1, part2 - mpart1] # 正为减
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        layer = AddCouple(True)
        return layer


class Scale(Layer):
    """尺度变换层
    """
    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[1]),
                                      initializer='glorot_normal',
                                      trainable=True)
    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel)) # 对数行列式
        return K.exp(self.kernel) * inputs
    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)


def build_basic_model(v_dim):
    """基础模型，即加性耦合层中的m
    """
    _in = Input(shape=(v_dim,))
    _ = _in
    for i in range(5):
        _ = Dense(1000, activation='relu')(_)
    _ = Dense(v_dim, activation='relu')(_)
    return Model(_in, _)


shuffle1 = Shuffle()
shuffle2 = Shuffle()
shuffle3 = Shuffle()
shuffle4 = Shuffle()

split = SplitVector()
couple = AddCouple()
concat = ConcatVector()
scale = Scale()

basic_model_1 = build_basic_model(original_dim//2)
basic_model_2 = build_basic_model(original_dim//2)
basic_model_3 = build_basic_model(original_dim//2)
basic_model_4 = build_basic_model(original_dim//2)


x_in = Input(shape=(original_dim,))
x = x_in

# 给输入加入负噪声
x = Lambda(lambda s: K.in_train_phase(s-0.01*K.random_uniform(K.shape(s)), s))(x)

x = shuffle1(x)
x1,x2 = split(x)
mx1 = basic_model_1(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle2(x)
x1,x2 = split(x)
mx1 = basic_model_2(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle3(x)
x1,x2 = split(x)
mx1 = basic_model_3(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle4(x)
x1,x2 = split(x)
mx1 = basic_model_4(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])
x = scale(x)

encoder = Model(x_in, x)
encoder.summary()
encoder.compile(loss=lambda y_true,y_pred: K.sum(0.5 * y_pred**2, 1),
                optimizer='adam')

checkpoint = ModelCheckpoint(filepath='./best_encoder.weights',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

encoder.fit(x_train,
            x_train,
            batch_size=128,
            epochs=30,
            validation_data=(x_test, x_test),
            callbacks=[checkpoint])

encoder.load_weights('./best_encoder.weights')

# 搭建逆模型（生成模型），将所有操作倒过来执行

x = x_in
x = scale.inverse()(x)

x1,x2 = concat.inverse()(x)
mx1 = basic_model_4(x1)
x1, x2 = couple.inverse()([x1, x2, mx1])
x = split.inverse()([x1, x2])
x = shuffle4.inverse()(x)

x1,x2 = concat.inverse()(x)
mx1 = basic_model_3(x1)
x1, x2 = couple.inverse()([x1, x2, mx1])
x = split.inverse()([x1, x2])
x = shuffle3.inverse()(x)

x1,x2 = concat.inverse()(x)
mx1 = basic_model_2(x1)
x1, x2 = couple.inverse()([x1, x2, mx1])
x = split.inverse()([x1, x2])
x = shuffle2.inverse()(x)

x1,x2 = concat.inverse()(x)
mx1 = basic_model_1(x1)
x1, x2 = couple.inverse()([x1, x2, mx1])
x = split.inverse()([x1, x2])
x = shuffle1.inverse()(x)

decoder = Model(x_in, x)

# 采样查看生成效果
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = np.array(np.random.randn(1, original_dim)) * 0.75 # 标准差取0.75而不是1
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


figure = np.clip(figure*255, 0, 255)
imageio.imwrite('test.png', figure)
```

</details>
<br>

实验中 20 个 Epoch 内可以跑到最优, 11 秒一个 Epoch (GTX 1070), 最终的损失约为 -2200.

相比于原论文的实现, 笔者做了一些改动:
加性耦合层采用逆变换即公式 (09) 为前向, 正变换为逆向, 这是因为 $m(x)$ 采用 ReLU 激活, 导致输出非负. 因为正向是编码器, 逆向是生成器, 那么选择公式 (07) 为逆向, 生成模型会更倾向于生成正数, 和所需的图像是 0~1 取值相吻合.

虽然我们最终希望从标准正态分布中采样随机数来生成样本, 但实际上对于训练好的模型, 理想的采样方差并不一定是 1, 而是在 1 的上下波动, 一般比 1 稍小.
最终采样的正态分布的标准差称为退火参数.
在实验中选择 0.75 为退火参数, 目测此时效果最佳.

NICE 模型还是比较庞大的, 按照上述架构, 模型参数两为 $4\times5\times 1000^2=2\times 10^7$, 两千万训练一个 MNIST 生成模型.

NICE 模型的整体还是比较简单粗暴的, 首先加性耦合本身比较简单, 其次模型 $m$ 只简单地用到了全连接层, 没结合其他模块.

RealNVP 和 Glow 就是它的两个改进版本.


