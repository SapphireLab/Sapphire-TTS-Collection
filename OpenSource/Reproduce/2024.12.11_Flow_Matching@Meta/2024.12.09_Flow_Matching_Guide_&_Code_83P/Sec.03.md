# 3·Flow Models: 流模型

<details>
<summary>展开原文</summary>

This section introduces ***flows***, the mathematical object powering the simplest forms of Flow Matching.
Later parts in the manuscript will discuss Markov processes more general than flows, leading to more sophisticated generative learning paradigms introducing many more design choices to the Flow Matching framework.
The reason we start with flows is three-fold: First, flows are arguably the simplest of all CTMPs --- being deterministic and having a compact parametrization via velocities---these models can transform any source distribution $p$ into any target distribution $q$, as long as these two have densities.
Second, flows can be sampled rather efficiently by approximating the solution of ODEs, compared, e.g., to the harder-to-simulate SDEs for diffusion processes.
Third, the deterministic nature of flows allows an unbiased model likelihood estimation, while more general stochastic processes require working with lower bounds.
To understand flows, we must first review some background notions in probability and differential equations theory, which we do next.

</details>
<br>

本节介绍***流 (Flows)***, 是流匹配中最简单的数学对象.
后续部分将讨论比流更一般的马尔可夫过程, 引出更复杂的生成学习范式, 为流匹配框架引入许多更多的设计选择.

我们从流 (Flows) 开始的原因有三点:
1. 流 (Flows) 可以说是所有连续时间马尔可夫过程 (CTMPs) 中最简单的一种: 它是确定性的, 且通过速度具有紧凑的参数化形式. 这些模型可以将任意源分布 $p$ 转换到任意目标分布 $q$, 只要这两个分布有密度.
2. 相比扩散过程中难以模拟的随机微分方程相比, 流 (Flows) 可以通过近似常微分方程的解来高效采样.
3. 流 (Flows) 的确定性特性可以获得无偏的模型似然估计, 而更一般的随机过程则需要使用下界.

为了理解流 (Flows), 我们首先需要回顾概率和微分方程理论中的一些背景概念.

## 3.1·Random Vectors: 随机向量

<details>
<summary>展开原文</summary>

Consider data in the $d$-dimensional Euclidean space $x=(x^1,\ldots,x^d)\in \mathbb{R}^d$ with the standard Euclidean inner product $\langle x,y\rangle=\sum_{i=1}^d x^i y^i$ and norm $\|x\|=\sqrt{\langle x,x\rangle}$.
We will consider random variables (RVs) $X\in\R^d$ with continuous probability density function (PDF), defined as a continuous function $p_X:\mathbb{R}^d\to \mathbb{R}_{\geq 0}$ providing event $A$ with probability

$$
\mathbb{P}(X\in A) = \int_A p_X(x) \text{d} x,
$$

where $\int p_X(x)\text{d} x = 1$.

By convention, we omit the integration interval when integrating over the whole space ($\int \equiv \int_{\mathbb{R}^d}$).
To keep notation concise, we will refer to the PDF $p_{X_t}$ of RV $X_t$ as simply $p_t$.
We will use the notation $X \sim p$ or $X \sim p(X)$ to indicate that $X$ is distributed according to $p$.

One common PDF in generative modeling is the $d$-dimensional isotropic Gaussian:

$$
\mathcal{N}(x|\mu,\sigma^2 I) = (2\pi\sigma^2)^{-\frac{d}{2}}\exp\left(-\frac{\|{x-\mu}_2\|^2}{2\sigma^2}\right),
$$

where $\mu\in \mathbb{R}^d$ and $\sigma \in \mathbb{R}_{>0}$ stand for the mean and the standard deviation of the distribution, respectively.

The expectation of a RV is the constant vector closest to $X$ in the least-squares sense:

$$
\mathbb{E}[X]=\argmin_{z\in\mathbb{R}^d} \int \|x-z\|^2 p_X(x)\text{d} x = \int x p_X(x)\text{d} x.
$$

One useful tool to compute the expectation of **functions of RVs** is the **Law of the Unconscious Statistician**:

$$
\mathbb{E} [f(X)] = \int f(x) p_X(x) \text{d} x.
$$

When necessary, we will indicate the random variables under expectation as $\mathbb{E}_{X} f(X)$.

</details>
<br>

考虑 $d$ 维欧氏空间中的数据 $x=(x^1,\ldots,x^d)\in \mathbb{R}^d$ 及标准欧氏内积 $\langle x,y\rangle=\sum_{i=1}^d x^i y^i$ 和范数 $\|x\|=\sqrt{\langle x,x\rangle}$.

我们将考虑**随机变量 (Random Variables, RVs)** $X\in\R^d$ 及其连续**概率密度函数 (Probability Density Function, PDF)**, 定义为一个连续函数 $p_X:\mathbb{R}^d\to \mathbb{R}_{\geq 0}$, 它为事件 $A$ 提供概率:

$$
\mathbb{P}(X\in A) = \int_A p_X(x) \text{d} x,
$$

其中积分和为 1: $\int p_X(x)\text{d} x = 1$.

按照惯例, 当在整个空间上积分时省略积分区间 ($\int \equiv \int_{\mathbb{R}^d}$).

为了保持符号简洁, 我们将随机变量 $X_t$ 的概率密度函数 $p_{X_t}$ 简写为 $p_t$.
我们将使用 $X \sim p$ 或 $X \sim p(X)$ 来表示 $X$ 服从分布 $p$.

---

生成式建模中常用的概率密度函数之一是 $d$ 维各向同性高斯分布:

$$
\mathcal{N}(x|\mu,\sigma^2 I) = (2\pi\sigma^2)^{-\frac{d}{2}}\exp\left(-\frac{\|{x-\mu}_2\|^2}{2\sigma^2}\right),
$$

其中 $\mu\in \mathbb{R}^d$ 和 $\sigma \in \mathbb{R}_{>0}$ 分别表示分布的均值和标准差.

---

随机变量的期望是在最小二乘意义下与 $X$ 最接近的常数向量:

$$
\mathbb{E}[X]=\argmin_{z\in\mathbb{R}^d} \int \|x-z\|^2 p_X(x)\text{d} x = \int x p_X(x)\text{d} x.
$$

---

随机变量函数的期望可以用**无意识统计学家定律 (Law of the Unconscious Statistician)** 进行计算:

$$
\mathbb{E} [f(X)] = \int f(x) p_X(x) \text{d} x.
$$

必要时, 我们将使用 $\mathbb{E}_{X} f(X)$ 来表示期望下的随机变量.

## 3.2·Conditional Densities and Expectations: 条件密度和期望

Given two random variables $X,Y\in \mathbb{R}^d$, their joint PDF $p_{X,Y}(x,y)$ has marginals

$$
    \int p_{X,Y}(x,y)\text{d} y = p_X(x) \text{ and } \int p_{X,Y}(x,y)\text{d} x = p_Y(y).
$$

See Figure.04 for an illustration of the joint PDF of two RVs in $\mathbb{R}$ ($d=1$).

The conditional PDF $p_{X|Y}$ describes the PDF of the random variable $X$ when conditioned on an event $Y=y$ with density $p_Y(y)>0$:

$$
    p_{X|Y}(x|y):=\frac{p_{X,Y}(x,y)}{p_Y(y)},
$$

and similarly for the conditional PDF $p_{Y|X}$.
Bayes' rule expresses the conditional PDF $p_{Y|X}$ with $p_{X|Y}$ by

$$
    p_{Y|X}(y|x) = \frac{p_{X|Y}(x|y)p_Y(y)}{p_X(x)},
$$

for $p_X(x)>0$.

The **conditional expectation** $\mathbb{E}[X | Y]$ is the best approximating **function** $g_\star(Y)$ to $X$ in the least-squares sense:

$$
\begin{aligned}
    g_\star &:= \argmin_{g:\mathbb{R}^d\to \mathbb{R}^d}\mathbb{E}[\|X-g(Y)\|^2]\\
    &= \argmin_{g:\mathbb{R}^d\to \mathbb{R}^d}\int \|x-g(y)\|^2 p_{X,Y}(x,y)\text{d} x \text{d} y \\
    &= \argmin_{g:\mathbb{R}^d\to \mathbb{R}^d} \int \left [\int \|x-g(y)\|^2p_{X|Y}(x|y)\text{d} x\right ] p_Y(y)\text{d} y.
\end{aligned}
$$

For $y\in \mathbb{R}^d$ such that $p_Y(y)>0$ the conditional expectation function is therefore

$$
    \mathbb{E}[X|Y=y] := g_\star(y) = \int x p_{X|Y}(x|y) \text{d} x,
$$

where the second equality follows from taking the minimizer of the inner brackets in \cref{e:cond_E_g_star} for $Y=y$, similarly to \cref{e:E}.

Composing $g_\star$ with the random variable $Y$, we get

$$
    \mathbb{E}[X|Y] := g_\star(Y),
$$

which is a random variable in $\mathbb{R}^d$.

Rather confusingly, both $\mathbb{E}[X|Y=y]$ and $\mathbb{E}[X|Y]$ are often called **conditional expectation**, but these are different objects.
In particular, $\mathbb{E}[X|Y=y]$ is a function $\mathbb{R}^d\to \mathbb{R}^d$, while $\mathbb{E}[X|Y]$ is a random variable assuming values in $\mathbb{R}^d$.
To disambiguate these two terms, our discussions will employ the notations introduced here.

The **tower property** is an useful property that helps simplify derivations involving conditional expectations of two RVs $X$ and $Y$:

$$
    \mathbb{E}[\mathbb{E}[X|Y]] = \mathbb{E}[X]
$$

Because $\mathbb{E}[X|Y]$ is a RV, itself a function of the RV $Y$, the outer expectation computes the expectation of $\mathbb{E}[X|Y]$.
The tower property can be verified by using some of the definitions above:

$$
\begin{aligned}
    \mathbb{E}[\mathbb{E}[X|Y]]
    &= \int \left (\int x p_{X|Y}(x|y) \text{d} x\right ) p_Y(y) \text{d} y \\
    &= \int \int x p_{X,Y}(x,y) \text{d} x\text{d} y \\
    &= \int x p_X(x)\text{d} x \\
    &= \mathbb{E} [X].
\end{aligned}
$$

Finally, consider a helpful property involving two RVs $f(X, Y)$ and $Y$, where $X$ and $Y$ are two arbitrary RVs.
Then, by using the Law of the Unconscious Statistician with \eqref{e:cond_E_func}, we obtain the identity

$$
    \mathbb{E}[f(X,Y)|Y=y] = \int f(x,y) p_{X|Y}(x|y) \text{d} x.
$$

## 3.3·Diffeomorphisms and Push-Forward Maps: 微分同胚和推前映射

We denote by $C^r(\mathbb{R}^m,\mathbb{R}^n)$ the collection of functions $f:\mathbb{R}^m\to \mathbb{R}^n$ with continuous partial derivatives of order $r$:

$$
    \frac{\partial^r f^k}{\partial x^{i_1}\cdots \partial x^{i_r}}, \qquad k\in [n], i_j\in [m],
$$

where $[n]:=\set{1,2,\dots,n}$.

To keep notation concise, define also $C^r(\mathbb{R}^n):= C^r(\mathbb{R}^m,\mathbb{R})$ so, for example, $C^1(\mathbb{R}^m)$ denotes the continuously differentiable scalar functions.
An important class of functions are the \highlight{$C^r$ diffeomorphism}; these are invertible functions $\psi\in C^r(\R^n,\R^n)$ with $\psi^{-1}\in C^r(\R^n,\R^n)$.

Then, given a RV $X\sim p_X$ with density $p_X$, let us consider a RV $Y=\psi(X)$, where $\psi:\mathbb{R}^d\to \mathbb{R}^d$ is a $C^1$ diffeomorphism.
The PDF of $Y$, denoted $p_Y$, is also called the **push-forward** of $p_X$.
Then, the PDF $p_Y$ can be computed via a change of variables:

$$
\begin{aligned}
\mathbb{E}[f(Y)]
&= \mathbb{E}[f(\psi(X))]\\
&= \int f(\psi(x)) p_X(x) \text{d} x \\
&= \int f(y) p_X(\psi^{-1}(y)) |\det \partial_y \psi^{-1} (y)| \text{d} y,
\end{aligned}
$$

where the third equality is due the change of variables $x=\psi^{-1}(y)$, $\partial_y \phi(y)$ denotes the Jacobian matrix (of first order partial derivatives), \ie,

$$
    [\partial_y \phi(y)]_{i,j} = \frac{\partial \phi^i}{\partial x^j}, \ i,j \in [d],
$$

and $\det A$ denotes the determinant of a square matrix $A\in\mathbb{R}^{d\times d}$.
Thus, we conclude that the PDF $p_Y$ is

$$
    p_Y(y) = p_X(\psi^{-1}(y))|\det \partial_y \psi^{-1} (y)|.
$$

We will denote the push-forward operator with the symbol $\sharp$, that is

$$
    [\psi_\sharp p_X](y) := p_X(\psi^{-1}(y))|\det \partial_y \psi^{-1} (y)|.
$$

## 3.4·Flows as Generative Models: 流作为生成模型

As mentioned in [Section 2](Main.md), the goal of generative modeling is to transform samples $X_0 = x_0$ from a \highlight{source distribution} $p$ into samples $X_1=x_1$ from a \highlight{target distribution} $q$.
In this section, we start building the tools necessary to address this problem by means of a flow mapping $\psi_t$.
More formally, a $C^r$ \highlight{flow} is a time-dependent mapping $\psi:[0,1]\times \mathbb{R}^d\to \mathbb{R}^d$ implementing $\psi : (t,x) \mapsto \psi_t(x)$.
Such flow is also a $C^r([0,1]\times\mathbb{R}^{d},\mathbb{R}^d)$ function, such that the function $\psi_t(x)$ is a $C^r$ diffeomorphism in $x$ for all $t \in [0, 1]$.
A \highlight{flow model} is a **continuous-time Markov process** $(X_t)_{0 \leq t \leq 1}$ defined by applying a flow $\psi_t$ to the RV $X_0$:

$$
    X_t = \psi_t(X_0), \quad t\in [0,1], \text{ where } X_0\sim p.
$$

See \Cref{fig:flow_model} for an illustration of a flow model.
To see why $X_t$ is Markov, note that, for any choice of $0\leq t < s \leq 1$, we have

$$
    X_s = \psi_s(X_0) = \psi_s(\psi_t^{-1}( \psi_t(X_0) ) ) = \psi_{s|t}(X_t),
$$

where the last equality follows from using \cref{e:flow_model} to set $X_t = \psi_t(X_0)$, and defining $\psi_{s|t}:= \psi_s\circ \psi_t^{-1}$, which is also a diffeomorphism.
$X_s=\psi_{s|t}(X_t)$ implies that states later than $X_t$ depend only on $X_t$, so $X_t$ is Markov.
In fact, for flow models, this dependence is **deterministic**.

In summary, the goal \highlight{generative flow modeling} is to find a flow $\psi_t$ such that

$$
    X_1 = \psi_1(X_0) \sim q.
$$

### 3.4.1·Equivalence between Flows and Velocity Fields

A $C^r$ flow $\psi$ can be defined in terms of a $C^r([0,1]\times \mathbb{R}^d,\mathbb{R}^d)$ **velocity field** $u:[0,1]\times \mathbb{R}^d \to  \mathbb{R}^d$ implementing $u : (t, x) \mapsto u_t(x)$ via the following ODE:

$$
\begin{aligned}
    \frac{\text{d}}{\text{d} t}\psi_{t}(x) &= u_t(\psi_{t}(x)) & \text{(flow ODE)}\\
    \psi_{0}(x)             &= x                & \text{(flow initial conditions)}
\end{aligned}
$$

See \cref{fig:flow} for an illustration of a flow together with its velocity field.

A standard result regarding the existence and uniqueness of solutions $\psi_t(x)$ to \cref{e:flow} is (see e.g., \cite{perko2013differential,coddington1956theory}):

> **Theorem 1 (Flow Local Existence and Uniqueness)**
> If $u$ is $C^r([0,1]\times\mathbb{R}^{d},\mathbb{R}^d)$, $r\geq 1$ (in particular, locally Lipschitz), then the ODE in \eqref{e:flow} has a unique solution which is a  $C^r(\Omega,\mathbb{R}^d)$ diffeomorphism $\psi_t(x)$ defined over an open set $\Omega$ which is super-set of $\set{0}\times \mathbb{R}^d$.

This theorem guarantees only the **local** existence and uniqueness of a $C^r$ flow moving each point $x\in\mathbb{R}^d$ by $\psi_t(x)$ during a potentially limited amount of time $t\in[0,t_x)$.
To guarantee a solution until $t=1$ for all $x\in\mathbb{R}^d$, one must place additional assumptions beyond local Lipschitzness.
For instance, one could consider global Lipschitness, guaranteed by bounded first derivatives in the $C^1$ case.
However, we will later rely on a different condition---namely, integrability---to guarantee the existence of the flow almost everywhere, and until time $t=1$.

So far, we have shown that a velocity field uniquely defines a flow.
Conversely, given a $C^1$ flow $\psi_t$, one can extract its defining velocity field $u_t(x)$ for arbitrary $x \in \mathbb{R}^d$ by considering the equation $\frac{\text{d}}{\text{d} t} \psi_t(x') = u_t(\psi_t(x'))$, and using the fact that $\psi_t$ is an invertible diffeomorphism for every $t \in [0, 1]$ to let $x'=\psi^{-1}_t(x)$.
Therefore, the unique velocity field $u_t$ determining the flow $\psi_t$ is

$$
u_t(x) = \dot{\psi}_t( \psi^{-1}_t(x)),
$$

where $\dot{\psi}_t:= \frac{\text{d}}{\text{d} t} \psi_t$.
In conclusion, we have shown the equivalence between $C^r$ flows $\psi_t$ and $C^r$ velocity fields $u_t$.

### 3.4.2·Computing Target Samples from Source Samples: 从源样本计算目标样本

Computing a target sample $X_1$---or, in general, any sample $X_t$---entails approximating the solution of the ODE in~\cref{e:flow} starting from some initial condition $X_0=x_0$.
Numerical methods for ODEs is a classical and well researched topic in numerical analysis, and a myriad of powerful methods exist \citep{iserles2009first}.
One of the simplest methods is the **Euler method**, implementing the update rule

$$
X_{t+h} = X_t + h u_t(X_t)
$$

where $h=n^{-1}>0$ is a step size hyper-parameter with $n \in \mathbb{N}$.

To draw a sample $X_1$ from the target distribution, apply the Euler method starting at some $X_0 \sim p$ to produce the sequence $X_h,X_{2h},\ldots,X_1$.
The Euler method coincides with first-order Taylor expansion of $X_t$:
$$X_{t+h}=X_t + h \dot{X}_t + o(h)=X_t + h u_t(X_t) + o(h),$$
where $o(h)$ stands for a function growing slower than $h$, that is, $o(h)/h \to  0$ as $h\to  0$.
Therefore, the Euler method accumulates $o(h)$ error per step, and can be shown to accumulate $o(1)$ error after $n=1/h$ steps.
Therefore, the error of the Euler method vanishes as we consider smaller step sizes $h\to  0$.
The Euler method is just one example among many ODE solvers.
\Cref{ex:euler_method} exemplifies another alternative, the second-order **midpoint method**, which often outperforms the Euler method in practice.

## 3.5·Probability paths and the Continuity Equation

We call a time-dependent probability $(p_t)_{0\leq t \leq 1}$ a \highlight{probability path}.
For our purposes, one important probability path is the marginal PDF of a flow model $X_t = \psi_t(X_0)$ at time $t$:

$$
X_t\sim p_t.
$$

For each time $t \in [0,1]$, these marginal PDFs are obtained via the push-forward formula in \cref{e:push-forward_p}, that is,

$$
p_t(x) = [\psi_{t\sharp}p](x).
$$

Given some arbitrary probability path $p_t$ we define

$$
u_t\ \text{generates}\ p_t \text{ if } X_t = \psi_t(X_0) \sim p_t \text{ for all } t \in[0,1).
$$

In this way, we establish a close relationship between velocity fields, their flows, and the generated probability paths, see Figure \ref{fig:ut_generates_pt} for an illustration.
Note that we use the time interval $[0,1)$, open from the right, to allow dealing with target distributions $q$ with compact support where the velocity is not defined precisely at $t=1$.

To verify that a velocity field $u_t$ generates a probability path $p_t$, one can verify if the pair $(u_t, p_t)$ satisfies a partial differential equation (PDE) known as the **Continuity Equation**:

$$
\frac{\text{d}}{\text{d} t} p_t(x)+ \text{div}(p_t u_t)(x) = 0,
$$

where $\text{div}(v)(x) = \sum_{i=1}^d \partial_{x^i} v^i (x)$, and $v(x)=(v^1(x),\ldots,v^d(x))$.

The following theorem, a rephrased version of the **Mass Conservation Formula** \citep{villani2009optimal}, states that a solution $u_t$ to the Continuity Equation generates the probability path $p_t$:

> Theorem 2 (Mass Conservation)
> Let $p_t$ be a probability path and $u_t$ a locally Lipchitz integrable vector field.
> Then, the following two statements are equivalent:
> - The Continuity Equation \eqref{e:continuity} holds for $t\in [0,1)$.
> - $u_t$ generates $p_t$, in the sense of \eqref{def:generates}.

In the previous theorem, local Lipschitzness assumes that there exists a local neighbourhood over which $u_t(x)$ is Lipschitz, for all $(t, x)$.
Assuming that $u$ is integrable means that:

$$
\int_0^1\int \|u_t(x)\|p_t(x)\text{d} x \text{d} t < \infty.
$$

Specifically, integrating a solution to the flow ODE \eqref{e:flow_flow} across times $[0,t]$ leads to the integral equation

$$
    \psi_t(x) = x + \int_0^t u_s(\psi_s(x))\text{d} s.
$$

Therefore, integrability implies

$$
\begin{aligned}
    \mathbb{E} \|X_t\|
    &= \int \|\psi_t(x)\|p(x)\text{d} x \\
    &= \int \|x + \int_0^t u_s(\psi_s(x))ds\|  p(x)\text{d} x\\
    &\leq \mathbb{E} \|X_0\| + \int_0^1\int \|u_s(x)\|p_t(x)\text{d} t \\
    &< \infty,
\end{aligned}
$$

where (i) follows from the triangle inequality, and (ii) assumes the integrability condition \eqref{e:integrable} and $\mathbb{E} \|X_0\| < \infty$.
In sum, integrability allows assuming that $X_t$ has bounded expected norm, if $X_0$ also does.

To gain further insights about the meaning of the Continuity Equation, we may write it in **integral form** by means of the Divergence Theorem---see \citet{matthews2012vector} for an intuitive exposition, and \citet{loomis1968advanced} for a rigorous treatment.
This result states that, considering some domain $\mathcal{D}$ and some smooth vector field $u:\mathbb{R}^d\to \mathbb{R}^d$, accumulating the divergences of $u$ inside $\mathcal{D}$ equals the **flux** leaving $\mathcal{D}$ by orthogonally crossing its boundary $\partial \mathcal{D}$:

$$
    \int_\mathcal{D} \text{div}(u)(x)\text{d} x = \int_{\partial \mathcal{D}} \langle u(y),n(y)\rangle\text{d} s_y,
$$

where $n(y)$ is a unit-norm normal field pointing outward to the domain's boundary $\partial\mathcal{D}$, and $\text{d} s_y$ is the boundary's area element.
To apply these insights to the Continuity Equation, let us integrate \eqref{e:continuity} over a small domain $\mathcal{D}\subset\mathbb{R}^d$ (for instance, a cube) and apply the Divergence Theorem to obtain

$$
\frac{\text{d}}{\text{d} t} \int_\mathcal{D} p_t(x) \text{d} x = -\int_\mathcal{D} \text{div}(p_t u_t)(x) \text{d} x = -\int_{\partial \mathcal{D}}  \langle p_t(y) u_t(y),n(y)\rangle\text{d} s_y.
$$

This equation expresses the rate of change of total probability mass in the volume $\mathcal{D}$ (left-hand side) as the negative probability **flux** leaving the domain (right-hand side).
The probability flux, defined as $j_t(y)=p_t(y)u_t(y)$, is the probability mass flowing through the hyperplane orthogonal to $n(y)$ per unit of time and per unit of (possibly high-dimensional) area.
See \cref{fig:continuity} for an illustration.

## 3.6·Instantaneous Change of Variables:

One important benefit of using flows as generative models is that they allow the tractable computation of **exact** likelihoods $\log p_1(x)$, for all $x \in \mathbb{R}^d$.
This feature is a consequence of the Continuity Equation called the **Instantaneous Change of Variables**~\citep{chen2018neural}:

$$
    \frac{\text{d}}{\text{d} t} \log p_t(\psi_t (x)) = -\text{div}(u_t)(\psi_t(x)).%
$$

This is the ODE governing the change in log-likelihood, $\log p_t(\psi_t(x))$, along a sampling trajectory $\psi_t(x)$ defined by the flow ODE \eqref{e:flow_flow}.
To derive \eqref{e:instant_div}, differentiate $\log p_t(\psi_t(x))$ with respect to time, and apply both the Continuity Equation \eqref{e:continuity} and the flow ODE \eqref{e:flow_flow}.
Integrating \eqref{e:instant_div} from $t=0$ to $t=1$ and rearranging, we obtain

$$
    \log p_1(\psi_1(x)) = \log p_0(\psi_0(x))-\int_0^1 \text{div}(u_t)(\psi_t(x))\text{d} t.
$$

In practice, computing $\text{div} (u_t)$, which equals the trace of the Jacobian matrix $\partial_x u_t(x) \in \mathbb{R}^{d\times d}$, is increasingly challenge as the dimensionality $d$ grows.
Because of this reason, previous works employ unbiased estimators such as Hutchinson's trace estimator~\citep{grathwohl2018ffjord}:

$$
    \text{div} (u_t)(x) = \text{tr} [\partial_x u_t(x)] = \mathbb{E}_Z\, \text{tr} [ Z^T \partial_x u_t(x) Z],
$$

where $Z\in\mathbb{R}^{d\times d}$ is any random variable with $\mathbb{E}[Z]=0$ and $Cov(Z,Z)=I$, (for example, $Z\sim\mathcal{N}(0,I)$), and $\text{tr}[Z] = \sum_{i=1}^d Z_{i,i}$.
By plugging the equation above into \eqref{e:div_int} and switching the order of integral and expectation, we obtain the following unbiased log-likelihood estimator:

$$
    \log p_1(\psi_1(x)) = \log p_0(\psi_0(x))-\mathbb{E}_Z\, \int_0^1  \text{tr} [ Z^T \partial_x u_t(\psi_t(x)) Z] \text{d} t.
$$

In contrast to $\text{div}(u_t)(\psi_t(x))$ in \eqref{e:instant_div}, computing  $\text{tr}[Z^T \partial_x u_t(\psi_t(x)) Z]$ for a fixed sample $Z$ in the equation above can be done with a single backward pass via a vector-Jacobian product (JVP).
E.g., see [PyTorch.org](https://pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html).

In summary, computing an unbiased estimate of $\log p_1(x)$ entails simulating the ODE

$$
\begin{aligned}
\frac{\text{d}}{\text{d} t}
\begin{bmatrix}
    f(t) \\ g(t)
\end{bmatrix}
&= \begin{bmatrix}
    u_t(f(t)) \\
    -\text{tr}[Z^T \partial_x u_t(f(t))Z]
\end{bmatrix},\\
\begin{bmatrix}
    f(1) \\ g(1)
\end{bmatrix}
&=
\begin{bmatrix}
    x\\ 0
\end{bmatrix},
\end{aligned}
$$

backwards in time, from $t=1$ to $t=0$, and setting:

$$
\widehat{\log p}_1(x) = \log p_0(f(0)) - g(0).
$$

See~\cref{ex:sample_with_likelihood} for an example on how to obtain log-likelihood estimates from a flow model using the \fmlibrary.

## 3.7·Training Flow Models with Simulation: 采用模拟训练流模型

The Instantaneous Change of Variables, and the resulting ODE system \eqref{e:log_p_1_ode_unbiased}, allows training a flow model by maximizing the log-likelihood of training data~\citep{chen2018neural,grathwohl2018ffjord}.
Specifically, let $u^\theta_t$ be a velocity field with learnable parameters $\theta\in\mathbb{R}^p$, and consider the problem of learning $\theta$ such that

$$
    p^\theta_1\approx q.
$$

We can pursue this goal, for instance, by minimizing the KL-divergence of $p^\theta_1$ and $q$:

$$
\mathcal{L}(\theta) = D_{\text{KL}}(q  , p^\theta_1) = -\mathbb{E}_{Y\sim q} \log p^\theta_1(Y) + \text{constant},
$$

where $p^\theta_1$ is the distribution of $X_1=\psi^\theta_1(X_0)$, $\psi_t^\theta$ is defined by $u_t^\theta$, and we can obtain an unbiased estimate of $\log p^\theta_1(Y)$ via the solution to the ODE system~\eqref{e:log_p_1_ode_unbiased}.
However, computing this loss---as well as its gradients---requires precise ODE simulations during training, where only errorless solutions constitute unbiased gradients.
In contrast, **Flow Matching**, presented next, is a simulation-free framework to train flow generative models without the need of solving ODEs during training.