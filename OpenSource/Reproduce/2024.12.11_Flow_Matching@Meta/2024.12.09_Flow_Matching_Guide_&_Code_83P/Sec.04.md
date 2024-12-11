# 4·Flow Matching: 流匹配

Given a source distribution $p$ and a target distribution $q$, Flow Matching (FM) [^1] [^2] [^3] is a scalable approach for training a flow model, defined by a learnable velocity $u_t^\theta$, and solving the **Flow Matching Problem**:

$$
\text{Find}\ u^\theta_t\ \text{generating}\ p_t, \quad\text{with} p_0=p\ \text{and}\ p_1=q.
$$

In the equation above, "generating" is in the sense of previous content:

$$
u_t\ \text{generates}\ p_t\ \text{if}\ X_t=\psi_t(X_0)\sim p_t\ \text{for all}\ t\in[0,1).
$$

Revisiting the Flow Matching blueprint from [Figure.02](Images/Fig.02.png), the FM framework (a) identifies a known source distribution $p$ and an unknown data target distribution $q$, (b) prescribes a probability path $p_t$ interpolating from $p_0=p$ to $p_1=q$, (c) learns a velocity field $u^\theta_t$ implemented in terms of a neural network and generating the path $p_t$, and (d) samples from the learned model by solving an ODE with $u^\theta_t$.

To learn the velocity field $u^\theta_t$ in step (c), FM minimizes the regression loss:

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{X_t\sim p_t} D(u_t(X_t) , u^\theta_t(X_t)),
$$

where $D$ is a dissimilarity measure between vectors, such as the squared $\ell_2$-norm $D(u, v) = \|u-v\|^2$.

Intuitively, the FM loss encourages our learnable velocity field $u_t^\theta$ to match the ground truth velocity field $u_t$ known to generate the desired probability path $p_t$.
\Cref{fig:diagram} depicts the main objects in the Flow Matching framework and their dependencies.
Let us start our exposition of Flow Matching by describing how to build $p_t$ and $u_t$, as well as a practical implementation of the FM loss.

[^1]: lipman2022flow,
[^2]: liu2022flow,
[^3]: albergo2022building

## 4.1·Data: 数据

To reiterate, let source samples be a RV $X_0 \sim p$ and target samples a RV $X_1 \sim q$.
Commonly, source samples follow a known distribution that is easy to sample, and target samples are given to us in terms of a dataset of finite size.
Depending on the application, target samples may constitute images, videos, audio segments, or other types of high-dimensional, richly structured data.
Source and target samples can be independent, or originate from a general joint distribution known as the **coupling**

$$
(X_0,X_1) \sim \pi_{0,1} (X_0,X_1),
$$

where, if no coupling is known, the source-target samples are following the independent coupling $\pi_{0,1} (X_0,X_1) = p(X_0)q(X_1)$.
One common example of independent source-target distributions is to consider the generation of images $X_1$ from random Gaussian noise vectors $X_0\sim \mathcal{N}(0,I)$.
As an example of a dependent coupling, consider the case of producing high-resolution images $X_1$ from their low resolution versions $X_0$, or producing colorized videos $X_1$ from their gray-scale counterparts $X_0$.

## 4.2·Building Probability Paths: 构建概率路径

Flow Matching drastically simplifies the problem of designing a probability path $p_t$---together with its corresponding velocity field $u_t$---by adopting a conditional strategy.
As a first example, consider conditioning the design of $p_t$ on a single target example $X_1=x_1$, yielding the **conditional probability path** $p_{t|1}(x|x_1)$ illustrated in [Figure.03 (a)](Images/Fig.03.png).
Then, we may construct the overall, **marginal probability path** $p_t$ by aggregating such conditional probability paths $p_{t|1}$:

$$
p_t(x) = \int p_{t|1}(x|x_1)q(x_1) \text{d} x_1,
$$

as illustrated in [Figure.03 (b)](Images/Fig.03.png).

To solve the Flow Matching Problem, we would like $p_t$ to satisfy the following boundary conditions:

$$
p_0=p, \quad p_1=q,
$$

that is, the marginal probability path $p_t$ interpolates from the source distribution $p$ at time $t=0$ to the target distribution $q$ at time $t=1$.
These boundary conditions can be enforced by requiring the conditional probability paths to satisfy

$$
p_{0|1}(x|x_1) = \pi_{0|1}(x|x_1), p_{1|1}(x|x_1)=\delta_{x_1}(x),
$$

where the conditional coupling $\pi_{0|1}(x_0|x_1)=\pi_{0,1}(x_0,x_1)/q(x_1)$ and $\delta_{x_1}$ is the delta measure centered at $x_1$.

For the independent coupling $\pi_{0,1}(x_0,x_1)=p(x_0)q(x_1)$, the first constraint above reduces to $p_{0|1}(x|x_1)=p(x)$.
Because the delta measure does not have a density, the second constraint should be read as $\int p_{t|1}(x|y)f(y)\text{d} y \to f(x)$ as $t \to 1$ for continuous functions $f$.
Note that the boundary conditions \eqref{e:p_q_interp} can be verified plugging \eqref{e:p_t_cond_boundary} into \eqref{e:p_t}.

A popular example of a conditional probability path satisfying the conditions in \eqref{e:p_t_cond_boundary} was given in \eqref{e:condot_path}:

$$
\mathcal{N}(\cdot | t x_1, (1-t)^2I)\to \delta_{x_1}(\cdot) \text{ as } t \to 1.
$$
