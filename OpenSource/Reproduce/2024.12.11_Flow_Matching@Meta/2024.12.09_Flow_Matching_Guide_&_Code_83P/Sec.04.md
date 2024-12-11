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