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

## 4.3·Deriving Generating Velocity Fields: 推导生成速度场

Equipped with a marginal probability path $p_t$, we now build a velocity field $u_t$ generating $p_t$.
The generating velocity field $u_t$ is an average of multiple \highlight{conditional velocity fields} $u_t(x|x_1)$, illustrated in \cref{fig:u_t}, and satisfying:

$$
u_{t}(\cdot|x_1) \text{ generates } p_{t|1}(\cdot|x_1).
$$

Then, the \highlight{marginal velocity field} $u_t(x)$, generating the marginal path $p_t(x)$,  illustrated in \cref{fig:marginal_u_t}, is given by averaging the conditional velocity fields $u_t(x|x_1)$ across target examples:

$$
u_t(x) = \int u_t(x|x_1)p_{1|t}(x_1|x) \text{d} x_1.
$$

To express the equation above using known terms,  recall Bayes' rule

$$
p_{1|t}(x_1|x) = \frac{p_{t|1}(x|x_1)q(x_1)}{p_t(x)},
$$

defined for all $x$ with $p_t(x)>0$.
\Cref{e:u_t} can be interpreted as the weighted average of the conditional velocities $u_t(x|x_1)$, with weights $p_{1|t}(x_1|x)$ representing the posterior probability of target samples $x_1$ given the current sample $x$.
Another interpretation of \eqref{e:u_t} can be given with conditional expectations (see \cref{sec:conditional_densities_and_expectations}).
Namely, if $X_t$ is \emph{any} RV such that $X_t\sim p_{t|1}(\cdot|X_1)$, or equivalently, the joint distribution of $(X_t,X_1)$ has density $p_{t,1}(x,x_1)=p_{t|1}(x|x_1)q(x_1)$ then using \eqref{e:f_x_y_cond_y} to write \eqref{e:u_t} as a conditional expectation, we obtain

$$
u_t(x) = \mathbb{E}[u_t(X_t|X_1)|X_t=x],
$$

which yields the useful interpretation of $u_t(x)$ as the least-squares approximation to $u_t(X_t|X_1)$ given $X_t=x$, see \cref{sec:conditional_densities_and_expectations}.
Note, that the $X_t$ in \eqref{e:u_t_cond_E} is in general a different RV that $X_t$ defined by the final flow model \eqref{e:flow_model}, although they share the same marginal probability $p_t(x)$.

## 4.4·General Conditioning & the Marginalization Trick: 一般条件与边际化技巧

To justify the constructions above, we need to show that the marginal velocity field $u_t$ from \cref{e:u_t,e:u_t_cond_E} generates the marginal probability path $p_t$ from \cref{e:p_t} under mild assumptions.
The mathematical tool to prove this is the Mass Conservation Theorem (\cref{thm:continuity}).
To proceed, let us consider a slightly more general setting that will be useful later in the manuscript.
In particular, there is nothing special about building conditional probability paths and velocity fields by conditioning on $X_1=x_1$.
As noted in~\cite{tong2023improving}, the analysis from the previous section carries through to conditioning on any arbitrary RV $Z\in\mathbb{R}^m$ with PDF $p_Z$.
This yields the \highlight{marginal probability path}

$$
p_t(x) = \int p_{t|Z}(x|z) p_Z(z) dz,
$$

which in turn is generated by the \highlight{marginal velocity field}

$$
u_t(x) = \int u_t(x|z) p_{Z|t}(z|x) dz = \mathbb{E}[u_t(X_t|Z)|X_t=x],
$$

where $u_t(\cdot|z)$ generates $p_{t|Z}(\cdot|z)$, $p_{Z|t}(z|x) = \frac{p_{t|Z}(x|z)p_Z(z)}{p_t(x)}$ follows from Bayes' rule given $p_t(x)>0$, and $X_t\sim p_{t|Z}(\cdot|Z)$.

Naturally, we can recover the constructions in previous sections by setting $Z=X_1$.
Before we prove the main result, we need some regularity assumptions, encapsulated as follows.

> Assumption 1.
> $p_{t|Z}(x|z)$ is $C^1([0,1)\times \mathbb{R}^d)$ and $u_t(x|z)$ is $C^1([0,1)\times \mathbb{R}^d,\mathbb{R}^d)$ as a function of $(t,x)$.
> Furthermore, $p_Z$ has bounded support, that is, $p_Z(x)=0$ outside some bounded set in $\mathbb{R}^m$.
> Finally, $p_t(x)>0$ for all $x\in\mathbb{R}^d$ and $t\in[0,1)$.

These are mild assumptions.

For example, one can show that $p_t(x) > 0$ by finding a condition $z$ such that $p_Z(z)>0$ and $p_{t|Z}(\cdot|z)>0$.
In practice, one can satisfy this by considering $(1-(1-t)\epsilon)p_{t|Z} + (1-t)\epsilon (0,I)$ for an arbitrarily small $\epsilon > 0$.
One example of $p_{t|Z}(\cdot|z)$ satisfying this assumption is the path in \eqref{e:condot_path}, where we let $Z=X_1$.
We are now ready to state the main result:

### Theorem 3 (Marginalization Trick)

Under Assumption 1, if $u_t(x|z)$ is conditionally integrable and generates the conditional probability path $p_t(\cdot|z)$, then the marginal velocity field $u_t$ generates the marginal probability path $p_t$, for all $t\in[0,1)$.

---

In the theorem above, conditionally integrable refers to a conditional version of the integrability condition from the Mass Conservation Theorem \eqref{e:integrable}, namely:

$$
\int_0^1\int\int \|u_t(x|z)\| p_{t|Z}(x|z) p_Z(x) dz \text{d} x \text{d} t < \infty.
$$

---

### Proof

The result follows from verifying the two conditions of the Mass Conservation in Theorem 2.
First, let us check that the pair $(u_t, p_t)$ satisfies the Continuity Equation \eqref{e:continuity}.
Because $u_t(\cdot|x_1)$ generates $p_t(\cdot|x_1)$, we have that

$$
\begin{aligned}
\frac{\text{d}}{\text{d} t}p_t(x)
&= \int \frac{\text{d}}{\text{d} t}p_{t|Z}(x|z)p_Z(x)dz \\
&= -\int \text{div}_x[u_t(x|z) p_{t|Z}(x|z)] p_Z(z) dz\\
&= -\text{div}_x \int u_t(x|z) p_{t|Z}(x|z)p_Z(z) dz\\
&= -\text{div}_x [u_t(x)p_t(x)].
\end{aligned}
$$

Equalities (i) follows from switching differentiation ($\frac{\text{d}}{\text{d} t}$ and $\text{div}_x$, respectively) and integration, as justified by Leibniz's rule, the fact that $p_{t|Z}(x|z)$ and $u_t(x|z)$ are $C^1$ in $t,x$, and the fact that $p_Z$ has bounded support (so all the integrands are integrable as continuous functions over bounded sets).
Equality (ii) follows from the fact that $u_t(\cdot|z)$ generates $p_{t|Z}(\cdot|z)$ and \cref{thm:continuity}.
Equality (iii) follows from multiplying and dividing by $p_t(x)$ (strictly positive by assumption) and using the formula \eqref{e:u_t_general} for $u_t$.

To verify the second and last condition from \cref{thm:continuity}, we shall prove that $u_t$ is integrable and locally Lipschitz.
Because $C^1$ functions are locally Lipschitz, it suffices to check that $u_t(x)$ is $C^1$ for all $(t, x)$.
This would follow from verifying that $u_t(x|z)$ and $p_{t|Z}(x|z)$ are $C^1$ and $p_t(x)>0$, which hold by assumption.
Furthermore, $u_t(x)$ is integrable because $u_t(x|z)$ is conditionally integrable:

$$
\int_0^1 \int \|u_t(x)\|p_t(x) \text{d} x \text{d} t \leq \int_0^1\int\int \|u_t(x|z)\|p_{t|Z}(x|z)p_Z(z) dz \text{d} x \text{d} t  < \infty,
$$

where the first inequality follows from vector Jensen's inequality.
