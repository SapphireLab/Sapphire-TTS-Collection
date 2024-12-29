# 5·Non-Euclidean Flow Matching: 非欧流匹配

This section extends Flow Matching from Euclidean spaces $\Real^d$ to general \emph{Riemannian manifolds} $\gM$.
Informally, Riemannian manifolds are spaces behaving locally like Euclidean spaces, and are equipped with a generalized notion of distances and angles.
Riemannian manifolds are useful to model various types of data.
For example, probabilities of natural phenomena on Earth can be modeled on the sphere \cite{mathieu2020riemannian}, and protein backbones are often parameterized inn terms of matrix Lie groups \cite{jumper2021highly}.
The extension of flows to Riemannian manifolds is due to \citet{mathieu2020riemannian,lou2020riemannian}.
However, their original training algorithms required expensive ODE simulations.
Following \citet{chen2024flow}, the Flow Matching solutions in this section provide a scalable, simulation-free training algorithm to learn generative models on Riemannian manifolds.

## 5.1·Riemannian Manifolds: 黎曼流形

We consider complete connected, smooth Riemannian manifolds $\gM$ with a metric $g$.
The tangent space at point $x\in\gM$, a vector space containing all tangent vectors to $\gM$ at $x$, is denoted with $T_x\gM$.
The Riemannian metric $g$ defines an inner product over $T_x \gM$ denoted by $\ip{u,v}_g$, for $u,v\in T_x\gM$.
Let $T\gM=\cup_{x\in \gM} \set{x}\times T_x\gM$ be the tangent bundle that collects all the tangent planes of the manifold.
In the following, vector fields defined on tangent spaces are important objects to build flows on manifolds with velocity fields.
We denote by $\gU=\set{u_t}$ the space of time-dependent smooth vector fields (VFs) $u_t :[0,1]\times \gM\too T\gM$, where $u_t(x)\in T_x\gM$ for all $x\in\gM$.
Also, $\divv_g(u_t)$ is the Riemannian divergence with respect to the spatial ($x$) argument.
Finally, we denote by $\dd \vol_x$ the volume element over $\gM$, and integration of a function $f:\gM\too\Real$ over $\gM$ is denoted $\int f(x) \dd \vol_x$.%

## 5.2·Probabilities, Flows and Velocities on Manifolds: 流形上的概率、流和速度

Probability density functions over a manifold $\gM$ are continuous non-negative functions $p:\gM\rightarrow\Real_{\geq 0}$ integrating to $1$, namely $\int_\gM p(x) \dd \vol_x=1$.
We define a probability path in time $p_t$ as a time-dependent curve in probability space $\gP$, namely $p_t:[0,1]\too\gP$.
A time-dependent flow, $\psi:[0,1]\times \gM\too\gM$, similar to the Euclidean space, defines a global diffeomorphism on $\gM$ for every $t$.

Remarkably, constructing flow-based models via velocity fields naturally applies to general Riemannian manifolds.
Formally, and rephrasing Proposition 1 from \cite{mathieu2020riemannian}:

### Theorem [Flow local existence and uniqueness]

Let $\gM$ a smooth complete manifold and a velocity field $u_t \in \gU$. If $u$ is $C^\infty([0,1]\times\gM,T\gM)$ (in particular, locally Lipschitz), then the ODE in \eqref{e:flow} has a unique solution which is a  $C^\infty(\Omega,\gM)$ diffeomorphism $\psi_t(x)$ defined over the open set $\Omega \supset \set{0}\times \gM$.

---

Similar to \cref{thm:ode_existence_and_uniqueness}, flow ODEs generally only define a local diffeomorphism on the manifold, meaning that $\psi_t(x)$ may be defined on a maximal interval in time $[0,t_x])$ for different values of $x\in\gM$. Similar to the Euclidean case we will work with the semi-open time interval $t\in[0,1)$ to allow $q$ to have compact support (for which $u_t$ is not everywhere defined). To ensure existence for the desired time interval, $[0,1)$, we add the integrability constraint (see \cref{thm:continuity}) and rely on the Mass Conservation theorem once again.
For a Riemannian manifold with metric $g$, the \highlight{Riemannian continuity equation} reads

$$
    \frac{\dd}{\dd t} p_t(x)+ \divv_g(p_t u_t)(x) = 0,%
$$

and the corresponding Manifold Mass Conservation theorem \citep{villani2009optimal} is stated as follows.

### Theorem [Manifold Mass Conservation]

Let $p_t$ be a probability path and $u_t\in\gU$ a locally Lipchitz integrable vector field over a Riemannian manifold $\gM$ with metric $g$. Then the following are equivalent

- The Continuity Equation \eqref{e:riemannian_continuity} holds for $t\in [0,1)$.
- $u_t$ generates $p_t$ in the sense of \ref{def:generates}.

---

In the previous result, by integrable $u$ we mean

$$\label{e:manifold_integrable}
 \int_0^1\int_\gM \norm{u_t(x)}p_t(x)\dd \vol_x \dd t < \infty. %
$$

Note that the assumptions of \cref{thm:riemannian_continuity} yield a global diffeomorphism on $\gM$, giving rise to the \highlight{Riemannian instantaneous change of variables} formula:

$$
    \label{e:r_instant_div} \frac{\dd}{\dd t} \log p_t(\psi_t (x)) = -\divv_g(u_t)(\psi_t(x)).%
$$

Finally, we say that $u_t$ generates $p_t$ from $p$ if

$$
    X_t=\psi_t(X_0)\sim p_t \text{ for } X_0\sim p.
$$

Having positioned flows as valid generative models on manifolds, it stands to reason that the FM principles can be transferred to this domain as well.
In the Riemannian version of FM we aim to find a velocity field $u_t^\theta\in\gU$ generating a target probability path $p_t:[0,1]\too\gP$ with marginal constraints $p_0=p$ and $p_1=q$, where $p,q$ denote the source and target distributions over the manifold $\gM$.
As the velocity field lies on the tangent spaces of the manifold, the \highlight{Riemannian Flow Matching loss} compares velocities using a Bregman divergence defined over the individual tangent planes of the manifold,

$$
    \gL_{\RFM}(\theta) = \E_{t,X_t\sim p_t}D_{X_t}\parr{u_t(X_t),u_t^\theta(X_t)}. %
$$

In the equation above, the expectation is now an integral over the manifold, that is $E[f(X)]=\int_\gM f(x)p_X(x)\dd \vol_x$, for a smooth function $f:\gM\too\gM$ and a random variable $X\sim p_X$.
The Bregman divergences, $D_x$, $x\in\gM$, are potentially defined with the Riemannian inner product and a strictly convex function assigned to each tangent space $\Phi_x:T_x\gM\too T_x\gM$, that is, $D_{x}(u,v) \defe \Phi_x(u) - \brac{ \Phi_x(v) + \langle u-v, \nabla_v \Phi_x(v) \rangle_g}$. For example, choosing the Riemannian metric $\Phi_x=\norm{\cdot}_g^2$ then  $D_x(u,v)=\norm{u-v}^2_g$  for $u,v\in T_x\gM$.
