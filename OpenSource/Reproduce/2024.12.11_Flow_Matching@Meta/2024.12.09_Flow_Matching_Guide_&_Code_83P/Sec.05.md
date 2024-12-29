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

## 5.3·Probability paths on manifolds

\highlight{Marginal probability paths} are built as in the Euclidean case \eqref{e:p_t}:

$$
    p_t(x) = \int_\gM p_t(x|x_1)q(x_1) \dd \vol_{x_1},%
$$

where $p_{t|1}(x|x_1)$ is the \highlight{conditional probability path} defined on the manifold.
We also require the boundary constraints

$$
    p_0=p, \quad p_1=q. %
$$

For instance, these constraints can be implemented by requiring the conditional path $p_{t|1}(x|x_1)$ to satisfy

$$
    p_{0|1}(x|x_1) = \pi_{0|1}(x|x_1), \text{ and  } p_{1|1}(x|x_1)=\delta_{x_1}(x),
$$

where $\pi_{0|1}$ is the conditional coupling, $\pi_{0|1}(x_0|x_1)=\pi_{0,1}(x_0,x_1)/q(x_1)$.

## 5.4·The Marginalization Trick for manifolds

The Marginalization Trick for the marginal velocity field (\cref{thm:fm_main}) readily applies to the Riemannian case.
Consider the \highlight{conditional velocity field} $u_t(x|x_1)\in\gU$ such that
$$
    u_{t}(\cdot|x_1) \text{ generates } p_{t|1}(\cdot|x_1).
$$

Then, the \highlight{marginal velocity} field $u_t(x)$ is given by the following averaging of the conditional velocities,
$$
    u_t(x) = \int_\gM u_t(x|x_1)p_{1|t}(x_1|x) \dd \vol_{x_1}, %
$$
where, by Bayes' Rule for PDFs, we obtain
$$
p_{1|t}(x_1|x) = \frac{p_{t|1}(x|x_1)q(x_1)}{p_t(x)},
$$
which is defined for all $x\in\gM$ for which $p_t(x)>0$.

The Marginalization Trick (\cref{thm:fm_main}) for the Riemannian case requires adjusting \cref{as:p_t} as follows:

### Assumption 2 [Riemannian p_t]

$p_{t|1}(x|x_1)$ is $C^\infty([0,1)\times \gM)$ and $u_t(x|x_1)$ is $C^\infty([0,1)\times \gM,\gM)$ as function of $(t,x)$. Furthermore, we assume either $q$ has bounded support, \ie, $q(x_1)=0$ outside some bounded set or $\gM$ is compact; and $p_t(x)>0$ for all $x\in\gM$ and $t\in[0,1)$.

---

We are now ready to state the \highlight{Manifold  Marginalization Trick} theorem:

### Theorem [Manifold Marginalization Trick]

Under Assumption \ref{as:riemannian_p_t}, if $u_t(x|x_1)$ is conditionally integrable and generates the conditional probability path $p_t(\cdot|x_1)$ then the marginal velocity field $u_t(\cdot)$ generates the marginal probability path $p_t(\cdot)$.

---

By \highlight{conditionally integrable}, we mean a conditioned version of the integrability condition from the Mass Conservation Theorem \eqref{e:manifold_integrable}:

$$
\begin{aligned}
\int^1_0\int_\gM\int_\gM \norm{u_t(x|x_1)}_g p_{t|1}(x|x_1) q(x_1) \dd \vol_{x_1} \dd \vol_{x} \dd t &< \infty
\end{aligned}
$$

The proof of \cref{thm:rfm_main} is repeating the arguments of \cref{thm:fm_main} and is given in \cref{a:manifold_marginalization_trick}.

## 5.5·Riemannian Flow Matching Loss

The \highlight{Riemannian Conditional Flow Matching (RCFM) loss} reads
\begin{equation}\label{e:rcfm_loss}
    \gL_{\RCFM}(\theta) = \E_{t,X_1,X_t\sim p_{t|1}(\cdot|X_1)}D_{X_t}\parr{u_t(X_t|X_1),u_t^\theta(X_t)}.%
\end{equation}
Once again, we have the equivalence:
\begin{myframe}
\begin{theorem}\label{thm:rcfm}
    The gradients of the Riemannian Flow Matching loss and the Riemannian Conditional Flow Matching loss coincide:
    \begin{equation}
        \nabla_\theta \gL_{\RFM}(\theta) = \nabla_\theta \gL_{\RCFM}(\theta).
    \end{equation}
\end{theorem}
\end{myframe}

The above theorem can be proved using \cref{prop:bregman_gradient} with $X=X_t$, $Y=u_t(X_t|X_1)$, $g^\theta(x)=u_t^\theta(x)$, and integrating w.r.t.~$t\in[0,1]$.

## 5.6·Conditional flows through premetrics


Having established how to learn a flow model with the RCFM loss, we are left with specifying the conditional probability path and its generating velocity field.
Similar  to \cref{s:conditional_flows}, we begin by stating the requirements for the corresponding conditional flow $\psi:[0,1)\times \gM \times \gM \too \gM$, such that $p_{t|1}(\cdot|x_1)$ satisfies the boundary conditions \eqref{e:riemannian_p_t_cond_boundary}.
The \highlight{conditional flow model} is
\begin{equation}
    X_{t|1} = \psi_t(X_0|x_1), \quad \text{ where } X_0\sim \pi_{0|1}(\cdot|x_1), %
\end{equation}
where the \highlight{conditional flow} is
\begin{equation}\label{e:riemannian_psi}
    \psi_t(x|x_1) = \begin{cases}
        x & t=0 \\
        x_1 & t=1
    \end{cases}, \ \text{ is smooth in } t,x \text{ and diffeomorphism in $x$ on } \gM. %
\end{equation}


Our analysis in Euclidean space focused on affine conditional flows, as these served as a rich class of easily computable (simulation-free) conditional flows.
Unfortunately,  combinations $\alpha_t x_1 + \sigma_t x_0$ for $\alpha_t+\sigma_t\ne1$ are not naturally defined on manifolds. The manifold analog for the case $\alpha_t+\sigma_t=1$ would be using  geodesic interpolation.
Indeed, \citet{chen2024flow} proposed building conditional flows by moving along geodesic curves, in particular, generalizing the conditional OT paths moving along straight lines in Euclidean space (see \cref{thm:cond_ot}). Geodesics represent the shortest paths between two points on a manifold, reducing to straight lines in Euclidean spaces.
For manifolds, we define the \highlight{geodesic conditional flow} as
\begin{equation}\label{e:exp_log}
    \psi_t (x_0|x_1)= \exp_{x_0} (\kappa(t) \log_{x_0}(x_1)), \quad t\in[0,1],%
\end{equation}
where $\kappa(t):[0,1]\too[0,1]$ is a monotonically increasing scheduler satisfying $\kappa(0)=0$ and $\kappa(1)=1$, making sure all $x_0$ are pushed to $x_1$ at $t=1$. The exponential map, evaluated at $x\in\gM$, $\exp_x:T_x\gM\too \gM$, $v\mapsto \exp_x(v)$, returns the endpoint at time $t=1$ of the unique geodesic starting at $x$ with initial speed $v$. The logarithmic map $\log_{x}:\gM\too  T_xM$, $y\mapsto \log_x(y)$, is the inverse of the exponential map. In Euclidean space, the exponential map is simply vector addition, and the logarithmic map is vector subtraction.
Now, if we plug these in \eqref{e:exp_log}, we get $\psi_t (x_0|x_1)= x_0 + \kappa(t)(x_1-x_0)$, and by choosing $\kappa(t)=t$ we recover the conditional OT flow.

For simple manifolds with closed-form exponential and logarithmic maps, this construction allows a simulation-free recipe for training flows on manifolds, an arguably clear advantage compared to diffusion models approaches built on manifolds \citep{de2022riemannian,huang2022rimannian,lou2023scaling}.
In particular, manifold diffusion models require in-training simulation to sample from $p_t$, and have to resort to approximations of the score function on the manifold.

Nevertheless, while building geodesic conditional flows is a natural construction, geodesics may be hard to compute for general manifolds that do not have closed-form exponential and logarithmic maps {and/or introduce undesired bias such as concentrating probability at boundary points.}
To overcome the difficulty in computing geodesics and/or inject a desired implicit bias, one may seek an alternative notion of smooth distance function, $\dist(\cdot,\cdot):\gM\times\gM\too\Real_{\geq 0}$, and require that the conditional flow satisfies
\begin{equation}\label{e:dist_cond_flow}
    \dist(\psi_t(x_0|x_1),x_1) = \bar{\kappa}(t)\dist(x_0,x_1),
\end{equation}
where $\bar{\kappa}(t)=1-\kappa(t)$. This will assure that the conditional flow concentrates all the probability at $x_1$ at time $t=1$ if the following conditions hold:
\begin{enumerate}
    \item \emph{Non-negative}: $\dist(x,y)\geq 0$ for all $x,y\in \gM$.
    \item \emph{Positive}: $\dist(x,y)=0$ if and only if $x=y$.
    \item \emph{Non-degenerate}: $\nabla \dist(x,y)\neq 0$ if and only if $x\neq y$.
\end{enumerate}

\citet{chen2024flow} showed that the minimal norm conditional velocity field corresponding to a flow that satisfies \eqref{e:dist_cond_flow} has the form:
\begin{equation}\label{e:riemannian_cond_ut}
    u_t(x|x_1) = \frac{d \log \bar{\kappa}(t)}{\dd t}\dist(x,x_1)\frac{\nabla \dist(x,x_1)}{\norm{\nabla \dist(x,x_1)}^2_g},
\end{equation}
\begin{wrapfigure}[13]{r}{0.4\textwidth}
  \begin{center} \vspace{-10pt}   \includegraphics[width=0.38\textwidth]{assets/rfm.pdf}
  \end{center}
  \caption{Conditional flows on the manifold $\gM$.}\label{fig:rfm}
\end{wrapfigure}
where the non-degeneracy requirement of the premetric ensures that the velocity field has no discontinuities, since $u_t(x|x_1)\propto 1/{\norm{\nabla \dist(x,x_1)}_g}$. In particular, note that the geodesic conditional flow in \eqref{e:exp_log} satisfies \eqref{e:dist_cond_flow} for the choice $\dist=\dist_g$, where $\dist_g$ is the geodesic distance. An example of a choice of alternative premetrics is using spectral distances on general geometries \citep{chen2024flow}, where the conditional velocity offers a way to sample from $p_t(x|x_1)$ by simulation. Importantly, although conditional flows with premetrics require in-training simulation---like diffusion models on manifolds---the velocity field can still be accurately recovered compared to approximations of the score function.

Another issue, is that both conditional flows defined via geodesic interpolation and premetric can suffer from singularities, \eg, for compact manifolds. For example on the 2-sphere the geodesic function $\dist(x,x_1)$ is not differentiable at the antipodal point $x=-x_1$. Furthermore, any smooth function such as $x\mapsto \dist(x,x_1)$ will showcase at-least two critical points (maximum and minimum) where the velocity in \eqref{e:riemannian_cond_ut} is not-defined. However, the set of such  problematic points is generally very small (in fact of zero volume usually). Therefore, this issue does not cause problems in practice, at-least in use cases we are aware of.


In any case, to deal with this issue, we can include an augmented scheduler in the geodesic conditional flow. That is, use  $\bar{\kappa}(t,x,x_1)$, that depends also on $x,x_1$ to make \eqref{e:exp_log} globally smooth. To deal with the zero gradient issue of the premetric conditional flow we can relax the non-degeneracy requirement as follows:
\begin{itemize}
    \item[3.]  \emph{Non-degenerate (relaxed)}: The volume of the set $\gA_y = \set{x\in \gM \, \vert \,  \nabla \dist(x,y)=0 \text{ and } x\ne y } $ is $0$ for all $y\in \gM$.
\end{itemize}


```
Training with geodesic flows on a Sphere using the CFM objective

import torch
from flow_matching.path import GeodesicProbPath, PathSample
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import Sphere

model = ...  # Define a trainable velocity model
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()  # Any Bregman divergence

manifold = Sphere()
scheduler = CondOTScheduler()
path = GeodesicProbPath(scheduler=scheduler, manifold=manifold)

for x_0, x_1 in dataloader:  # Samples from $\pi_{0,1}$ of shape [batch_size, *data_dim]
    t = torch.rand(batch_size)  # Randomize time $t \sim U[0,1]$
    sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)  # Sample the conditional path

    model_output = model(sample.x_t, sample.t)
    projected_model_output = manifold.proju(sample.x_t, model_output)  # Project to tangent space

    loss = loss_fn(projected_model_output, sample.dx_t)  # CFM loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
