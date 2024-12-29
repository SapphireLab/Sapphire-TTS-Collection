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
