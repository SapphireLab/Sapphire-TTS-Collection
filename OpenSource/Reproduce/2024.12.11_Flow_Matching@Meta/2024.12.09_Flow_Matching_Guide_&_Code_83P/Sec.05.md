# 5·Non-Euclidean Flow Matching: 非欧流匹配

This section extends Flow Matching from Euclidean spaces $\Real^d$ to general \emph{Riemannian manifolds} $\gM$.
Informally, Riemannian manifolds are spaces behaving locally like Euclidean spaces, and are equipped with a generalized notion of distances and angles.
Riemannian manifolds are useful to model various types of data.
For example, probabilities of natural phenomena on Earth can be modeled on the sphere \cite{mathieu2020riemannian}, and protein backbones are often parameterized inn terms of matrix Lie groups \cite{jumper2021highly}.
The extension of flows to Riemannian manifolds is due to \citet{mathieu2020riemannian,lou2020riemannian}.
However, their original training algorithms required expensive ODE simulations.
Following \citet{chen2024flow}, the Flow Matching solutions in this section provide a scalable, simulation-free training algorithm to learn generative models on Riemannian manifolds.
