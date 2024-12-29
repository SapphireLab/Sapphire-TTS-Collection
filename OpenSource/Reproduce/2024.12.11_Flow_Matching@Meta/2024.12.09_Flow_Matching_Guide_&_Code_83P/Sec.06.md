# 6·Continuous Time Markov Chain Models

This section presents the \highlight{Continuous Time Markov Chains (CTMCs)} as an alternative generative model to flow, with the use-case of generating discrete data, \ie, data residing in a discrete (and finite) state space.
CTMC are Markov processes that form the building blocks behind the generative model paradigm of Discrete Flow Matching (DFM) \cite{campbell2024generative,gat2024discrete}, later discussed in~\cref{sec:discreteflow}.
Therefore, this section is analogous to \cref{s:flow_models}, where we presented flows as the building blocks behind the generative model paradigm of Flow Matching (FM).

## 6.1·Discrete state spaces and random variables

Consider a finite version of $\Real^d$ as our state space $\gS = \gT^d$, where $\gT = [K] = \set{1,2,\ldots,K}$, sometimes called \highlight{vocabulary}.
Samples and states are denoted by $x=(x^1,\ldots,x^d)\in \gS$, where $x^i\in \gT$ is single coordinate or a \highlight{token}. We will similarly use states $y,z\in\gS$.
Next, $X$ denotes a random variable taking values in the state space $\gS$, with probabilities governed by the \highlight{probability mass function (PMF)} $p_X:\gS\too\Real_{\geq 0}$, such that $\sum_{x\in \gS}p_X(x) = 1$, and the probability of an event $A\subset \gS$ being
\begin{equation}
    \sP(X\in A) = \sum_{x\in A} p_X(x).
\end{equation}
The notations $X\sim p_X$ or $X\sim p_X(X)$ indicate that $X$ has the PMF $p_X$.
The $\delta$ PMF in the discrete case is defined by
\begin{equation}
    \delta(x,z) = \begin{cases}
        1 & x=z, \\ 0 & \text{else}.
    \end{cases}
\end{equation}
where we sometimes also define $\delta$ PMFs on tokens, such as in $\delta(x^i,y^i)$, for some $x^i,y^i\in \gT$.
