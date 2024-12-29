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

## 6.2·The CTMC generative model

The \highlight{CTMC model} is an $\gS$-valued time-dependent family of random variables $(X_t)_{0\leq t \leq 1}$ that a form a Markov chain characterized by the \highlight{probability transition kernel} $\tkernel_{t+h|t}$ defined via
\begin{myframe}
\begin{equation}
    \tkernel_{t+h|t}(y|x) \defe \sP(X_{t+h}=y \vert X_t = x) = \delta(y,x) + h u_t(y,x) + o(h), \text{ and }
    \sP(X_0 = x)= p(x),\label{e:ctmc_model} %
    \end{equation}
\end{myframe}

\begin{wrapfigure}[13]{r}{0.3\textwidth}
  \includegraphics[width=0.3\textwidth]{assets/ctmc/ctmc_general.pdf}
  \caption{The CTMC model is defined by prescribing rates (velocities) of probability between states.}\label{fig:ctmc_general}
\end{wrapfigure}

where the PMF $p$ indicates the initial distribution of the process at time $t=0$, and $o(h)$ is an arbitrary function satisfying $o(h)/h\too 0$ as $t\too 0$.
The values $u_t(y,x)$, called \highlight{rates} or \highlight{velocities}, indicate the speed at which the probability transitions between states as a function of time.
By fully characterized, we mean that all the joints $\sP(X_{t_1}=x_1,\ldots,X_{t_n}=x_n)$, for arbitrary $0\leq t_1 < \cdots < t_n \leq 1$ and $x_{i}\in\gS$, $i\in[n]$, are defined this way.


To make sure the transition probabilities $p_{t+h|t}(y|x)$ are defined via \eqref{e:ctmc_model}, velocities needs to satisfy the following \highlight{rate conditions}:
\begin{equation}
    u_t(y,x)\geq 0 \text{ for all } y\ne x\text{, and } \sum_y u_t(y,x)=0. \label{e:rate_conds}%
\end{equation}
If one of these conditions were to fail, then the transition probabilities $p_{t+h|t}(\cdot|x)$ would become negative or sum to $c \neq 1$ for arbitrary small $h>0$.
\Cref{e:ctmc_model} plays he same role as \cref{e:flow_model} and \cref{e:flow} when we were defining the flow generative modeling.
The \emph{marginal probability} of the process $X_t$ is denoted by the PMF $p_t(x)$ for time $t \in[0,1]$.
Then, similarly to \cref{def:generates} for the case of flows, we say that
\par %
\begin{myframe}
\begin{equation}\label{def:generates_ctmc}
    u_t \text{ \highlight{generates} } p_t \text{ if there exists } p_{t+h|t} \text{ satisfying } \eqref{e:ctmc_model} \text{ with marginals } p_t.
\end{equation}
\end{myframe}

\paragraph{Simulating CTMC.}

To sample $X_t$, sample $X_0\sim p$ and take steps using the \highlight{(naive) Euler method}:
\begin{equation}\label{e:ctmc_euler_naive}
    \sP(X_{t+h} = y \ \vert \ X_t ) =  \delta(y,X_t) + hu_t(y,X_t). %
\end{equation}
According to \eqref{e:ctmc_model}, these steps introduce $o(h)$ errors to the update probabilities.
In practice, this means that we would need a sufficiently small $h>0$ to ensure that the right-hand side in \eqref{e:ctmc_euler_naive} remains a valid PMF.
One possible remedy to assure that any choice of $h>0$ results in a valid PMF, and maintains the $o(h)$ local error in probabilities is the following \highlight{Euler method}:
\begin{align}\label{e:ctmc_euler}
    \sP(X_{t+h} = y \ \vert \ X_t ) =  \begin{cases}
       \exp\brac{h u_t(X_t,X_t)} & y=X_t\\
       \frac{u_t(y,X_t)}{\abs{u_t(X_t,X_t)}}\parr{1-\exp\brac{hu_t(X_t,X_t)}} & y\ne X_t
    \end{cases}.%
\end{align}