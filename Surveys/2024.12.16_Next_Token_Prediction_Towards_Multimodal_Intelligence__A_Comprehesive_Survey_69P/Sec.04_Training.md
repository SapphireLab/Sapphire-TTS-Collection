# 4·Training with Unified Multimodal Task Representation

Once content from various modalities has been tokenized into a sequence of tokens, with a unified backbone model, typically a decoder-only transformer model~\cite{vaswani2017attention}, we can undergo training to tackle a wide array of downstream understanding and generation tasks following different training objectives (refer to Section~\ref{sub:training_obj}). The training tasks are primarily divided into two categories, which resemble the training of large language models: Pretraining (refer to Section~\ref{subsub-ssl}) and Finetuning (refer to Section~\ref{subsub-sl}).

For a sequence of input tokens $x_{1\sim i-1} = \{ x_1, x_2, \ldots, x_{i-1} \}$ , the model predicts the next token $x_i \in V$. The general loss function $f$ for a single prediction could be written as:

$$
    L(\theta) = f\left( y_i , p_{\theta}\left(x_i \mid x_{1\sim i-1} \right)\right),
$$

where:

- $L(\theta)$ is the loss, parameterized by the model parameters $\theta$ and loss function $f$.
- $V$ is the total vocabulary. We use $V_T$, $V_M$ to denote text split and multimdoal split of the full vocabulary, $V_S$ to denote the continuous tokens which are continuous vectors.
- $y_i$ represents the target output for the next token. In supervised training, $y_i$ is typically derived from labeled data, whereas in self-supervised training, $y_i$ can be constructed from the data itself without explicit labels, often using the true next token from the input sequence. In special cases, $y_i$ could involve multiple tokens, enabling parallel prediction of next tokens.
- $f$ is cross-entropy loss when $y_i$ is the discrete token distribution. $f$ can also have different forms like mean-square error if $y_i$ belongs to continuous tokens.

Different training tasks differ in the organization of given sequence $x_{1\sim i-1}$ and target label $y_i$. For self-supervised training, the sequence itself provides the target $y_i$, with the correct next token being used as the label. This allows the model to learn from the vast amounts of unlabeled multimodal data available, which consumes larger training resources. Supervised training would require explicit labeling of the next tokens, which can improve more specific downstream tasks at the cost of being more labor-intensive in the data collection period.