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

## Training Objectives

Based on what kind of target token $y_i$ to predict, the NTP training objectives could be further categorized into two classes: \textbf{ Discrete Token Prediction}, \textbf{Continuous Token Prediction} or a combination of them.

$$
y_i \in \left\{
    \begin{array}{lr}
    V_D, &\text{Discrete Token Prediction}  \\
    V_S, &\text{Continuous Token Prediction}
    \end{array}
\right.
$$

In Fig~\ref{fig:training_obj}, we give an example using the task of text-to-image generation to show the difference between the two training objectives.

### Discrete-Tokens Prediction (DTP)

Discrete token Prediction (DTP) refers learn to predict the next discrete token given the context. The next token could belong to text or different modalities. This approach extends the conventional Causal Language Modeling (CLM), which typically deals with a unimodal text sequence, to accommodate inputs and outputs that interleave text with other data modalities, such as images. DTM enables the model to understand and generate different content from different modalities in a unified way. The training objective is to minimize the average cross-entropy loss among tokens.

Focusing on multimodal understanding ability, a majority of multimodal LLMs (e,g. Flamingo~\citep{madureira-2021-flamingos}, GPT4V~\citep{gpt4v}, MiniGPT4~\citep{zhu2023minigpt4}, Qwen-VL~\citep{QwenVL} and LLaVA~\citep{liu2023llava}) only predict language tokens $V_T$ given multimodal inputs. It leverages the powerful reasoning ability and world knowledge of LLMs to support various multimodal understanding tasks without re-pretraining the model.

Enlarging the output token space to discrete multimodal tokens $V_M$ like quantization codes would enable multimodal generation ability.  In this approach, multimodal contents are first converted into discrete tokens, utilizing cross-entropy loss as the loss function. A major line of works is auto-regressive multimodal information generation, such as DALLE~\cite{ramesh2021zeroshot}, CogView~\cite{CogView}, Unified-IO~\cite{lu2022unifiedio}, LVM~\cite{bai2023sequential} and Video-Poet~\cite{kondratyuk2023videopoet}.

Merging the two output spaces ($V_T$ and $V_M$) into one model is an intriguing direction~\cite{lu2022unifiedio,lu2023unifiedio2,liu2023world}, which naturally unifies multimodal understanding and generation tasks. However, some related research~\cite{zhang2023pretrained} shows that learning to predict text tokens have no benifit for predicting multimdoal tokens and sometimes lead to strong conflict. Under the NTP training framework, whether multimodal generation helps understanding ability also remains unclear. Consequently, effectively integrating the output spaces of text and multimodal tokens presents itself as one of the main challenges in the domain, underscoring the need for innovative and scalable approaches to harness the full potential of NTP models in the realm of multimodal learning.

A variant of standard next token prediction is to predict multiple tokens at one time, disobeying the causal order. Recent researches~\cite{MaskGIT,magvit2,tian2024VAR} have found that parallel prediction is more effective for visual domains such as images and videos than simple raster-based prediction, which predicts the image tokens from left to right and top to down. MaskGIT~\citep{MaskGIT} and MAGVIT~\citep{magvit2} predict a portion of tokens at each prediction step according to a dynamic confidence threshold. VAR~\citep{tian2024VAR} predicts the visual tokens in a resolution-autoregressive manner, which predicts tokens in the same resolution in parallel and predict low-to-high images in sequential. Those approaches inject different inductive bias for different modality during NTP modeling, which is also an important challenge when unifying multiple modalities in multimodal NTP framework.


### Continuous Token Prediction (CTP)

In addition to discrete multimodal tokens, the multimodal information can also be represented as continuous vectors, referred to as Continuous-tokens. The Continuous-tokens can be viewed as conditions for external model such as stable diffusion model for better generation quality. The continuous tokens are usually predicted auto-regressively with MSE loss~\cite{sun2023emu1,sun2023generative,zheng2023minigpt5,koh2023GILL,tang2023codi2}. For example, Emu-1 and Emu-2~\citep{sun2023emu1,sun2023generative} leverage a large language model to generate continuous tokens, which are used as condition for a pretrained diffusion model to generate images. The language model and diffusion model are trained simultaneously during the text-to-image instruction tuning stage. This method utilizes the powerful image generation ability of open-source diffusion model and unlocks the multimodal generation ability of large language model with modest additional cost.

Beyond utilizing continuous tokens as conditions for external models, some researches explored using continuous tokens to directly generate images, replacing discrete tokens with continuous tokens throughout the NTP training paradigm.  \citet{AIM} reveals that when trained with L2 loss, a patch-based image Transformer exhibits scaling properties akin to those of LLMs. \citep{li2024denoising} represents image with continuous tokens and involves diffusion loss during training the causal transformer model. However, these models are trained solely on single modality such as image. Whether different training objectives for different modalities can coexist harmoniously in one NTP model remains under-explored.
