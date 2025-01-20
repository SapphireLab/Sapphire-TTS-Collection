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
    V_D, |\text{Discrete Token Prediction}  |
|    V_S, |\text{Continuous Token Prediction}
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

## Pretraining: Modality Alignment

Large Language Models have demonstrated their effectiveness and scalability in the pure language domain. In a similar vein, pioneering research is exploring the use of the abundant supply of multimodal data in training large multimodal models in NTP framework. The major focus of pretraining in LMM is to align the representation space of different modality with language space, which could be categorized into alignment in understanding (Section~\ref{sec: alignment understand}) and generation (Section~\ref{sec: alignment generation}) task.

### Modality Alignment in Understanding

Modality alignment is a critical process that endeavors to represent inputs from diverse modalities within a shared space for subsequent processing.
Given the inherent differences in the nature of various modalities, dedicated encoders tailored to each modality transform raw inputs into a vector representation, which is then aligned in the shared space.
For instance, the alignment training of vision-language models typically occurs on a large-scale corpus $\mathcal{C} = \{(C, I)\}$ comprising image-text pairs with the image denoted as $I$ and its corresponding caption as $C$.
The modality alignment objective typically adheres to a conditional language modeling format, expressed as:

$$
    L(\theta_\mathcal{M}) = f\left( y_i, p_{\theta}\left(x_i \mid x_{1\sim i-1}, I\right)\right),
$$

where the parameter of the modality encoder module $\theta_\mathcal{M}$—such as a CLIP vision encoder responsible for mapping multi-modal inputs into vectors in the shared space—is exclusively trained to enhance stability.

It is noteworthy that the modality condition $I$ for images can be seamlessly adapted to other modalities, such as videos and audios, with corresponding training corpora like WebVid~\citep{webvid} for video-text alignment and Clotho~\citep{drossos2020clotho} for audio-text alignment, CroCo~\citep{croco} for 3D views and embodiment Habitat~\citep{habitat}.
Besides, it is also possible that the text and the image are interleaved with each other, and the objective can be adjusted accordingly~\citep{awadalla2023openflamingo,laurencon2023obelics}.
We provide a comprehensive list of modality alignment training in the later section~(\S~\ref{subsec:pretrain_dataset}).

### Modality Alignment in Generation

\label{sec: alignment generation}

The alignment objective can be easily adapted to the generative scenarios by replacing the one-hot word index $y_i$ with corresponding modality tokens, which might be learned via a pre-defined codebook or optimized via regression.
Take the traditional text-to-image task as an example, given a description-image pair  $(C, I)$,
the alignment objective becomes:

$$
    L(\theta_\mathcal{M}) = f\left( y_i, p_{\theta}\left(t_i \mid t_{1\sim i-1}, C\right)\right).
$$

In DTM, the $y_i$ could be a targeted discrete visual token learned via an off-shelf model such as VQGAN, and the image content would be reconstructed by mapping the token back to the image space via the codebook.
In CSM, the $y_i$ is instead a contiguous modality vector that can be further decoded by a decoder to produce the image pixels~\citep{sun2024emu}.
Besides, the objective can also be implemented in a span corruption style for a better reconstruction of specific modalities~\citep{lu2022unifiedio}.

Given that a primary objective in the alignment stage is to harmonize the semantics of concepts expressed across different modalities, comprehensive coverage of the training corpus becomes imperative. Consequently, the alignment training is often performed on web-scale datasets. For example, the visual-text alignment is usually conducted on up to millions and even billions of pairs on Laion400M~\citep{laion400m} and Laion5B~\citep{laion5b}.

## Finetuning: Instruction and Preference

After modality alignment training, LMMs acquire a foundational understanding of the semantics associated with various modalities in a unified semantic space.
To further enhance LMMs' ability to comprehend and perform complex user queries, such as image understanding and generation, researchers employ \emph{instruction tuning} on meticulously curated datasets. Subsequently, \emph{preference alignment training} is utilized to refine model behaviors with implicit human preferences and address potential issues that may have emerged during earlier training phases.
In the following discussion, we will discuss recent advancements in instruction tuning (\S\ref{subsec:sft_understanding} and \S\ref{subsec:sft_generation}) and alignment training (\S\ref{subsec:alignment_understanding} and \S\ref{subsec:alignment_gen}), as well as explore promising avenues for future research in these domains.


### Instruction Tuning in Understanding

\label{subsec:sft_understanding}

After the modality alignment training, different modality inputs now can be represented in a unified embedding space for the backbone LLM to perform complex tasks. Instruction tuning~(\emph{alias} supervised fine-tuning) plays a crucial role in activating this potential of multi-modal language models.
Specifically, the instruction tuning aims to improve the model's ability to satisfy user queries.
Again, take the vision language models as an example.
The visual instruction tuning involves training the model on a
dataset that usually consists of a multi-modal triplet $(I, Q, A)$ of an image $I$, a
user query $Q$, and a desired response $A$. This still can be achieved by the previous training object:

$$
    L(\theta) = f\left( A, p_{\theta}\left(x_i \mid x_{1\sim i-1}, I\right)\right).
$$

Different from the previous alignment training, the instruction tuning stage involves a more challenging objective to reason over the modalities, motivating the model to explore the inner interaction between different modalities to increase the likelihood of the preferred answers.
It has been shown that the quality of the instruction tuning is the key to the ability~\citep{liu2023llava15}. Pilot studies explore various methods for constructing high-quality instruction tuning datasets such as
adapting publicly available multi-modal benchmarks~\citep{li2023m3it,visionFlan2023,xu2022multiinstruct}, synthesizing datasets using self-instruction with ChatGPT/GPT-4~\citep{liu2023llava,chen2023sharegpt4v,zhao2023svit,zhao2023chatbridge}. Furthermore, mixing the multi-modal instruction dataset with
text-only query-response pairs is also shown to be effective for improving the instruction following ability~\citep{xu2022multiinstruct,liu2023llava}.
For
A curated list of these instruction tuning dataset can also be found in later section.

### Instruction Tuning in Generation

Similar to the practice in understanding, the key to improving the generation ability after alignment is collecting high-quality and diverse task datasets, where the reconstruction targets vary according to the task requirements.

However, most training objectives still fall into the token modeling paradigm with different tokenization schemas. The desired output such as textual sentences, images/videos and audios, is represented in a sequence of $N$ tokens $S = (s_0, \ldots, s_N)$, given the conditioned user queries $Q$ specifying the requirements on the target outputs. During the instruction tuning stage, the following objective is optimized:

$$
    L(\theta) = f\left( y_i, p_{\theta}\left(s_i \mid s_{1\sim i-1}, Q\right)\right),
$$

where $y_i$ would be the corresponding discrete token or contiguous vector processed as in the alignment training objective.
To provide wide coverage of the generation ability, previous work~\citep{lu2022unifiedio} ensembles a massive multi-tasking dataset and the sampling ratio during training would be balanced to better expose the model to underrepresented tasks. AnyGPT~\citep{Zhan2024AnyGPT}
utilizes commercial image-generation and music-generation systems to construct a large-scale high-quality text-to-multimodal instruction tuning datasets.

### Preference Alignment Training in Understanding

Despite the progress made by previous training stages, misalignment issues that pose a potential risk of generating misleading content without anchoring to the provided visual context~\citep{li2023hallucinate,2023llavarlhf}, or biased responses against minority groups~\citep{gpt4v}, still exist.
To further align with human preference for LMMs, pilot studies draw insights from LLMs and
apply alignment techniques such as Reinforcement Learning with Human Feedback (RLHF)~\citep{RLHF} and Direct Preference Optimization (DPO)~\citep{DPO} for LMMs.

LLaVA-RLHF~\citep{2023llavarlhf} first explores the RLHF for VLM, by training a factuality-oriented reward model on a synthesized dataset to guide the VLM to produce outputs that anchor with the visual context better.
Formally, let $x$ be a prompt containing both images and text inputs, and $y_i$ denotes the corresponding response generated by model $\pi_i$. The RLHF process can be formulated as:

\begin{equation*}
    \max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}\left[r(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right],
\end{equation*}
where $r$ is the reward model and the KL term penalizes deviations of the current model $\pi_{\theta}$ from the initial model $\pi_{\mathrm{ref}}$. $\beta$ is a hyper-parameter. The RLHF process aims to finetune the model to achieve higher rewards from the reward model, all while preserving the majority of its original knowledge.


As training the reward model can be difficult due to the stability issue, there has been a DPO method to tackle these challenges.
The key insight behind DPO is that the optimal policy $\pi^*$ has a closed-form solution with regard to a reward function $r$ and initial policy $\pi_{\mathrm{ref}}$:
\begin{equation*}
    r(x, y)=\beta \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x),
\end{equation*}
where $Z$ is the partition function.

Under the  Bradley-Terry (BT) preference model~\citep{Bradley1952RankAO}, the objective becomes:
$$
\label{eq:dpo}
    \max _{\pi_\theta} \mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}} \log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right),
$$
where $\sigma$ denotes the sigmoid function.
RLHF-V\citep{Yu2023RLHFVTT}  collects human preference in the form of segment-level corrections on hallucinations, and performs dense direct preference optimization
over the human feedback.
\citet{2023vlfeedback} build VLFeedback by annotating the preference with GPT-4V models and applies DPO on Qwen-VL-Chat showing clear advantages.

### Preference Alignment Training in Generation

Due to the computation cost and the difficulty of collecting large-scale comparison datasets (i.e., creating slightly different images), there are few explorations on preference alignment in generative unified multimodal models.
There are pilot studies investigating preference alignment for diffusion models, where the expected reward of a generated sequence $\boldsymbol{x}_{1:T}$ given a condition $\boldsymbol{c}$ and initial latent $\boldsymbol{x}_0$ is:
$$
r\left(\boldsymbol{c}, \boldsymbol{x}_0\right)=\mathbb{E}_{p_\theta\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_0, \boldsymbol{c}\right)}\left[R\left(\boldsymbol{c}, \boldsymbol{x}_{0: T}\right)\right]
$$

Similar to the alignment training in understanding tasks, the objective is to maximize the expected reward while minimizing the KL divergence between the learned distribution $p_\theta$ and a reference distribution $p_\text{ref}$:
$$
\begin{aligned}
|\max _{p_\theta} \mathbb{E}_{\boldsymbol{c} \sim \mathcal{D}_c, \boldsymbol{x}_{0: T} \sim p_\theta\left(\boldsymbol{x}_{0: T} \mid \boldsymbol{c}\right)}\left[r\left(\boldsymbol{c}, \boldsymbol{x}_0\right)\right] |
||-\beta \mathbb{D}_{\mathrm{KL}}\left[p_\theta\left(\boldsymbol{x}_{0: T} \mid \boldsymbol{c}\right) \| p_{\mathrm{ref}}\left(\boldsymbol{x}_{0: T} \mid \boldsymbol{c}\right)\right]
\end{aligned}
$$
Current methods for aligning image generative models mainly adopt DPO to bypass the cumbersome reward modeling process.
\citet{diffusion_dpo1} re-formulate DPO to account for the intractable likelihood in diffusion models, where the evidence lower bound (ELBO) is employed to derive a differentiable objective function for optimization.
The final DPO-Diffusion loss function encourages the model to improve the denoising process for preferred images more than for non-preferred images.

$$
\begin{aligned}
L_{\text {DPO-Diffusion }}(\theta) |= -\mathbb{E}_{\left(\boldsymbol{x}_0^w, \boldsymbol{x}_0^l\right) \sim \mathcal{D}, t \sim \mathcal{U}(0, T),
\boldsymbol{x}_{t-1, t}^w \sim p_\theta\left(\boldsymbol{x}_{t-1, t}^w \mid \boldsymbol{x}_0^w\right),
\boldsymbol{x}_{t-1, t}^l \sim p_\theta\left(\boldsymbol{x}_{t-1, t}^l \mid \boldsymbol{x}_0^l\right)} |
||\log \sigma\left(\beta T \log \frac{p_\theta\left(\boldsymbol{x}_{t-1}^w \mid \boldsymbol{x}_t^w\right)}{p_{\mathrm{ref}}\left(\boldsymbol{x}_{t-1}^w \mid \boldsymbol{x}_t^w\right)}-\beta T \log \frac{p_\theta\left(\boldsymbol{x}_{t-1}^l \mid \boldsymbol{x}_t^l\right)}{p_{\mathrm{ref}}\left(\boldsymbol{x}_{t-1}^l \mid \boldsymbol{x}_t^l\right)}\right),
\end{aligned}
$$
where condition $\mathbf{c}$ is omitted for brevity.
The models are trained  on the Pick-a-Pic~\citep{kirstain2024pick} dataset, which contains pairwise preferences for images generated by SDXL-beta and
Dreamlike, a fine-tuned version of Stable Diffusion 1.5.

D3PO~\citep{yang2023d3po} instead treats diffusion generation as the multi-step decision problem. Under mild assumptions, the model is trained by the preference objective at the image segment level.
The human annotators are asked about the final image quality and D3PO assumes that any state-action
pair of the preferred image is better than that of the rejected image.

## Inference: Enhancing Multimodal Task Performance via Prompt Engineering

After the pretraining and finetuning stages, MMNTP models can also benefit from prompt engineering techniques, much like LLMs. Stemming from research in prompt engineering \citep{vatsal2024surveypromptengineeringmethods}, In-Context Learning (ICL) \citep{icl_survey} and Chain-of-Thought reasoning (CoT) \citep{wei2022chain} are key methods that significantly enhance the performance of LLMs on complex tasks, such as mathematical reasoning \citep{cobbe2021trainingverifierssolvemath}.

Although prompt engineering techniques have had huge success in LLMs~\citep{vatsal2024surveypromptengineeringmethods}, their application in multimodal remains largely underexplored so far. Table~\ref{table:multimodal_ICL_summary} lists the related work on multimodal ICL and CoT research.

### Multimodal In-Context Learning

| MultiModal ICL Method |Year |Modality |Backbone Model |Task |
| --- | --- | ---| --- | --- |
|Frozen~\cite{tsimpoukelli2021multimodal}  |2021 |Image |GPT2 Architecture~\citep{gpt2} | Understanding  |
|Flamingo~\cite{alayrac2022flamingo}  |2022 |Image  |GPT2 Architecture~\citep{gpt2} | Understanding  |
|MMICL~\citep{zhao2023mmicl} |2023 |Image  |InstructBLIP~\citep{dai2023instructblip} |Understanding|
|EILeV~\cite{yu2023efficient} |2023 |Image  |- | Understanding |
|Open-Flamingo~\cite{awadalla2023openflamingo}  |2023 |Image  |Flamingo Architecture~\cite{alayrac2022flamingo} | Understanding |
|LCL~\cite{tai2023linkcontext} |2023 | Image |Otter~\citep{li2023mimicit}, OpenFlamingo~\cite{awadalla2023openflamingo} | Understanding |
|Med-Flamingo~\cite{moor2023medflamingo}  |2023 |Image  |Open-Flamingo~\cite{awadalla2023openflamingo} | Understanding |
|MIMIC-IT~\cite{li2023mimicit}  |2023 |Image |OpenFlamingo~\cite{awadalla2023openflamingo} | Understanding |
|LVM~\cite{bai2023sequential} |2023 |Image |LLaMA Architecture~\cite{touvron2023llama} |Understanding\|Generation  |
|LWM~\cite{liu2023world}  |2023 |Image, Video |LLaMA Architecture~\cite{touvron2023llama} |Understanding \|Generation |
|~\citet{yang2024exploring} |2024 |Image  |Open-Flamingo~\cite{awadalla2023openflamingo}  | Understanding |
|VisualICL~\cite{zhou2024visual} |2024 |Image  |LLaVA~\citep{liu2023llava} |Understanding |
|Many-Shots ICL~\citep{jiang2024manyshotincontextlearningmultimodal} |2024 |Image |GPT4-o~\citep{gpt4o}, Gemini1.5~\citep{team2024gemini} |Understanding |
|CoBSAT~\cite{zeng2024mllms}  |2024 |Image |Emu~\citep{sun2023emu1} |Generation |
|Video ICL~\citep{zhang2024videoincontextlearning} |2024 |Video |LLaMA Architecture~\citep{touvron2023llama} |Generation |
|Emu~\cite{sun2024emu}  |2024 | Image, Video |LLaMA~\cite{touvron2023llama} |Understanding \|Generation |
|Emu2~\cite{sun2024generative} |2024 |Image, Video | LLaMA-33B~\cite{touvron2023llama} |Understanding \|Generation  |
|Yang et al.~\citep{sheng2024unified} |2024 |Image |GPT2 Architecture~\citep{gpt2} |Understanding \|Generation  |
|VALL-E~\cite{wang2023neural} |2023 |Audio |-|Generation |
|MELLE~\cite{meng2024autoregressive} |2024 |Audio |-|Generation|
|Seed-TTS~\cite{anastassiou2024seed} |2024 |Audio |-|Generation|
|Audio Flamingo~\cite{kong2024audio} |2024 |Audio |OPT-IML-MAX-1.3B~\cite{iyer2022opt}|Understanding|
|Moshi~\cite{defossez2024moshi} |2024 |Audio |Helium~\cite{defossez2024moshi}|Understanding \|Generation |

Multimodal In-Context Learning (ICL) is an emerging paradigm in which models leverage a few demonstration examples incorporating visual, textual, and other optional modalities to perform multimodal tasks. In this learning paradigm, the input processed by the Large Multimodal Model is divided into two components: the query \( x_q \) and the context \( C \). The LMM needs to generate a sequence of tokens as outputs \(y_q\) based on these two parts:
$$
y_q = LMM(x_q, C)
$$
The context \( C \) consists of a set of input-output ICL examples:
$$
C = \{(x_i, y_i)\}_{i=1}^n
$$
Adopting the notation from Todd et al. \cite{todd2024function}, we represent the generic template for organizing the context \( C \) as follows:
$$
 Q:\{x_1\}\textbackslash n \textbf{ }A:\{y_1\} \textbackslash n \textbackslash n \textbf{ }\dots Q:\{x_n\} \textbackslash n \textbf{ }A:\{ y_n\},
$$
where \( Q \) and \( A \) symbolize the question and answer template structures respectively, and \( x_i \) and \( y_i \) denote the question and answer of the \( i \)-th demonstration respectively.

Multimodal ICL introduces unique challenges compared to unimodal ICL, particularly in integrating and aligning diverse modalities such as text, images, and videos~\citep{shukor2023beyond}~\citep{zhao2023mmicl}\citep{baldassini2024makes}. In multimodal ICL, both the query \( x_q \) and context \( x_i \) may vary in modality, conveying complementary yet distinct information that can lead to imbalanced or inefficient learning. A primary challenge, as noted in recent studies \citep{awadalla2023openflamingo} \citep{baldassini2024makes}, is that performance in many multimodal ICL systems remains largely text-driven, with other modalities—such as images or videos—contributing minimally to overall task performance.

To address this challenge, several approaches~\citep{awadalla2023openflamingo,yu2023efficient,yu2023efficient,zhao2023mmicl} focus on enhancing the model's ability to generalize across diverse multimodal tasks. EILEV~\citep{yu2023efficient} proposes new training methods for video understanding. MMICL~\citep{zhao2023mmicl} and CoBSAT~\citep{zeng2024mllms} use specialized datasets and prompt engineering to enhance multimodal reasoning. Recent work further extends these efforts by exploring large-scale models for more effective in-context learning with interleaved multimodal inputs, ~\citep{EMU2,liu2023world,EMU2,laurenccon2024obelics}.

### Multimodal Chain-of-Thought Prompting

| MultiModal CoT Method |Year |Modality |Backbone Model |Task |
| --- | --- | ---| --- | --- |
| MM-CoT~\citep{zhang2023multimodal}  | 2023 | Image | T5-770M~\citep{raffel2020exploring} |  Understanding  |
|DDCoT~\citep{zheng2023ddcot}  | 2023 | Image | ChatGPT~\citep{ouyang2022training}/GPT-3~\citep{gpt3} |  Understanding  |
|VCDM~\citep{harvey2023visual}  | 2023 | Image |  Stable Diffusion~\citep{diffusion_dpo1} |   Generation   |
|V*~\citep{wu2024v}  | 2024 | Image | Vicuna-7B~\citep{vicuna2023} |  Understanding  |
|CogCoM~\citep{qi2024cogcom}  | 2024 | Image | Vicuna-7B~\citep{vicuna2023} |  Understanding  |
|VisualCoT~\citep{shao2024visual}  | 2024 | Image | Vicuna-7B/13B~\citep{vicuna2023} |  Understanding  |
|CCoT~\citep{Mitra_2024_CVPR}  | 2024 | Image | - |  Understanding  |
|VideoCoT~\citep{wang2024videocot}  | 2024 | Video | - |  Understanding  |
|VoT~\citep{fei2024videoofthought}  | 2024 | Video | Vicuna-7B~\citep{vicuna2023} |  Understanding  |
|WavLLM~\citep{hu2024wavllm}  | 2024 | Audio | LLaMA Architecture~\citep{touvron2023llama2} | Understanding|
|SpeechVerse~\citep{das2024speechverse}  | 2024 | Audio |  Flan-T5-XL~\citep{JMLR:v25:23-0870} | Understanding|
|CoT-ST~\citep{du2024cot}  | 2024 | Audio | - |  Understanding|
|AST-CoT~\citep{hu2024chain11}  | 2024 | Audio | T5~\citep{raffel2020exploring} |  Understanding|

Multimodal Chain-of-Thought (CoT) is a method that enables models to perform complex reasoning and decision-making in a multimodal setting through step-by-step derivation and coherent thinking. Pioneered by~\citet{zhang2023multimodal}, MM-CoT introduces Chain-of-Thought prompting into visual domains, raising the challenge of labor-intensive annotation, as multimodal data often demands expensive and complex human-labeled information. MM-CoT employs ScienceQA~\citep{lu2022scienceqa}, a dataset focused on scientific questions involving multiple modalities with annotated rationales, while VoT~\citep{fei2024videoofthought} tackles the annotation challenge in video tasks by combining machine and human expertise through active learning.

Another challenge lies in mitigating language hallucinations~\citep{alayrac2022flamingo,ji2023survey,maynez2020faithfulness,rawte2023survey,zhang2023sirenssongaiocean,chen2024pcabench,zhao2024lookingtextreducinglanguage}, which are exacerbated due to the lack of necessary and fine-grained visual context when multimodal information is provided simultaneously. To better inject visual information, V*~\citep{wu2024v} addresses this by dynamically focusing on key visual regions, ensuring that visual details are accurately attended to, particularly in high-resolution images. CCoT~\citep{Mitra_2024_CVPR} generates scene graphs instead of simple captions, explicitly reasoning over visual features to avoid misinterpretation. Moreover, DDCoT ~\citep{zheng2023ddcot} introduces a new CoT prompting method that divides the roles of reasoning and visual recognition between language and visual models, thereby enhancing reasoning clarity and reducing hallucinations.

Subsequent work~\citep{wang2024videocot}~\citep{fei2024videoofthought}~\citep{du2024cot}~\citep{raffel2020exploring} has extended the method beyond images to include video and audio. For instance, the CoT-ST~\citep{du2024cot}framework adapts chain-of-thought reasoning for speech translation, breaking the process into distinct steps to improve accuracy and fluency. Video-CoT~\citep{wang2024videocot}  focus on complex video reasoning, aiming to achieve human-level video comprehension.