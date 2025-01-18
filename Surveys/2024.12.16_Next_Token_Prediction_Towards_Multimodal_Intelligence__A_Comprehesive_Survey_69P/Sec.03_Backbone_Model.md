# Backbone Model for Multimodal Next Token Prediction

After multimodal information is tokenized into sequential tokens, we need a model capable of handling multimodal information.
In the literature, two classic MMNTP model structures are depicted in Fig.~\ref{fig:two type of MMNTP models}: 1) the Compositional Model and 2) the Unified Model.
The key distinction lies in their design: the Compositional Model relies on heavily trained external encoders and decoders (such as ~\citep{radford2021clip}), and Diffusion models~\citep{ho2020denoising}, for understanding and generation tasks respectively.
In contrast, the Unified Model features lightweight encoders and decoders, with multimodal understanding and generation tasks primarily occurring within the backbone model, typically a large transformer decoder.
A categorization of current MMNTP models is shown in Table~\ref{table:mmntp_structure_summary}.
We will introduce the general structure of MMNTP model in Section~\ref{sec: general structure}, the recent advances in compostional and unified models in Sections~\ref{sec:comp model} and~\ref{sec:unified model}, and compare them in Section~\ref{sec:comparision}.
