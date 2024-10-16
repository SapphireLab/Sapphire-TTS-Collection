
### 5.4 Speech synthesis for unseen speakers and unseen styles

Building a speech synthesis model that supports multiple speakers or styles can be achieved by training TTS model with a multi-speaker multi-style dataset.
However, generating speech for an unseen speaker or style is a challenging task for which several solutions have been proposed in the literature.
A popular approach is to fine-tune the averaged TTS model with some samples from the unseen target speaker or style.
The fine-tuning process may require a single sample from the unseen speaker or style (referred to as one-shot models) or a few samples(referred to as few-shot models).
There are also models that do not require any fine-tuning steps, and these are known as zero-shot TTS models.

For instance, the fine-tuning process proposed in [112] focused on sentences used in the process to ensure phonetic coverage, meaning that each phoneme should appear at least once in these sentences.
The proposed model requires about 5 minutes of recordings from the unseen target speaker to clone the voice and allow for manipulation of some voice features (such as F0 and duration) by the model at the phoneme-level.

Another approach to address the problem of unseen data is to employ specific structures in the TTS model,as proposed in [52] [96] [97] [107].
As an example, in [107], a cycle consistency network is proposed with two Variational Autoencoders (VAEs).
The model incorporates two training paths: a paired path and an unpaired path.
The unpaired path refers to training scenarios where the reference audio differs from the output (target) speech in terms of text, style, or speaker.
Two separate style encoders are utilized in the model, with one dedicated to each path.
This structure facilitates style transfer among intra-speaker, inter-speaker, and unseen speaker scenarios.

In [52], the U-net structure proposed for the TTS model supports one-shot speech synthesis for unseen styles and speakers.
The U-net structure is used between the style encoder and the mel decoder of the TTS model,with an opposite flow between them.
Both the style encoder and decoder consist of multiple modules with the main building unit as ResCnn1D and instance normalization (IN) layers.
The decoder receives phoneme embedding and produces the Mel-spectrogram as output.
In parallel, the style encoder receives the reference audio and produces its linguistic content with guidance from the content (text) encoder.
The style encoder modules produce latent variables, i.e., mean, and standard deviation, for the hidden inputs in the IN layers.
These latent variables are used to bias and scale the normalized hiddens of the corresponding module layers in the decoder.

A separate encoder (reference encoder) has been used in [96] to extract speaker-related information besides the prosody encoder (extractor) that encodes prosody features into the prosody embedding.
A prosody predictor is also trained to predict the prosody embedding based on the phoneme-embedding.
While the instance normalization (IN) layer is utilized by the prosody extractor to remove global (speaker) information and to keep prosody-related information, the speaker encoder is designed with a special structure (Conv2D layers, residual blocks(GLU with fully connected layers), and a multi-head self-attention unit) for better extraction of speaker information.
Moreover, instead of concatenation or summation with the decoder input, the speaker embedding is adaptively affine transformed to the different FFT blocks of the decoder through a Speaker-Adaptive Linear Modulation (SALM) network that is inspired by Feature-wise Linear Modulation (FiLM) [141].
The speaker encoder and conditioning of decoder blocks with speaker embedding allow the model to generate natural speech for unseen speakers with only a single reference sample(zero-shot).

The attention unit used in seq2seq TTS models aims at mapping the different length between text and audio pairs.
However, it can get unstable when the input is not seen during training [97].
The STYLER model has addressed this issue by using a linear compression or expansion of the audio to match the text’s length via a method named Mel Calibrator.
With this simplification of the alignment process as a scaling method, the unseen data robustness issue is alleviated and all audio-related style factors become dependent only on the audio.

Similarly, in [119], the Householder Normalizing Flow [169] is incorporated into the VAE-based baseline model [77].
The Householder normalizing flow applies a series of easily invertible affine transformations to align the VAE’s latent vectors (style embeddings) with a full covariance Gaussian distribution.
As a result, the correlation among the latent vectors is improved.
Generally, this architecture enhances the disentanglement capability of the baseline model and enables it to generate embedding for unseen style with just a single (one-shot) utterance of around one second length.

The Multi-SpectroGAN TTS model proposed in [98] is a multi-speaker model trained based on adversarial feedback.
The model supports the generation of speech for unseen styles/speakers by introducing adversarial style combination (ASC) during the training process.
Style combinations result from mixing/interpolating style embeddings from different source speakers.
The model is then trained with adversarial feedback using mixed-style mel-spectrograms.
Two mixing methods are employed:binary selection or manifold mix-up via linear combination.
This training strategy enables the model to generate more natural speech for unseen speakers.

Lastly, recent TTS models based on in-context learning ([NaturalSpeech2 (2022)](../../Models/Diffusion/2023.04.18_NaturalSpeech2.md), [VALL-E (2023)](../../Models/Speech_LLM/2023.01.05_VALL-E.md), [Voicebox](../../Models/Speech_LLM/2023.06.23_VoiceBox.md)) all share the capability to perform zero-shot speech synthesis, as explained in Section 4.4.
In fact, the in-context training strategy underlies the ability of these models to synthesize speech given only a style prompt with the input text.
Specifically, the synthesis process treats the provided prompt/reference as part of the desired output speech.
Therefore, the model’s goal is to predict the rest of this speech in the same style as the given part (prompt) and with the input text.
In Table 5 we list papers addressing each challenge.