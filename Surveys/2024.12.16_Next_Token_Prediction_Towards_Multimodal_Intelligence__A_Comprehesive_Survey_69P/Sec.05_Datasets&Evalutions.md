# 5·Datasets | Evaluation

In this section, we delve into several crucial aspects of training and evaluating MMNTP models. The subdivision begins with an exploration of the training datasets (Section~\ref{sec: training_dataset}), categorized into pre-training and fine-tuning datasets. The pre-training datasets are further divided based on modality into text-only, image-based, video-based, and audio-based data, which are essential for modality alignment and the establishment of a unified multimodal representation. Following this, fine-tuning datasets are described, focusing on their specific applications in multimodal understanding and multimodal generation tasks.

Additionally, we discuss the evaluation of MMMNTP models (Section~\ref{sec: evaluation_dataset}), which is pivotal in measuring their effectiveness and capability across various modalities. This aspect is divided into holistic evaluation and emerging evaluation benchmarks. Holistic evaluation benchmarks, such as MME~\citep{fu2023mme} and SEED-Bench~\citep{li2023seedbench}, comprehensively assess the integration and interplay between different modalities like image, text, and video. Emergent benchmarks, including SparklesEval~\citep{huang2023sparkles} and HallusionBench~\citep{guan2024hallusionbench}, push the boundaries further by testing specialized capabilities like conversational competence, mathematical reasoning, and mitigation of hallucinations in model outputs.

## Training Datasets

Depending on the stage of training, we categorize data into pre-training data and fine-tuning data. Pre-training data can be classified into uni-modal data and multimodal data based on modality. Fine-tuning data is categorized based on its usage scenario into multimodal understanding data and multimodal generation data.

### Pre-training Datasets

Unlike large language models that are pre-trained only on pure text data, multimodal models require pre-training on a variety of different modalities of data, which demands a significant quantity and diversity of multimodal data. In this section, we briefly summarize several multimodal datasets widely used for training multimodal models. Based on the type of modality, we categorize these data into four groups: Text-Only, Image-Based, Video-Based, and Audio-Based.

**Text-Only**

Although pure text data is commonly utilized in language models, it also plays a crucial role in enhancing the language expression and reasoning abilities of multimodal models. For this purpose, pure text data is integrated into the pre-training corpus. One of the most extensively used datasets in this context is C4~\cite{habernal2016c4corpus}, a filtered open-source dataset derived from web crawls. Its multilingual variant, mC4~\cite{xue2021mt5}, encompasses natural text in 101 languages, sourced from the public Common Crawl web archive. Additionally, the Wikipedia dataset~\cite{guo2020wiki}, which consists of cleaned articles in multiple languages, is created from language-specific segments of the Wikipedia dump. Another significant contribution to this field is The Pile, an expansive and diverse open-source dataset for language modeling. Amassing a total of 825 GiB, The Pile~\cite{gao2020pile} is an amalgamation of 22 distinct, high-quality smaller datasets, providing a rich resource for language model pretraining. Recently, RedPajama~\cite{together2023redpajama}, an open dataset with 30 trillion tokens for training large language models, has also been introduced, contributing significantly to the resources available for developing advanced language models. Furthermore, FineWeb~\cite{penedo2024finewebdatasetsdecantingweb} release a new, large-scale (15-trillion tokens) dataset for LLM pretraining. FineWeb is derived from 96 CommonCrawl snapshots and produces better-performing LLMs. Dolma~\citep{soldaini2024dolmaopencorpustrillion} is a high-quality open dataset from a diverse mix of web content, academic publications, code, books, and encyclopedic materials, covering 3T tokens.

**Image-Based**

Multimodal data is key for models to perform modality alignment, that is, to map different modal representations into a unified space. CLIP~\citep{radford2021clip} was developed using 400 million image-text pairs sourced from the internet. Subsequent models like ALIGN~\citep{Jia2021ALIGN}, BASIC~\citep{pham2023combined}, and Florence~\citep{yuan2021florence} were trained on even larger and more diverse datasets with noisier image-text pairs. However, the majority of these extensive datasets remain inaccessible to the public. In the academic community, researchers recommend using several million image-text pairs for multimodal model pre-training, including CC12M~\citep{changpinyo2021conceptual}, RedCaps~\citep{desai2021redcaps}, YFCC~\citep{thomee2016yfcc100m}, WIT~\citep{srinivasan2021wit}, and Capsfusion~\cite{yu2023capsfusion}. Publicly accessible datasets of a relatively smaller scale include SBU~\cite{ordonez2011im2text}, MSCOCO~\cite{lin2014microsoft}, VG~\cite{krishna2017visual_genome}, and CC3M~\citep{sharma2018conceptual}. Among the larger-scale image-text datasets available to the public are FILIP~\citep{filip}, LAION-400M~\citep{laion400m}, COYO-700M~\citep{kakaobrain2022coyo-700m}, SA-1B~\citep{wang2023all}, and LAION-5B~\citep{laion5b}, among others. Additionally, some studies have emphasized the importance of data quality in building robust multimodal models, such as DataComp~\citep{gadre2023datacomp}, Shutterstock~\citep{nguyen2022quality}, and ShareGPT4V~\citep{chen2023sharegpt4v}.
Beyond sourcing image-text data from the web, there has been a growing interest in compiling datasets that interleave images and text, a concept pioneered by the M3W~\citep{alayrac2022flamingo} dataset featured in Flamingo~\citep{alayrac2022flamingo}. Notable examples of such datasets are MMC4~\citep{zhu2023multimodal} and OBELISC~\cite{laurenccon2023obelisc}. Additionally, there's an emerging trend in research to focus on the extraction and association of text segments in captions with specific areas in images, leading to the formation of grounded image-text pairs. Datasets like GRIT-20M~\citep{peng2023kosmos} and CapsFusion-grounded~\citep{sun2023generative} exemplify this methodology.

**Video-Based**

MSR-VTT~\citep{xu2016msrvtt} features 10K diverse web video clips and 200K clip-sentence pairs spanning a wide range of categories. HowTo100M~\citep{miech2019howto100m} expands this landscape with 1.22 million YouTube videos on topics like cooking and crafting, enriched with subtitles from ASR systems or manual input. ACAV100M~\citep{lee2021acav100m} provides a vast 100 million video library, ideal for self-supervised learning with high audio-visual correspondence. WebVid~\citep{webvid} enhances video data with manually crafted, accurate captions. Ego4D~\citep{grauman2022ego4d} offers an extensive collection of diverse egocentric video footage for research. HD-VILA~\citep{xue2022advancing} introduces a high-resolution video-language dataset with varied content. YT-Temporal~\cite{zellers2022merlot}, sourced from public YouTube videos, focuses on broadening understanding of objects, actions, and scenes. VideoCC3M~\cite{nagrani2022learning} utilizes a new pipeline to transfer image captions to videos without extra manual labor. Youku-mPLUG~\citep{xu2023youku} has released the largest public Chinese video-language dataset, prioritizing safety, diversity, and quality. Most recently, InternVid~\citep{wang2023internvid} demonstrates a scalable method for building high-quality video-text datasets using large language models, effectively enhancing video language representation learning.

**Audio-Based**

Audio-based pretraining datasets can be primarily categorized into three types: speech pretraining datasets, music pretraining datasets, and general audio pretraining datasets.
Librilight~\cite{kahn2020libri} includes more than 60k hours unlabeled speech data and is widely used by audio pretraining~\cite{wang2023neural,zhang2024speechlm}.
Libriheavy~\cite{kang2024libriheavy} introduces a refined pipeline for audio alignment and segmentation and detailed annotations with punctuation and capitalization, reflecting more natural speech patterns, to the mostly unlabeled Librilight.
Wenetspeech~\cite{zhang2022wenetspeech} is the largest Mandarin speech pretraining corpus, collecting over 22,400 hours of audio, with 10,000+ hours of high-quality labeled speech, 2,400+ hours of weakly labeled speech, and roughly 10,000 hours of unlabeled speech from diverse sources such as YouTube and podcasts.
Yodas~\cite{li2023yodas} offer over 500,000 hours of speech data in more than 100 languages, significantly benefiting the multilingual nature of the audio pretrain community.
Other widely-used speech pretraining datasets include librispeech~\cite{panayotov2015librispeech}, libritts~\cite{zen2019libritts} and gigaspeech~\cite{chen2021gigaspeech}.
Music pretraining is a growing research area~\cite{dhariwal2020jukebox,zhu2021musicbert,li2022map,li2023mert,lu2023musecoco,hussain2023m,qu2024mupt}.
Million Song Dataset (MSD)~\cite{bertin2011million} is one of the largest publicly available collections of audio features and metadata for a million contemporary popular music tracks.
FMA (Free Music Archive) Dataset~\cite{defferrard2016fma} is a well-curated collection of over 100,000 tracks from various artists and genres available under Creative Commons licenses.
Other widely-used music pretraining datasets include disco10m~\cite{lanzendorfer2024disco}, mtg-jamendo~\cite{bogdanov2019mtg}, and Lp-musiccaps~\cite{doh2023lp}.
General audio pretraining datasets, including wavcaps~\cite{mei2023wavcaps}, audioset~\cite{gemmeke2017audio}, vggsound~\cite{chen2020vggsound}, and clotho~\citep{drossos2020clotho}, mainly focus on boosting the performance of localizing audio-visual correspondence and audio-text intermodal translation tasks (not speech-to-text).

| Datasets | Tags | Doc/Img/Vid/Aud | Source | Time |
| --- | --- | --- | --- | --- |
|C4~\cite{habernal2016c4corpus} | Text-Only | 8.2M/-/-/- | CommonCrawl | Apr-2019  |
|mC4~\cite{xue2021mt5}| Text-Only | 2.1M/-/-/- | CommonCrawl | Oct-2020 |
|Pile~\cite{gao2020pile} | Text-Only  | 211M/-/-/- | Other | Dec-2020 |
|Wikipedia~\cite{guo2020wiki} |  Text-Only  | 13.4M/-/-/- | Wikipedia | Mar-2023  |
|RedPajama~\cite{together2023redpajama} | Text-Only | 100B/-/-/- | CommonCrawl  | Oct-2023 |
|Dolma~\citep{soldaini2024dolmaopencorpustrillion} |  Text-Only | 4.4B/-/-/- | Common Crawl, GitHub, Reddit, ...  | Jan-2024 |
|FineWeb~\cite{penedo2024finewebdatasetsdecantingweb} | Text-Only | 22.7B/-/-/- | CommonCrawl | May-2024 |
SBU~\cite{ordonez2011im2text}  | Image-Based   | 1M/1M/-/-  | Flickr | Dec-2011  |
|YFCC~\cite{thomee2016yfcc100m}  | Image-Based   |  100M/99.2M/0.8M/-  | Flickr | Jan-2016  |
|MS-COCO~\cite{lin2014mscoco}   | Image-Based   | 1M/200K/-/-  | HumanCurated | Jul-2018 |
|VG~\cite{krishna2017visual_genome}  | Image-Based | 5.4M/108K/-/- | HumanCurated | Feb-2016 |
|CC3M~\cite{sharma2018conceptual}  | Image-Based   | 3.3M/3.3M/-/- | Web Crawl | Jul-2018 |
|CC12M~\cite{changpinyo2021conceptual} | Image-Based | 12M/12M/-/-  | Web Crawl | Feb-2021 |
|WIT~\cite{srinivasan2021wit}  | Image-Based  |  37.6M/11.5M/-/- | Wikipedia | Jul-2021 |
|RedCaps~\cite{desai2021redcaps}  | Image-Based  | 12M/12M/-/- | Reddit links | Nov-2021 |
|FILIP300M~\cite{yao2021filip} |	Image-Based | 300M/300M/-/-	| Web Crawl | Nov-2021 |
|LAION-400M~\cite{laion400m}  | Image-Based  | 400M/400M/-/- |  CommonCrawl  | Nov-2021 |
|Shutterstock~\cite{nguyen2022quality}  | Image-Based  | 15M/15M/-/-| Shutterstock  | Aug-2022|
|Coyo-700M~\cite{kakaobrain2022coyo-700m}  | Image-Based  | 747M/747M/-/- | CommonCrawl  | Aug-2022 |
|Laion-5B~\cite{laion5b}  | Image-Based   | 5B/5B/-/-  | CommonCrawl | Oct-2022 |
|DataComp~\cite{gadre2023datacomp}  | Image-Based   | 1.4B/1.4B/-/- | Web Crawl | Apr-2023 |
|SA-1B~\cite{wang2023all}  | Image-Based   | 1.1B/11M/-/-  | Photo Company | Aug-2023  |
|Capsfusion~\cite{yu2023capsfusion}  | Image-Based | 120M/120M/-/- | Other | Oct-2023 |
|ShareGPT4V~\cite{chen2023sharegpt4v}  | Image-Based  | 1.2M/1.2M/-/- | Other | Nov-2023 |
|M3W~\cite{alayrac2022flamingo}  | Image-Based (Interleaved)   | -/185M/-/- | Web Crawl | Apr-2022|
|MMC4~\cite{zhu2023multimodal}  | Image-Based (Interleaved)  | 103M/585M/-/- | Other | Apr-2023 |
|Obelisc~\cite{laurenccon2023obelisc}  | Image-Based  (Interleaved)  | 141M/353M/-/- | Web Crawl| Jun-2023 |
|GRIT-20M~\cite{peng2023kosmos} |  Image-Based (Grounded)  | 20M/20/-/- | Other | Jun-2023 |
|CapsFusion-grounded~\cite{sun2023generative} | Image-Based  (Grounded)  | 100M/100M/-/- | Other | Dec-2023 |
|MSR-VTT~\cite{xu2016msr}  |  Video-Based  | 200K/-/10K/-  | HumanCurated | Jun-2016 |
|HowTo100M~\cite{miech2019howto100m} |  Video-Based  |  136M/-/1.2M/-  | Youtube | Jun-2019 |
|ACAV~\cite{lee2021acav100m}  |  Video-Based  | -/-/100M/100M | Web Crawl | Jan-2021  |
|WebVid~\cite{webvid}  |  Video-Based  |  10M/-/10M/-  | Stock Footage | Jan-2021 |
|Ego4D~\cite{grauman2022ego4d}  |  Video-Based  | -/-/-/-  | HumanCurated | Oct-2021  |
|HD-VILA~\cite{xue2022advancing}  |  Video-Based  |  100M/-/3.3M/-  | YouTube | Nov-2021 |
|YT-Temporal~\cite{zellers2022merlot}  |  Video-Based  | 1B/-/20M/-  | YouTube | Jan-2022 |
|VideoCC3M~\cite{nagrani2022learning} | Video-Based  | 10.3M/-/6.3M/- | Other | Apr-2022 |
|Youku-mPLUG~\cite{xu2023youku} |  Video-Based |  10M/-/10M/-  | Youku | Jun-2023  |
|InternVid~\cite{wang2023internvid}  |  Video-Based  |  234M/-/7.1M/-  | YouTube  | Jul-2023 |
|Million Song Dataset~\cite{bertin2011million} | Audio-Based | -/-/-/1M | The Echo Nest | Feb 2011|
|MTT~\cite{law2009evaluation} | Audio-Based | -/-/-/25.8k | Web Crawl | June 2013|
|LibriSpeech~\cite{panayotov2015librispeech} | Audio-Based | 155.8k/-/-/1k hours | Audio Books | Jun 2015|
|FMA~\cite{defferrard2016fma} | Audio-Based | -/-/-/106k | Free Music Archive | Dec 2016|
|Audio Set~\cite{gemmeke2017audio}  | Audio-Based | 2.1M/-/-/2.1M | YouTube  | Mar-2017|
|LibriTTS~\cite{zen2019libritts} | Audio-Based | 2.4k/-/-/0.58k hours | Audio Books | Apr 2019|
|MTG-Jamendo~\cite{bogdanov2019mtg} | Audio-Based | -/-/-/55k | Jamendo | Jun 2019|
|Clotho~\cite{drossos2020clotho} | Audio-Based | 25k/-/-/5k | FreeSound Platform | Oct 2019|
|Librilight~\cite{kahn2020libri} | Audio-Based | -/-/-/60k hours | Audio Books | Dec 2019|
|VGGSound~\cite{chen2020vggsound} | Video-Based | 309/-/200k/200k | Web Crawl | Apr 2020|
|Gigaspeech~\cite{chen2021gigaspeech} | Audio-Based | -/-/-/40k hours | Audio Books | Jun 2021|
|LAION-Audio-630k~\cite{wu2023large} | Audio-Based | 630k/-/-/630k | Web Crawl | Nov 2021|
|wenetspeech~\cite{zhang2022wenetspeech} | Audio-Based | -/-/-/22.4k hours | Youtube | Feb 2022|
|WavCaps~\cite{mei2023wavcaps} | Audio-Based | 400k/-/-/400k | Other | Mar-2023|
|LP-MusicCaps~\cite{doh2023lp} | Audio-Based | 2.2M/-/-/0.5M | Web Crawl | Jul 2023|
|LibriHeavy~\cite{kang2024libriheavy} | Audio-Based | 9M/-/-/50k hours | Audio Books | Sep 2023|
|disco-10m~\cite{lanzendorfer2024disco} | Audio-Based | -/-/-/15.2M | Youtube | 2023|
|yodas~\cite{li2023yodas} | Audio-Based | -/-/-/500k hours | Youtube | Dec 2023|

### Fine-tuning Datasets

**Multimodal Understanding**

The inaugural work in applying instruction tuning to the multi-modal domain was presented by MultiInstruct~\citep{xu2022multiinstruct}, which successfully combined multi-modal learning into a single-format benchmark dataset incorporating 62 diverse tasks. Concurrently, LLaVA~\citep{liu2023llava} harnessed the capabilities of the language-centric GPT-4 to generate datasets for multi-modal, instruction-based tasks involving both text and images. MiniGPT-4~\citep{zhu2023minigpt4} precisely assembled a dataset rich in detailed image descriptions to facilitate the convergence of visual and linguistic elements.

Further advancements were marked by LMeye~\citep{li2023lmeye}, MMEvol~\citep{luo2024mmevol}, PF-1M~\citep{chen2023visual}, and SVIT~\citep{zhao2023svit}, which scaled up the magnitude of instruction tuning. The domains of video content were also explored by Video-Chat~\citep{li2023videochat} and Video-ChatGPT~\citep{maaz2023video}, which adapted instruction tuning to this dynamic format. In the specialized medical sector, PMC-VQA~\citep{zhang2023pmc} and LLaVA-Med~\citep{li2024llava} crafted datasets for instruction tuning by leveraging existing medical data repositories.
Object detection tasks were ingeniously integrated into instruction tuning through the efforts of DetGPT~\citep{pi2023detgpt} and MGVLID~\citep{zhao2023chatspot}. GPT4Tools~\citep{yang2024gpt4tools} was developed to enhance open-source large language models (LLMs) by equipping them with the versatility to utilize an array of tools effectively, while M$^3$IT expanded the reach of multi-modal instruction tuning across multiple languages.
Expanding the horizon further, X-LLM~\citep{chen2023x}, MIMIC-IT~\citep{li2023mimicit}, MotionGPT~\citep{jiang2024motiongpt}, Macaw-LLM~\citep{lyu2023macaw}, and BuboGPT~\citep{zhao2023bubogpt} ventured into new modalities, enhancing the scope of instruction tuning. The integration of 3D tasks into this domain was initiated by LAMM~\citep{yin2024lamm} and M3DBench~\citep{li2023m3dbench}, enriching the complexity and applicability of instruction tuning.
Meanwhile, LLaVAR~\citep{zhang2023llavar} leveraged publicly accessible OCR tools to harvest text-rich images from the LAION~\citep{laion400m} dataset, thus enhancing visual instruction tuning processes. To address the phenomenon of hallucinations, HalDetect~\citep{gunjal2023detecting} developed a pioneering multi-modal dataset focused on accurate image descriptions. In the pursuit of robustness, GAVIE~\citep{liu2023mitigating} introduced a mix of positive and negative instructions, fortifying the training for visual instruction tuning.
StableLLaVA~\citep{li2023stablellava} combined the generative prowess of ChatGPT with text-to-image models to produce a versatile and diversified dataset featuring a wide range of image content. Sparkles~\citep{huang2023sparkles} introduced the first machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions. The project LVIS-INSTRUCT4V~\citep{wang2023see} capitalized on the improved visual processing strengths of GPT-4 to achieve a higher precision in image detail capture and instruction annotation accuracy.

**Multimodal Generation**

Additionally, some instruction-based image editing datasets focus on image generation. A typical dataset is InstructPix2Pix~\citep{brooks2023instructpix2pix}, which initially uses GPT-3~\citep{gpt3} to generate the text for edits, and then utilizes Stable Diffusion~\citep{Rombach_Blattmann_Lorenz_Esser_Ommer_2022} along with Prompt2Prompt~\citep{Hertz2022PrompttoPromptIE} technology to generate the corresponding edited images to construct the dataset. Furthermore, HIVE~\citep{zhang2023hive} introduces a larger number of training triplets and incorporates human ranking results, providing stronger supervision signals for more effective model training. Building on these advancements, MagicBrush~\citep{zhang2024magicbrush} introduces the first large-scale, manually annotated dataset specifically designed for instruction-guided real image editing. Expanding further, HQ-Edit~\citep{hui2024hq} provides a high-quality instruction-based image editing dataset consisting of approximately 200,000 edits. Unlike previous methods that relied on attribute guidance or human feedback to build datasets, HQ-Edit employs a scalable data collection pipeline that leverages advanced foundation models, specifically GPT-4V and DALL-E 3.
