# 🎨 Multimodal Models: Vision, Language & Beyond

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-18+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2021--2025-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of groundbreaking papers bridging vision, language, and other modalities

_From foundational vision-language models to cutting-edge video understanding and generation systems._

</div>

---

## 📑 Table of Contents

- [👁️ Vision-Language Models](#-vision-language-models)
- [🎨 Image Generation](#-image-generation)
- [🎬 Video Understanding & Generation](#-video-understanding--generation)
- [🆕 Recent Breakthroughs](#-recent-breakthroughs)

---

## 👁️ Vision-Language Models

### 📄 [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Radford et al. (OpenAI)  
**Contribution:** `🔗 Vision-Language Foundation`

> Introduced **Contrastive Language-Image Pre-training (CLIP)**, a revolutionary approach that learns visual concepts from natural language supervision. By training on 400 million image-text pairs from the internet, CLIP learns a joint embedding space where images and text can be directly compared. This enables remarkable zero-shot transfer to downstream tasks—the model can classify images into arbitrary categories just by being given their text descriptions, without any task-specific training.

### 📄 [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Liu et al. (Microsoft Research, University of Wisconsin-Madison)  
**Contribution:** `🎯 Visual Instruction Following`

> Pioneered **visual instruction tuning** by connecting a vision encoder (CLIP) with a large language model (LLaMA/Vicuna) through a simple projection layer. LLaVA demonstrated that instruction-following capabilities could be extended to the visual domain using GPT-4 generated multimodal instruction data. This efficient approach achieved impressive multimodal chat capabilities, sparking a wave of open-source vision-language models.

### 📄 [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** OpenAI  
**Contribution:** `🏆 Multimodal Intelligence`

> Introduced **GPT-4V**, the first large-scale commercial model to seamlessly integrate vision and language understanding. GPT-4V demonstrated unprecedented capabilities in visual reasoning, document understanding, and complex image analysis. It can interpret charts, solve visual puzzles, read handwritten text, and engage in nuanced discussions about image content, setting a new benchmark for multimodal AI assistants.

### 📄 [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Alayrac et al. (DeepMind)  
**Contribution:** `🦩 Few-Shot Visual Learning`

> Introduced **Flamingo**, a family of visual language models capable of rapid adaptation to new tasks from just a few examples. By using a novel architecture that interleaves frozen pre-trained vision and language models with learnable cross-attention layers, Flamingo achieved state-of-the-art few-shot performance on a wide range of vision-language tasks, demonstrating that in-context learning extends powerfully to the multimodal domain.

---

## 🎨 Image Generation

### 📄 [Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)](https://arxiv.org/abs/2204.06125)

**Authors:** Ramesh et al. (OpenAI)  
**Contribution:** `🖼️ Text-to-Image Generation`

> Introduced **DALL-E 2**, a two-stage text-to-image system that first generates CLIP image embeddings from text captions, then uses a diffusion model to decode these embeddings into photorealistic images. The model demonstrated remarkable capabilities in generating diverse, high-quality images from complex text prompts, and introduced the ability to edit existing images through inpainting and variations.

### 📄 [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752)

**Authors:** Rombach et al. (LMU Munich, Runway)  
**Contribution:** `🎨 Efficient Diffusion`

> Introduced **Latent Diffusion Models (LDMs)**, which perform the diffusion process in a compressed latent space rather than pixel space. This dramatically reduced computational requirements while maintaining high image quality, making high-resolution image generation accessible on consumer hardware. The open-source release of Stable Diffusion democratized AI image generation and sparked an explosion of creative applications and fine-tuning techniques.

---

## 🎬 Video Understanding & Generation

### 📄 [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)

**Authors:** Tong et al. (Nanjing University, Tencent)  
**Contribution:** `🎥 Video Self-Supervision`

> Extended masked autoencoding to video, demonstrating that **extremely high masking ratios (90-95%)** work remarkably well for video due to temporal redundancy. VideoMAE showed that self-supervised pre-training on video can achieve competitive results with far less data than supervised approaches, establishing an efficient paradigm for video representation learning.

### 📄 [Video Generation Models as World Simulators (Sora Technical Report)](https://openai.com/index/video-generation-models-as-world-simulators/)

**Authors:** OpenAI  
**Contribution:** `🌍 World Simulation`

> Introduced **Sora**, a diffusion transformer capable of generating high-fidelity videos up to a minute long with remarkable temporal consistency and physical understanding. The technical report frames video generation models as "world simulators" that learn implicit physics and 3D consistency from video data alone. Sora represents a paradigm shift in video generation, demonstrating emergent capabilities in simulating complex real-world dynamics.

### 📄 [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)

**Authors:** Hong et al. (Tsinghua University, BAAI)  
**Contribution:** `📹 Text-to-Video`

> Pioneered **large-scale text-to-video generation** using a transformer-based architecture built upon CogView2. CogVideo introduced multi-frame-rate hierarchical training and demonstrated that the text-to-image generation paradigm could be effectively extended to video, generating coherent video clips from text descriptions while maintaining temporal consistency.

---

## 🆕 Recent Breakthroughs

### 📄 [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530) 🆕

**Authors:** Gemini Team (Google)  
**Contribution:** `📚 Long-Context Multimodal`

> Introduced **Gemini 1.5**, featuring a mixture-of-experts architecture capable of processing up to 10 million tokens of multimodal context. This unprecedented context length enables entirely new capabilities: analyzing hour-long videos, entire codebases, or hundreds of pages of documents in a single prompt. The model demonstrates near-perfect recall across its massive context window while maintaining strong performance on standard benchmarks.

### 📄 [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) 🆕

**Authors:** Liu et al. (ByteDance, University of Wisconsin-Madison)  
**Contribution:** `📈 Enhanced Visual Reasoning`

> Built upon LLaVA with significant improvements in **visual reasoning, OCR, and world knowledge**. LLaVA-NeXT introduced dynamic high-resolution image processing, allowing the model to handle images of varying sizes and aspect ratios more effectively. It achieved GPT-4V-level performance on several benchmarks while remaining fully open-source, advancing the state of open multimodal models.

### 📄 [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) 🆕

**Authors:** Chen et al. (Shanghai AI Lab, Tsinghua University)  
**Contribution:** `🔬 Scaled Vision-Language`

> Introduced **InternVL**, a large-scale vision-language foundation model that scales the vision encoder to 6 billion parameters. By progressively aligning the vision encoder with LLMs through a multi-stage training process, InternVL achieved state-of-the-art performance across a wide range of vision-language tasks. The model demonstrates that scaling vision encoders, not just language models, is crucial for multimodal understanding.

### 📄 [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) 🆕

**Authors:** Meta FAIR  
**Contribution:** `🔀 Early Fusion`

> Unlike previous models that "glue" a vision encoder to an LLM, Chameleon **tokenizes images and text together from the start**. This "early fusion" approach processes all modalities in a unified token space, allowing it to generate mixed text-and-image content (e.g., a webpage layout with icons) natively. Represents a fundamental architectural shift toward true multimodal models rather than late-fusion approaches.

### 📄 [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611) 🆕

**Authors:** Apple  
**Contribution:** `🔬 Scientific Analysis`

> A crucial "science of AI" paper that rigorously **ablates architectural choices** for multimodal models—image encoder resolution, token counts, connector types, and pre-training strategies. MM1 provides a recipe book for building SOTA multimodal models, demonstrating what works and what doesn't through systematic experimentation. This research accelerated the field by de-risking design decisions for future models.

### 📄 [Janus-Pro: Unified Multimodal Understanding and Generation](https://arxiv.org/pdf/2501.17811) 🆕

**Authors:** DeepSeek AI (Jan 2025)  
**Contribution:** `🎭 Decoupled Pathways`

> Solves the "Jack of all trades, master of none" problem in multimodal models by **decoupling visual encoding** for understanding vs. generation. Uses separate pathways: SigLIP for understanding and VQ tokenizer for generation, but processes them in a single unified transformer. This architectural separation enables excellence in both tasks without the typical trade-offs of unified approaches.

### 📄 [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234) 🆕

**Authors:** Wei, Sun, Li (DeepSeek AI, Oct 2025)  
**Contribution:** `🗜️ Vision-Text Compression`

> Introduces a novel approach to **vision-text compression** using optical 2D mapping. DeepSeek-OCR achieves ~97% OCR precision at <10x compression ratio (10 text tokens → 1 vision token), enabling efficient processing of long documents. Can process over 200,000 pages per day on a single A100-40G GPU, making it practical for historical document digitization and LLM memory enhancement.

### 📄 [VL-JEPA: Joint Embedding Predictive Architecture for Video/World Modeling](https://arxiv.org/pdf/2512.10942) 🆕

**Authors:** Meta AI  
**Contribution:** `🌍 World Models`

> Meta's **joint embedding predictive architecture** for video and world modeling that learns physics without reconstruction. Unlike generative models that predict pixels, VL-JEPA learns abstract representations by predicting embeddings in latent space. This approach enables learning physical dynamics and causal relationships from video data more efficiently than pixel-level prediction methods.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more multimodal papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

This list celebrates the researchers pushing the boundaries of multimodal AI.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
