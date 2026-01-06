# 📚 RAG & Knowledge: Retrieval-Augmented Generation

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-14+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2020--2025-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### Papers on retrieval-augmented generation, long context modeling, and memory-augmented systems

_From foundational RAG architectures to million-token context windows and persistent memory systems._

</div>

---

## 📑 Table of Contents

- [🔍 RAG Foundations](#-rag-foundations)
- [📏 Long Context](#-long-context)
- [🧠 Memory Systems](#-memory-systems)

---

## 🔍 RAG Foundations

### 📄 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Lewis et al. (Facebook AI, UCL, NYU)  
**Contribution:** `🔍 RAG Architecture`

> The foundational **RAG paper** that introduced the paradigm of combining parametric (neural network) and non-parametric (retrieval) memory. By retrieving relevant documents from a knowledge base and conditioning generation on them, RAG models can access and leverage external knowledge without storing everything in model parameters, dramatically improving factual accuracy and enabling knowledge updates without retraining.

### 📄 [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)

**Authors:** Guu et al. (Google Research)  
**Contribution:** `🎓 Pre-training with Retrieval`

> Pioneered the concept of **pre-training language models with retrieval**. REALM learns to retrieve documents that help predict masked tokens during pre-training, creating a model that inherently knows how to use external knowledge. This end-to-end approach to learning retrieval alongside language modeling laid crucial groundwork for modern RAG systems.

### 📄 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

**Authors:** Asai et al. (University of Washington, IBM Research)  
**Contribution:** `🪞 Self-Reflective RAG` 🆕

> Introduced a framework where the model learns to **adaptively retrieve and self-critique** its outputs. Self-RAG trains a single LM to generate special reflection tokens that decide when to retrieve, assess relevance of retrieved passages, and critique its own generations. This self-reflective approach significantly improves factuality and citation accuracy over standard RAG.

### 📄 [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

**Authors:** Sarthi et al. (Stanford University)  
**Contribution:** `🌳 Hierarchical Retrieval` 🆕

> Proposed a novel approach to organizing retrieved information in a **hierarchical tree structure**. RAPTOR recursively clusters and summarizes text chunks, creating multi-level abstractions that enable retrieval at different granularities. This allows the model to answer questions requiring both fine-grained details and high-level synthesis across large document collections.

---

## 📏 Long Context

### 📄 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

**Authors:** Su et al. (Zhuiyi Technology)  
**Contribution:** `🔄 Position Encoding`

> Introduced **Rotary Position Embedding (RoPE)**, a revolutionary approach to encoding positional information in Transformers. RoPE encodes positions through rotation matrices, naturally capturing relative positions while being compatible with linear attention. This technique has become the de facto standard for modern LLMs and is crucial for extending context lengths through interpolation methods.

### 📄 [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)

**Authors:** Chen et al. (CUHK, MIT)  
**Contribution:** `⚡ Efficient Long Context` 🆕

> Developed an efficient method to **extend context windows** of pre-trained LLMs with minimal computational cost. LongLoRA combines shifted sparse attention during training with LoRA for parameter efficiency, enabling extension to 100k+ tokens while using a fraction of the compute required by full fine-tuning. This democratized long-context capabilities for the research community.

### 📄 [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

**Authors:** Liu et al. (UC Berkeley)  
**Contribution:** `♾️ Infinite Context` 🆕

> Introduced **Ring Attention**, a technique that enables training and inference on sequences of virtually unlimited length by distributing attention computation across multiple devices in a ring topology. By overlapping communication with computation and using blockwise attention, it removes memory constraints as a bottleneck, enabling context windows in the millions of tokens.

### 📄 [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)

**Authors:** Chen et al. (Meta AI)  
**Contribution:** `📐 Context Extension` 🆕

> Proposed **Position Interpolation (PI)**, a simple yet effective method to extend the context window of RoPE-based LLMs. Instead of extrapolating positions beyond training, PI downscales position indices to fit within the original range. This elegant approach enables extending context from 2K to 32K+ tokens with minimal fine-tuning, becoming a standard technique for context extension.

---

## 🧠 Memory Systems

### 📄 [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

**Authors:** Packer et al. (UC Berkeley)  
**Contribution:** `💾 Virtual Memory` 🆕

> Introduced a paradigm-shifting approach that treats LLM context as a **virtual memory system**. MemGPT manages context like an OS manages memory, with a hierarchy of main context (fast, limited) and external storage (slow, unlimited). The LLM learns to page information in and out, enabling unbounded conversation history and document analysis within fixed context windows.

### 📄 [Memorizing Transformers](https://arxiv.org/abs/2203.08913)

**Authors:** Wu et al. (Google Research)  
**Contribution:** `🗄️ External Memory`

> Augmented Transformers with a **kNN-based external memory** that stores and retrieves past key-value pairs. This approach allows the model to attend over a massive corpus of past activations without increasing computational cost proportionally. The technique demonstrated significant improvements on language modeling tasks, especially for rare patterns and long-range dependencies.

### 📄 [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174)

**Authors:** Wang et al. (UC Santa Barbara, Microsoft)  
**Contribution:** `🧠 Long-Term Memory` 🆕

> Proposed **LongMem**, a framework for augmenting LLMs with a decoupled long-term memory module. The system uses a frozen backbone LLM with a trainable memory encoder and retriever, enabling the model to access information from arbitrarily long histories. This architecture allows for efficient memory updates and retrieval without modifying the base model.

### 📄 [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)

**Authors:** Munkhdalai et al. (Google)  
**Contribution:** `∞ Compressive Memory` 🆕

> Introduced **Infini-attention**, which combines local attention with a compressive memory mechanism in a single Transformer block. The approach maintains a compressed representation of the entire history while performing standard attention on local context. This enables processing of infinitely long sequences with bounded memory and compute, achieving strong results on long-context benchmarks.

### 📄 [GraphRAG: Unlocking LLM Discovery on Narrative Private Data](https://arxiv.org/abs/2404.16130) 🆕

**Authors:** Microsoft  
**Contribution:** `🕸️ Knowledge Graph RAG`

> Moves beyond simple vector similarity search by **building a knowledge graph from data first**, then using it for retrieval. GraphRAG constructs entity-relationship graphs and community summaries, allowing LLMs to answer "global" questions (e.g., "What are the main themes in this dataset?") that standard RAG fails at. Particularly effective for complex reasoning over large private document collections where understanding relationships between concepts is crucial.

### 📄 [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study](https://arxiv.org/abs/2407.16833) 🆕

**Authors:** Li et al. (EMNLP 2024/2025)  
**Contribution:** `🔬 RAG vs Long-Context`

> The definitive study answering: **"Do we still need RAG if models have 1M+ context?"** Through comprehensive experiments, the paper shows that while long-context models are powerful, RAG remains far cheaper and often more accurate for "needle-in-a-haystack" retrieval tasks. Provides empirical guidance on when to use each approach: RAG for specific fact retrieval, long context for deep document understanding.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more RAG, long context, or memory papers!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

Thanks to all researchers pushing the boundaries of how LLMs access and utilize knowledge.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
