# 🏗️ Model Architectures: The Building Blocks of LLMs

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-16+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2017--2025-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of papers defining the core architectures that power modern language models

_From the original Transformer to State Space Models and Mixture of Experts, these papers represent the fundamental innovations in neural network design for language understanding._

</div>

---

## 📑 Table of Contents

- [🏛️ Foundational Architectures](#-foundational-architectures)
- [🐍 State Space Models](#-state-space-models)
- [🧩 Mixture of Experts](#-mixture-of-experts)
- [🆕 Recent Breakthroughs](#-recent-breakthroughs)

---

## 🏛️ Foundational Architectures

### 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Vaswani et al. (Google)  
**Contribution:** `🏗️ Transformer`

> The paper that started it all. Introduced the **Transformer architecture**, replacing recurrent layers entirely with self-attention mechanisms. This breakthrough enabled massive parallelization during training and superior handling of long-range dependencies. Every modern LLM—from GPT to LLaMA—is built upon this foundational work.

### 📄 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Devlin et al. (Google)  
**Contribution:** `🧠 Bidirectional`

> Revolutionized NLP by introducing **bidirectional pre-training** with the Masked Language Model (MLM) objective. Unlike left-to-right models, BERT learns context from both directions simultaneously, achieving breakthrough performance on understanding tasks like question answering and sentiment analysis. Spawned an entire family of encoder-based models.

### 📄 [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Radford et al. (OpenAI)  
**Contribution:** `🎯 Decoder-Only`

> Demonstrated that a sufficiently large **decoder-only Transformer** could perform diverse tasks in a zero-shot setting without task-specific training. GPT-2 proved that scaling up autoregressive language models unlocks emergent capabilities, establishing the architectural paradigm that would dominate with GPT-3, GPT-4, and beyond.

---

## 🐍 State Space Models

### 📄 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Gu & Dao (CMU, Princeton)  
**Contribution:** `🐍 SSM Architecture`

> Introduced a **selective state space model** that achieves Transformer-quality performance with linear-time complexity. By making the state space parameters input-dependent (selective), Mamba can dynamically filter information based on content. This enables 5x faster inference than Transformers and efficient processing of sequences up to 1M tokens.

### 📄 [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)](https://arxiv.org/abs/2405.21060)

**Authors:** Dao & Gu (Princeton, CMU)  
**Contribution:** `🔄 Unified Theory`

> Revealed a deep theoretical connection between Transformers and state space models through **Structured State Space Duality (SSD)**. Mamba-2 leverages this insight to achieve 2-8x faster training than the original Mamba while maintaining competitive performance. Provides a unified framework for understanding both architectural families.

---

## 🧩 Mixture of Experts

### 📄 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Fedus et al. (Google)  
**Contribution:** `🧩 Sparse MoE`

> Successfully scaled **Mixture-of-Experts (MoE)** to over a trillion parameters while keeping compute manageable. The key innovation is routing each token to just one expert (instead of multiple), dramatically simplifying training. This sparse activation pattern means the model can be massive while only using a fraction of parameters per forward pass.

### 📄 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

**Authors:** Jiang et al. (Mistral AI)  
**Contribution:** `⚡ Efficient MoE`

> 🆕 Released a highly efficient **Sparse Mixture of Experts** model that outperforms Llama 2 70B and GPT-3.5 on most benchmarks while using only 13B active parameters (from 46.7B total). Mixtral 8x7B routes each token to 2 of 8 experts, achieving excellent performance-to-compute ratio and setting a new standard for open MoE models.

---

## 🆕 Recent Breakthroughs

### 📄 [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

**Authors:** Lieber et al. (AI21 Labs)  
**Contribution:** `🔀 Hybrid Architecture`

> 🆕 Successfully combines **Transformer attention layers with Mamba (State Space Model) layers** in a single architecture, breaking the "Transformer monopoly." This hybrid approach enables handling massive context windows (up to 256K+) with high throughput on a single GPU—something pure Transformers struggle with. By interleaving attention and SSM layers, Jamba achieves the best of both worlds: the modeling power of attention and the efficiency of state space models.

### 📄 [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Ghorbani, Behrouz, Zandieh, Karbasi, Mirrokni, Farajtabar (Google Research)  
**Contribution:** `🧠 Neural Memory`

> 🆕 Introduces a revolutionary **memory-augmented architecture** that can learn to memorize and retrieve information at test time. Titans combines attention with a neural long-term memory module, enabling the model to handle context lengths beyond 2M tokens while maintaining the ability to recall specific details. Represents a fundamental shift in how models handle long-range dependencies.

### 📄 [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models](https://arxiv.org/abs/2404.02258)

**Authors:** Raposo et al. (Google DeepMind)  
**Contribution:** `⚡ Dynamic Compute`

> 🆕 Proposes **dynamic depth allocation** where the model learns to route tokens through different numbers of layers based on their complexity. Simple tokens skip layers while complex tokens get full processing. This achieves comparable performance to baseline Transformers while using 12-50% less compute, offering a new dimension of efficiency beyond MoE.

### 📄 [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

**Authors:** Sun et al. (Microsoft Research, Tsinghua University)  
**Contribution:** `🔄 Retention Mechanism`

> Introduces **RetNet**, which achieves training parallelism (like Transformers), low-cost inference (like RNNs), and linear complexity (like state space models) simultaneously. The key innovation is the retention mechanism that supports parallel, recurrent, and chunkwise computation modes. Offers O(1) inference complexity while maintaining competitive performance.

### 📄 [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)

**Authors:** Poli et al. (Stanford, Hazy Research)  
**Contribution:** `🌊 Subquadratic Attention`

> Proposes **Hyena**, a subquadratic replacement for attention based on long convolutions and data-controlled gating. Achieves comparable quality to Transformers while reducing compute requirements significantly for long sequences. Demonstrates that attention is not the only path to high-quality language modeling, opening new architectural possibilities.

### 📄 [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

**Authors:** Peng et al. (RWKV Foundation)  
**Contribution:** `🔁 Linear RNN`

> Combines the **efficient parallelizable training of Transformers with the efficient inference of RNNs**. RWKV uses a linear attention mechanism that can be formulated as either an RNN (for inference) or a Transformer (for training). Achieves competitive performance with Transformers while enabling constant memory and linear time inference.

### 📄 [DeepSeek-V3.2: Reasoning Rival](https://arxiv.org/pdf/2512.02556)

**Authors:** DeepSeek AI  
**Contribution:** `� Reasoning Excellence`

> 🆕 Open-source reasoning model rivaling GPT-5 in mathematics and logic at 1/10th the cost. Demonstrates breakthrough reasoning capabilities through innovative training approaches.

### 📄 [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/pdf/2512.24880)

**Authors:** DeepSeek AI  
**Contribution:** `📐 Stable Residuals`

> 🆕 Introduces **Manifold-Constrained Hyper-Connections**, a framework that stabilizes and scales residual connection architectures by projecting connections onto the Birkhoff Polytope manifold. This ensures identity mapping properties are preserved while enabling diversified connectivity patterns. Achieves 2.1% improvement on BBH benchmarks for 27B models with only 6.7% additional training overhead, solving critical stability issues in deep architectures.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more architecture papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

This collection celebrates the researchers pushing the boundaries of neural network design for language understanding.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
