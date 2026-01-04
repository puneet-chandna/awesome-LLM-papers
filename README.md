# 🧠 Awesome LLM Papers

<div align="center">

![Daily LLM Papers](https://img.shields.io/badge/Awesome%20LLM%20Papers-Breaking%20Research%20Daily-blue?style=for-the-badge&logo=arxiv&logoColor=white)
[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Stars](https://img.shields.io/github/stars/puneet-chandna/awesome-LLM-papers?style=for-the-badge&color=yellow)](https://github.com/puneet-chandna/awesome-LLM-papers/stargazers)
[![Updates](https://img.shields.io/badge/Updates-Weekly%20@%209AM%20IST-green?style=for-the-badge)](https://github.com/puneet-chandna/awesome-LLM-papers)

<!-- [![Newsletter](https://img.shields.io/badge/Newsletter-500+%20Subscribers-red?style=for-the-badge)](https://your-newsletter-link.com) -->

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

<h3>Read what matters. Skip the noise. 🎯</h3>

<p align="center">
  <a href="#-todays-pick">Today's Pick</a> •
  <a href="#-this-weeks-essential-reads">This Week</a> •
  <a href="#-must-read-papers-hall-of-fame">Hall of Fame</a> •
  <a href="#-browse-by-category">Categories</a> •
  <a href="#-contributing">Contribute</a>
</p>

---

<img src="https://img.shields.io/badge/📄%20Papers%20Curated-123-blue?style=flat-square&labelColor=000000" alt="Papers">
<img src="https://img.shields.io/badge/🆕%20Added%20This%20Week-8-green?style=flat-square&labelColor=000000" alt="New Papers">
<img src="https://img.shields.io/badge/👥%20Contributor-1-orange?style=flat-square&labelColor=000000" alt="Contributors">
<img src="https://img.shields.io/badge/⭐%20Community%20Rated-10+-red?style=flat-square&labelColor=000000" alt="Ratings">

</div>

---

## 🔥 Today's Pick

### 🏆 **[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)**

<table>
<tr>
<td width="70%">

**Authors:** _Behrooz Ghorbani, Ali Behrouz, Amir Zandieh, Amin Karbasi, Vahab Mirrokni, Mehrdad Farajtabar_ • Google Research

**Why this matters:** This groundbreaking paper introduces a new architecture that combines the power of Transformers with learnable memory modules, enabling models to memorize and recall information at test time. Titans achieve superior performance on long-context tasks while maintaining efficiency.

**Key Innovations:**

- 🔸 Neural long-term memory module that learns to memorize at test time
- 🔸 Three architectural variants: Memory as Context (MAC), Memory as Gate (MAG), and Memory as Layer (MAL)
- 🔸 Outperforms Transformers and modern linear recurrent models on language modeling, commonsense reasoning, and needle-in-haystack tasks
- 🔸 Scales effectively to context windows over 2M tokens with persistent memory

</td>
<td width="30%">

**Resources:**

- 🔗 [**Paper**](https://arxiv.org/pdf/2501.00663)
- 📊 [**Google Research**](https://research.google/)
- 🐦 [**Thread**](https://x.com/behrouz_ali/status/1878859086227255347?s=20)

**Impact Score:**

```diff
+ Performance: █████████░ 94%
+ Innovation:  ██████████ 96%
+ Practicality:█████████░ 90%
```

</td>
</tr>
</table>

---

## 🔥 Trending Topics

_"Hot research areas this month - Where the field is moving"_

<div align="center">

| Topic                             | Papers                                                       | Why It's Trending                                                        |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| 🧠 **Memory-Augmented LLMs**      | [10 papers](categories/architectures.md#-memory-systems)     | Titans architecture - test-time memorization over 2M+ token contexts     |
| 🤖 **Agentic AI & Autonomy**      | [15 papers](categories/reasoning.md#-agent-systems)          | Self-play SWE-RL, autonomous coding agents managing full repositories    |
| 🚀 **MoE & Sparse Architectures** | [12 papers](categories/architectures.md#-mixture-of-experts) | DeepSeek V3 - 671B params, 37B active per inference, crushing benchmarks |
| 🔄 **Recursive Language Models**  | [6 papers](categories/training.md#-self-improvement)         | Context folding, sub-LLM calling - managing infinite context elegantly   |

</div>

---

## 📆 This Week's Essential Reads

<details open>
<summary><b>Click to expand this week's papers</b> (December 23-30, 2025)</summary>

<br>

| Day     | Paper                                                                                        | Impact                  | TL;DR                                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Mon** | [🧮 🧠 DeepSeek-V3.2: Reasoning Rival](https://arxiv.org/pdf/2512.02556)                     | `reasoning` `math`      | Open-source reasoning model rivaling GPT-5 in mathematics and logic at 1/10th the cost                                 |
| **Tue** | [👁️ 🌍 VLJEPA: World Model Architecture](https://arxiv.org/pdf/2512.10942)                   | `vision` `architecture` | Meta's joint embedding predictive architecture for video/world modeling - learning physics without reconstruction      |
| **Wed** | [⚡ Titans: Learning to Memorize at Test Time](https://arxiv.org/pdf/2501.00663)             | `architecture` `memory` | Google's memory-augmented architecture handles 2M+ token context with persistent long-term memory                      |
| **Thu** | [🏗️ Claude Opus 4.5: Engineering Excellence](https://www.anthropic.com/news/claude-opus-4-5) | `coding` `engineering`  | New benchmark for software engineering agents - outperforming human candidates in internal capability tests            |
| **Fri** | [🤖 Self-Play SWE-RL: Autonomous Coding](https://arxiv.org/pdf/2512.18552)                   | `agents` `RL`           | Meta's system where agents learn to code by generating and fixing their own bugs, creating superhuman coding abilities |

</details>

---

## 📚 Must-Read Papers (Hall of Fame)

> 🏛️ **Papers that fundamentally changed the field**

<table>
<thead>
<tr>
<th width="30%">Paper</th>
<th width="15%">Impact</th>
<th width="40%">Why Essential</th>
<th width="15%">Resources</th>
</tr>
</thead>
<tbody>

<tr>
<td>

**[Attention Is All You Need](summaries/Attention%20Is%20All%20You%20Need%20.md)**
<br>_Vaswani et al., 2017_

</td>
<td align="center">

🏆 **Foundational**
<br>`architecture`

</td>
<td>

Created the Transformer architecture that powers all modern LLMs. Replaced RNNs with self-attention, enabling parallelization and scaling.

</td>
<td align="center">

[📄](https://arxiv.org/abs/1706.03762) [💻](https://github.com/tensorflow/tensor2tensor) [📊](http://jalammar.github.io/illustrated-transformer/)

</td>
</tr>

<tr>
<td>

**[GPT-3: Language Models are Few-Shot Learners](summaries/GPT-3%20Language%20Models%20are%20Few-Shot%20Learners.md)**
<br>_Brown et al., 2020_

</td>
<td align="center">

🚀 **Scale**
<br>`emergence`

</td>
<td>

Proved that scale leads to emergent abilities. In-context learning without fine-tuning revolutionized how we use LLMs.

</td>
<td align="center">

[📄](https://arxiv.org/abs/2005.14165) [🔍](https://openai.com/api/) [📊](https://gpt3demo.com)

</td>
</tr>

<tr>
<td>

**[Constitutional AI: Harmlessness from AI Feedback](summaries/Constitutional%20AI%20Harmlessness%20from%20AI%20Feedback.md)**
<br>_Bai et al., 2022_

</td>
<td align="center">

🛡️ **Safety**
<br>`alignment`

</td>
<td>

Introduced RLAIF - training AI systems to be helpful and harmless using AI feedback instead of human feedback.

</td>
<td align="center">

[📄](https://arxiv.org/abs/2212.08073) [🏠](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

</td>
</tr>

<tr>
<td>

**[Chain-of-Thought Prompting](summaries/Chain-of-Thought%20Prompting.md)**
<br>_Wei et al., 2022_

</td>
<td align="center">

🧮 **Reasoning**
<br>`prompting`

</td>
<td>

Simple prompting technique that dramatically improves reasoning by asking models to think step-by-step.

</td>
<td align="center">

[📄](https://arxiv.org/abs/2201.11903) [📝](https://www.promptingguide.ai/techniques/cot) [💡](https://learnprompting.org/docs/intermediate/chain_of_thought)

</td>
</tr>

<tr>
<td>

**[RLHF: Training with Human Feedback](summaries/RLHF%20Training%20with%20Human%20Feedback.md)**
<br>_Ouyang et al., 2022_

</td>
<td align="center">

🎯 **Alignment**
<br>`training`

</td>
<td>

The technique behind ChatGPT's success. Aligns model outputs with human preferences through reinforcement learning.

</td>
<td align="center">

[📄](https://arxiv.org/pdf/2203.02155) [💻](https://github.com/openai/following-instructions-human-feedback) [📚](https://huggingface.co/blog/rlhf)

</td>
</tr>

</tbody>
</table>

[View all foundational papers →](categories/Hall-of-fame.md)

---

## 📚 Browse by Category

<div align="center">
<table>
<tr>
<td width="50%" valign="top">

### [🏗️ **Model Architectures**](categories/architectures.md)

> _Transformers, SSMs, MoE, Novel designs_  
> **📄 10 papers** &nbsp;|&nbsp; 🔥

</td>
<td width="50%" valign="top">

### [🧮 **Reasoning & Agents**](categories/reasoning.md)

> _CoT, Planning, Tool use, Autonomous systems_  
> **📄 6 papers** &nbsp;|&nbsp; 🔥🔥

</td>
</tr>

<tr>
<td width="50%" valign="top">

### ⚡ [**Efficiency & Scaling**](categories/efficiency.md)

> _Quantization, Pruning, Fast inference_  
> **📄 7 papers** &nbsp;|&nbsp; 🔥

</td>
<td width="50%" valign="top">

### 🎯 [**Training & Alignment**](categories/training.md)

> _RLHF, DPO, Fine-tuning, PEFT methods_  
> **📄 5 papers** &nbsp;|&nbsp; →

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🎨 [**Multimodal Models**](categories/multimodal.md)

> _Vision-Language, Audio, Video, Any-to-any_  
> **📄 8 papers** &nbsp;|&nbsp; 🔥

</td>
<td width="50%" valign="top">

### 📚 [**RAG & Knowledge**](categories/rag.md)

> _Retrieval systems, Long context, Memory_  
> **📄 8 papers** &nbsp;|&nbsp; 🔥🔥🔥

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🛡️ [**Safety & Security**](categories/safety.md)

> _Jailbreaks, Alignment, Robustness, Ethics_  
> **📄 4 papers** &nbsp;|&nbsp; →

</td>
<td width="50%" valign="top">

### 🔬 [**Analysis & Theory**](categories/analysis.md)

> _Interpretability, Mechanistic, Evaluations_  
> **📄 3 papers** &nbsp;|&nbsp; →

</td>
</tr>
</table>
</div>

---

### 🏗️ Model Architectures

<details>
<summary><b>View papers</b> (10 total) • <code>🔥 Hot area</code></summary>

#### Latest Additions:

- 🆕 **[Mamba-2: Improved State Space Models](https://arxiv.org/abs/2405.21060)** - 5x faster than Transformers at 100K+ context
- 🆕 **[Mixture of Depths: Dynamic Layer Selection](https://arxiv.org/abs/2404.02258)** - Skip layers adaptively based on input
- **[Retentive Networks: Retention Replaces Attention](https://arxiv.org/abs/2307.08621)** - O(1) memory complexity breakthrough

#### Foundational Papers:

- 🏆 **[Attention Is All You Need](summaries/Attention%20Is%20All%20You%20Need%20.md)** - The paper that started everything
- **[BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** - Revolutionized NLP pre-training
- **[GPT-3: Few-Shot Learning](https://arxiv.org/abs/2005.14165)** - Proved scale leads to emergence

**Tags:** `transformer` `state-space` `mixture-of-experts` `attention-mechanisms`

[View all architecture papers →](categories/architectures.md)

</details>

---

### 🧮 Reasoning & Agents

<details>
<summary><b>View papers</b> (6 total) • <code>🔥🔥 Very hot area</code></summary>

#### Agent Systems:

- 🔥 **[ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)** - Combines reasoning with action execution
- 🔥 **[Reflexion: Self-Reflecting Agents](https://arxiv.org/abs/2303.11366)** - Learn from mistakes autonomously
- **[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)** - Self-taught tool use

#### Reasoning Methods:

- 🏆 **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** - Simple technique for complex reasoning
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)** - Explore multiple reasoning paths
- **[Graph of Thoughts](https://arxiv.org/abs/2308.09687)** - Non-linear reasoning structures

**Tags:** `agents` `chain-of-thought` `planning` `tool-use` `reasoning`

[View all reasoning papers →](categories/reasoning.md)

</details>

---

### ⚡ Efficiency & Scaling

<details>
<summary><b>View papers</b> (7 total) • <code>🔥 Hot area</code></summary>

#### Inference Optimization:

- 🆕 **[Speculative Decoding](https://arxiv.org/abs/2211.17192)** - 3x faster inference without quality loss
- **[FlashAttention-2: Faster Attention](https://arxiv.org/abs/2307.08691)** - Optimized attention for inference
- **[vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)** - 24x throughput improvement

#### Model Compression:

- 🏆 **[QLoRA: 4-bit Quantization](https://arxiv.org/abs/2305.14314)** - Fine-tune 65B models on single GPU
- **[GPTQ: Accurate Quantization](https://arxiv.org/abs/2210.17323)** - 3-4 bit quantization with minimal loss
- **[SparseGPT: 50% Sparsity](https://arxiv.org/abs/2301.00774)** - Remove half the weights, keep performance

**Tags:** `quantization` `pruning` `distillation` `inference` `deployment`

[View all efficiency papers →](categories/efficiency.md)

</details>

---

### 🎯 Training & Alignment

<details>
<summary><b>View papers</b> (5 total)</summary>

#### Alignment Methods:

- 🏆 **[RLHF: Human Feedback Training](https://arxiv.org/abs/1706.03741)** - The technique behind ChatGPT
- 🆕 **[DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)** - Simpler alternative to RLHF
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** - AI feedback for harmless assistants

#### Fine-tuning Techniques:

- **[LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)** - Parameter-efficient fine-tuning
- **[Prefix Tuning](https://arxiv.org/abs/2101.00190)** - Tune only prefix parameters
- **[FLAN: Instruction Tuning](https://arxiv.org/abs/2109.01652)** - Teaching models to follow instructions

**Tags:** `rlhf` `fine-tuning` `peft` `instruction-tuning` `alignment`

[View all training papers →](categories/training.md)

</details>

---

### 🎨 Multimodal Models

<details>
<summary><b>View papers</b> (8 total) • <code>🔥 Hot area</code></summary>

#### Vision-Language:

- 🆕 **[GPT-4V: Visual Understanding](https://arxiv.org/abs/2303.08774)** - State-of-the-art visual reasoning
- **[CLIP: Contrastive Vision-Language](https://arxiv.org/abs/2103.00020)** - Foundation for modern multimodal
- **[LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)** - Open-source GPT-4V alternative

#### Generation:

- **[DALL-E 2: Text-to-Image](https://arxiv.org/abs/2204.06125)** - Photorealistic generation
- **[Stable Diffusion](https://arxiv.org/abs/2112.10752)** - Open-source image generation
- **[Sora: Text-to-Video](https://openai.com/index/video-generation-models-as-world-simulators/)** - Minute-long video generation

**Tags:** `vision-language` `image-generation` `video` `audio` `multimodal`

[View all multimodal papers →](categories/multimodal.md)

</details>

---

### 📚 RAG & Knowledge

<details>
<summary><b>View papers</b> (4 total) • <code>🔥🔥🔥 Hottest area</code></summary>

#### RAG Systems:

- 🔥 **[Self-RAG: Self-Reflection](https://arxiv.org/abs/2310.11511)** - Adaptive retrieval and generation
- 🔥 **[RAPTOR: Recursive Abstractive Processing](https://arxiv.org/abs/2401.18059)** - Tree-based retrieval
- **[RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)** - The foundational RAG paper

#### Long Context:

- 🆕 **[Ring Attention: Million Token Context](https://arxiv.org/abs/2310.01889)** - Process entire books
- **[LongLoRA: Efficient Long Context](https://arxiv.org/abs/2309.12307)** - Extend context to 100k+
- **[Infini-attention: Infinite Context](https://arxiv.org/abs/2404.07143)** - Never-ending conversations

**Tags:** `retrieval` `rag` `long-context` `memory` `knowledge-bases`

[View all RAG papers →](categories/rag.md)

</details>

---

### 🛡️ Safety & Security

<details>
<summary><b>View papers</b> (4 total)</summary>

#### Safety Research:

- 🚨 **[Universal Adversarial Attacks](https://arxiv.org/abs/2307.15043)** - Attacks that work on all models
- **[Circuit Breakers](https://arxiv.org/abs/2406.04313)** - Built-in safety mechanisms
- **[Representation Engineering](https://arxiv.org/abs/2310.01405)** - Control model behavior directly

#### Alignment:

- **[Sleeper Agents](https://arxiv.org/abs/2401.05566)** - Hidden model behaviors
- **[Scalable Oversight via Debate](https://arxiv.org/abs/1805.00899)** - Supervising superhuman AI
- **[Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390)** - Superalignment research

**Tags:** `jailbreaks` `alignment` `safety` `robustness` `red-teaming`

[View all safety papers →](categories/safety.md)

</details>

---

### 🔬 Analysis & Theory

<details>
<summary><b>View papers</b> (3 total)</summary>

- **[Language Models are Few-Shot Learners (Analysis)](https://arxiv.org/abs/2005.14165)** - (2020) Scaling laws and emergent abilities
- **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)** - (2022) Investigating non-linear scaling behavior
- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)** - (2021) Reverse engineer several toy, attention-only models by Anthropic

**Tags:** `interpretability` `mechanistic` `theory` `analysis` `evaluation`

[View all analysis papers →](categories/analysis.md)

</details>

---

<!--
## 🏷️ Explore by Tags

<div align="center">

<!-- Trending Tags Section
<p align="center">
<a href="#"><img src="https://img.shields.io/badge/🔥_mixture--of--experts-12_papers-ff6b6b?style=for-the-badge&labelColor=000000" alt="moe"/></a>
<a href="#"><img src="https://img.shields.io/badge/🔥_long--context-8_papers-4ecdc4?style=for-the-badge&labelColor=000000" alt="long-context"/></a>
<a href="#"><img src="https://img.shields.io/badge/🔥_rag-15_papers-45b7d1?style=for-the-badge&labelColor=000000" alt="rag"/></a>
<a href="#"><img src="https://img.shields.io/badge/🔥_agents-11_papers-96ceb4?style=for-the-badge&labelColor=000000" alt="agents"/></a>
</p>


**📊 Most Used Tags**

<kbd><a href="#">transformer (67)</a></kbd> • <kbd><a href="#">efficient (45)</a></kbd> • <kbd><a href="#">reasoning (38)</a></kbd> • <kbd><a href="#">open-source (35)</a></kbd> • <kbd><a href="#">multimodal (28)</a></kbd>

<kbd><a href="#">production-ready (25)</a></kbd> • <kbd><a href="#">breakthrough (22)</a></kbd> • <kbd><a href="#">sota (20)</a></kbd> • <kbd><a href="#">chain-of-thought (18)</a></kbd> • <kbd><a href="#">rlhf (15)</a></kbd>


**🎯 Quick Filters**

**By Impact:** [`🏆 breakthrough`](#) • [`⭐ sota`](#) • [`🚀 production-ready`](#) • [`🧪 experimental`](#)

**By Org:** [`🟠 openai`](#) • [`🔷 anthropic`](#) • [`🔴 google`](#) • [`🔵 meta`](#) • [`🎓 academic`](#)

**By Size:** [`<1B`](#) • [`1B-7B`](#) • [`7B-30B`](#) • [`30B+`](#)

</div>
-->

## 📈 Research Trends Dashboard

<div align="center">

```mermaid
%%{init: {'theme':'dark'}}%%
graph TD
    Q1[2025 Q1] -->|RAG Revolution| Q2[2025 Q2]
    Q2 -->|Agents & Tools| Q3[2025 Q3]
    Q2 -->|1M+ Context| Q3
    Q2 -->|MoE Everything| Q3
    Q3 -->|Reasoning Models| Q4[2025 Q4]
    Q3 -->|Multimodal Fusion| Q4
    Q4 -->|Memory Architectures| C[2026 Q1]
    Q4 -->|Agentic AI Boom| D[2026 Q1]
    Q4 -->|MoE at Scale| E[2026 Q1]
    C -->|Recursive LMs| F[2026 Q2]
    D -->|Self-Improving Agents| F
    E -->|200K+ Context| F

    style Q1 fill:#0f172a
    style Q2 fill:#1e293b
    style Q3 fill:#334155
    style Q4 fill:#475569
    style C fill:#4b5563
    style D fill:#6b7280
    style E fill:#9ca3af
    style F fill:#059669
```

**January 2026 Momentum:**

- 📈 **Rising:** Agentic AI (+520%), Memory Architectures (+480%), MoE & Sparse Models (+380%), Recursive Language Models (+350%)
- 📉 **Cooling:** Vanilla RAG (-45%), Basic Prompting (-70%), Static Context Windows (-55%)
- 🔮 **Next Wave:** World Models, Self-play Training, Neuromorphic Chips, Context Folding
- **Current Hot Topics:** 🔥 Titans & Test-Time Memory | 🔥 DeepSeek Reasoning | 🔥 Autonomous Code Agents | 🔥 200K+ Token Windows

</div>

---

## 🤝 Contributing

<div align="center">

### **Add a paper in 30 seconds!**

<a href="https://github.com/puneet-chandna/awesome-LLM-papers/issues/new?assignees=&labels=new-paper&template=new-paper.yml&title=%5BPaper%5D%3A+">
  <img src="https://img.shields.io/badge/Submit%20Paper-Click%20Here-blue?style=for-the-badge&logo=github" alt="Submit Paper">
</a>

**Just need:** Paper link + One sentence on why it matters

✅ Reviewed within 24 hours | 🏆 Contributors get credit | 💬 Join discussions

</div>

<!--


## 📧 Stay Updated

<div align="center">

| **Daily Updates** | **Weekly Digest** | **Community** |
|:---:|:---:|:---:|
| ⭐ Star & Watch this repo | 📧 [Newsletter](https://newsletter-link.com) (500+ subscribers) | 💬 [Discord](https://discord.gg/xxxxx) (200+ members) |
| Get notifications for daily picks | Curated weekend reading | Discuss papers with researchers |

</div>

---

## 🙏 Acknowledgments

<div align="center">

### Top Contributors This Month

<a href="https://github.com/contributor1"><img src="https://github.com/contributor1.png" width="50" height="50" style="border-radius: 50%;" alt="contributor1"/></a>
<a href="https://github.com/contributor2"><img src="https://github.com/contributor2.png" width="50" height="50" style="border-radius: 50%;" alt="contributor2"/></a>
<a href="https://github.com/contributor3"><img src="https://github.com/contributor3.png" width="50" height="50" style="border-radius: 50%;" alt="contributor3"/></a>
<a href="https://github.com/contributor4"><img src="https://github.com/contributor4.png" width="50" height="50" style="border-radius: 50%;" alt="contributor4"/></a>
<a href="https://github.com/contributor5"><img src="https://github.com/contributor5.png" width="50" height="50" style="border-radius: 50%;" alt="contributor5"/></a>

 **Special thanks to our [23 contributors](https://github.com/yourusername/daily-papers-llm/graphs/contributors)**

</div>
 -->

---

<div align="center">

**[⬆ Back to Top](#-awesome-LLM-papers)**

Made with ❤️ for the AI Research Community

_Last updated: January 2, 2026, 7:00 PM IST_

[![Follow on Twitter](https://img.shields.io/twitter/follow/puneet_chandna_?style=social)](https://x.com/puneet_chandna_)
[![GitHub followers](https://img.shields.io/github/followers/puneet-chandna?style=social)](https://github.com/puneet-chandna)

</div>
