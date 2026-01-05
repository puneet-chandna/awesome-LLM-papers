# 🔬 Analysis & Theory: Understanding How LLMs Work

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-11+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2020--2024-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of papers on interpretability, mechanistic analysis, and evaluation of Large Language Models

_Understanding the inner workings of LLMs—from circuit-level analysis to emergent behaviors and rigorous evaluation methodologies._

</div>

---

## 📑 Table of Contents

- [🔍 Interpretability](#-interpretability)
- [🪄 Emergent Abilities](#-emergent-abilities)
- [📊 Evaluation](#-evaluation)

---

## 🔍 Interpretability

### 📄 [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

**Authors:** Elhage et al. (Anthropic)  
**Contribution:** `🔬 Mechanistic Interpretability`

> Established the foundational **mathematical framework for understanding Transformers as computational circuits**. This seminal work introduced key concepts like the residual stream, attention heads as information movers, and MLPs as memory stores. It laid the groundwork for mechanistic interpretability, enabling researchers to reverse-engineer how specific computations emerge from model weights.

### 📄 [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

**Authors:** Olsson et al. (Anthropic)  
**Contribution:** `🧩 Circuit Discovery`

> Identified **induction heads**—a specific circuit pattern responsible for in-context learning in Transformers. This paper demonstrated that a simple two-attention-head circuit can implement pattern matching and copying, explaining a core mechanism behind few-shot learning. It provided concrete evidence that complex behaviors emerge from identifiable, interpretable circuits.

### 📄 [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/) 🆕

**Authors:** Templeton et al. (Anthropic)  
**Contribution:** `🔎 Feature Extraction`

> Applied **sparse autoencoders at unprecedented scale** to extract millions of interpretable features from a production-grade model (Claude 3 Sonnet). This work demonstrated that even the largest models contain monosemantic features—neurons that respond to specific, human-understandable concepts—providing a scalable path toward understanding what knowledge LLMs encode and how they represent it.

### 📄 [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)

**Authors:** Zou et al. (Center for AI Safety)  
**Contribution:** `🎛️ Representation Control`

> Introduced **Representation Engineering (RepE)**, a framework for understanding and controlling LLM behavior by directly manipulating internal representations. Rather than analyzing individual neurons, RepE identifies high-level concepts in activation space and enables steering model behavior by adding or subtracting these concept vectors, offering a practical approach to AI transparency and control.

---

## 🪄 Emergent Abilities

### 📄 [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

**Authors:** Wei et al. (Google, DeepMind, Stanford, UNC)  
**Contribution:** `🪄 Emergence Theory`

> Formalized and documented **emergent abilities**—capabilities that appear suddenly and unpredictably as models scale. The paper catalogued numerous examples where performance on specific tasks remained near-random until a critical scale threshold, after which it jumped dramatically. This work shaped our understanding of why scaling matters and what surprises larger models might hold.

### 📄 [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)

**Authors:** Schaeffer et al. (Stanford)  
**Contribution:** `🔬 Critical Analysis`

> Challenged the prevailing narrative of emergent abilities by demonstrating that **apparent emergence can be an artifact of metric choice**. The paper showed that when using linear or continuous metrics instead of discontinuous ones (like exact-match accuracy), the sharp transitions disappear and performance scales smoothly. This critical analysis reshaped how researchers interpret and measure model capabilities.

### 📄 [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)

**Authors:** Turpin et al. (Anthropic, NYU)  
**Contribution:** `⚠️ Faithfulness Analysis`

> Revealed that **Chain-of-Thought explanations can be systematically unfaithful** to the model's actual reasoning process. By introducing biasing features that influenced model answers without appearing in the explanations, this work demonstrated that CoT outputs may post-hoc rationalize rather than reveal true reasoning, raising important questions about interpretability through natural language explanations.

---

## 📊 Evaluation

### 📄 [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300)

**Authors:** Hendrycks et al. (UC Berkeley)  
**Contribution:** `📏 Benchmark`

> Introduced **MMLU**, a comprehensive benchmark covering 57 subjects across STEM, humanities, social sciences, and more. With questions ranging from elementary to professional difficulty, MMLU became the de facto standard for measuring broad knowledge and reasoning capabilities in LLMs, providing a single metric that captures multitask understanding across diverse domains.

### 📄 [Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-bench)](https://arxiv.org/abs/2206.04615)

**Authors:** Srivastava et al. (Google, et al.)  
**Contribution:** `🎯 Comprehensive Evaluation`

> Created **BIG-bench**, a collaborative benchmark with over 200 tasks contributed by 450+ researchers. Designed to probe capabilities beyond standard benchmarks, it includes tasks testing linguistic knowledge, reasoning, world knowledge, and social understanding. BIG-bench revealed that model capabilities scale predictably on some tasks while showing emergent behavior on others.

### 📄 [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)

**Authors:** Liang et al. (Stanford)  
**Contribution:** `🔄 Holistic Assessment`

> Proposed **HELM**, a framework for evaluating LLMs across multiple dimensions simultaneously—accuracy, calibration, robustness, fairness, efficiency, and more. Rather than optimizing for a single metric, HELM provides a comprehensive view of model capabilities and limitations, enabling more informed comparisons and highlighting trade-offs between different aspects of performance.

### 📄 [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

**Authors:** Zheng et al. (UC Berkeley, UCSD, CMU, Stanford)  
**Contribution:** `⚖️ LLM Evaluation`

> Introduced **MT-Bench** and the **Chatbot Arena** methodology for evaluating conversational AI. This work demonstrated that strong LLMs can serve as reliable judges of other models' outputs, correlating well with human preferences. The Chatbot Arena's crowdsourced pairwise comparisons created a dynamic leaderboard that became the gold standard for comparing chat models.

### 📄 [The Illusion of Thinking: Understanding Strengths and Limitations of Reasoning Models](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) 🆕

**Authors:** Shojaee et al. (Apple Machine Learning Research, June 2025)  
**Contribution:** `🔬 Reasoning Analysis`

> A critical analysis of **Large Reasoning Models (LRMs)** like o1/R1 that reveals fundamental limitations. The paper identifies three performance regimes: (1) **low-complexity**: standard LLMs outperform LRMs; (2) **medium-complexity**: LRMs show advantage through extended thinking; (3) **high-complexity**: both experience **complete collapse**. Most importantly, it reveals that LRMs "give up" on very hard tasks—their reasoning abruptly shrinks despite available token budget. Demonstrates the apparent reasoning is often an "illusion," particularly as problem complexity increases.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more analysis and theory papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

This list honors the researchers working to understand and evaluate the systems that are reshaping our world.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
