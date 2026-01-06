# 🎯 Training & Alignment: Methods for Building Better LLMs

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-16+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2017--2025-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of papers on training methodologies, alignment techniques, and fine-tuning strategies

_From RLHF to parameter-efficient methods, these papers define how we train and align modern language models._

</div>

---

## 📑 Table of Contents

- [🎯 RLHF & Alignment](#-rlhf--alignment)
- [🔧 Parameter-Efficient Fine-Tuning (PEFT)](#-parameter-efficient-fine-tuning-peft)
- [📚 Instruction Tuning](#-instruction-tuning)
- [🔄 Self-Improvement](#-self-improvement)
- [🆕 Recent Breakthroughs](#-recent-breakthroughs)

---

## 🎯 RLHF & Alignment

### 📄 [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Christiano et al. (OpenAI, DeepMind)  
**Contribution:** `🏗️ Foundation`

> The foundational paper that introduced **Reinforcement Learning from Human Feedback (RLHF)**. This work demonstrated that agents could learn complex behaviors by training on human preferences between pairs of trajectory segments, rather than requiring explicit reward functions. This paradigm of learning from comparative human feedback became the cornerstone of modern LLM alignment.

---

### 📄 [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Ouyang et al. (OpenAI)  
**Contribution:** `🧑‍🏫 Instruction Following`

> Detailed the **three-step RLHF methodology** that powered ChatGPT. The process involves: (1) supervised fine-tuning on human demonstrations, (2) training a reward model on human preference comparisons, and (3) optimizing the policy using PPO against the reward model. This paper made models dramatically more helpful, truthful, and aligned with human intent.

---

### 📄 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Rafailov et al. (Stanford)  
**Contribution:** `⚡ Simplified Alignment`

> Introduced **Direct Preference Optimization**, a simpler alternative to RLHF that eliminates the need for a separate reward model and reinforcement learning. DPO directly optimizes the language model on preference data using a clever reparameterization of the reward function. This approach is more stable, computationally efficient, and has become the go-to method for preference-based fine-tuning.

---

### 📄 [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Bai et al. (Anthropic)  
**Contribution:** `🛡️ Scalable Safety`

> Introduced **Reinforcement Learning from AI Feedback (RLAIF)** and the concept of Constitutional AI. Instead of relying solely on human labelers, this method uses AI-generated feedback based on a set of principles (a "constitution") to train models to be helpful and harmless. This approach offers a more scalable and consistent path to safety alignment.

---

## 🔧 Parameter-Efficient Fine-Tuning (PEFT)

### 📄 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Hu et al. (Microsoft)  
**Contribution:** `💡 Efficient Adaptation`

> Introduced **Low-Rank Adaptation (LoRA)**, a breakthrough technique for efficient fine-tuning. Instead of updating all model parameters, LoRA freezes the pre-trained weights and injects trainable low-rank decomposition matrices into each layer. This reduces trainable parameters by 10,000x while maintaining or improving model quality, making fine-tuning accessible on consumer hardware.

---

### 📄 [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

**Authors:** Li & Liang (Stanford)  
**Contribution:** `🎛️ Soft Prompts`

> Proposed **Prefix-Tuning**, an early parameter-efficient method that prepends trainable continuous vectors ("prefixes") to the input at each Transformer layer. By keeping the entire LLM frozen and only training these small prefix parameters, the method achieves competitive performance with full fine-tuning while requiring only 0.1% of the parameters.

---

### 📄 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

**Authors:** Dettmers et al. (University of Washington)  
**Contribution:** `🗜️ Memory Efficiency`

> Combined 4-bit quantization with LoRA to enable **fine-tuning of 65B parameter models on a single 48GB GPU**. QLoRA introduced innovations like 4-bit NormalFloat quantization and Double Quantization, reducing memory usage without sacrificing performance. This democratized fine-tuning of large models, making it accessible to researchers without massive compute resources.

---

## 📚 Instruction Tuning

### 📄 [Finetuned Language Models are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Wei et al. (Google)  
**Contribution:** `🎯 Task Generalization`

> Demonstrated that **instruction tuning**—fine-tuning on a diverse collection of NLP tasks phrased as natural language instructions—dramatically improves zero-shot performance on unseen tasks. FLAN showed that models could learn to follow instructions in general, not just perform specific tasks, laying the groundwork for general-purpose AI assistants.

---

### 📄 [Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416)

**Authors:** Chung et al. (Google)  
**Contribution:** `📈 Scaling Instructions`

> Scaled instruction tuning to 1,800+ tasks and demonstrated that **instruction tuning benefits scale with both model size and number of tasks**. The resulting Flan-T5 and Flan-PaLM models showed substantial improvements over their base versions, with Flan-PaLM achieving state-of-the-art on many benchmarks and outperforming larger models.

---

### 📄 [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)

**Authors:** Taori et al. (Stanford)  
**Contribution:** `🦙 Accessible Fine-tuning`

> Demonstrated that a strong instruction-following model could be created by **fine-tuning LLaMA on 52K instruction-response pairs generated by GPT-3.5**. Alpaca showed that high-quality instruction-tuned models could be produced at low cost (~$600), sparking a wave of open-source instruction-tuned models and making LLM customization accessible to the broader community.

---

### 📄 [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

**Authors:** Wang et al. (University of Washington)  
**Contribution:** `🔄 Self-Generated Data`

> Introduced a framework for **improving instruction-following capabilities using self-generated data**. The method bootstraps from a small seed set of instructions, using the model itself to generate new instructions, inputs, and outputs. This approach reduces reliance on expensive human annotation and enables continuous improvement of instruction-following abilities.

---

## 🔄 Self-Improvement

### 📄 [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)](https://arxiv.org/abs/2401.01335)

**Authors:** Chen et al. (UCLA)  
**Contribution:** `♾️ Self-Play`

> Introduced **Self-Play Fine-Tuning (SPIN)**, where a model improves by playing against itself. The model learns to distinguish its own responses from human-generated ones, iteratively improving without additional human annotation. This self-play mechanism enables continuous improvement and has shown to boost performance on various benchmarks.

---

### 📄 [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)

**Authors:** Yuan et al. (Meta AI)  
**Contribution:** `🎯 Self-Reward`

> Introduced **Self-Rewarding Language Models**, where the model acts as its own reward model during training. The LLM generates responses, then judges and scores its own outputs to create preference pairs for DPO training. This iterative self-improvement loop enables models to continuously enhance themselves without external feedback.

---

### 📄 [Direct Language Model Alignment from Online AI Feedback (OAIF)](https://arxiv.org/abs/2402.04792)

**Authors:** Guo et al. (Google DeepMind)  
**Contribution:** `🔄 Online Learning`

> Proposed **Online AI Feedback (OAIF)**, which generates preference data on-the-fly during training using an LLM judge. Unlike offline methods like DPO that use static preference datasets, OAIF continuously generates fresh comparisons, leading to better alignment and reduced overfitting. This online approach bridges the gap between offline preference optimization and online RL methods.

---

## 🆕 Recent Breakthroughs

### 📄 🆕 [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

**Authors:** Hong et al. (KAIST)  
**Contribution:** `🎯 Simplified Training`

> Introduced **Odds Ratio Preference Optimization (ORPO)**, which combines supervised fine-tuning and preference alignment into a single training stage. Unlike DPO, ORPO doesn't require a reference model, reducing memory requirements and training complexity. This monolithic approach achieves competitive or superior results while being more efficient to train.

---

### 📄 🆕 [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

**Authors:** Ethayarajh et al. (Stanford, Contextual AI)  
**Contribution:** `📊 Human-Aligned Loss`

> Introduced **Kahneman-Tversky Optimization (KTO)**, which aligns LLMs using only binary feedback (good/bad) rather than pairwise preferences. Based on prospect theory from behavioral economics, KTO models how humans actually perceive gains and losses. This approach is more practical since binary feedback is easier to collect than preference pairs, while achieving comparable results to DPO.

### 📄 🆕 [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

**Authors:** Meng et al. (Princeton University)  
**Contribution:** `⚡ Reference-Free Alignment`

> A successor to DPO that removes the need for a **reference model** during training, making alignment significantly more memory-efficient and stable. SimPO uses length-normalized rewards and a target reward margin to directly optimize policy models on preference data. This simpler approach became the default fine-tuning method for many open-source models (like Llama 3 variants) due to its superior efficiency and comparable or better performance than DPO.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more training and alignment papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

Thanks to all researchers advancing the science of training and aligning language models.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
