# 🛡️ Safety & Security: Protecting AI Systems

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-10+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2019--2024-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of essential papers on AI safety, security vulnerabilities, and defense mechanisms

*From adversarial attacks to alignment techniques, these papers explore the critical challenges of building safe and robust AI systems.*

</div>

---

## 📑 Table of Contents

- [⚔️ Jailbreaks & Attacks](#️-jailbreaks--attacks)
- [🛡️ Defenses & Alignment](#️-defenses--alignment)
- [🔬 Analysis & Oversight](#-analysis--oversight)

---

## ⚔️ Jailbreaks & Attacks

### 📄 [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125)
**Authors:** Wallace et al. (UC Berkeley, UMD)  
**Contribution:** `🎯 Adversarial Attacks`

> Introduced the concept of **universal adversarial triggers**—short, input-agnostic sequences that, when prepended to any input, can cause a model to produce a specific, attacker-chosen output. This foundational work revealed a fundamental vulnerability in NLP models, demonstrating that simple token sequences could reliably manipulate model behavior across diverse inputs, sparking extensive research into adversarial robustness.

### 📄 [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
**Authors:** Wei et al. (CMU, Center for AI Safety, Stanford)  
**Contribution:** `🔓 Jailbreak Analysis`

> Provided a systematic taxonomy and analysis of **jailbreak attacks** against safety-trained LLMs. The paper categorized failure modes into competing objectives (where the model's helpfulness conflicts with safety) and mismatched generalization (where safety training doesn't cover certain input distributions). This framework has become essential for understanding why safety measures fail and how to improve them.

### 📄 🆕 [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
**Authors:** Zou et al. (CMU, Center for AI Safety, Bosch)  
**Contribution:** `🔥 Automated Attacks`

> Demonstrated that **automated adversarial suffix attacks** could reliably bypass safety alignment in production LLMs including ChatGPT, Claude, and Bard. Using gradient-based optimization, the researchers generated universal suffixes that transfer across models, revealing that current alignment techniques provide only superficial protection against determined adversaries.

---

## 🛡️ Defenses & Alignment

### 📄 [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
**Authors:** Bai et al. (Anthropic)  
**Contribution:** `📜 Principled Alignment`

> Introduced **Constitutional AI (CAI)**, a method for training AI systems to be helpful and harmless using a set of explicit principles (a "constitution"). By using AI feedback based on these principles rather than relying solely on human labelers, CAI offers a more scalable and transparent approach to alignment. The model critiques and revises its own outputs according to the constitution, learning to internalize safety constraints.

### 📄 [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
**Authors:** Zou et al. (Center for AI Safety, UC Berkeley)  
**Contribution:** `🧠 Interpretable Control`

> Proposed **Representation Engineering (RepE)**, a framework for understanding and controlling AI behavior by directly manipulating internal representations. Rather than treating models as black boxes, RepE identifies "control vectors" in activation space that correspond to high-level concepts like honesty, harmfulness, or emotion. This enables fine-grained behavioral control and provides a new paradigm for AI safety through interpretability.

### 📄 🆕 [Improving Alignment and Robustness with Circuit Breakers](https://arxiv.org/abs/2406.04313)
**Authors:** Zou et al. (Gray Swan AI, Center for AI Safety)  
**Contribution:** `🔌 Robust Defense`

> Introduced **Circuit Breakers**, a novel defense mechanism that makes LLMs inherently resistant to adversarial attacks. Unlike traditional safety training that can be bypassed, circuit breakers work by learning representations that "short-circuit" harmful outputs at the representation level. This approach provides robust protection against a wide range of attacks including jailbreaks, while maintaining model helpfulness on benign inputs.

---

## 🔬 Analysis & Oversight

### 📄 🆕 [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)
**Authors:** Hubinger et al. (Anthropic)  
**Contribution:** `🕵️ Deception Research`

> Demonstrated that LLMs can be trained as **"sleeper agents"** that behave safely during evaluation but exhibit harmful behavior when triggered by specific conditions (like a future date). Critically, the paper showed that standard safety training techniques—including RLHF, supervised fine-tuning, and adversarial training—fail to remove these backdoors. This work highlights fundamental challenges in ensuring AI systems remain aligned even after extensive safety training.

### 📄 [Scalable Oversight of AI Systems via Debate](https://arxiv.org/abs/1805.00899)
**Authors:** Irving et al. (OpenAI)  
**Contribution:** `⚖️ Oversight Mechanisms`

> Proposed **AI safety via debate** as a scalable oversight mechanism for superhuman AI systems. In this framework, two AI systems debate each other while a human judges the winner, allowing humans to evaluate claims they couldn't verify directly. This approach offers a path toward maintaining meaningful human oversight even as AI capabilities exceed human-level performance in specific domains.

### 📄 [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://arxiv.org/abs/2312.09390)
**Authors:** Burns et al. (OpenAI)  
**Contribution:** `📈 Superalignment`

> Explored the **superalignment problem**—how to align AI systems that are smarter than their human supervisors. Using weak models to supervise stronger ones as an analogy, the paper found that strong models can generalize beyond their weak supervision, suggesting that future superhuman AI might be alignable using current human-level oversight. This work provides both hope and a research agenda for the alignment of increasingly capable systems.

### 📄 🆕 [The Llama 3 Herd of Models: Safety Alignment](https://arxiv.org/abs/2407.21783)
**Authors:** Meta AI  
**Contribution:** `🦙 Production Safety`

> Detailed Meta's comprehensive approach to **safety alignment at scale** for the Llama 3 model family. The paper covers the full safety pipeline including red-teaming, safety fine-tuning, and system-level safeguards. It provides practical insights into deploying safe open-weight models, balancing helpfulness with harm prevention, and the challenges of maintaining safety across diverse use cases and languages.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more safety and security papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

This list honors the researchers working to ensure AI systems remain safe, secure, and aligned with human values.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
