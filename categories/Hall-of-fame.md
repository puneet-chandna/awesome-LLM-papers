# 🏆 Hall of Fame: Foundational LLM Papers

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-40+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2017--2025-green.svg)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of pivotal papers that shaped the landscape of Large Language Models

*Each entry represents a significant milestone, from foundational architectures to the latest breakthroughs in reasoning, safety, and scale.*

</div>

---

## 📑 Table of Contents

- [🏗️ 2017: The Foundation](#-2017-the-foundation)
- [🌱 2018: The Pre-training Era](#-2018-the-pre-training-era)
- [🚀 2019: Scaling Begins](#-2019-scaling-begins)
- [📈 2020: The Science of Scale](#-2020-the-science-of-scale)
- [🧮 2021: Emergence & Efficiency](#-2021-emergence--efficiency)
- [💡 2022: Reasoning & Alignment](#-2022-reasoning--alignment)
- [🌍 2023: Democratization](#-2023-democratization)
- [🥇 2024: Open Excellence](#-2024-open-excellence)
- [🔮 2025: Self-Improvement](#-2025-self-improvement)

---

## 🏗️ 2017: The Foundation

### 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
**Authors:** Vaswani et al. (Google)  
**Contribution:** `🏗️ Architecture`

> Created the **Transformer architecture**, the bedrock of all modern LLMs. It fundamentally broke from the sequential processing of recurrent networks (RNNs) by introducing self-attention. This allowed the model to weigh the importance of all words in the input simultaneously, capturing complex, long-range dependencies. More importantly, this design enabled massive parallelization during training, unlocking the ability to build and train the gigantic models we see today.

### 📄 [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
**Authors:** Christiano et al. (OpenAI, DeepMind)  
**Contribution:** `🎯 Alignment`

> Introduced the core ideas behind **Reinforcement Learning from Human Feedback (RLHF)**. This paper demonstrated that models could learn complex behaviors not from explicit goals, but by being trained on human preferences between two different outputs. This concept of learning from comparative feedback laid the essential groundwork for aligning powerful models like ChatGPT to be more helpful, harmless, and aligned with human values.

---

## 🌱 2018: The Pre-training Era

### 📄 [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
**Authors:** Radford et al. (OpenAI)  
**Contribution:** `🌱 Pre-training`

> Introduced the highly effective strategy of **generative pre-training** for language models. It established a powerful two-stage process: first, an unsupervised pre-training phase on a vast and diverse text corpus to learn general world knowledge and language structure, followed by a supervised fine-tuning phase for specific tasks. This set the paradigm for the GPT series and many subsequent models.

### 📄 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
**Authors:** Devlin et al. (Google)  
**Contribution:** `🧠 Understanding`

> Revolutionized natural language understanding by introducing the **bidirectional Transformer (BERT)**. Unlike previous models that processed text in a left-to-right or shallowly combined manner, BERT's Masked Language Model (MLM) objective allowed it to learn deep context from both directions simultaneously. This led to a profound leap in performance on understanding-based tasks like question answering and sentiment analysis.

---

## 🚀 2019: Scaling Begins

### 📄 [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
**Authors:** Radford et al. (OpenAI)  
**Contribution:** `🎯 Versatility`

> Demonstrated that a sufficiently large language model could perform a wide range of tasks—like translation, summarization, and question answering—in a **zero-shot setting**. This meant it could tackle these tasks without any explicit, task-specific training, purely through clever prompting. This discovery hinted that a single, generalist model could be a viable path forward for AI.

### 📄 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
**Authors:** Shoeybi et al. (NVIDIA)  
**Contribution:** `🛠️ Engineering`

> Pioneered and popularized techniques for training **multi-billion parameter models** that were too large to fit on a single GPU. By introducing efficient tensor and pipeline parallelism strategies, this work provided the crucial engineering blueprint that made training truly massive models not just possible, but practical across large GPU clusters.

### 📄 [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
**Authors:** Rasley et al. (Microsoft)  
**Contribution:** `💾 Memory Optimization`

> Introduced a family of powerful **memory optimization strategies (ZeRO)** that significantly reduced the memory footprint required for large-scale training. By cleverly partitioning the model states (optimizer states, gradients, and parameters) across data-parallel processes, it made it feasible to train models with trillions of parameters on existing hardware, democratizing access to large-scale training.

### 📄 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)
**Authors:** Raffel et al. (Google)  
**Contribution:** `🔄 Framework`

> Proposed a **unified and elegant text-to-text framework**, reframing every NLP task as a text generation problem (e.g., for sentiment analysis, the model would generate the string "positive" or "negative"). This simplification, combined with a comprehensive study on pre-training objectives and datasets, pushed the limits of transfer learning and influenced the design of many subsequent models.

---

## 📈 2020: The Science of Scale

### 📄 [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
**Authors:** Kaplan et al. (OpenAI)  
**Contribution:** `📈 Scaling`

> Empirically established that model performance scales predictably and follows a **power law** with increases in model size, dataset size, and compute. These "scaling laws" provided a crucial, quantitative roadmap for the industry, justifying the massive investment in larger models by showing that better performance was not just a matter of chance, but an expected outcome of scaling up.

### 📄 [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
**Authors:** Brown et al. (OpenAI)  
**Contribution:** `🚀 Emergence`

> Proved that pure scale could unlock surprising, **"emergent" abilities** not present in smaller models. GPT-3's remarkable ability to perform tasks from just a few examples provided in the prompt (in-context learning)—without any fine-tuning or gradient updates—revolutionized how we interact with and build upon LLMs, shifting the focus from model training to prompt engineering.

---

## 🧮 2021: Emergence & Efficiency

### 📄 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
**Authors:** Fedus et al. (Google)  
**Contribution:** `🧩 Efficiency`

> Successfully introduced the **Mixture-of-Experts (MoE)** architecture to Transformers at an unprecedented scale. This allowed models to grow to over a trillion parameters while keeping the computational cost manageable. For any given input, the model would only activate a small subset of its weights (the "experts"), dramatically improving training and inference efficiency and paving the way for models like Mixtral.

### 📄 [Evaluating Large Language Models Trained on Code (Codex)](https://arxiv.org/abs/2107.03374)
**Authors:** Chen et al. (OpenAI)  
**Contribution:** `💻 Code Generation`

> Demonstrated the phenomenal power of LLMs when trained on vast amounts of public code from sources like GitHub. This research, which directly led to the creation of **GitHub Copilot**, proved that complex, logical reasoning abilities could be learned from the structure and patterns inherent in code, extending the capabilities of LLMs far beyond natural language.

### 📄 [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)
**Authors:** Bommasani et al. (Stanford)  
**Contribution:** `📖 Concept`

> Coined the term **"Foundation Models"** to describe large models trained on broad data that can be adapted to a wide range of downstream tasks. The paper provided a thorough analysis of the paradigm shift they represent, creating a shared vocabulary and framework for discussing their capabilities, homogenization effects, and profound societal impact.

### 📄 [Finetuned Language Models are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652)
**Authors:** Wei et al. (Google)  
**Contribution:** `🎯 Instruction Tuning`

> Showed that fine-tuning language models on a massive collection of diverse NLP tasks expressed as natural language instructions dramatically improves their zero-shot performance on unseen tasks. This **"instruction tuning"** made models more usable and better at generalizing to what users intended, a key step towards helpful assistants.

### 📄 [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)
**Authors:** Rae et al. (DeepMind)  
**Contribution:** `📊 Scaling Insights`

> Detailed the training of the **280B parameter Gopher model**, providing one of the most comprehensive analyses of the effects of scale on model performance. The paper explored performance across 152 different tasks, identifying not only the benefits of scale but also its limitations, particularly in areas requiring deep reasoning or factual recall, guiding future research directions.

---

## 💡 2022: Reasoning & Alignment

### 📄 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
**Authors:** Wei et al. (Google)  
**Contribution:** `🧮 Reasoning`

> A simple but profound discovery that unlocked new capabilities in LLMs. By simply prompting a model to **"think step-by-step"** and generate intermediate reasoning steps before the final answer, this technique dramatically improved its ability to solve complex multi-step problems in domains like arithmetic, commonsense, and symbolic reasoning, without any change to the model itself.

### 📄 [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239)
**Authors:** Thoppilan et al. (Google)  
**Contribution:** `💬 Dialogue`

> Introduced a family of models specifically designed for **open-ended dialogue**. LaMDA was pre-trained on dialogue data and fine-tuned on metrics like quality, safety, and groundedness, allowing it to produce more sensible, specific, and interesting conversational responses compared to general-purpose models of its time.

### 📄 [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
**Authors:** Ouyang et al. (OpenAI)  
**Contribution:** `🧑‍🏫 Instruction Following`

> Detailed the **three-step RLHF methodology** that was the engine behind ChatGPT. It refined the process of aligning powerful LLMs to follow human instructions by using human-written demonstrations and preference-labeled comparisons. This made models significantly more helpful, truthful, and harmless, setting the standard for AI assistants.

### 📄 [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
**Authors:** Hoffmann et al. (DeepMind)  
**Contribution:** `⚖️ Optimization`

> Challenged the prevailing wisdom of "bigger is always better" for model size. The Chinchilla paper showed that for a given compute budget, the best performance is achieved by **training a smaller model on much more data**. This revised the original scaling laws and caused a major industry shift towards prioritizing high-quality, large-scale datasets over simply increasing parameter counts.

### 📄 [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
**Authors:** Chowdhery et al. (Google)  
**Contribution:** `🌟 Performance`

> Showcased the power of extreme scale (**540B parameters**) combined with Google's efficient Pathways system. PaLM achieved state-of-the-art performance on hundreds of language tasks and demonstrated breakthrough capabilities in few-shot learning and, notably, chain-of-thought reasoning, solving problems previously thought to be beyond the reach of LLMs.

### 📄 [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
**Authors:** Wei et al. (Google, DeepMind, Stanford, UNC)  
**Contribution:** `🪄 Emergence`

> Formalized and provided compelling evidence for the concept of **emergent abilities**—complex abilities that are not present in smaller models but appear, often unpredictably, in larger ones. This paper highlighted that the relationship between scale and capability is non-linear, explaining why simply making models bigger can unlock qualitatively new skills.

### 📄 [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)
**Authors:** BigScience Workshop  
**Contribution:** `🌍 Open Science`

> A massive, collaborative effort involving over 1,000 researchers to build and release a powerful, **open-access multilingual model**. BLOOM stands as a testament to transparent and reproducible research in the era of large-scale AI, providing a powerful artifact for the global research community to study and build upon.

### 📄 [Constitutional AI: Harmlessness from AI Feedback (RLAIF)](https://arxiv.org/abs/2212.08073)
**Authors:** Bai et al. (Anthropic)  
**Contribution:** `🛡️ Safety`

> Introduced **Reinforcement Learning from AI Feedback (RLAIF)**. This groundbreaking technique trains models to be helpful and harmless using AI-generated preference labels based on a simple "constitution" (a set of principles). This reduces the reliance on expensive and potentially biased human labeling for safety alignment, offering a more scalable path to safe AI.

---

## 🌍 2023: Democratization

### 📄 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
**Authors:** Touvron et al. (Meta)  
**Contribution:** `🌐 Open Source`

> Released a family of powerful yet efficient foundation models to the research community. While not fully open-source initially, LLaMA's availability **democratized access to state-of-the-art LLMs**, sparking a massive wave of innovation in open-source fine-tuning, quantization, and local deployment that continues to this day.

### 📄 [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
**Authors:** OpenAI  
**Contribution:** `🏆 Multimodality`

> Set a new standard for large-scale models, demonstrating exceptional performance across a wide array of benchmarks. As a **large multimodal model**, GPT-4 exhibited human-level performance on various professional and academic exams and showcased deep reasoning and the ability to seamlessly process both text and image inputs, expanding the scope of what AI assistants can do.

### 📄 [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
**Authors:** Touvron et al. (Meta)  
**Contribution:** `🔓 Open & Safe`

> A major step forward for open-source AI. Llama 2 not only improved upon the original models but was also released with a **commercial-use license** and was fine-tuned for dialogue. The release included a detailed safety report, making powerful, aligned chat models widely accessible for both research and commercial applications.

### 📄 [Mistral 7B](https://arxiv.org/abs/2310.06825)
**Authors:** Jiang et al. (Mistral AI)  
**Contribution:** `⚡ Efficiency`

> Proved that smaller, smartly designed models can outperform much larger ones. Mistral 7B used novel attention mechanisms like **Grouped-Query Attention** and **Sliding Window Attention** to achieve performance superior to Llama 13B on many benchmarks, setting a new bar for what is possible with efficient model architecture and inference.

### 📄 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
**Authors:** Gu & Dao (CMU, Princeton)  
**Contribution:** `🐍 New Architecture`

> Introduced Mamba, a **selective state space model (SSM)** that achieves Transformer-level performance but with linear-time complexity and much faster inference. By using a state-based approach instead of quadratic self-attention, it represents a major architectural challenger to the Transformer's dominance, especially for handling extremely long sequences.

---

## 🥇 2024: Open Excellence

### 📄 [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
**Authors:** Groeneveld et al. (AI2)  
**Contribution:** `🔬 Reproducibility`

> Championed a truly open approach to AI research by releasing a state-of-the-art language model while providing not just the model weights, but also the **complete training data, code, and evaluation tools**. OLMo is designed explicitly to empower researchers to dissect, study, and advance the science of language models.

### 📄 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
**Authors:** Meta AI  
**Contribution:** `🥇 SOTA Open Models`

> Released a new family of **state-of-the-art open models**, including 8B and 70B parameter versions, that set a new standard for performance at their scale. Trained on a massive, high-quality dataset, Llama 3 demonstrated significant improvements in reasoning, code generation, and instruction following, rivaling many closed-source models.

---

## 🔮 2025: Self-Improvement

### 📄 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
**Authors:** DeepSeek AI  
**Contribution:** `🧠 Advanced Reasoning`

> Introduced a novel **Reinforcement Learning framework** that directly rewards models for generating correct intermediate reasoning steps, not just the final answer. This incentivizes the model to learn more robust and generalizable reasoning paths, significantly boosting performance on complex logic, math, and coding tasks where the process is as important as the result.

### 📄 [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2501.13024)
**Authors:** Salesforce AI Research  
**Contribution:** `♾️ Self-Improvement`

> A paradigm shift in training that moves beyond human data. This paper introduces a method where a model can **autonomously generate its own reasoning problems**, attempt to solve them, and use the verifiable outcomes to improve itself through reinforcement learning. This self-play loop marks a key step towards self-improving, data-independent AI systems that can continuously enhance their own capabilities.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more foundational papers or improve existing entries!

### 📜 License

This repository is licensed under MIT License.

### 🙏 Acknowledgments

This list is a tribute to the researchers and engineers who paved the way for the AI revolution.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>