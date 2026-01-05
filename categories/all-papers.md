# 📚 All Papers - Chronological Index

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-141-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2017--2025-green.svg)](https://github.com)

**Complete chronological listing of all LLM papers in this repository**

_Sorted from newest to oldest • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)_

[2025](#-2025) • [2024](#-2024) • [2023](#-2023) • [2022](#-2022) • [2021](#-2021) • [2020](#-2020) • [2019](#-2019) • [2018](#-2018) • [2017](#-2017)

</div>

---

## 🔥 2025

### [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)

**Salesforce AI** • `Reasoning` `Self-Improvement` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Model autonomously generates reasoning problems and improves through self-play without human data

### [VL-JEPA: Joint Embedding Predictive Architecture for Video/World Modeling](https://arxiv.org/abs/2512.10942)

**Meta AI** • `Multimodal` `World Models`

> Learns physics and dynamics from video by predicting embeddings rather than pixels

### [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)

**DeepSeek AI** • `Multimodal` `Compression`

> Achieves 97% OCR accuracy at 10x compression, processing 200K+ pages/day on single GPU

### [The Illusion of Thinking: Understanding Strengths and Limitations of Reasoning Models](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)

**Apple ML Research** • `Analysis` `Reasoning`

> Reveals that reasoning models "give up" on hard tasks and reasoning is often an illusion

### [HAPE: Hardware-Aware LLM Pruning For Efficient On-Device Inference](https://dl.acm.org/doi/epdf/10.1145/3744244)

**TODAES** • `Efficiency` `Pruning`

> Hardware-specific pruning considering device constraints for efficient edge deployment

### [Janus-Pro: Unified Multimodal Understanding and Generation](https://arxiv.org/pdf/2501.17811)

**DeepSeek AI** • `Multimodal` `Architecture`

> Decouples visual encoding for understanding vs generation, excelling at both without trade-offs

### [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/pdf/2512.24880)

**DeepSeek AI** • `Architecture` `Stability`

> Stabilizes residual connections at scale through manifold projection

### [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)

**DeepSeek AI** • `Reasoning` `RL` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Rewards correct reasoning steps, not just final answers, boosting problem-solving

### [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

**Google Research** • `Architecture` `Memory` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Revolutionary memory architecture handling 2M+ tokens with persistent test-time memory

---

## 🚀 2024

### [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

**Princeton** • `Training` `Alignment`

> Reference-free successor to DPO with superior stability and memory efficiency

### [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)

**Meta FAIR** • `Multimodal` `Early Fusion`

> Tokenizes images and text together from start, generating mixed content natively

### [Mixture-of-Depths: Dynamically Allocating Compute](https://arxiv.org/abs/2404.02258)

**Google DeepMind** • `Architecture` `Efficiency`

> Tokens skip layers based on complexity, saving 12-50% compute

### [GraphRAG: Unlocking LLM Discovery on Narrative Data](https://arxiv.org/abs/2404.16130)

**Microsoft** • `RAG` `Knowledge Graph`

> Builds knowledge graphs first, enabling "global" questions standard RAG can't answer

### [Leave No Context Behind: Infini-attention](https://arxiv.org/abs/2404.07143)

**Google** • `RAG` `Memory`

> Combines local attention with compressive memory for infinite-length processing

### [Many-Shot Jailbreaking](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf)

**Anthropic** • `Safety` `Attacks`

> Shows 100+ fake examples in context override safety training via in-context learning

### [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

**Meta AI** • `Architecture` `Open Models` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> State-of-the-art open models rivaling closed-source systems

### [RAG vs Long-Context LLMs: A Comprehensive Study](https://arxiv.org/abs/2407.16833)

**EMNLP** • `RAG` `Analysis`

> Proves RAG still needed despite 1M+ context - cheaper and more accurate

### [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

**AI21 Labs** • `Architecture` `Hybrid`

> Successfully combines Transformer and Mamba layers for 256K+ context

### [Gemini 1.5: Multimodal Understanding Across Millions of Tokens](https://arxiv.org/abs/2403.05530)

**Google** • `Multimodal` `Long-Context`

> MoE architecture processing 10M tokens of multimodal context

### [MM1: Methods, Analysis & Insights from Multimodal Pre-training](https://arxiv.org/abs/2403.09611)

**Apple** • `Multimodal` `Analysis`

> Rigorous ablation study providing recipe book for multimodal models

### [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)

**KAIST** • `Training` `Alignment`

> Combines SFT and preference alignment without reference model

### [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)

**Meta, ETH Zurich** • `Efficiency` `Pruning`

> Up to half of LLM layers can be removed with minimal impact

### [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)

**Microsoft, UW-Madison** • `Multimodal` `Instruction`

> Connects vision encoder to LLM, pioneering visual instruction following

### [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

**OpenAI** • `Multimodal` `Intelligence` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Large multimodal model with human-level performance on professional exams

### [The Era of 1-bit LLMs: BitNet b1.58](https://arxiv.org/abs/2402.17764)

**Microsoft Research** • `Efficiency` `Quantization`

> Ternary weights {-1, 0, 1} matching FP16 performance

### [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

**Stanford, Contextual AI** • `Training` `Alignment`

> Uses binary feedback instead of pairwise preferences

### [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)

**AI2** • `Architecture` `Open Science` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Fully open with complete training data, code, and tools

### [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

**ByteDance, UW-Madison** • `Multimodal` `Vision-Language`

> GPT-4V-level performance with dynamic high-resolution processing

### [Self-Play Fine-Tuning (SPIN)](https://arxiv.org/abs/2401.01335)

**UCLA** • `Training` `Self-Improvement`

> Model improves by distinguishing its outputs from human ones

### [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)

**Meta AI** • `Training` `Self-Improvement`

> Model judges its own outputs for continuous self-improvement

### [RAPTOR: Recursive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

**Stanford** • `RAG` `Hierarchical`

> Builds hierarchical tree of summaries for multi-level retrieval

### [DeepSeek-V3.2: Reasoning Rival](https://arxiv.org/pdf/2512.02556)

**DeepSeek AI** • `Reasoning` `Math`

> Open-source reasoning model rivaling GPT-5 in mathematics and logic at 1/10th the cost

---

## 🌍 2023

### [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238)

**Shanghai AI Lab** • `Multimodal` `Vision-Language`

> Scales vision encoder to 6B parameters for SOTA multimodal understanding

### [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

**CMU, Princeton** • `Architecture` `SSM` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Selective state space model achieving Transformer quality with linear complexity

### [Mistral 7B](https://arxiv.org/abs/2310.06825)

**Mistral AI** • `Architecture` `Efficiency` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Outperforms Llama 13B using Grouped-Query and Sliding Window Attention

### [MemGPT: LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

**UC Berkeley** • `RAG` `Memory`

> Treats context as virtual memory with OS-like paging

### [Ring Attention: Near-Infinite Context](https://arxiv.org/abs/2310.01889)

**UC Berkeley** • `RAG` `Infinite Context`

> Distributes attention across devices for virtually unlimited sequences

### [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)

**UW, IBM** • `RAG` `Self-Reflection`

> Model learns when to retrieve and self-critiques with reflection tokens

### [Representation Engineering: Top-Down AI Transparency](https://arxiv.org/abs/2310.01405)

**Center for AI Safety** • `Safety` `Interpretability`

> Directly manipulates internal representations for behavioral control

### [LongLoRA: Efficient Fine-tuning of Long-Context LLMs](https://arxiv.org/abs/2309.12307)

**CUHK, MIT** • `RAG` `Long-Context`

> Extends context to 100K+ tokens with minimal compute

### [Efficient Memory Management with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)

**UC Berkeley** • `Efficiency` `Inference`

> Virtual memory paging for attention, 2-4x throughput increase

### [Graph of Thoughts: Solving Elaborate Problems](https://arxiv.org/abs/2308.09687)

**ETH Zurich** • `Reasoning` `Structured`

> Models thoughts as graphs for merging and refining reasoning

### [Universal Transferable Adversarial Attacks on Aligned LLMs](https://arxiv.org/abs/2307.15043)

**CMU, Center for AI Safety** • `Safety` `Attacks`

> Automated suffixes bypass safety in ChatGPT, Claude, Bard

### [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

**Princeton** • `Efficiency` `Attention`

> 2x speedup over FlashAttention with optimized parallelism

### [Retentive Network: A Successor to Transformer](https://arxiv.org/abs/2307.08621)

**Microsoft, Tsinghua** • `Architecture` `Retention`

> Training parallelism, low-cost inference, and linear complexity simultaneously

### [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

**Meta** • `Architecture` `Open Models` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Open models with commercial license and comprehensive safety

### [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)

**CMU, Center for AI Safety** • `Safety` `Analysis`

> Systematic taxonomy of jailbreak attacks and failure modes

### [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

**Meta AI** • `Reasoning` `Tool Use`

> Models autonomously learn when and how to use external tools

### [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174)

**UC Santa Barbara, Microsoft** • `RAG` `Memory`

> Decoupled memory module for arbitrarily long histories

### [Wanda: Simple and Effective Pruning](https://arxiv.org/abs/2306.11695)

**CMU, Meta** • `Efficiency` `Pruning`

> Pruning by weights and activations, no retraining needed

### [Extending Context via Positional Interpolation](https://arxiv.org/abs/2306.15595)

**Meta AI** • `RAG` `Context Extension`

> Position Interpolation extends from 2K to 32K+ tokens

### [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685)

**UC Berkeley, UCSD, CMU, Stanford** • `Analysis` `Evaluation`

> Strong LLMs as reliable judges, creating Chatbot Arena standard

### [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

**MIT, NVIDIA** • `Efficiency` `Quantization`

> Protects critical 1% of weights for better 4-bit accuracy

### [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601)

**Princeton, Google DeepMind** • `Reasoning` `Structured`

> Tree-structured exploration with backtracking for complex problems

### [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

**University of Washington** • `Efficiency` `Training`

> 4-bit quantization with LoRA for fine-tuning 65B on single GPU

### [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

**RWKV Foundation** • `Architecture` `Linear RNN`

> Parallelizable training with efficient RNN inference

### [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)

**Stanford** • `Training` `Alignment`

> Simpler alternative to RLHF without separate reward model

### [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)

**Anthropic, NYU** • `Analysis` `Faithfulness`

> CoT explanations can be systematically unfaithful to actual reasoning

### [Reflexion: Language Agents with Verbal RL](https://arxiv.org/abs/2303.11366)

**Northeastern, MIT** • `Reasoning` `Self-Reflection`

> Agents learn from mistakes through verbose self-reflection

### [Scaling Monosemanticity: Extracting Interpretable Features](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

**Anthropic** • `Analysis` `Interpretability`

> Sparse autoencoders extract millions of monosemantic features from Claude

### [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

**OpenAI** • `Multimodal` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Multimodal model with human-level performance on professional exams

### [LLaMA: Open and Efficient Foundation Models](https://arxiv.org/abs/2302.13971)

**Meta** • `Architecture` `Open Source` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Democratized access to SOTA LLMs, sparking open-source revolution

### [Hyena Hierarchy: Towards Larger Convolutional LMs](https://arxiv.org/abs/2302.10866)

**Stanford, Hazy Research** • `Architecture` `Subquadratic`

> Subquadratic attention replacement using long convolutions

### [SparseGPT: Massive Models Can Be Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

**IST Austria** • `Efficiency` `Pruning`

> One-shot pruning to 50-60% sparsity without retraining

---

## 💡 2022

### [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

**Anthropic** • `Safety` `Alignment` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> RLAIF using AI-generated feedback based on constitutional principles

### [Self-Instruct: Aligning with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

**University of Washington** • `Training` `Instruction`

> Bootstraps from small seed set using self-generated instructions

### [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)

**Stanford** • `Analysis` `Evaluation`

> Multi-dimensional evaluation across accuracy, robustness, fairness, efficiency

### [BLOOM: 176B-Parameter Open-Access Multilingual Model](https://arxiv.org/abs/2211.05100)

**BigScience Workshop** • `Architecture` `Open Science` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Collaborative effort with 1000+ researchers for open multilingual model

### [Scaling Instruction-Finetuned LMs (Flan-T5/PaLM)](https://arxiv.org/abs/2210.11416)

**Google** • `Training` `Instruction`

> Instruction tuning benefits scale with model size and task count

### [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)

**IST Austria** • `Efficiency` `Quantization`

> One-shot 3-4 bit quantization enabling local LLM deployment

### [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

**Princeton, Google** • `Reasoning` `Agents`

> Interleaves reasoning traces with actions for better decision-making

### [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

**Google, DeepMind, Stanford, UNC** • `Analysis` `Emergence` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Formalized emergent abilities appearing unpredictably at scale

### [Beyond the Imitation Game: BIG-bench](https://arxiv.org/abs/2206.04615)

**Google, et al.** • `Analysis` `Evaluation`

> 200+ tasks from 450+ researchers probing beyond standard benchmarks

### [CogVideo: Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)

**Tsinghua, BAAI** • `Multimodal` `Video`

> Large-scale text-to-video generation with temporal consistency

### [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

**Stanford** • `Efficiency` `Attention`

> IO-aware attention 2-4x faster with 5-20x less memory

### [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

**Google** • `Architecture` `Performance` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> 540B parameters achieving SOTA with breakthrough reasoning capabilities

### [Flamingo: Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)

**DeepMind** • `Multimodal` `Few-Shot`

> Cross-attention between frozen vision and language models

### [DALL-E 2: Hierarchical Text-Conditional Image Generation](https://arxiv.org/abs/2204.06125)

**OpenAI** • `Multimodal` `Generation`

> Two-stage system generating photorealistic images from text

### [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)

**Stanford** • `Analysis` `Critical`

> Apparent emergence can be artifact of metric choice

### [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)

**DeepMind** • `Architecture` `Optimization` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Smaller models on more data outperform larger models

### [VideoMAE: Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)

**Nanjing University, Tencent** • `Multimodal` `Video`

> Extremely high masking ratios (90-95%) work well for video

### [InstructGPT: Training with Human Feedback](https://arxiv.org/abs/2203.02155)

**OpenAI** • `Training` `Alignment` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Three-step RLHF methodology powering ChatGPT

### [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

**Google** • `Reasoning` `Prompting` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> "Think step-by-step" dramatically improves reasoning without model changes

### [LaMDA: Language Models for Dialog](https://arxiv.org/abs/2201.08239)

**Google** • `Architecture` `Dialogue` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Pre-trained on dialogue data for open-ended conversations

---

## 🧮 2021

### [Stable Diffusion: Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

**LMU Munich, Runway** • `Multimodal` `Generation`

> Diffusion in compressed latent space for efficient high-res generation

### [Scaling Language Models: Training Gopher](https://arxiv.org/abs/2112.11446)

**DeepMind** • `Architecture` `Scaling` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Comprehensive 280B parameter analysis across 152 tasks

### [Finetuned Language Models are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652)

**Google** • `Training` `Instruction` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Instruction tuning improves zero-shot on unseen tasks

### [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)

**Stanford** • `Analysis` `Concept` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Coined "Foundation Models" term with comprehensive analysis

### [Codex: Evaluating LLMs Trained on Code](https://arxiv.org/abs/2107.03374)

**OpenAI** • `Architecture` `Code` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Demonstrated code reasoning abilities, led to GitHub Copilot

### [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)

**Microsoft** • `Training` `PEFT`

> Efficient fine-tuning with 10,000x fewer parameters

### [Switch Transformers: Scaling to Trillion Parameters](https://arxiv.org/abs/2101.03961)

**Google** • `Architecture` `MoE` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Sparse MoE enabling trillion-parameter models

### [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190)

**Stanford** • `Training` `PEFT`

> Trainable continuous prefixes keeping LLM frozen

### [Learning Transferable Visual Models (CLIP)](https://arxiv.org/abs/2103.00020)

**OpenAI** • `Multimodal` `Foundation`

> Joint embedding space enabling zero-shot visual classification

---

## 📈 2020

### [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

**OpenAI** • `Architecture` `Emergence` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Proved scale unlocks emergent in-context learning abilities

### [REALM: Retrieval-Augmented LM Pre-Training](https://arxiv.org/abs/2002.08909)

**Google Research** • `RAG` `Pre-training`

> Pre-training with retrieval for learning to use external knowledge

### [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401)

**Facebook AI, UCL, NYU** • `RAG` `Architecture`

> Foundational RAG combining parametric and non-parametric memory

### [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**OpenAI** • `Analysis` `Scaling` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Established predictable power-law scaling relationships

### [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300)

**UC Berkeley** • `Analysis` `Benchmark`

> Comprehensive benchmark across 57 subjects

### [Memorizing Transformers](https://arxiv.org/abs/2203.08913)

**Google Research** • `RAG` `Memory`

> kNN-based external memory for attending over past activations

---

## 🚀 2019

### [Exploring Transfer Learning with T5](https://arxiv.org/abs/1910.10683)

**Google** • `Architecture` `Framework` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Unified text-to-text framework for all NLP tasks

### [ZeRO: Memory Optimizations Toward Trillion Parameters](https://arxiv.org/abs/1910.02054)

**Microsoft** • `Efficiency` `Memory` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Partitioning model states enabling trillion-parameter training

### [Megatron-LM: Training Multi-Billion Parameter Models](https://arxiv.org/abs/1909.08053)

**NVIDIA** • `Architecture` `Engineering` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Tensor and pipeline parallelism for multi-billion parameter models

### [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**OpenAI** • `Architecture` `Zero-Shot` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Demonstrated zero-shot task performance through scaling

### [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

**Zhuiyi Technology** • `Architecture` `Position Encoding`

> RoPE becoming standard for modern LLMs and context extension

### [Universal Adversarial Triggers for Attacking NLP](https://arxiv.org/abs/1908.07125)

**UC Berkeley, UMD** • `Safety` `Attacks`

> Short sequences causing models to produce attacker-chosen outputs

### [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192)

**Google** • `Efficiency` `Inference`

> Draft model generates candidates verified in parallel, 2-3x speedup

---

## 🌱 2018

### [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

**Google** • `Architecture` `Understanding` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Bidirectional pre-training with MLM revolutionizing NLP understanding

### [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

**OpenAI** • `Architecture` `Pre-training` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Established generative pre-training paradigm for language models

---

## 🏗️ 2017

### [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Google** • `Architecture` `Foundation` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Created Transformer architecture, foundation of all modern LLMs

### [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)

**OpenAI, DeepMind** • `Training` `Alignment` • ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

> Foundational RLHF framework learning from human preferences

---

<div align="center">

## 🌟 Contributing

Found a missing paper? [Submit it here](https://github.com/puneet-chandna/awesome-LLM-papers/issues/new)!

---

⭐ If you find this index helpful, please star the repository!

**[⬆ Back to Top](#-all-papers---chronological-index)**

</div>
