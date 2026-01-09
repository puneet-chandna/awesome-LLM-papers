# Awesome Efficiency & Scaling

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Essential papers on making LLMs faster, smaller, and more deployable.

_From quantization breakthroughs to attention optimization, these papers enable running powerful models on limited hardware._

---

## 📑 Table of Contents

- [🔢 Quantization](#-quantization)
- [⚡ Attention Optimization](#-attention-optimization)
- [🚀 Inference Optimization](#-inference-optimization)
- [✂️ Pruning & Sparsity](#-pruning--sparsity)

---

## 🔢 Quantization

### 📄 [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

**Authors:** Frantar et al. (IST Austria)  
**Contribution:** `🔢 Post-Training Quantization`

> Introduced a highly accurate **one-shot weight quantization** method that can compress models to 3-4 bits with minimal accuracy loss. GPTQ uses approximate second-order information to quantize weights layer-by-layer, enabling billion-parameter models to run on consumer GPUs. This breakthrough made local LLM deployment practical for the first time.

### 📄 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

**Authors:** Dettmers et al. (University of Washington)  
**Contribution:** `🎯 Efficient Fine-tuning`

> Combined **4-bit quantization with Low-Rank Adaptation (LoRA)** to enable fine-tuning of 65B parameter models on a single 48GB GPU. QLoRA introduced novel techniques like 4-bit NormalFloat and Double Quantization, reducing memory requirements by up to 75% while matching full 16-bit fine-tuning performance. This democratized LLM customization for researchers with limited compute.

### 📄 [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

**Authors:** Lin et al. (MIT, NVIDIA)  
**Contribution:** `🧠 Activation-Aware Compression`

> Proposed an **activation-aware quantization** approach that identifies and protects the most important weights based on activation patterns. AWQ achieves better accuracy than GPTQ at the same bit-width by recognizing that only ~1% of weights are critical for preserving model quality. This insight led to more efficient 4-bit models with minimal degradation.

### 📄 [The Era of 1-bit LLMs: All Large Language Models Are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

**Authors:** Ma et al. (Microsoft Research)  
**Contribution:** `🔢 1-bit Quantization`

> 🆕 Introduced **BitNet b1.58**, proving that model weights can be quantized to ternary values {-1, 0, 1} (effectively ~1.58 bits) while matching the performance of full-precision FP16 models. This revolutionary approach eliminates the need for expensive matrix multiplications, replacing them with simple addition operations. Potentially transforms hardware requirements for LLM deployment, enabling massive models to run on resource-constrained devices.

---

## ⚡ Attention Optimization

### 📄 [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Dao et al. (Stanford)  
**Contribution:** `💾 IO-Aware Algorithm`

> Revolutionized attention computation by making it **IO-aware**, reducing memory reads/writes between GPU high-bandwidth memory and on-chip SRAM. FlashAttention computes exact attention 2-4x faster while using 5-20x less memory than standard implementations. This breakthrough enabled training with much longer sequences and became the foundation for efficient Transformer implementations.

### 📄 [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

**Authors:** Dao (Princeton)  
**Contribution:** `🚀 Optimized Parallelism`

> Built upon FlashAttention with **improved work partitioning** and parallelism strategies, achieving up to 2x additional speedup. FlashAttention-2 better utilizes GPU resources by reducing non-matmul FLOPs and optimizing thread block scheduling. It has become the de facto standard for attention computation in modern LLM frameworks.

---

## 🚀 Inference Optimization

### 📄 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

**Authors:** Leviathan et al. (Google)  
**Contribution:** `🎯 Parallel Decoding`

> Introduced **speculative decoding**, a technique that uses a smaller "draft" model to generate candidate tokens that are then verified in parallel by the larger target model. This approach can achieve 2-3x speedup in inference without any change to model outputs, exploiting the fact that verification is much cheaper than generation in autoregressive models.

### 📄 🆕 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**Authors:** Kwon et al. (UC Berkeley)  
**Contribution:** `💾 Memory Management`

> Introduced **PagedAttention**, a novel attention algorithm inspired by virtual memory paging in operating systems. By storing attention keys and values in non-contiguous memory blocks, vLLM achieves near-zero memory waste and enables flexible memory sharing across requests. This innovation increased serving throughput by 2-4x compared to existing systems, making it the backbone of modern LLM serving infrastructure.

---

## ✂️ Pruning & Sparsity

### 📄 [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

**Authors:** Frantar & Alistarh (IST Austria)  
**Contribution:** `✂️ One-Shot Pruning`

> Demonstrated that **massive language models can be pruned to 50-60% sparsity** in a single pass without any retraining. SparseGPT uses an efficient approximate sparse regression solver to remove weights while minimizing output error. This enables significant speedups on sparse-aware hardware while maintaining model quality, opening new paths for efficient deployment.

### 📄 [Wanda: A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

**Authors:** Sun et al. (CMU, Meta)  
**Contribution:** `🎯 Simple Pruning`

> Proposed **Pruning by Weights and Activations (Wanda)**, an extremely simple yet effective pruning method that requires no retraining or weight updates. By considering both weight magnitudes and input activations, Wanda matches or exceeds SparseGPT performance while being orders of magnitude faster to compute. This simplicity makes it highly practical for real-world deployment.

### 📄 🆕 [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)

**Authors:** Gromov et al. (Meta, ETH Zurich)  
**Contribution:** `🔬 Layer Pruning`

> Revealed that **up to half of the layers in popular LLMs can be removed** with minimal impact on performance across various benchmarks. This surprising finding suggests significant redundancy in current model architectures and opens new avenues for model compression. The paper provides practical guidance for layer pruning strategies that maintain model quality.

### 📄 🆕 [HAPE: Hardware-Aware LLM Pruning For Efficient On-Device Inference](https://dl.acm.org/doi/epdf/10.1145/3744244)

**Authors:** Wenqian Zhao  
**Contribution:** `⚙️ Hardware-Specific Pruning`

> Moves beyond generic pruning by **incorporating hardware-specific constraints directly into the pruning process**. HAPE considers memory bandwidth, compute capabilities, and power constraints of target devices (phones, laptops, edge devices) when determining which weights to prune. This hardware-aware approach enables massive models to run efficiently on consumer devices with minimal latency and energy consumption.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more efficiency papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

Thanks to all researchers pushing the boundaries of efficient AI.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
