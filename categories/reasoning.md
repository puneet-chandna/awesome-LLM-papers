# 🧠 Reasoning & Agents: From Chain-of-Thought to Autonomous Systems

<div align="center">

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![Papers](https://img.shields.io/badge/Papers-10+-blue.svg)](https://github.com)
[![Years](https://img.shields.io/badge/Years-2022--2025-green.svg)](https://github.com)
[![License: CC0](https://img.shields.io/badge/License-CC0-yellow.svg)](https://opensource.org/licenses/CC0-1.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### A curated collection of papers on reasoning techniques and autonomous AI agents

_From prompting strategies that unlock step-by-step thinking to systems that can plan, use tools, and act autonomously._

</div>

---

## 📑 Table of Contents

- [💭 Chain-of-Thought Reasoning](#-chain-of-thought-reasoning)
- [🤖 Agent Systems](#-agent-systems)
- [🆕 Recent Breakthroughs](#-recent-breakthroughs)

---

## 💭 Chain-of-Thought Reasoning

### 📄 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** Wei et al. (Google)  
**Contribution:** `🧮 Reasoning`

> A simple but profound discovery that unlocked new capabilities in LLMs. By simply prompting a model to **"think step-by-step"** and generate intermediate reasoning steps before the final answer, this technique dramatically improved its ability to solve complex multi-step problems in domains like arithmetic, commonsense, and symbolic reasoning, without any change to the model itself.

---

### 📄 [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

**Authors:** Yao et al. (Princeton, Google DeepMind)  
**Contribution:** `🌳 Structured Reasoning`

> Extended chain-of-thought by introducing a **tree-structured exploration** of reasoning paths. Instead of a single linear chain, Tree of Thoughts allows the model to consider multiple different reasoning branches, evaluate their promise, and backtrack when necessary. This deliberate search process enables solving problems that require exploration, strategic lookahead, and planning.

---

### 📄 [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)

**Authors:** Besta et al. (ETH Zurich)  
**Contribution:** `🕸️ Graph Reasoning`

> Generalized the reasoning paradigm further by modeling thoughts as a **graph structure**. This allows for combining, refining, and aggregating multiple reasoning paths in ways that trees cannot capture. Graph of Thoughts enables more complex reasoning patterns where partial solutions can be merged and improved iteratively, achieving superior results on tasks like sorting and set operations.

---

## 🤖 Agent Systems

### 📄 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

**Authors:** Yao et al. (Princeton, Google)  
**Contribution:** `🔄 Reasoning + Action`

> Introduced the influential **ReAct paradigm** that interleaves reasoning traces with actions. Instead of just thinking or just acting, ReAct models generate verbal reasoning to track goals and plans, then take actions in an environment (like searching Wikipedia), and use observations to inform further reasoning. This synergy dramatically improves performance on knowledge-intensive and decision-making tasks.

---

### 📄 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

**Authors:** Shinn et al. (Northeastern, MIT)  
**Contribution:** `🪞 Self-Reflection`

> Enabled agents to **learn from their mistakes through verbal self-reflection**. After failing a task, Reflexion agents generate natural language feedback about what went wrong and store these reflections in memory. On subsequent attempts, they use this accumulated experience to avoid past errors, achieving significant improvements without any weight updates—a form of in-context reinforcement learning.

---

### 📄 [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

**Authors:** Schick et al. (Meta AI)  
**Contribution:** `🔧 Tool Use`

> Demonstrated that language models can **autonomously learn when and how to use external tools**. Toolformer was trained to decide when calling an API (calculator, search engine, translator) would be helpful, insert the appropriate API call, and incorporate the result. This self-supervised approach to tool use opened the door to more capable and grounded AI systems.

---

## 🆕 Recent Breakthroughs

### 📄 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** DeepSeek AI  
**Contribution:** `🧠 Advanced Reasoning`

> Introduced a novel **Reinforcement Learning framework** that directly rewards models for generating correct intermediate reasoning steps, not just the final answer. This incentivizes the model to learn more robust and generalizable reasoning paths, significantly boosting performance on complex logic, math, and coding tasks where the process is as important as the result.

### 📄 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) ![Hall of Fame](https://img.shields.io/badge/⭐-Hall%20of%20Fame-ff1493?style=flat&labelColor=000000)

**Authors:** DeepSeek AI  
**Contribution:** `🧮 Mathematical Reasoning`

> Introduced **DeepSeekMath 7B**, the first open model to approach GPT-4 on MATH benchmark (51.7%). This paper introduced **Group Relative Policy Optimization (GRPO)**—a memory-efficient RL algorithm that became hugely influential and was adopted by DeepSeek-R1 and other reasoning models.

---

### 📄 [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)

**Authors:** OpenAI  
**Contribution:** `🎯 Reasoning at Scale`

> Detailed the safety evaluation and capabilities of OpenAI's **o1 reasoning model**, which uses extended chain-of-thought at inference time to solve complex problems. The system card reveals how scaling test-time compute through longer reasoning chains enables breakthrough performance on math, coding, and scientific reasoning benchmarks, while also discussing new safety considerations for reasoning models.

---

<div align="center">

### 🌟 Contributing

Feel free to submit PRs to add more reasoning and agent papers or improve existing entries!

### 📜 License

This repository is licensed under CC0 License.

### 🙏 Acknowledgments

Thanks to all the researchers pushing the boundaries of AI reasoning and autonomous systems.

---

⭐ If you find this repository helpful, please consider giving it a star!

</div>
