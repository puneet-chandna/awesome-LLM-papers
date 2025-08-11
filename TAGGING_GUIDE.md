# 🏷️ Paper Tagging System

> **Consistent tags = Better discovery** | Make papers findable with the right tags

## 📌 Quick Reference

Every paper should have:
1. **One primary category** (e.g., 🏗️ Model Architectures)
2. **2-5 descriptive tags** (specific attributes)
3. **Optional meta tags** (organization, size, etc.)

### Example Entry:
```markdown
### [Mixtral 8x7B](https://arxiv.org/...)
**Category:** 🏗️ Model Architectures
**Tags:** `mixture-of-experts` `efficient` `open-source` `production-ready`
**Meta:** `mistral` `45B-params` `apache-2.0`
```

---

## 🎯 Tag Categories

### 1️⃣ **Technical Tags** (What it does)

#### Architecture Types
- `transformer` - Transformer-based models
- `state-space` - SSM models like Mamba
- `mixture-of-experts` - MoE architectures
- `diffusion` - Diffusion models
- `gan` - Generative adversarial networks
- `hybrid` - Combined architectures

#### Training Methods
- `rlhf` - Reinforcement learning from human feedback
- `dpo` - Direct preference optimization  
- `supervised` - Standard supervised learning
- `self-supervised` - Self-supervised learning
- `constitutional` - Constitutional AI training
- `instruction-tuning` - Instruction following
- `few-shot` - Few-shot learning
- `zero-shot` - Zero-shot capabilities

#### Optimization Techniques
- `quantization` - Model quantization (specify: 4-bit, 8-bit)
- `pruning` - Weight pruning
- `distillation` - Knowledge distillation
- `lora` - Low-rank adaptation
- `peft` - Parameter-efficient fine-tuning
- `adapter` - Adapter methods
- `flash-attention` - Optimized attention

#### Capabilities
- `reasoning` - Logical reasoning
- `chain-of-thought` - CoT prompting
- `tool-use` - Can use external tools
- `code-generation` - Generates code
- `math` - Mathematical capabilities
- `multimodal` - Multiple modalities
- `vision` - Visual understanding
- `audio` - Audio processing
- `video` - Video understanding
- `long-context` - Extended context (specify: 32k, 100k, 1M)

### 2️⃣ **Impact Tags** (Why it matters)

#### Performance Level
- `sota` - State-of-the-art results
- `breakthrough` - Major advancement
- `incremental` - Minor improvements
- `competitive` - Matches existing SOTA

#### Readiness Level
- `production-ready` - Can deploy now
- `experimental` - Research prototype
- `theoretical` - Concept/theory only
- `reproducible` - Code and weights available

#### Innovation Type
- `novel-architecture` - New model design
- `novel-method` - New technique/approach
- `novel-application` - New use case
- `benchmark` - New evaluation method
- `analysis` - Understanding/interpretation

### 3️⃣ **Meta Tags** (Additional context)

#### Organization
- `openai` `anthropic` `google` `meta` `microsoft`
- `deepmind` `mistral` `stability-ai` `cohere`
- `academic` `independent` `open-source-community`

#### Model Size
- `<1B` - Under 1 billion parameters
- `1B-7B` - Small models
- `7B-30B` - Medium models  
- `30B-100B` - Large models
- `100B+` - Very large models
- `size-unknown` - Parameters not disclosed

#### License
- `open-source` - Fully open
- `open-weights` - Weights available
- `api-only` - Only API access
- `proprietary` - Closed source
- `mit` `apache-2.0` `cc-by` (specific licenses)

#### Resource Requirements
- `consumer-gpu` - Runs on consumer hardware
- `single-gpu` - Needs one GPU
- `multi-gpu` - Requires multiple GPUs
- `datacenter` - Datacenter scale
- `edge-device` - Runs on edge/mobile

---

## 🔍 Tag Usage Examples

### Example 1: Efficient Model
```markdown
**[Phi-2: Small Language Model](link)**
Category: 🏗️ Model Architectures
Tags: `transformer` `efficient` `2.7B` `open-source` `microsoft`
Why: Small but capable model for edge deployment
```

### Example 2: Reasoning Paper
```markdown
**[Tree of Thoughts](link)**
Category: 🧮 Reasoning & Agents
Tags: `reasoning` `chain-of-thought` `prompting` `novel-method`
Why: Explores multiple reasoning paths for better problem-solving
```

### Example 3: Safety Research
```markdown
**[Constitutional AI](link)**
Category: 🛡️ Safety & Security
Tags: `alignment` `rlhf` `constitutional` `anthropic` `breakthrough`
Why: AI systems training themselves to be harmless
```

---

## 📊 Tag Combinations

### Common Powerful Combinations

| Combination | Meaning | Example Papers |
|------------|---------|----------------|
| `open-source` + `production-ready` | Deployable open models | Llama 2, Mistral |
| `efficient` + `sota` | Best performance with less resources | Flash Attention |
| `reasoning` + `tool-use` | Agents that can think and act | ReAct, AutoGPT |
| `long-context` + `efficient` | Handle long inputs efficiently | LongLoRA |
| `multimodal` + `open-source` | Open vision-language models | LLaVA, CLIP |

---

## 🚫 Tag Anti-Patterns (Avoid These)

### ❌ Too Generic
- Bad: `good` `interesting` `new` `ai` `llm`
- Good: `breakthrough` `novel-architecture` `transformer`

### ❌ Too Specific
- Bad: `arxiv-2401.12345` `january-15-2025`
- Good: `2025` `recent`

### ❌ Redundant
- Bad: `llm` + `language-model` + `large-language-model`
- Good: Just use primary category

### ❌ Opinion-Based
- Bad: `must-read` `amazing` `game-changer`
- Good: `breakthrough` `sota` `influential`

---

## 🔄 Tag Lifecycle

### Adding New Tags
1. Check if existing tag covers the concept
2. Propose in issue/discussion if truly new
3. Add to this guide once approved
4. Backfill relevant papers

### Deprecating Tags
1. Mark as deprecated in guide
2. Map to replacement tag
3. Update existing papers gradually
4. Remove after 3 months

## 🎯 Tag Best Practices

### For Contributors
1. **Use 2-5 tags** - Not too few, not too many
2. **Mix tag types** - Technical + Impact + Meta
3. **Be specific** - `lora` better than `efficient`
4. **Check existing papers** - Maintain consistency
5. **Propose new tags** - If truly needed

### For Maintainers
1. **Regular audits** - Check tag consistency monthly
2. **Update guide** - Add new tags as field evolves
3. **Merge similar** - Consolidate redundant tags
4. **Track usage** - Remove unused tags
5. **Communicate changes** - Announce tag updates

---

## 🔗 Integration Examples

### In Paper Entries
```markdown
### [Paper Title](link)
**Category:** 🏗️ Model Architectures
**Tags:** `mixture-of-experts` `efficient` `open-source`
**Quick Take:** One-line summary
```

### In Category Pages
```markdown
## 🏗️ Model Architectures

### Filter by tags:
[`transformer`](#) [`state-space`](#) [`moe`](#) [All](#)

### Papers:
<!-- Papers automatically sorted by tags -->
```

### In Search/Filter UI
```javascript
// Tag-based filtering
const papers = filterPapers({
  mustHave: ['open-source', 'production-ready'],
  anyOf: ['efficient', 'small-model'],
  exclude: ['api-only']
});
```

---

## 📝 Examples for Each Category

### 🏗️ Model Architectures
Common tags: `transformer`, `state-space`, `mixture-of-experts`, `novel-architecture`

### 🧮 Reasoning & Agents  
Common tags: `chain-of-thought`, `tool-use`, `planning`, `agent`

### ⚡ Efficiency & Scaling
Common tags: `quantization`, `pruning`, `flash-attention`, `efficient`

### 🎯 Training & Alignment
Common tags: `rlhf`, `dpo`, `instruction-tuning`, `peft`

### 🎨 Multimodal Models
Common tags: `vision`, `audio`, `video`, `multimodal`

### 📚 RAG & Knowledge
Common tags: `retrieval`, `long-context`, `memory`, `rag`

### 🛡️ Safety & Security
Common tags: `jailbreak`, `alignment`, `safety`, `robustness`

### 🔬 Analysis & Theory
Common tags: `interpretability`, `mechanistic`, `analysis`, `benchmark`

---

## 🚀 Quick Start for New Contributors

1. **Read a paper**
2. **Identify category** (pick one from 8 main categories)
3. **Add 2-5 tags**:
   - What technique? (technical tag)
   - Why important? (impact tag)  
   - Who made it? (meta tag)
4. **Check this guide** for approved tags
5. **Submit!**

---

*Last updated: January 2025 | [Suggest new tags](https://github.com/yourusername/daily-papers-llm/issues)*

