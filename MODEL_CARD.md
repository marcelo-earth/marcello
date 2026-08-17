---
language:
  - es
  - en
license: MIT
tags:
  - grpo
  - style-transfer
  - reinforcement-learning
  - lora
  - text-generation
base_model:
  - Qwen/Qwen2.5-1.5B
  - microsoft/deberta-v3-small
datasets:
  - marcelo-earth/marcello-writing-samples
pipeline_tag: text-generation
---

# Model Card for MarceLLo

RL-based writing style transfer system that fine-tunes an LLM to capture a personal writing style using GRPO, guided by a style classifier as reward signal.

## Model Details

### Model Description

MarceLLo uses Group Relative Policy Optimization (GRPO) to let a base LLM discover a writing style through reinforcement learning, rather than memorizing examples via standard fine-tuning. A DeBERTa-v3-small classifier trained on the author's writing serves as the reward model, scoring "how much does this sound like Marcelo."

- **Developed by:** Marcelo
- **Model type:** Causal LM with LoRA adapter, trained via GRPO
- **Language(s):** Spanish, English
- **License:** MIT
- **Finetuned from:** [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)

### Model Sources

- **Repository:** [marcelo-earth/marcello](https://github.com/marcelo-earth/marcello)

## Uses

### Direct Use

Text generation in the author's writing style. Given a seed sentence or topic, the model continues the text with stylistic patterns learned via RL.

### Out-of-Scope Use

This model captures one person's writing style from a small corpus. It is not a general-purpose writing assistant and should not be used for impersonation or generating content attributed to others.

## Bias, Risks, and Limitations

- Trained on a small, single-author corpus (35 texts). The model may overfit to specific topics (introspection, learning, curiosity, poetry).
- Bilingual (es/en) but with uneven coverage — more poetic content in Spanish, more expository in English.
- The style classifier reward can be gamed: the model may learn superficial stylistic tics rather than deeper voice characteristics.
- Not evaluated for factual accuracy — this is a style model, not a knowledge model.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("marcelo-earth/marcello-qwen2.5-1.5b-grpo")
tokenizer = AutoTokenizer.from_pretrained("marcelo-earth/marcello-qwen2.5-1.5b-grpo")
```

For best results, use the control-token prompt template:

```bash
python scripts/generate.py \
  --model outputs/grpo/final \
  --prompt "The night felt larger than the street below." \
  --format-prompts \
  --style standard
```

## Training Details

### Training Data

- **Positive samples:** 27 poems/prose pieces (es) + 8 blog posts (en) = 35 total
- **Negative samples:** 23 pre-written contrastive texts (same topics, generic voice)
- **Split:** 85/15 train/val, seed 42

### Training Procedure

Two-phase pipeline:

**Phase 1 — Style Classifier (Reward Model)**

DeBERTa-v3-small fine-tuned as a binary classifier (Marcelo vs. not-Marcelo).

- Full fine-tune (no frozen layers)
- lr: 2e-5, batch: 16, epochs: 5, warmup: 10%, max_length: 512

**Phase 2 — GRPO Training**

Qwen2.5-1.5B with LoRA, optimized via GRPO using the classifier as reward signal.

#### Training Hyperparameters

- **LoRA:** r=16, alpha=32, dropout=0.05
- **Training regime:** fp32/auto, lr: 5e-7, batch: 4, grad_accum: 4, epochs: 3
- **GRPO:** group size (G): 8, max_new_tokens: 256, temperature: 0.8, top_p: 0.95
- **Regularization:** KL coef: 0.1, clip range: 0.2, max_grad_norm: 1.0

#### Reward Weights

| Signal | Weight |
|--------|--------|
| Style score | 0.65 |
| Prompt relevance | 0.20 |
| Repetition penalty | 0.15 |
| Reference copy penalty | 0.15 |
| Length bonus | 0.10 |
| Prompt echo penalty | 0.10 |

Target length: 60 tokens, measured with the policy's own tokenizer. Reference n-gram size: 8.

Prompt relevance and the prompt echo penalty both look at seed reuse, so they are defined to
grade disjoint halves of it. Vocabulary the completion reuses inside a run copied verbatim from
the seed (4 content tokens or longer) is charged by the echo penalty and earns no relevance;
vocabulary re-embedded in the completion's own phrasing is paid by relevance and charged
nothing. Relevance is scored against a fixed budget of 8 carried content tokens rather than
against the seed's own vocabulary, so its scale does not follow seed length, which varies a lot
once poems fall back to a line-based split. `scripts/analyze_reward_independence.py` checks on a
finished eval run that the two really do move apart.

The corpus positives have a median length of 51 tokens (49 in Spanish, 54 in English) and a
p90 of 85, so the bonus peaks close to how Marcelo actually writes and reaches zero at 120,
well inside the 256-token generation budget.

## Evaluation

### Metrics

- **Style Score:** average classifier P(Marcelo) on generated text
- **Perplexity:** fluency measure via base model (lower = more fluent)
- **Distinct-N:** lexical diversity (ratio of unique n-grams)
- **Length stats:** average/min/max word count

### Testing Data

20 eval prompts (10 es + 10 en) covering introspection, learning, curiosity, creation, and poetry. Available in `data/eval_prompts.txt`.

## Environmental Impact

- **Hardware Type:** NVIDIA T4 (Kaggle free tier)
- **Cloud Provider:** Kaggle

## Technical Specifications

### Model Architecture and Objective

- **Base LLM:** Qwen2.5-1.5B (causal LM) with LoRA adapter
- **Reward Model:** DeBERTa-v3-small with mean-pooling + classification head
- **Objective:** GRPO — group-relative policy optimization with clipped surrogate + KL penalty
- **Prompt format:** `<style:X> <lang:X> <task:continue>` control tags

### Compute Infrastructure

#### Hardware

Kaggle T4 GPU (16 GB VRAM), free tier

#### Software

- PyTorch >= 2.1
- Transformers >= 4.40
- TRL >= 0.8
- PEFT >= 0.10

## Version History

### v0.1 — 2026-03-28

First end-to-end run. Classifier trained, reward pipeline stable, eval prompts in place.

## Model Card Authors

Marcelo
