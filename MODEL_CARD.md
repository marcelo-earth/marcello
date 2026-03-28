# MarceLLo Model Card

## Current Version: v0.1

First end-to-end GRPO run. Classifier trained, reward pipeline stable, eval prompts in place.

---

## v0.1

**Date:** 2026-03-28

### Models

| Component | Base | HuggingFace |
|-----------|------|-------------|
| Style Classifier | `microsoft/deberta-v3-small` | [`marcelo-earth/marcello-style-classifier`](https://huggingface.co/marcelo-earth/marcello-style-classifier) |
| Fine-tuned LLM | `Qwen/Qwen2.5-1.5B` + LoRA | [`marcelo-earth/marcello-qwen2.5-1.5b-grpo`](https://huggingface.co/marcelo-earth/marcello-qwen2.5-1.5b-grpo) |

### Data

- **Positive samples:** 20 poems (es) + 8 blog posts (en) + messages
- **Negative samples:** pre-written contrastive texts
- **Split:** 85/15 train/val, seed 42

### Classifier

- DeBERTa-v3-small, full fine-tune (no frozen layers)
- lr: 2e-5, batch: 16, epochs: 5, warmup: 10%, max_length: 512

### GRPO Training

- LoRA r=16, alpha=32, dropout=0.05
- lr: 5e-7, batch: 4, grad_accum: 4, epochs: 3
- Group size (G): 8, max_new_tokens: 256
- KL coef: 0.1, clip range: 0.2
- Temperature: 0.8, top_p: 0.95

### Reward Weights

| Signal | Weight |
|--------|--------|
| Style score | 0.65 |
| Prompt relevance | 0.20 |
| Repetition penalty | 0.15 |
| Reference copy penalty | 0.15 |
| Length bonus | 0.10 |
| Prompt echo penalty | 0.10 |

Target length: 180 tokens. Reference n-gram size: 8.

### Eval

- 20 eval prompts (10 es + 10 en) in `data/eval_prompts.txt`
- Metrics: style score, distinct-1/2, perplexity, length stats

### Notes

- Trained on Kaggle T4 GPU (free tier)
- Reward shaping iterated over several runs to reduce echo and memorization
- Prompt template uses `<style:X> <lang:X> <task:continue>` control tags
