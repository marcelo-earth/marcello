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

MarceLLo uses Group Relative Policy Optimization (GRPO) to let a base LLM discover a writing style through reinforcement learning. A DeBERTa-v3-small classifier trained on the author's writing serves as the reward model, scoring "how much does this sound like Marcelo." Whether this beats plain supervised fine-tuning is not yet measured: an SFT baseline exists in the repo as the control, and the comparison has not been run.

**Status:** the published weights are v0.1 and predate the corpus and classifier fixes described below. No GRPO run has been made against the current classifier.

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

- The reward classifier is trained on a small, single-author corpus (253 positives, 253 negatives after balancing; the validation split is 76 texts), so every metric carries a wide interval. The model may overfit to specific topics (introspection, learning, curiosity, poetry).
- Bilingual (es/en) but with uneven coverage — more poetic content in Spanish, more expository in English.
- The style classifier reward can be gamed: the model may learn superficial stylistic tics rather than deeper voice characteristics. Reading the top-reward completions after every run is part of the plan for exactly this reason.
- The classifier sits at the frozen-feature ceiling (see below), so a better reward needs a different approach, not more head training.
- The negative pool relies on rewrites from a 1.5B model, which may carry grammar artifacts the classifier could learn as "negative". Part of the verse negatives is written for the purpose rather than quoted.
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

- **Positive samples:** 297 paragraph-level pieces from the author (poems, prose, blog posts; es and en), 253 kept after balancing
- **Negative pool:** 405 texts from 7 sources, 253 kept after balancing:

| Source | Count |
|--------|-------|
| Pre-written prose | 69 |
| Wikipedia | 21 |
| Neutral rewrites (same content, voice stripped) | 89 |
| Poetic rewrites (another poetic voice, same theme) | 89 |
| Curated Spanish verse (not the author's) | 85 |
| English tech blog prose | 28 |
| German, Italian, French, Portuguese | 24 |

- **Balancing:** positives and negatives are matched on length bin, language and verse form together, so every surface cell is equal (112/112 English prose, 22/22 Spanish prose, 119/119 Spanish verse). Each of these was added because something visible without reading the text separated the classes.
- **Split:** 85/15 train/val, seed 42 (506 samples total)
- **Reserved for the probe:** Bécquer and Darío are never used as negatives, so the sanity probe keeps measuring generalization.

### Training Procedure

Pipeline: style classifier, sanity probe gate, then GRPO. An SFT baseline and a separate judge classifier support the evaluation.

**Phase 1: Style Classifier (Reward Model)**

DeBERTa-v3-small as a binary classifier (Marcelo vs. not-Marcelo). The encoder is fully frozen and only the head is trained.

- Frozen encoder (`freeze_encoder_layers: 6`), LayerNorm plus per-dimension feature standardization before the head
- lr: 1e-3, batch: 16, epochs: 20, warmup: 10%, weight decay: 0.01, max_length: 512

Per-dimension standardization (a mean and std fitted on the cached training features and shipped as buffers) fixed a head that was dominated by high-variance dimensions: on the same split, logistic regression on the same frozen features reached AUC 0.903 against the head's 0.827.

Validation metrics on the final corpus: accuracy 0.8026, AUC-ROC 0.8878, F1 0.8101. These dropped every time a shortcut was removed from the data, which is intended: 0.80 with no shortcuts left is worth more than 0.90 where counting words was enough.

**Sanity probe (hard gate)**

Before any GRPO run, the classifier scores out-of-distribution texts (encyclopedic prose, public-domain poetry, generic LLM prose, English tech blogs, German, Italian) against held-out real samples as control. The gate passed on 2026-07-26:

| Measure | Value |
|---------|-------|
| Mean OOD score | 0.155 |
| Max OOD score | 0.392 |
| Control (held-out real samples) | 0.824 |
| Margin | +0.669 |
| Failures | 0 |

The margin on the original corpus was -0.063 (11 of 11 failures). The Italian probe passes at 0.3918 against a 0.4 threshold, so it is the one text that would flip the gate. The probe has to be re-run after any change to the corpus or classifier.

**Frozen-feature ceiling**

5-fold CV on frozen features, 506 samples (`scripts/compare_backbones.py`). This is what a frozen-encoder head can reach:

| Backbone | Accuracy | AUC |
|----------|----------|-----|
| microsoft/deberta-v3-small | 0.820 ± 0.034 | 0.885 ± 0.030 |
| FacebookAI/xlm-roberta-base | 0.793 ± 0.033 | 0.884 ± 0.027 |
| intfloat/multilingual-e5-small | 0.779 ± 0.037 | 0.864 ± 0.029 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.709 ± 0.048 | 0.807 ± 0.049 |

The trained classifier (0.8026 / 0.8878) is at this ceiling.

**Judge classifier**

A second classifier trained on a different split (`--resplit-seed 1337`) scores completions at evaluation time only, so the policy is never optimized against the model that grades it.

**SFT baseline (control, not yet run)**

LoRA adapter trained on positives only (74 control-prompt to continuation pairs), r=16, alpha=32, lr 1e-4, batch 2 x 4 accumulation, 3 epochs.

**Phase 2: GRPO Training**

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
grade disjoint halves of it. A completion's content tokens are first split by whether they
belong to a run copied verbatim from the seed:

- **The echo penalty charges** the share of the seed's content tokens that ended up inside a
  copied run of 4 or longer (`echo_ngram_size`, clamped down if the seed itself is shorter),
  minus a floor of 0.35 that leaves room for a handoff. The denominator is the seed, which is
  the same for every completion in a GRPO group, so the group is ranked on comparable values.
- **Relevance pays** for seed vocabulary that appears *outside* any copied run of 2 or longer,
  against a fixed budget of 8 carried content tokens. The budget does not follow seed length,
  which varies a lot once poems fall back to a line-based split.

Relevance excludes copies from a shorter run length than the penalty charges for, and the gap
is deliberate: with a single window, copying exactly 3 content tokens was charged nothing and
paid in full, which made trimmed verbatim copying the best-paying move under the pair.

What this does not promise: reuse under the floor is charged nothing and is still not paid, so
copying more of the seed can lower a completion's relevance. That is the intended direction,
but it means relevance is monotone in how much of the seed a completion *rewrites*, not in how
much it touches. `scripts/analyze_reward_independence.py` checks on a finished eval run that the
two components do move apart in practice.

The corpus positives have a median length of 51 tokens (49 in Spanish, 54 in English) and a
p90 of 85, so the bonus peaks close to how Marcelo actually writes and reaches zero at 120,
well inside the 256-token generation budget.

## Evaluation

**Results are held back.** No GRPO run has been made against the current classifier, so this card reports no generation scores. Reward-classifier scores alone are not evidence of style transfer, since the model can be optimized against that classifier. The comparison table (base, few-shot, SFT, GRPO, GRPO from SFT) will be added once independent judges and a blinded human evaluation have run.

Metrics the evaluation produces:

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

First end-to-end run. Classifier trained, reward pipeline stable, eval prompts in place. The published weights are from this version and predate the corpus and classifier fixes above.

## Model Card Authors

Marcelo
