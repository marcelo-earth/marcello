# MarceLLo — research plan and status

The order of work comes from `FEEDBACK.md`. This file tracks where each step
stands and what the measurements said. Update it after every run.

## Step 1 — Classifier sanity probe

**Status: done, and it failed.**

`python scripts/sanity_probe.py` scores out-of-distribution texts against the
trained classifier: encyclopedic prose, public-domain poetry (Bécquer, Darío),
generic LLM prose, English tech blogs, German and Italian. Held-out real
samples are the control.

Result on `outputs/classifier/best` (2026-07-24):

| group | mean P(Marcelo) |
|---|---|
| out-of-distribution (11 texts) | 0.734 |
| held-out real samples (6 texts) | 0.671 |

Every OOD text scored above the 0.4 threshold, and the separation margin is
**negative**: unrelated Spanish prose scores *higher* than real writing. The
classifier is unusable as a reward signal, so no GRPO run based on it means
anything.

Two independent causes, both now addressed:

1. **The training loop could not fit the data.** Frozen DeBERTa features plus
   logistic regression separate this corpus at AUC 1.000 (`scripts/compare_backbones.py`),
   yet the trained head reached AUC 0.335 and collapsed to a constant. Fixed by
   normalizing the pooled features and keeping a fully frozen encoder in eval
   mode (commit `09da361`).
2. **The corpus itself is separable for the wrong reason.** Positives are short
   poems and posts, negatives are Wikipedia-style prose: topic, register and
   length all differ, so any classifier learns topic detection. Fixed by
   generating content-matched negatives (`scripts/generate_negatives.py`).

Backbone comparison, 5-fold CV on frozen features (all near-perfect, which is
itself the evidence that the corpus is too easy):

| backbone | accuracy | AUC |
|---|---|---|
| microsoft/deberta-v3-small | 0.989 ± 0.014 | 1.000 ± 0.000 |
| FacebookAI/xlm-roberta-base | 0.989 ± 0.022 | 0.999 ± 0.002 |
| intfloat/multilingual-e5-small | 0.978 ± 0.044 | 0.996 ± 0.009 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.961 ± 0.038 | 0.993 ± 0.011 |

**Gate: the probe must pass before any GRPO run.**

## Step 2 — Independent evaluation infrastructure

**Status: in place, judge not yet trained.**

- `evaluate.py --judge-classifier <path>` scores completions with a second model.
- `scripts/train_classifier.py --resplit-seed 1337 --output-dir outputs/classifier/judge`
  trains that judge on a different split.
- Per-component reward breakdown exists in `reward.py`.
- Blinded human eval: not written yet.
- `data/eval_prompts.txt` is frozen. Do not edit it.

## Step 3 — Grow and fix the corpus

**Status: in progress.**

- 90 positives, paragraph-level. Within the 80-120 target.
- Wikipedia negatives: 90. Keep them, but they cannot be the only negatives.
- Rephrased negatives via `scripts/generate_negatives.py` (Qwen2.5-3B-Instruct,
  one per positive). Same content, neutral voice.
- Open risk: the rephrases carry grammar errors from a small model, so the
  classifier could learn "broken Spanish = negative" instead of voice. The
  sanity probe is what catches this.

## Step 4 — SFT baseline

**Status: code ready, not run.**

`python scripts/train_sft.py --config configs/sft.yaml` trains a LoRA adapter on
74 (control prompt -> continuation) pairs, using the same prompt format as GRPO.

## Step 5 — GRPO from base

Blocked on Step 1 passing.

## Step 6 — GRPO from the SFT checkpoint

Blocked on Steps 4 and 5.

## Step 7 — Read the samples

After every run, read the top-10 highest-reward completions and look for
classifier gaming. A consistent hack is a result: add negatives for that
pattern and retrain.
