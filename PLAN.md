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

Backbone comparison, 5-fold CV on frozen features. On the old corpus every
backbone was near-perfect, which is itself the evidence that the corpus was too
easy. On the rebuilt corpus the same measurement drops to a believable range:

| backbone | accuracy (old) | AUC (old) | accuracy (new) | AUC (new) |
|---|---|---|---|---|
| microsoft/deberta-v3-small | 0.989 | 1.000 | 0.847 ± 0.004 | 0.919 ± 0.006 |
| FacebookAI/xlm-roberta-base | 0.989 | 0.999 | 0.813 ± 0.013 | 0.902 ± 0.013 |
| intfloat/multilingual-e5-small | 0.978 | 0.996 | 0.780 ± 0.033 | 0.869 ± 0.031 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.961 | 0.993 | not rerun | not rerun |

That 0.919 is the ceiling a frozen-encoder head can reach, and the number the
trained classifier has to be judged against.

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

**Status: done. 536 samples, 268 per class.**

The collector yields 297 paragraph-level positives, but class balancing used to
discard down to whatever the negative pool could match — 90. With 268 negatives
the corpus is now 455 train / 81 val.

Negative pool (`scripts/generate_negatives.py`, Qwen2.5-1.5B-Instruct):

| source | count | what it forces |
|---|---|---|
| prewritten prose | 69 | generic voice, unrelated topics |
| Wikipedia | 21 | encyclopedic register |
| neutral rewrites | 89 | same content and language, voice stripped |
| poetic rewrites | 89 | another poetic voice on the same theme |

Two spurious signals found and removed while building this:

- **Language drift.** A Spanish system prompt made the model translate the
  English samples: 39 of the first 90 negatives came back in Spanish, which
  would have taught the classifier "English means Marcelo". Prompts are now
  chosen per source language, and a rewrite that changes language is retried
  and then discarded. Both sets: 0 mismatches.
- **Poetry as a proxy.** Most positives are poems, so prose-only negatives let
  "poetry" stand in for "Marcelo". The poetic variant closes that gap.

Open risks:

- The poetic negatives run long (median 53 words against the positives' 33.5),
  so length is the cue still worth watching. Length-matched sampling is the fix
  if the probe shows the classifier leaning on it.
- Rewrites carry grammar errors from a 1.5B model; the classifier could learn
  "awkward phrasing = negative". A larger rephraser on a GPU would be better.
- Qwen2.5-3B-Instruct stalled mid-download, hence 1.5B.

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
