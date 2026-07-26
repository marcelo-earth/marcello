# MarceLLo — research plan and status

The order of work comes from `FEEDBACK.md`. This file tracks where each step
stands and what the measurements said. Update it after every run.

## Step 1 — Classifier sanity probe

**Status: PASSED on 2026-07-26. The gate is open.**

`python scripts/sanity_probe.py` scores out-of-distribution texts against the
trained classifier: encyclopedic prose, public-domain poetry (Bécquer, Darío),
generic LLM prose, English tech blogs, German and Italian. Held-out real
samples are the control.

Every fix came from finding something that separates the classes without
anyone reading the text. Five rounds:

| run | mean OOD | max OOD | control | margin | failures |
|---|---|---|---|---|---|
| original corpus | 0.734 | — | 0.671 | **−0.063** | 11 of 11 |
| content-matched negatives | 0.380 | 0.774 | 0.674 | +0.293 | 5 |
| length-matched balancing | 0.254 | 0.722 | 0.599 | +0.345 | 2 |
| form-matched balancing | 0.276 | 0.532 | 0.681 | +0.405 | 2 |
| per-dimension standardisation | 0.155 | 0.392 | 0.824 | **+0.669** | **0** |

Bécquer went 0.750 → 0.611 → 0.276 → **0.053**, Darío 0.774 → **0.241**.

Thin margin worth watching: the Italian probe sits at 0.3918 against a 0.4
threshold. It is the one text that would flip the gate, and there are only 24
non-Spanish non-English negatives holding it down.

**Gate: the probe must pass before any GRPO run. Re-run it after any change to
the corpus or the classifier, not just once.**

### The shortcuts found

Five causes, all measured rather than guessed at:

1. **The training loop could not fit the data.** Frozen DeBERTa features plus
   logistic regression separate this corpus at AUC 1.000 (`scripts/compare_backbones.py`),
   yet the trained head reached AUC 0.335 and collapsed to a constant. Fixed by
   normalizing the pooled features and keeping a fully frozen encoder in eval
   mode (commit `09da361`).
2. **The corpus was separable by topic and register.** Positives were poems and
   posts, negatives were Wikipedia-style prose, so any classifier learns topic
   detection. Fixed by generating content-matched negatives
   (`scripts/generate_negatives.py`).
3. **Length.** Negatives ran 44 words at the median against the positives' 34.
   Score correlated with word count at −0.341 across the corpus and −0.394
   *within the negatives alone*, so short text scored as Marcelo whatever it
   said. That is what the four-line probe poems were reading. Fixed by binning
   on word count and balancing each bin (`src/marcello/data/balance.py`).
4. **Verse form and language.** Spanish verse held 113 positives to 74
   negatives, a 60% base rate for Marcelo from the shape of the text alone.
   Bécquer came back at 0.611, which is that base rate almost exactly. Fixed by
   adding 85 Spanish verse negatives and making the balancer stratify on
   (length bin, language, verse) together. Every surface cell is now equal:
   112/112 English prose, 22/22 Spanish prose, 119/119 Spanish verse.

5. **The head was not reading the features it was given.** Nothing to do with
   the corpus. LayerNorm normalises each sample across its own 768 dimensions,
   which is not the same as putting the dimensions on a common scale across the
   corpus, so a high-variance dimension still dominated the linear layer. On the
   identical split, logistic regression on the same frozen features reached AUC
   0.903 against the head's 0.827, and was flat across C from 0.01 to 10, so it
   was never regularisation strength. Fixed by fitting a per-dimension mean and
   std on the cached training features and shipping them as buffers.

The pattern across the first four: whenever something visible *without reading
the text* correlates with the label, the head takes it. Balancing it away is the
fix, and the probe is what makes the next one visible. The fifth is the mirror
image, and worth remembering separately: the corpus can be clean and the
classifier still fail, so measure the ceiling before blaming the data.

### Cost of removing the shortcuts

Val metrics dropped every time a shortcut was removed, which is the point, then
recovered once the head could actually use the features:

| corpus | samples | val accuracy | AUC-ROC | F1 |
|---|---|---|---|---|
| content-matched | 536 | 0.9012 | 0.9476 | 0.9091 |
| length-matched | 510 | 0.8701 | 0.9170 | 0.8750 |
| form-matched | 506 | 0.7500 | 0.8269 | 0.7467 |
| form-matched + standardised | 506 | **0.8026** | **0.8878** | **0.8101** |

0.80 on a task with no shortcuts left is worth more than 0.90 on one where
counting words was enough.

### The ceiling

`scripts/compare_backbones.py`, 5-fold CV on frozen features, 506 samples. This
is what a frozen-encoder head can reach, and the number to judge training
against:

| backbone | accuracy | AUC |
|---|---|---|
| microsoft/deberta-v3-small | 0.820 ± 0.034 | 0.885 ± 0.030 |
| FacebookAI/xlm-roberta-base | 0.793 ± 0.033 | 0.884 ± 0.027 |
| intfloat/multilingual-e5-small | 0.779 ± 0.037 | 0.864 ± 0.029 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.709 ± 0.048 | 0.807 ± 0.049 |

The trained classifier is at 0.8026 / 0.8878, so it now sits at that ceiling and
there is nothing left to win from the head alone. Note that deberta-v3-small
still wins on a corpus that is over half Spanish, despite being English-only
pretrained; the multilingual backbones did not overtake it once the task got
hard, which is the opposite of what I expected before measuring.

## Step 2 — Independent evaluation infrastructure

**Status: in place, judge not yet trained.**

- `evaluate.py --judge-classifier <path>` scores completions with a second model.
- `scripts/train_classifier.py --resplit-seed 1337 --output-dir outputs/classifier/judge`
  trains that judge on a different split.
- Per-component reward breakdown exists in `reward.py`.
- Blinded human eval: not written yet.
- `data/eval_prompts.txt` is frozen. Do not edit it.

## Step 3 — Grow and fix the corpus

**Status: done. 506 samples, 253 per class, every surface cell matched.**

The collector yields 297 paragraph-level positives. The negative pool is now
405, and balancing keeps 253 of each after matching on length, language and
form.

| source | count | what it forces |
|---|---|---|
| prewritten prose | 69 | generic voice, unrelated topics |
| Wikipedia | 21 | encyclopedic register |
| neutral rewrites | 89 | same content and language, voice stripped |
| poetic rewrites | 89 | another poetic voice on the same theme |
| curated Spanish verse | 85 | classical and modernist verse that is not his |
| English blog prose | 28 | first-person tech writing, the failing register |
| German/Italian/French/Portuguese | 24 | languages he does not write |

The last three were added because the probe named them. English tech blogging
went 0.53/0.45 → 0.26/0.20 and German 0.58 → 0.23 once the pool contained any.

Rules that came out of building this:

- **Never let a rewrite change language.** A Spanish system prompt made the
  model translate the English samples: 39 of the first 90 negatives came back
  in Spanish, which would have taught "English means Marcelo". Prompts are now
  per source language, and a rewrite that changes language is retried and
  discarded.
- **Bécquer and Darío are reserved for the probe.** No negative may use them,
  or the probe stops measuring generalisation. The curated verse uses Góngora,
  Quevedo, Sor Juana, Machado, Manrique, Garcilaso, Hernández and Lorca.
- **Match the surface before trusting the score.** See Step 1.

Open risks:

- Rewrites carry grammar errors from a 1.5B model; the classifier could learn
  "awkward phrasing = negative". A larger rephraser on a GPU would be better.
  Qwen2.5-3B-Instruct stalled mid-download, hence 1.5B.
- Roughly half the curated verse is written for the purpose rather than quoted.
  It fills the form cell honestly, but it is one author's idea of a classical
  register, not a sample of one.
- 253 per class is small. Every metric here carries a wide interval and the
  val split is 76 texts.

## Step 4 — SFT baseline

**Status: code ready, not run.**

`python scripts/train_sft.py --config configs/sft.yaml` trains a LoRA adapter on
74 (control prompt -> continuation) pairs, using the same prompt format as GRPO.

## Step 5 — GRPO from base

**Unblocked.** Step 1 passed on 2026-07-26, so a run from here means something.
Re-run the probe against whatever classifier the run actually uses first.

## Step 6 — GRPO from the SFT checkpoint

Blocked on Steps 4 and 5.

## Step 7 — Read the samples

After every run, read the top-10 highest-reward completions and look for
classifier gaming. A consistent hack is a result: add negatives for that
pattern and retrain.
