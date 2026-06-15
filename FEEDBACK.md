# MarceLLo — Detailed Code Review and Research Feedback

This document records a full technical review of the MarceLLo project as of June 2026.
It covers what is working, what has structural problems, and what to do in priority order.
Nothing is summarized or glossed over.

---

## What the project is doing

MarceLLo is a two-phase pipeline:

1. **Style Classifier (reward model):** DeBERTa-v3-small fine-tuned as a binary classifier (label 1 = Marcelo's writing, label 0 = generic writing).
2. **GRPO Training:** Qwen2.5-1.5B with LoRA, optimized via Group Relative Policy Optimization using the classifier as the reward signal. The policy generates 8 completions per prompt, all 8 are scored, and the advantage of each completion is its score minus the group mean divided by the group standard deviation.

The reward function (`src/marcello/grpo/reward.py`) is a weighted sum of six signals:
- Style score (classifier probability): weight 0.65
- Length bonus (penalty for deviating from 180 tokens): weight 0.10
- Prompt relevance (seed token overlap): weight 0.20
- Repetition penalty (repeated bigrams): weight 0.15 subtracted
- Prompt echo penalty (4-gram overlap with seed): weight 0.10 subtracted
- Reference copy penalty (8-gram overlap with training corpus): weight 0.15 subtracted

---

## What is genuinely well done

**Engineering quality is above average for a school project and above many professional repos:**

- Clean module structure with clear separation: `data/`, `classifier/`, `grpo/`, `eval/`, `utils/`.
- Unit tests exist and CI runs lint + tests on push. This is rare for ML projects of this size.
- Configs are YAML files separated from code, not hardcoded. Every hyperparameter is auditable.
- The model card (`MODEL_CARD.md`) follows the HuggingFace spec, includes version history, and honestly states limitations. The specific limitation named is important: "the style classifier reward can be gamed — the model may learn superficial stylistic tics rather than deeper voice characteristics." This is the correct diagnosis. The issue is that the project doesn't yet act on it.
- The `push_to_hub.py` script with `--dry-run`, `--merge-weights`, and per-artifact flags is thoughtful.
- The eval harness (`scripts/evaluate.py`, `scripts/compare_runs.py`) with `--only-regressions` and `--show-completions` flags is well designed.
- The reference-copy penalty in the reward function (8-gram overlap with training corpus) directly applies the lesson from automatic-downlink that "duplication teaches memorization." You penalize the model for copying your own texts verbatim.
- The `NegativeSampler` has four strategies already implemented: prewritten, llm_rephrase, shuffle_sentences, random_corpus. The right strategy (`llm_rephrase`) is already in the code, just not the one being used.
- Temperature scaling on reward probability (`_temperature_scale`) is numerically stable (converts to logit space) and prevents the reward from being either saturated or collapsed.

---

## Problem 1: The framing of GRPO vs. SFT is wrong, and it drives the whole architecture in the wrong direction

**What the README says:** "Standard fine-tuning (SFT) memorizes examples. MarceLLo uses GRPO to let the model discover writing style through reinforcement learning."

**What is actually true:**

SFT with LoRA does not simply memorize. SFT shifts the entire output distribution of the model toward your writing. With a small corpus and a low-rank adapter, SFT is regularized enough that it generalizes beyond the specific texts it saw. This is the standard, widely-deployed technique for style personalization, and it is strong.

More importantly: GRPO and SFT are not alternatives. They are sequential stages. DeepSeek-R1, which is cited in the README as the inspiration for this approach, used SFT cold-start data before GRPO. The GRPO stage in R1 refined a model that already knew how to do the task in roughly the right format. The R1 GRPO also worked because its reward was verifiable: the answer to a math problem is either correct or not. A learned classifier scoring "how much does this sound like Marcelo" is noisy, gameable, and off-distribution in ways a math checker is not.

**The consequence of this framing:**

The entire architecture is designed around avoiding SFT, when SFT is actually the most important thing to run. The project has no SFT baseline, which means:
- There is no measurement of what GRPO contributes on top of the base model.
- There is no measurement of what GRPO contributes on top of SFT.
- If GRPO is beating the base model (which it likely is), that still says nothing useful because SFT almost certainly beats the base model too, and probably more simply.

**What to do:**

1. Train SFT LoRA on all 35 positive texts. Config: lr=2e-4, LoRA r=8 alpha=16, 5-10 epochs, batch 4. Takes under 30 minutes on a T4. Call this `outputs/sft/final`.
2. Evaluate SFT with the independent eval (see Problem 3 below).
3. Run GRPO starting from the SFT checkpoint, not from the base model. In `configs/grpo.yaml`, the `model.name` field should point to `outputs/sft/final` for the GRPO warm-start run.
4. Add a few-shot prompting baseline: paste 5 of your real texts into a frontier model (Claude or GPT-4) and ask it to continue a seed sentence in the same voice. No fine-tuning needed. Score it with the independent eval.
5. The comparison table you want in your final report: `Base Model vs. Few-Shot vs. SFT vs. GRPO-from-base vs. GRPO-from-SFT`. That table is a real result no matter which system wins.

---

## Problem 2: 35 positive samples and 23 negatives fail the minimum count threshold

**The current corpus:**
- Positive: 27 Spanish poems/prose pieces + 8 English blog posts = **35 total**
- Negative: **23 pre-written contrastive texts**
- After 85/15 split: ~30 positive train samples, ~5 validation samples

**Why this number is structurally too small:**

The rule from the previous project (automatic-downlink) was: if the rare class has fewer than roughly 50 distinct real examples, you have a data collection problem, not a training problem. The same rule applies here, and 35 is below the threshold even for a non-rare class.

**The specific failure mode for a classifier:**

With 49 training texts after splitting (30 positive + ~19 negative in train), a fully fine-tuned DeBERTa will reach near-perfect binary accuracy by latching onto the cheapest separating signal in the data. The cheapest signals here are:

1. **Language:** Your Spanish positive texts are mostly poems. Your English positive texts are blog posts. Your negatives cover both. The classifier may learn "this is Marcelo if it reads like Spanish poetry" rather than "this is Marcelo."
2. **Topic:** Your poems cover love, learning, time, stars, identity. Your negatives cover similar themes (the filenames match: `01_searching_stars.txt` vs. `01_buscando.txt`, `11_human_freedom.txt` vs. `11_la_libertad_humana.txt`). If topics overlap, the classifier has to look at voice. If they don't, it classifies on topic.
3. **Line breaks and length:** Poems have short lines. Prose has long ones. The classifier may learn from token count per line.
4. **Style tokens:** Your writing uses specific punctuation, sentence structures, and vocabulary patterns. These will be learned. The question is whether they generalize to novel generations or get gamed.

**Why the negatives make it worse:**

Your current negatives are all pre-written human texts with their own style. During GRPO, the classifier scores *Qwen2.5-1.5B generated text*, which looks nothing like either set. The classifier was never trained on generated text, so its behavior on generated text is off-distribution and unpredictable. It may assign high scores to generations that have artifacts of language model output (repetitive sentence structure, hedge words, generic phrasing) if those artifacts happen to correlate with patterns in your training set.

**What to do:**

Growing the positive corpus:

- The easiest source is segmentation. If your 35 texts average 200 words, splitting each into 3-4 paragraph-level samples gives you ~100-140 samples from the same writing. This is legitimate: the classifier needs sentence-to-paragraph-level patterns, not full documents.
- Add journal entries, long messages, notes, old school essays, any writing in your own voice. Target 80-120 distinct pieces before splitting.
- If you have writing from before and after a style change (e.g., writing from 3 years ago sounds different), decide which period is "Marcelo" and label accordingly.

Rebuilding the negatives to be harder and closer to what the reward model will actually judge:

1. **Base-model generations on your training prompts.** Run Qwen2.5-1.5B with no fine-tuning on the first 2 sentences of each of your texts. These look like language model output. The classifier needs to score them low. This is the most important negative type to add.
2. **LLM-rephrased versions of your own texts.** You already have `llm_rephrase` in `NegativeSampler`. Use it. This forces the classifier to look at voice and rhythm, not content.
3. **Real texts by other authors in matched genres.** Other Spanish poets, other English tech bloggers. The classifier should score them low not because they are low-quality but because they are not you.
4. Do not rely on negatives from only one strategy. Mix all three with the existing pre-written set.

**Sanity probe to run before trusting the classifier at all:**

Take texts the classifier was never trained on: a Wikipedia article, a poem by Pablo Neruda, a passage from a random blog, and a raw Qwen2.5-1.5B generation on a neutral prompt. Score them with the trained classifier. They should all score low (below 0.3 ideally). If any score above 0.5, the classifier has a spurious signal and anything the GRPO training produces with "high style score" is meaningless.

---

## Problem 3: The evaluation is circular

This is the most important structural problem.

**What is happening:**

The headline metric in `evaluate.py` and `metrics.py` is `style_score`, which is the probability assigned by the DeBERTa classifier. That classifier is the same model used as the GRPO reward. GRPO's explicit objective is to maximize that classifier's score. After training, the model's style score will go up almost by definition, because going up was the optimization target.

This is identical to the automatic-downlink mistake of reporting "94.8% bandwidth savings" from class collapse: the metric looked like success while measuring whether the optimization ran, not whether the output was good.

A rising style score after GRPO training tells you:
- The optimization is numerically stable (useful to know)
- The policy learned to increase classifier probability (by definition)

It does not tell you:
- Whether the output actually sounds like you
- Whether the model found a genuine style signal or a classifier artifact
- Whether GRPO is doing anything that SFT wouldn't also do

**What independent evaluation looks like:**

**Tier 1 — Held-out judge classifier (cheapest, most automated):**

Train a second style classifier on a different random seed and a different train/val split of the same corpus. Never use this classifier as a reward signal. Use it only at evaluation time. If the GRPO model improves on this held-out classifier, the improvement is more likely to be real voice signal rather than overfitting to the reward model's specific quirks.

Implementation: add a `--judge-classifier` argument to `evaluate.py` that loads a second classifier from a different path and reports its scores separately from the reward classifier scores.

**Tier 2 — LLM judge (cheap, moderately trustworthy):**

Give Claude or GPT-4 a system prompt like: "You are evaluating whether text sounds like a specific person. Here are 5 examples of their writing: [paste 5 of your actual texts]. Now score the following text from 0 to 10 on how much it matches the author's voice, considering sentence rhythm, vocabulary choices, emotional register, and structural patterns. Do not consider topic similarity." Run this on 20 generated samples per model version. Average the scores. This judge is never exposed to the training process and cannot be gamed by the policy.

**Tier 3 — Blinded human evaluation (most credible, best for a school submission):**

Create a test set: 10 real Marcelo paragraphs interleaved with 10 generated paragraphs from each model version (base, SFT, GRPO). Show the mixed set to 3-5 people (friends, family, anyone who doesn't know which is which) and ask them to mark which ones they think are written by a real person vs. an AI, or which ones sound like the same voice as a given reference paragraph. Report the confusion rate: if humans can only identify your real text 60% of the time when mixed with the best model's output, that is a meaningful result. If they identify it 100% of the time, the model hasn't learned your voice.

This is the gold-standard evaluation for this task. It is also the most understandable result for a teacher or judge who is not an ML expert.

**Add to `evaluate.py` now:** log each reward component separately per run, not just the total reward. The fields `style_score`, `length_bonus`, `prompt_relevance`, `repetition_penalty`, `prompt_echo_penalty`, `reference_copy_penalty` should each appear in the output JSON. Right now they are computed but not individually logged. If the model's total reward goes up but style_score is flat while length_bonus is the thing increasing, you've found a degenerate optimization.

---

## Problem 4: Learning rate is likely too conservative for GRPO with LoRA

**Current config:** `learning_rate: 5e-7`

**Why this is suspect:**

For GRPO on top of LoRA, typical ranges in published work are 1e-6 to 1e-5. The KL coefficient of 0.1 provides regularization, so you have some room. At 5e-7 the policy may not move enough in 3 epochs to produce measurable improvement, especially with a small dataset that means few gradient updates per epoch.

**How to detect this:**

Log the total reward mean per epoch. If it is flat across all 3 epochs or increasing by less than 0.01 per epoch, the learning rate is too low for any learning to show up. If it is unstable or collapsing, it is too high. For this corpus size, you want to see steady improvement of 0.02-0.05 reward per epoch.

**What to try:** Run one sweep with lr=2e-6 and one with lr=5e-6, keeping everything else constant. Three epochs each. Compare reward curves and separately evaluate with the independent judge.

---

## Problem 5: Validation set is too small to report meaningful classifier accuracy

**Current split:** 85/15 of 58 total samples (35 positive + 23 negative) = ~49 train, ~9 validation.

With 9 validation samples, one misclassification moves accuracy by 11 percentage points. Reported val accuracy from a 9-sample set is not a metric, it is a coin flip.

**Fix:** Use stratified k-fold cross-validation (k=5) on the full 58 samples. This gives you 5 held-out sets of ~12 samples each. Average accuracy and report the standard deviation. With 5-fold CV your effective test set is the full corpus, and the reported accuracy is far more reliable.

The `sklearn` implementation is 5 lines of code. The `train.py` script should optionally run k-fold and report results when `--cross-validate` is passed.

---

## Problem 6: Reward component weights sum to more than 1.0 in ways that hide reward magnitude

**Current weights in `grpo.yaml`:**

```
style_weight: 0.65
length_bonus_weight: 0.10
prompt_relevance_weight: 0.20
repetition_penalty_weight: 0.15  # subtracted
prompt_echo_penalty_weight: 0.10  # subtracted
reference_copy_penalty_weight: 0.15  # subtracted
```

The maximum possible reward (style=1.0, length perfect, prompt relevant, no penalties) is 0.65 + 0.10 + 0.20 = 0.95. The minimum is clamped at -1.0. But the penalty weights sum to 0.40, which means a generation with moderate style score (0.5) and all penalties firing could reach 0.65*0.5 + 0.10 + 0.20 - 0.15 - 0.10 - 0.15 = 0.325 + 0.30 - 0.40 = 0.225. This is positive, so there's no strong negative signal for a mediocre generation. The GRPO advantage calculation divides by group standard deviation, so what matters is the *spread* within each group of 8, not the absolute magnitude. But if all 8 completions are mediocre and they all score 0.2-0.3, the variance is tiny and the gradients are near-zero.

This is not a critical bug, but it means the reward shaping may be less effective than intended. Consider: after GRPO is running stably and the reward curves are logged per component, revisit the weights empirically based on what components are actually varying across completions in a group.

---

## The recommended order of work

**Step 1: Add the classifier sanity probe (1 hour)**

Before anything else. Score 5 out-of-distribution texts (Wikipedia paragraph, Neruda poem, base Qwen generation, English tech blog, a text in a language you don't write) with the existing classifier. If any score above 0.4, the classifier has a spurious signal and all subsequent GRPO training is meaningless. This determines whether you need to rebuild the corpus first or can proceed to training.

**Step 2: Build the independent evaluation infrastructure (1 day)**

- Train a second judge classifier on a different seed and train/val split. Save to `outputs/classifier/judge`.
- Add `--judge-classifier` flag to `evaluate.py`.
- Write 2-3 blinded human eval prompts for friends: instructions + mixed set of real and generated paragraphs. Create this artifact even before training anything. Freeze the eval prompts in `data/eval_prompts.txt` (20 prompts already exist; they are good).
- Add per-component reward logging to the eval output JSON.

**Step 3: Grow the corpus (1-2 days)**

- Segment existing 35 texts into paragraph-level samples. Aim for 80-120 positive samples.
- Add base-model generations as negatives (run `generate.py` with `--model Qwen/Qwen2.5-1.5B` on each seed sentence, save outputs to `data/raw/negative_samples/generated/`).
- Enable `llm_rephrase` in `negative_sampler.py` for a subset of your texts.
- Rerun classifier training with the larger corpus. Run k-fold CV.

**Step 4: Run SFT baseline (30 minutes to train, rest is evaluation)**

- Train SFT LoRA on your corpus.
- Evaluate with the independent judge classifier, LLM judge, and if possible the blinded human test.
- Write down the number.

**Step 5: Run GRPO from base model (already done in v0.1; re-evaluate with new independent eval)**

**Step 6: Run GRPO from SFT checkpoint**

- Point `model.name` in `grpo.yaml` to `outputs/sft/final`.
- Train. Evaluate with independent eval.
- Try lr=2e-6 and lr=5e-6 sweeps. Log reward per epoch and per component.

**Step 7: Read generated samples at each stage**

After every training run, manually read the top-10 highest-reward completions. Look for:
- Does it sound like you, or does it just sound like a language model with some stylistic quirks?
- Are there repeated phrases or structures that seem to game the classifier?
- Does the language and rhythm feel right?

If you find a consistent hack (e.g., the model always ends with a short rhetorical question and that pattern scores high), that is a result: add that pattern's negative examples to the classifier and retrain. This adversarial iteration is the strongest form of what this project can produce as a research finding.

---

## Notes on specific files

**`src/marcello/grpo/reward.py` — overall well implemented**

- The `_temperature_scale` method correctly handles edge cases (clips probability away from 0/1 before computing logit).
- The `_load_reference_ngrams` method correctly skips label-0 samples.
- The `min_reward` and `max_reward` clamps are appropriate.
- One gap: `score()` does not log which component drove the reward for any given text. Add an optional `return_breakdown=True` mode that returns a dict of component scores alongside the total. This is needed to diagnose reward hacking.

**`src/marcello/data/negative_sampler.py` — the right strategies are there, wrong one is active**

- `PREWRITTEN` is the current default. It should not be the primary strategy because it produces a classifier that has never seen generated text.
- `LLM_REPHRASE` is the most important strategy to enable. It forces the classifier to distinguish voice from content.
- `SHUFFLE_SENTENCES` is a cheap augmentation strategy, not a high-quality negative. Shuffled sentences of your own text still carry your vocabulary and rhythm. The classifier may correctly classify them as Marcelo (they are Marcelo's sentences, just reordered), which is not a useful negative.

**`src/marcello/eval/metrics.py` — correct implementation**

- `perplexity()` correctly uses the base model (Qwen2.5-1.5B) to score fluency, not the fine-tuned model. This is the right design.
- `distinct_n()` at n=1 and n=2 is a good diversity metric and will catch repetitive collapse if it happens.
- Missing: report these metrics separately for Spanish vs. English completions, since the two subsets have different expected distributions and mixing them hides per-language regressions.

**`configs/grpo.yaml` — see Problem 4 above for lr concern**

- `num_generations: 8` is appropriate. At group size 4, advantage variance is too high. At 16, it is slow on a single T4. 8 is the right tradeoff.
- `kl_coef: 0.1` is on the lower end. If the model's style diverges from the base model in ways that produce incoherent outputs, raising this to 0.2 is the first fix.

**`data/eval_prompts.txt` — 20 prompts (10 es + 10 en) are good**

Do not change these prompts at any point. They are the frozen evaluation set. If you want to experiment with prompt formats, add a separate `data/exploratory_prompts.txt`. The frozen eval prompts must remain untouched across all runs to keep comparisons valid.

---

## What the project needs to say by the end

The strongest version of this school project is not "I built a model that sounds like me." It is:

"I ran a controlled comparison of four approaches to personal writing style capture: few-shot prompting, SFT, GRPO from base, and GRPO from SFT. I evaluated all four with an independent judge classifier, an LLM judge, and a blinded human test where participants tried to distinguish my real writing from each model's output. Here is what each approach contributes and where GRPO specifically helps or doesn't."

That is a real research result. It's credible whether GRPO wins or loses. It shows scientific discipline. And it directly builds on the lessons from automatic-downlink (evaluate the real task before training, use independent evaluation, run the baselines first).

---

*Review conducted June 2026. Based on full read of: README.md, MODEL_CARD.md, src/marcello/grpo/reward.py, src/marcello/grpo/prompting.py, src/marcello/data/negative_sampler.py, src/marcello/eval/metrics.py, configs/grpo.yaml, configs/classifier.yaml, data/ file tree, and git log.*
