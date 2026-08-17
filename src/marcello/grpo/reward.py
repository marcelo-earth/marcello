"""Reward function that wraps the style classifier for GRPO training.

The reward function scores generated text on how much it sounds like
Marcelo's writing style. This is the signal that GRPO uses to update
the policy (base LLM).
"""

from __future__ import annotations

import math
import re

import torch
from datasets import load_from_disk

from marcello.classifier.model import StyleClassifier
from marcello.grpo.prompting import extract_seed_text

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "con",
    "de",
    "del",
    "el",
    "en",
    "for",
    "in",
    "la",
    "los",
    "of",
    "para",
    "por",
    "the",
    "to",
    "un",
    "una",
    "with",
    "y",
}


class StyleReward:
    """Wraps a trained StyleClassifier as a reward function for GRPO.

    Reward = classifier_probability * temperature + length_bonus

    The temperature parameter controls reward sharpness:
      - temperature > 1.0: smoother rewards (more exploration)
      - temperature < 1.0: sharper rewards (more exploitation)
      - temperature = 1.0: raw classifier probability
    """

    def __init__(
        self,
        classifier_path: str,
        temperature: float = 1.0,
        style_weight: float = 0.65,
        length_bonus_weight: float = 0.1,
        prompt_relevance_weight: float = 0.2,
        repetition_penalty_weight: float = 0.15,
        prompt_echo_penalty_weight: float = 0.1,
        reference_copy_penalty_weight: float = 0.15,
        target_length: int = 60,
        tokenizer=None,
        reference_texts_path: str | None = None,
        reference_ngram_size: int = 8,
        echo_ngram_size: int = 4,
        echo_floor: float = 0.35,
        relevance_exclusion_run: int = 2,
        relevance_target_tokens: int = 8,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ):
        if length_bonus_weight > 0 and tokenizer is None:
            raise ValueError(
                "length_bonus_weight > 0 requires a tokenizer: target_length is measured "
                "in tokens, so the bonus cannot be computed without one. Pass the same "
                "tokenizer the policy generates with, or set length_bonus_weight=0."
            )

        self.classifier = StyleClassifier.from_pretrained(classifier_path)
        self.classifier.eval()
        self.temperature = temperature
        self.style_weight = style_weight
        self.length_bonus_weight = length_bonus_weight
        self.prompt_relevance_weight = prompt_relevance_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.prompt_echo_penalty_weight = prompt_echo_penalty_weight
        self.reference_copy_penalty_weight = reference_copy_penalty_weight
        self.target_length = target_length
        self.tokenizer = tokenizer
        self.reference_texts_path = reference_texts_path
        self.reference_ngram_size = max(3, reference_ngram_size)
        self.echo_ngram_size = max(2, echo_ngram_size)
        self.echo_floor = echo_floor
        # never longer than what the penalty charges, or copying would go unpriced
        # in the gap between the two run lengths
        self.relevance_exclusion_run = min(max(2, relevance_exclusion_run), self.echo_ngram_size)
        self.relevance_target_tokens = max(1, relevance_target_tokens)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reference_ngrams = self._load_reference_ngrams(reference_texts_path)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()
        elif torch.backends.mps.is_available() and hasattr(self.classifier, "to"):
            self.classifier = self.classifier.to(torch.device("mps"))

    def _load_reference_ngrams(self, dataset_path: str | None) -> set[tuple[str, ...]]:
        """Load positive train-set n-grams to penalize memorized generations."""
        if not dataset_path:
            return set()

        dataset = load_from_disk(dataset_path)
        ngrams: set[tuple[str, ...]] = set()
        labels = dataset["label"] if "label" in dataset.column_names else [1] * len(dataset)
        for text, label in zip(dataset["text"], labels):
            if label != 1:
                continue
            tokens = self._content_tokens(text)
            ngrams.update(self._ngrams(tokens, self.reference_ngram_size))
        return ngrams

    def _length_bonus(self, text: str) -> float:
        """Small bonus for outputs near target length. Prevents degenerate short/long outputs.

        Length is measured in tokens, the same unit as `max_new_tokens`, so the peak of
        the bonus is a length the policy can actually reach. Measuring in words made the
        target unreachable in Spanish (1.62 tokens per word) and turned the bonus into a
        monotone "longer is better" signal. See issue #17.
        """
        length = len(self.tokenizer.encode(text, add_special_tokens=False))
        diff = abs(length - self.target_length) / self.target_length
        return max(0.0, 1.0 - diff)

    def _temperature_scale(self, prob: float) -> float:
        """Apply temperature scaling to a probability in a numerically stable way."""
        clipped = min(max(prob, 1e-5), 1 - 1e-5)
        logit = math.log(clipped / (1 - clipped))
        return 1.0 / (1.0 + math.exp(-(logit / max(self.temperature, 1e-3))))

    def _content_tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Záéíóúñü']+", text.lower())
        return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]

    def _ngrams(self, tokens: list[str], n: int) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _copied_spans(
        self, seed_tokens: list[str], text_tokens: list[str], min_run: int
    ) -> tuple[set[int], set[int]]:
        """Which positions on each side belong to a run of `min_run` copied verbatim.

        Working in positions rather than in sets of n-grams is what lets the echo
        penalty and `_prompt_relevance` split the same input cleanly: each token is
        either inside a copied run or it is not, and the two components can then read
        that one decision at different run lengths without ever grading a token twice.

        `min_run` is clamped to the seed, never to the completion. Clamping it to the
        completion made the window depend on the sample being scored, so padding a
        verbatim copy with filler widened the window until the copy no longer matched,
        and inside a GRPO group the denominator changed from completion to completion.
        """
        n = min(min_run, len(seed_tokens))
        if n < 2 or len(text_tokens) < n:
            return set(), set()

        starts_by_window: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(seed_tokens) - n + 1):
            starts_by_window.setdefault(tuple(seed_tokens[i : i + n]), []).append(i)

        seed_hits: set[int] = set()
        text_hits: set[int] = set()
        for j in range(len(text_tokens) - n + 1):
            starts = starts_by_window.get(tuple(text_tokens[j : j + n]))
            if not starts:
                continue
            text_hits.update(range(j, j + n))
            for i in starts:
                seed_hits.update(range(i, i + n))
        return seed_hits, text_hits

    def _echoed_tokens(self, seed_tokens: list[str], text_tokens: list[str]) -> set[str]:
        """Tokens the completion took from the seed as part of a copied run.

        The run length here is `relevance_exclusion_run`, which is shorter than the
        run the echo penalty charges for. Matching the two exactly left a gap the
        policy could sit in: with a single window of 4, copying exactly three content
        tokens was charged nothing and paid full relevance, so the best-paying strategy
        under the pair was still verbatim copying, trimmed to just under the window.
        Everything the penalty charges is excluded here, and so is the near miss.
        """
        _, text_hits = self._copied_spans(seed_tokens, text_tokens, self.relevance_exclusion_run)
        return {text_tokens[i] for i in text_hits}

    def _prompt_relevance(self, prompt: str, text: str) -> float:
        """Reward carrying the seed's subject forward in the completion's own words.

        Two properties this needs, both from issue #16:

        Disjoint from `_prompt_echo_penalty`. Tokens the completion took as part of a
        run copied from the seed are already the echo penalty's business, so counting
        them here paid for the behaviour the penalty charged for and the policy could
        not learn which one it was being graded on. They are dropped from the numerator,
        leaving only vocabulary that was re-embedded in the completion's own phrasing.

        Note what this does not claim. Copying is never paid, so adding a copied run to
        a completion can lower its relevance: the run absorbs neighbouring tokens that
        were carried on their own before. That is the intended direction, not a bug, but
        it does mean relevance is not monotone in how much of the seed a completion
        touches. It is monotone in how much of it the completion rewrites.

        Scale-free in the seed. The old denominator was the whole seed vocabulary, so
        the score was recall over it: a three-word seed was maxed out by carrying one
        word, and a long seed put the ceiling out of reach. Seed length varies a lot,
        because `split_seed_and_continuation` falls back to a line split for poems
        (`src/marcello/grpo/prompting.py:74`). The denominator is now the fixed budget
        `relevance_target_tokens`, so one carried token is worth the same everywhere.

        The budget is deliberately not capped at the seed's own vocabulary size. That
        variant reads better (a short seed could still reach 1.0) but it re-creates the
        saturation this is meant to remove, and saturation is worse than a low ceiling
        under GRPO: advantages are normalized inside a group that shares one prompt, so
        a component every completion maxes out contributes no spread and therefore no
        gradient. A short seed earns less from this component; it still ranks its
        completions against each other.
        """
        seed_tokens = self._content_tokens(extract_seed_text(prompt))
        text_tokens = self._content_tokens(text)
        if not seed_tokens or not text_tokens:
            return 0.0

        carried = set(seed_tokens) & set(text_tokens)
        carried -= self._echoed_tokens(seed_tokens, text_tokens)
        if not carried:
            return 0.0

        return min(1.0, len(carried) / self.relevance_target_tokens)

    def _repetition_penalty(self, text: str) -> float:
        tokens = self._content_tokens(text)
        if len(tokens) < 4:
            return 0.0
        bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        repeated = len(bigrams) - len(set(bigrams))
        return repeated / max(len(bigrams), 1)

    def _prompt_echo_penalty(self, prompt: str, text: str) -> float:
        """Charge for how much of the seed the completion hands back verbatim.

        The charge is the share of the seed's content tokens that ended up inside a run
        of `echo_ngram_size` or longer copied into the completion, above a floor that
        leaves room for a handoff. The denominator is the seed, which is fixed for every
        completion in a GRPO group, so the values a group is ranked on are comparable;
        the old denominator was the count of shared n-grams and moved per completion.

        `_prompt_relevance` never pays for a token counted here, so no single reuse is
        both paid and charged. The converse does not hold and is not meant to: reuse
        under the floor is charged nothing, and is still not paid.
        """
        seed_tokens = self._content_tokens(extract_seed_text(prompt))
        text_tokens = self._content_tokens(text)
        if not seed_tokens or not text_tokens:
            return 0.0

        seed_hits, _ = self._copied_spans(seed_tokens, text_tokens, self.echo_ngram_size)
        if not seed_hits:
            return 0.0
        return max(0.0, len(seed_hits) / len(seed_tokens) - self.echo_floor)

    def _reference_copy_penalty(self, text: str) -> float:
        if not self.reference_ngrams:
            return 0.0
        text_ngrams = self._ngrams(self._content_tokens(text), self.reference_ngram_size)
        if not text_ngrams:
            return 0.0
        overlap = len(text_ngrams & self.reference_ngrams) / len(text_ngrams)
        return overlap

    def score(
        self,
        texts: list[str],
        prompts: list[str] | None = None,
        return_breakdown: bool = False,
    ) -> list[float] | list[dict]:
        """Score a batch of generated texts.

        Returns rewards in [min_reward, max_reward] range.
        When return_breakdown=True, returns a list of dicts with per-component
        values alongside the total, which is useful for diagnosing reward hacking.
        """
        style_probs = self.classifier.predict(texts)
        results = []

        for idx, (text, prob) in enumerate(zip(texts, style_probs)):
            prompt = prompts[idx] if prompts else None

            style_component = self.style_weight * self._temperature_scale(prob)
            length_component = (
                self.length_bonus_weight * self._length_bonus(text)
                if self.length_bonus_weight > 0
                else 0.0
            )
            relevance_component = (
                self.prompt_relevance_weight * self._prompt_relevance(prompt, text)
                if (prompt and self.prompt_relevance_weight > 0)
                else 0.0
            )
            repetition_component = (
                self.repetition_penalty_weight * self._repetition_penalty(text)
                if self.repetition_penalty_weight > 0
                else 0.0
            )
            echo_component = (
                self.prompt_echo_penalty_weight * self._prompt_echo_penalty(prompt, text)
                if (prompt and self.prompt_echo_penalty_weight > 0)
                else 0.0
            )
            refcopy_component = (
                self.reference_copy_penalty_weight * self._reference_copy_penalty(text)
                if self.reference_copy_penalty_weight > 0
                else 0.0
            )

            reward = (
                style_component
                + length_component
                + relevance_component
                - repetition_component
                - echo_component
                - refcopy_component
            )
            reward = max(self.min_reward, min(self.max_reward, reward))

            if return_breakdown:
                results.append(
                    {
                        "total": reward,
                        "raw_style_prob": prob,
                        "style_score": style_component,
                        "length_bonus": length_component,
                        "prompt_relevance": relevance_component,
                        "repetition_penalty": repetition_component,
                        "prompt_echo_penalty": echo_component,
                        "reference_copy_penalty": refcopy_component,
                    }
                )
            else:
                results.append(reward)

        return results

    def __call__(self, texts: list[str], prompts: list[str] | None = None) -> list[float]:
        """Score texts. Compatible with TRL's reward function interface."""
        return self.score(texts, prompts=prompts)
