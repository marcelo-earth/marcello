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

    def _shared_ngrams(
        self, seed_tokens: list[str], text_tokens: list[str]
    ) -> set[tuple[str, ...]]:
        """N-grams the completion copies verbatim from the seed.

        Both sides are cut at the same n. The previous version sized the seed and
        the completion n-grams independently, so a seed shorter than the window
        produced 3-grams against the completion's 4-grams and the intersection was
        empty by construction: short seeds could never be charged for echoing.
        """
        n = min(self.echo_ngram_size, len(seed_tokens), len(text_tokens))
        if n < 2:
            return set()
        return self._ngrams(seed_tokens, n) & self._ngrams(text_tokens, n)

    def _echoed_tokens(self, seed_tokens: list[str], text_tokens: list[str]) -> set[str]:
        """Seed tokens the completion reuses inside a verbatim run, not on its own."""
        return {token for ngram in self._shared_ngrams(seed_tokens, text_tokens) for token in ngram}

    def _prompt_relevance(self, prompt: str, text: str) -> float:
        """Reward carrying the seed's subject forward in the completion's own words.

        Two properties this needs, both from issue #16:

        Disjoint from `_prompt_echo_penalty`. Tokens the completion reuses inside a
        run copied verbatim from the seed are already charged by the echo penalty, so
        counting them here paid for the same behaviour that the penalty charged for.
        The policy could not learn which one it was being graded on. They are dropped
        from the numerator, leaving only vocabulary that was re-embedded in new phrasing.

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
        """Charge for copying the seed verbatim, above a floor that allows a handoff.

        This is the only component that grades verbatim reuse; `_prompt_relevance`
        excludes everything counted here, so the two never move on the same tokens.
        """
        seed_tokens = self._content_tokens(extract_seed_text(prompt))
        text_tokens = self._content_tokens(text)
        if not seed_tokens or not text_tokens:
            return 0.0

        n = min(self.echo_ngram_size, len(seed_tokens), len(text_tokens))
        seed_ngrams = self._ngrams(seed_tokens, n) if n >= 2 else set()
        if not seed_ngrams:
            return 0.0
        overlap = len(self._shared_ngrams(seed_tokens, text_tokens)) / len(seed_ngrams)
        return max(0.0, overlap - self.echo_floor)

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
