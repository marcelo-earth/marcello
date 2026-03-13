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
        length_bonus_weight: float = 0.0,
        prompt_relevance_weight: float = 0.2,
        repetition_penalty_weight: float = 0.15,
        prompt_echo_penalty_weight: float = 0.1,
        reference_copy_penalty_weight: float = 0.15,
        target_length: int = 200,
        reference_texts_path: str | None = None,
        reference_ngram_size: int = 8,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ):
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
        self.reference_texts_path = reference_texts_path
        self.reference_ngram_size = max(3, reference_ngram_size)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reference_ngrams = self._load_reference_ngrams(reference_texts_path)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

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
        """Small bonus for outputs near the target length. Prevents degenerate short/long outputs."""
        length = len(text.split())
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

    def _prompt_relevance(self, prompt: str, text: str) -> float:
        seed_tokens = set(self._content_tokens(extract_seed_text(prompt)))
        text_tokens = set(self._content_tokens(text))
        if not seed_tokens or not text_tokens:
            return 0.0
        return len(seed_tokens & text_tokens) / len(seed_tokens)

    def _repetition_penalty(self, text: str) -> float:
        tokens = self._content_tokens(text)
        if len(tokens) < 4:
            return 0.0
        bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        repeated = len(bigrams) - len(set(bigrams))
        return repeated / max(len(bigrams), 1)

    def _prompt_echo_penalty(self, prompt: str, text: str) -> float:
        seed_tokens = self._content_tokens(extract_seed_text(prompt))
        text_tokens = self._content_tokens(text)
        if not seed_tokens or not text_tokens:
            return 0.0

        seed_ngrams = self._ngrams(seed_tokens, min(4, len(seed_tokens)))
        text_ngrams = self._ngrams(text_tokens, min(4, len(text_tokens)))
        if not seed_ngrams or not text_ngrams:
            return 0.0
        overlap = len(seed_ngrams & text_ngrams) / len(seed_ngrams)
        return max(0.0, overlap - 0.35)

    def _reference_copy_penalty(self, text: str) -> float:
        if not self.reference_ngrams:
            return 0.0
        text_ngrams = self._ngrams(self._content_tokens(text), self.reference_ngram_size)
        if not text_ngrams:
            return 0.0
        overlap = len(text_ngrams & self.reference_ngrams) / len(text_ngrams)
        return overlap

    def score(self, texts: list[str], prompts: list[str] | None = None) -> list[float]:
        """Score a batch of generated texts.

        Returns rewards in [min_reward, max_reward] range.
        """
        style_probs = self.classifier.predict(texts)
        rewards = []

        for idx, (text, prob) in enumerate(zip(texts, style_probs)):
            prompt = prompts[idx] if prompts else None
            reward = self.style_weight * self._temperature_scale(prob)

            if self.length_bonus_weight > 0:
                reward += self.length_bonus_weight * self._length_bonus(text)
            if prompt and self.prompt_relevance_weight > 0:
                reward += self.prompt_relevance_weight * self._prompt_relevance(prompt, text)
            if self.repetition_penalty_weight > 0:
                reward -= self.repetition_penalty_weight * self._repetition_penalty(text)
            if prompt and self.prompt_echo_penalty_weight > 0:
                reward -= self.prompt_echo_penalty_weight * self._prompt_echo_penalty(prompt, text)
            if self.reference_copy_penalty_weight > 0:
                reward -= self.reference_copy_penalty_weight * self._reference_copy_penalty(text)

            reward = max(self.min_reward, min(self.max_reward, reward))
            rewards.append(reward)

        return rewards

    def __call__(self, texts: list[str], prompts: list[str] | None = None) -> list[float]:
        """Score texts. Compatible with TRL's reward function interface."""
        return self.score(texts, prompts=prompts)
