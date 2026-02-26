"""Reward function that wraps the style classifier for GRPO training.

The reward function scores generated text on how much it sounds like
Marcelo's writing style. This is the signal that GRPO uses to update
the policy (base LLM).
"""

from __future__ import annotations

import torch

from marcello.classifier.model import StyleClassifier


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
        length_bonus_weight: float = 0.0,
        target_length: int = 200,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ):
        self.classifier = StyleClassifier.from_pretrained(classifier_path)
        self.classifier.eval()
        self.temperature = temperature
        self.length_bonus_weight = length_bonus_weight
        self.target_length = target_length
        self.min_reward = min_reward
        self.max_reward = max_reward

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

    def _length_bonus(self, text: str) -> float:
        """Small bonus for outputs near the target length. Prevents degenerate short/long outputs."""
        length = len(text.split())
        diff = abs(length - self.target_length) / self.target_length
        return max(0.0, 1.0 - diff)

    def score(self, texts: list[str]) -> list[float]:
        """Score a batch of generated texts.

        Returns rewards in [min_reward, max_reward] range.
        """
        style_probs = self.classifier.predict(texts)
        rewards = []

        for text, prob in zip(texts, style_probs):
            # apply temperature scaling
            reward = prob / self.temperature

            # optional length bonus
            if self.length_bonus_weight > 0:
                reward += self.length_bonus_weight * self._length_bonus(text)

            # clamp to range
            reward = max(self.min_reward, min(self.max_reward, reward))
            rewards.append(reward)

        return rewards

    def __call__(self, texts: list[str]) -> list[float]:
        """Score texts. Compatible with TRL's reward function interface."""
        return self.score(texts)
