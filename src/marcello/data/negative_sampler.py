"""Generate negative (non-Marcelo) samples for contrastive classifier training.

Strategies:
  - prewritten: Load pre-written negative samples from a directory (best quality, no GPU)
  - llm_rephrase: Use an LLM to rewrite Marcelo's text in a generic style
  - shuffle_sentences: Randomly shuffle sentence order (destroys personal voice)
  - random_corpus: Sample from a generic text corpus (e.g., Wikipedia, C4)
"""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path

from datasets import Dataset


class NegativeStrategy(str, Enum):
    PREWRITTEN = "prewritten"
    LLM_REPHRASE = "llm_rephrase"
    SHUFFLE_SENTENCES = "shuffle_sentences"
    RANDOM_CORPUS = "random_corpus"


REPHRASE_PROMPT = (
    "Rewrite the following text in a completely generic, neutral style. "
    "Remove all personal voice, quirks, and distinctive patterns. "
    "Keep the same meaning but make it sound like anyone could have written it.\n\n"
    "Original:\n{text}\n\n"
    "Generic rewrite:"
)


class NegativeSampler:
    """Generates negative samples to pair with positive (Marcelo's) writing.

    The classifier needs both positive and negative examples to learn what
    makes Marcelo's style distinctive. This class creates those negatives.
    """

    def __init__(
        self,
        strategy: NegativeStrategy = NegativeStrategy.PREWRITTEN,
        num_negatives_per_positive: int = 2,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        prewritten_path: str = "data/raw/negative_samples",
        seed: int = 42,
    ):
        self.strategy = strategy
        self.num_negatives = num_negatives_per_positive
        self.model_name = model_name
        self.prewritten_path = Path(prewritten_path)
        self.seed = seed
        self._rng = random.Random(seed)
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the text generation pipeline (only for llm_rephrase)."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
        return self._pipeline

    def _rephrase_with_llm(self, text: str) -> str:
        """Use an LLM to rewrite text in a generic style."""
        pipe = self._get_pipeline()
        prompt = REPHRASE_PROMPT.format(text=text)
        result = pipe(prompt, return_full_text=False)
        return result[0]["generated_text"].strip()

    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle sentence order to destroy personal narrative flow."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            # can't shuffle a single sentence — reverse words as fallback
            words = text.split()
            mid = len(words) // 2
            return " ".join(words[mid:] + words[:mid])
        self._rng.shuffle(sentences)
        return " ".join(sentences)

    def _load_prewritten(self) -> list[str]:
        """Load pre-written negative samples from the directory."""
        texts = []
        for path in sorted(self.prewritten_path.rglob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                # split by double newline to get paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                texts.extend(paragraphs)
        return texts

    def generate_negatives(self, positive_texts: list[str]) -> list[str]:
        """Generate negative samples from a list of positive texts."""
        if self.strategy == NegativeStrategy.PREWRITTEN:
            negatives = self._load_prewritten()
            self._rng.shuffle(negatives)
            return negatives

        negatives = []
        for text in positive_texts:
            for _ in range(self.num_negatives):
                if self.strategy == NegativeStrategy.LLM_REPHRASE:
                    neg = self._rephrase_with_llm(text)
                elif self.strategy == NegativeStrategy.SHUFFLE_SENTENCES:
                    neg = self._shuffle_sentences(text)
                elif self.strategy == NegativeStrategy.RANDOM_CORPUS:
                    raise NotImplementedError(
                        "Random corpus sampling requires a corpus path. "
                        "Use collect_from_corpus() instead."
                    )
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                negatives.append(neg)

        return negatives

    def build_contrastive_dataset(self, positive_dataset: Dataset) -> Dataset:
        """Take a dataset of positive samples and create a balanced contrastive dataset.

        Returns a dataset with columns: text, label (1=Marcelo, 0=generic)
        """
        positive_texts = positive_dataset["text"]

        negative_texts = self.generate_negatives(positive_texts)

        all_texts = list(positive_texts) + negative_texts
        all_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

        # shuffle together
        combined = list(zip(all_texts, all_labels))
        self._rng.shuffle(combined)
        texts, labels = zip(*combined)

        return Dataset.from_dict({"text": list(texts), "label": list(labels)})
