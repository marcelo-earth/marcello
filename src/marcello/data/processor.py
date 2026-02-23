"""Text processing and cleaning pipeline for writing samples."""

from __future__ import annotations

import re
import hashlib

from datasets import Dataset


class TextProcessor:
    """Cleans and normalizes text samples for classifier training.

    Pipeline:
      1. Normalize unicode and whitespace
      2. Optionally strip URLs
      3. Optionally strip email addresses
      4. Collapse repeated punctuation
      5. Trim to max_length
      6. Deduplicate by content hash
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        max_length: int = 2048,
        deduplicate: bool = True,
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.max_length = max_length
        self.deduplicate = deduplicate
        self._seen_hashes: set[str] = set()

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to a single text."""
        if self.remove_urls:
            text = re.sub(r"https?://\S+", "", text)

        if self.remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", "", text)

        if self.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        # collapse repeated punctuation (e.g. "!!!" -> "!!")
        text = re.sub(r"([!?.]){3,}", r"\1\1", text)

        # trim to max length at word boundary
        if len(text) > self.max_length:
            text = text[: self.max_length]
            last_space = text.rfind(" ")
            if last_space > self.max_length * 0.8:
                text = text[:last_space]

        return text

    def is_duplicate(self, text: str) -> bool:
        """Check if we've already seen this text (by content hash)."""
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def process_dataset(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """Clean and deduplicate an entire HuggingFace dataset."""
        self._seen_hashes.clear()

        cleaned_texts = []
        keep_indices = []

        for i, text in enumerate(dataset[text_column]):
            cleaned = self.clean(text)
            if not cleaned:
                continue
            if self.deduplicate and self.is_duplicate(cleaned):
                continue
            cleaned_texts.append(cleaned)
            keep_indices.append(i)

        # select non-text columns for kept rows, then replace text
        result = dataset.select(keep_indices)
        result = result.remove_columns([text_column])
        result = result.add_column(text_column, cleaned_texts)

        return result
