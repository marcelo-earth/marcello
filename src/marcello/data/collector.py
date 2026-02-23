"""Collect writing samples from various sources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset


@dataclass
class WritingSample:
    """A single writing sample with metadata."""

    text: str
    source: str
    source_type: str = "unknown"  # txt, jsonl, csv
    metadata: dict = field(default_factory=dict)


class WritingSampleCollector:
    """Collects writing samples from text files, JSONL, and other sources.

    Supports:
      - Plain .txt files (one sample per file, or split by blank lines)
      - JSONL files (one JSON object per line with a configurable text field)
      - Directories (recursively collects all supported files)
    """

    def __init__(self, min_length: int = 50, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length
        self._samples: list[WritingSample] = []

    def collect_from_txt(self, path: Path, split_on_blank_lines: bool = True) -> int:
        """Load samples from a plain text file.

        If split_on_blank_lines is True, each paragraph (separated by blank
        lines) becomes its own sample. Otherwise the whole file is one sample.
        """
        text = path.read_text(encoding="utf-8")
        added = 0

        if split_on_blank_lines:
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        else:
            chunks = [text.strip()] if text.strip() else []

        for chunk in chunks:
            if self.min_length <= len(chunk) <= self.max_length:
                self._samples.append(
                    WritingSample(text=chunk, source=str(path), source_type="txt")
                )
                added += 1
        return added

    def collect_from_jsonl(self, path: Path, text_field: str = "text") -> int:
        """Load samples from a JSONL file. Each line should be a JSON object."""
        added = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if self.min_length <= len(text) <= self.max_length:
                    self._samples.append(
                        WritingSample(
                            text=text,
                            source=str(path),
                            source_type="jsonl",
                            metadata={k: v for k, v in obj.items() if k != text_field},
                        )
                    )
                    added += 1
        return added

    def collect_from_directory(self, directory: Path) -> int:
        """Recursively collect from all .txt and .jsonl files in a directory."""
        directory = Path(directory)
        added = 0
        for path in sorted(directory.rglob("*")):
            if path.suffix == ".txt":
                added += self.collect_from_txt(path)
            elif path.suffix == ".jsonl":
                added += self.collect_from_jsonl(path)
        return added

    @property
    def samples(self) -> list[WritingSample]:
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def to_dataset(self) -> Dataset:
        """Convert collected samples to a HuggingFace Dataset."""
        return Dataset.from_dict(
            {
                "text": [s.text for s in self._samples],
                "source": [s.source for s in self._samples],
                "source_type": [s.source_type for s in self._samples],
                "label": [1] * len(self._samples),  # all positive (Marcelo's writing)
            }
        )
