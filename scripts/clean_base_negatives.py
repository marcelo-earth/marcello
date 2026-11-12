"""Turn raw base-model generations into negatives (issue #12).

Raw output lives in data/raw/base_generations/, outside the negative pool, so
nothing trains on it by accident. This script cuts each generation at its first
tag, trims it to a length drawn from the positives, and writes the result into
data/raw/negative_samples/ where NegativeSampler reads it.

Usage:
    python scripts/clean_base_negatives.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk
from rich.console import Console

from marcello.data.base_generation import clean_generations

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw/base_generations")
    parser.add_argument(
        "--output", default="data/raw/negative_samples/base_generations_clean.jsonl"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=50)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    corpus = concatenate_datasets(
        [load_from_disk(processed / "train"), load_from_disk(processed / "val")]
    )
    positive_counts = [
        len(text.split()) for text, label in zip(corpus["text"], corpus["label"]) if label == 1
    ]

    rows = []
    for path in sorted(Path(args.raw_dir).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    cleaned = clean_generations(
        [row["text"] for row in rows], positive_counts, seed=args.seed, min_chars=args.min_chars
    )

    out = Path(args.output)
    kept = 0
    with out.open("w", encoding="utf-8") as handle:
        for row, text in zip(rows, cleaned):
            if text is None:
                continue
            handle.write(json.dumps({**row, "text": text}, ensure_ascii=False) + "\n")
            kept += 1

    console.print(f"{kept} of {len(rows)} generations kept, written to {out}")


if __name__ == "__main__":
    main()
