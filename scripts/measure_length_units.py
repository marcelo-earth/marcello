"""Measure the corpus length in words and in tokens, per language.

The length bonus in the reward peaks at `target_length`. That number is only
meaningful if it is expressed in the same unit the reward measures (tokens) and
sits where Marcelo's real writing actually sits. This script reports both, so
the target is picked from data instead of guessed.

Spanish and English are reported separately because Qwen's tokenizer splits
them at very different rates, which is what made the old word-based target
unreachable inside `max_new_tokens`. See issue #17.

Usage:
    python scripts/measure_length_units.py
    python scripts/measure_length_units.py --model Qwen/Qwen2.5-1.5B --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import statistics

from datasets import load_from_disk
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

console = Console()

# short function words, enough to separate the two languages Marcelo writes in
ES_MARKERS = {"de", "la", "el", "que", "y", "en", "los", "un", "una", "para", "con", "por", "es"}
EN_MARKERS = {"the", "of", "and", "to", "in", "is", "that", "it", "for", "with", "on", "a"}


def detect_language(text: str) -> str:
    """Spanish or English by function-word count. Ties go to Spanish."""
    words = [word.strip(".,;:!?¿¡\"'()").lower() for word in text.split()]
    spanish = sum(1 for word in words if word in ES_MARKERS)
    english = sum(1 for word in words if word in EN_MARKERS)
    return "es" if spanish >= english else "en"


def collect(splits: list[str], tokenizer) -> dict[str, list[tuple[int, int]]]:
    """Word and token counts for every positive sample, bucketed by language."""
    buckets: dict[str, list[tuple[int, int]]] = {"es": [], "en": []}
    for split in splits:
        dataset = load_from_disk(split)
        labels = dataset["label"] if "label" in dataset.column_names else [1] * len(dataset)
        for text, label in zip(dataset["text"], labels):
            if label != 1:
                continue
            words = len(text.split())
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            buckets[detect_language(text)].append((words, tokens))
    return buckets


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["data/processed/train", "data/processed/val"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    buckets = collect(args.splits, tokenizer)

    table = Table(title=f"positive-sample length under {args.model}")
    table.add_column("lang")
    table.add_column("n", justify="right")
    table.add_column("tokens/word", justify="right")
    table.add_column("words p50", justify="right")
    table.add_column("tokens p50", justify="right")
    table.add_column("tokens p90", justify="right")
    table.add_column("tokens max", justify="right")
    table.add_column(f"over {args.max_new_tokens}", justify="right")

    for lang, rows in buckets.items():
        if not rows:
            continue
        tokens = [token_count for _, token_count in rows]
        words = [word_count for word_count, _ in rows]
        ratios = [t / w for w, t in rows if w]
        over = sum(1 for t in tokens if t > args.max_new_tokens) / len(tokens)
        table.add_row(
            lang,
            str(len(rows)),
            f"{statistics.mean(ratios):.2f}",
            f"{statistics.median(words):.0f}",
            f"{statistics.median(tokens):.0f}",
            str(percentile(tokens, 0.9)),
            str(max(tokens)),
            f"{over:.0%}",
        )

    console.print(table)

    pooled = [token_count for rows in buckets.values() for _, token_count in rows]
    if not pooled:
        console.print("[red]no positive samples found[/]")
        return

    console.print(f"\npooled median: [bold]{statistics.median(pooled):.0f}[/] tokens")
    console.print(f"pooled p90:    [bold]{percentile(pooled, 0.9)}[/] tokens")
    console.print(
        "\ntarget_length should sit near the pooled median, and 2x it (where the "
        f"bonus reaches zero) should stay under max_new_tokens={args.max_new_tokens}."
    )


if __name__ == "__main__":
    main()
