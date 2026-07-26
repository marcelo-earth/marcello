"""Class balancing that also removes length as a shortcut.

Plain undersampling equalises how many positives and negatives there are, but
says nothing about how long they are. In this corpus the negatives ran a full
ten words longer than the positives at the median, and the classifier picked
that up: within the negatives alone, score correlated with length at -0.394.
Short text scored as Marcelo no matter what it said, which is exactly why the
sanity probe kept failing on four-line poems.

Length matching fixes the cause instead of the symptom. Texts are binned by
word count and each bin is trimmed until it holds as many positives as
negatives, so word count carries no information about the label and the head
has to read the text.
"""

from __future__ import annotations

import random


def _word_count(text: str) -> int:
    return len(text.split())


def _bin_edges(counts: list[int], num_bins: int) -> list[int]:
    """Quantile edges, deduplicated so heavily repeated lengths collapse."""
    ordered = sorted(counts)
    edges = {ordered[int(len(ordered) * i / num_bins)] for i in range(1, num_bins)}
    return sorted(edges)


def _bin_of(count: int, edges: list[int]) -> int:
    index = 0
    for edge in edges:
        if count < edge:
            break
        index += 1
    return index


def length_matched_indices(
    texts: list[str],
    labels: list[int],
    seed: int = 42,
    num_bins: int = 8,
) -> list[int]:
    """Indices of a subset where each length bin is class balanced.

    Returns positions into `texts`, sorted. Bins holding only one class drop
    out entirely: a length only one class ever reaches is a giveaway, so there
    is nothing to learn there that would generalise.
    """
    counts = [_word_count(text) for text in texts]
    edges = _bin_edges(counts, num_bins)

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (count, label) in enumerate(zip(counts, labels)):
        buckets.setdefault((_bin_of(count, edges), label), []).append(index)

    rng = random.Random(seed)
    kept: list[int] = []
    for bin_index in {key[0] for key in buckets}:
        positives = buckets.get((bin_index, 1), [])
        negatives = buckets.get((bin_index, 0), [])
        keep = min(len(positives), len(negatives))
        if not keep:
            continue
        kept += rng.sample(positives, keep) + rng.sample(negatives, keep)

    return sorted(kept)


def undersample_indices(labels: list[int], seed: int = 42) -> list[int]:
    """Plain class balancing: trim the majority class at random."""
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    keep = min(len(positives), len(negatives))

    rng = random.Random(seed)
    if len(positives) > keep:
        positives = rng.sample(positives, keep)
    else:
        negatives = rng.sample(negatives, keep)

    return sorted(positives + negatives)
