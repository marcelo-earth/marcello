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

Length was not the only such cue. Language and verse-versus-prose are just as
visible without reading, and Spanish verse ran 113 positives against 74
negatives, so form alone made a poem 60% likely to be Marcelo. The balancer
therefore strata on all three surface features together.
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


def is_verse(text: str) -> bool:
    """Verse is the shape, not the content: several short hard-wrapped lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 3


def _surface_strata(texts: list[str], num_bins: int) -> list[tuple]:
    """The surface features a classifier can exploit without reading anything.

    Length, language and verse-versus-prose are all visible before a single
    word is understood. Whenever one of them correlates with the label the
    head will take it, so each has to be balanced away rather than trusted.
    """
    from marcello.grpo.prompting import infer_language

    counts = [_word_count(text) for text in texts]
    edges = _bin_edges(counts, num_bins)

    return [
        (_bin_of(count, edges), infer_language(text), is_verse(text))
        for text, count in zip(texts, counts)
    ]


def length_matched_indices(
    texts: list[str],
    labels: list[int],
    seed: int = 42,
    num_bins: int = 8,
    match_form: bool = True,
) -> list[int]:
    """Indices of a subset where every surface stratum is class balanced.

    Returns positions into `texts`, sorted. Strata holding only one class drop
    out entirely: a length or a form only one class ever reaches is a giveaway,
    so there is nothing there that would generalise.

    With `match_form`, language and verse-versus-prose join word count as
    strata. Length alone was not enough: Spanish verse ran 113 positives to 74
    negatives, a 60% base rate for Marcelo that the probe read back almost
    exactly as Becquer's 0.61.
    """
    counts = [_word_count(text) for text in texts]
    if match_form:
        strata = _surface_strata(texts, num_bins)
    else:
        edges = _bin_edges(counts, num_bins)
        strata = [(_bin_of(count, edges),) for count in counts]

    buckets: dict[tuple, list[int]] = {}
    for index, (stratum, label) in enumerate(zip(strata, labels)):
        buckets.setdefault((stratum, label), []).append(index)

    rng = random.Random(seed)
    kept: list[int] = []
    for stratum in sorted({key[0] for key in buckets}, key=repr):
        positives = buckets.get((stratum, 1), [])
        negatives = buckets.get((stratum, 0), [])
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
