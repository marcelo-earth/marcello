"""Tests for class balancing."""

from __future__ import annotations

import statistics

from marcello.data.balance import is_verse, length_matched_indices, undersample_indices


def _text(words: int) -> str:
    return " ".join(["palabra"] * words)


def _verse(words: int) -> str:
    line = " ".join(["palabra"] * max(1, words // 4))
    return "\n".join([line] * 4)


def test_undersample_equalises_class_counts():
    labels = [1] * 30 + [0] * 10
    kept = undersample_indices(labels, seed=0)

    assert len(kept) == 20
    assert sum(labels[i] for i in kept) == 10


def test_undersample_returns_sorted_indices():
    labels = [1] * 5 + [0] * 5
    assert undersample_indices(labels, seed=0) == sorted(undersample_indices(labels, seed=0))


def test_length_matching_closes_the_median_gap():
    """The gap this exists to remove: negatives 10 words longer at the median."""
    texts = [_text(n) for n in range(20, 50)] + [_text(n) for n in range(30, 60)]
    labels = [1] * 30 + [0] * 30

    before_pos = statistics.median(len(texts[i].split()) for i in range(30))
    before_neg = statistics.median(len(texts[i].split()) for i in range(30, 60))
    assert before_neg - before_pos == 10

    kept = length_matched_indices(texts, labels, seed=0)
    after_pos = statistics.median(len(texts[i].split()) for i in kept if labels[i] == 1)
    after_neg = statistics.median(len(texts[i].split()) for i in kept if labels[i] == 0)

    assert abs(after_neg - after_pos) <= 2


def test_length_matching_keeps_classes_balanced():
    texts = [_text(n) for n in range(10, 60)] + [_text(n) for n in range(40, 90)]
    labels = [1] * 50 + [0] * 50

    kept = length_matched_indices(texts, labels, seed=0)
    assert sum(labels[i] for i in kept) == len(kept) // 2


def test_length_matching_drops_bins_holding_one_class():
    """A length only one class reaches is a giveaway, so it must not survive."""
    texts = [_text(5), _text(6), _text(200), _text(210)] + [_text(5), _text(6)]
    labels = [1, 1, 1, 1, 0, 0]

    kept = length_matched_indices(texts, labels, seed=0, num_bins=2)
    kept_lengths = {len(texts[i].split()) for i in kept}

    assert 200 not in kept_lengths and 210 not in kept_lengths


def test_is_verse_needs_several_hard_wrapped_lines():
    assert is_verse("uno\ndos\ntres")
    assert not is_verse("una sola linea de prosa corrida sin saltos")
    assert not is_verse("uno\ndos")


def test_form_matching_removes_the_verse_base_rate():
    """Verse ran 60% Marcelo, so form alone predicted the label. It must not."""
    texts = [_verse(30) for _ in range(60)] + [_text(30) for _ in range(40)]
    texts += [_verse(30) for _ in range(20)] + [_text(30) for _ in range(80)]
    labels = [1] * 100 + [0] * 100

    kept = length_matched_indices(texts, labels, seed=0, match_form=True)
    verse_labels = [labels[i] for i in kept if is_verse(texts[i])]

    assert verse_labels, "verse should not be eliminated entirely"
    assert sum(verse_labels) == len(verse_labels) // 2


def test_form_matching_can_be_switched_off():
    """Same lengths on both sides, so only the verse/prose split separates them."""
    texts = [_verse(n) for n in range(20, 80)] + [_text(n) for n in range(20, 80)]
    labels = [1] * 60 + [0] * 60

    # matching on form sees one class per stratum and keeps nothing
    assert not length_matched_indices(texts, labels, seed=0, match_form=True)
    # ignoring it, the length bins are mixed and the shortcut survives
    assert length_matched_indices(texts, labels, seed=0, match_form=False)


def test_length_matching_is_deterministic_for_a_seed():
    texts = [_text(n) for n in range(10, 60)] + [_text(n) for n in range(20, 70)]
    labels = [1] * 50 + [0] * 50

    assert length_matched_indices(texts, labels, seed=7) == length_matched_indices(
        texts, labels, seed=7
    )
