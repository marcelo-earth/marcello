"""Tests for evaluation metrics."""

from __future__ import annotations

from marcello.eval.metrics import distinct_n, length_stats, style_score


class FakeClassifier:
    def predict(self, texts):
        return [0.8 if "marcelo" in t.lower() else 0.2 for t in texts]


def test_distinct_1_fully_unique():
    texts = ["one two three four"]
    assert distinct_n(texts, n=1) == 1.0


def test_distinct_1_fully_repeated():
    texts = ["the the the the"]
    assert distinct_n(texts, n=1) == 0.25


def test_distinct_2_empty():
    assert distinct_n([], n=2) == 0.0


def test_distinct_2_single_word():
    # no bigrams possible from a single word
    assert distinct_n(["hello"], n=2) == 0.0


def test_distinct_n_multiple_texts():
    texts = ["a b c", "a b c"]
    # 2 total bigrams (a,b) and (b,c), 1 unique each → 2 unique / 4 total
    assert distinct_n(texts, n=2) == 0.5


def test_length_stats_single_text():
    result = length_stats(["one two three"])
    assert result["avg_words"] == 3.0
    assert result["min_words"] == 3
    assert result["max_words"] == 3


def test_length_stats_multiple_texts():
    result = length_stats(["one two", "a b c d"])
    assert result["min_words"] == 2
    assert result["max_words"] == 4
    assert result["avg_words"] == 3.0


def test_style_score_mean_and_range():
    classifier = FakeClassifier()
    texts = ["marcelo writes", "generic text"]
    result = style_score(texts, classifier)

    assert result["style_score_mean"] == 0.5
    assert result["style_score_min"] == 0.2
    assert result["style_score_max"] == 0.8
    assert result["style_score_std"] > 0


def test_style_score_uniform():
    class UniformClassifier:
        def predict(self, texts):
            return [0.6] * len(texts)

    result = style_score(["a", "b", "c"], UniformClassifier())
    assert result["style_score_mean"] == 0.6
    assert result["style_score_std"] == 0.0
    assert result["style_score_min"] == 0.6
    assert result["style_score_max"] == 0.6
