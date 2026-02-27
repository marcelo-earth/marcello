"""Evaluation metrics for style-aligned text generation.

Metrics:
  - Style Score: average classifier probability on generated text
  - Perplexity: fluency measure (lower = more fluent)
  - Distinct-N: lexical diversity (higher = more diverse vocabulary)
  - Length stats: average length of generated outputs
"""

from __future__ import annotations

from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def style_score(texts: list[str], classifier) -> dict:
    """Compute style score using the trained classifier.

    Returns mean, std, min, max of classifier probabilities.
    """
    probs = classifier.predict(texts)
    return {
        "style_score_mean": sum(probs) / len(probs),
        "style_score_std": (sum((p - sum(probs) / len(probs)) ** 2 for p in probs) / len(probs)) ** 0.5,
        "style_score_min": min(probs),
        "style_score_max": max(probs),
    }


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Compute Distinct-N: ratio of unique n-grams to total n-grams.

    Higher values indicate more diverse, less repetitive text.
    """
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def perplexity(texts: list[str], model_name: str = "Qwen/Qwen2.5-1.5B") -> float:
    """Compute perplexity of generated texts using a reference LLM.

    Lower perplexity = more fluent/natural text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        total_loss += outputs.loss.item() * encoded["input_ids"].size(1)
        total_tokens += encoded["input_ids"].size(1)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return torch.exp(torch.tensor(avg_loss)).item()


def length_stats(texts: list[str]) -> dict:
    """Compute basic length statistics."""
    word_counts = [len(t.split()) for t in texts]
    return {
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }


def compute_style_metrics(
    texts: list[str],
    classifier=None,
    compute_perplexity: bool = False,
    reference_model: str = "Qwen/Qwen2.5-1.5B",
) -> dict:
    """Compute all style metrics for a list of generated texts."""
    metrics = {}

    if classifier is not None:
        metrics.update(style_score(texts, classifier))

    metrics["distinct_1"] = distinct_n(texts, n=1)
    metrics["distinct_2"] = distinct_n(texts, n=2)
    metrics.update(length_stats(texts))

    if compute_perplexity:
        metrics["perplexity"] = perplexity(texts, model_name=reference_model)

    return metrics
