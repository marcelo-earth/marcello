"""Corpus manifest hashing, shared by collect_data.py and verify_corpus.py."""

from __future__ import annotations

import hashlib


def corpus_hash(texts: list[str]) -> str:
    """SHA-256 over the sorted texts, so a rebuild that drifts is detectable.

    Sorting first means the hash depends only on which texts are present,
    not on the order the collector or sampler happened to produce them in.
    """
    return hashlib.sha256("\n".join(sorted(texts)).encode("utf-8")).hexdigest()
