"""Tests for the corpus manifest hash and the verify_corpus drift check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import Dataset

from marcello.data.manifest import corpus_hash

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import verify_corpus  # noqa: E402


def test_corpus_hash_is_order_independent():
    assert corpus_hash(["a", "b", "c"]) == corpus_hash(["c", "a", "b"])


def test_corpus_hash_changes_when_a_text_changes():
    assert corpus_hash(["a", "b"]) != corpus_hash(["a", "c"])


def _write_corpus(root: Path, train_texts: list[str], val_texts: list[str], hash_value: str):
    Dataset.from_dict({"text": train_texts, "label": [1] * len(train_texts)}).save_to_disk(
        root / "train"
    )
    Dataset.from_dict({"text": val_texts, "label": [1] * len(val_texts)}).save_to_disk(root / "val")
    (root / "manifest.json").write_text(json.dumps({"text_hash": hash_value}), encoding="utf-8")


def _run_verify(root: Path, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["verify_corpus.py", "--path", str(root)])
    try:
        verify_corpus.main()
    except SystemExit as exc:
        return exc.code or 0
    return 0


def test_verify_corpus_passes_when_hash_matches(tmp_path, monkeypatch, capsys):
    matching_hash = corpus_hash(["train text", "val text"])
    _write_corpus(tmp_path, ["train text"], ["val text"], matching_hash)

    assert _run_verify(tmp_path, monkeypatch) == 0
    assert "matches" in capsys.readouterr().out


def test_verify_corpus_fails_on_drift(tmp_path, monkeypatch, capsys):
    _write_corpus(tmp_path, ["train text"], ["val text"], "not-the-real-hash")

    assert _run_verify(tmp_path, monkeypatch) != 0
    assert "DRIFT" in capsys.readouterr().out


def test_verify_corpus_fails_when_manifest_missing(tmp_path, monkeypatch, capsys):
    Dataset.from_dict({"text": ["x"], "label": [1]}).save_to_disk(tmp_path / "train")
    Dataset.from_dict({"text": ["y"], "label": [1]}).save_to_disk(tmp_path / "val")

    assert _run_verify(tmp_path, monkeypatch) != 0
    assert "No manifest" in capsys.readouterr().out
