"""Tests for which probe items can fail the sanity probe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sanity_probe import gate_failures  # noqa: E402


def _item(id_: str, **extra) -> dict:
    return {"id": id_, "category": "c", "lang": "es", "text": "t", **extra}


def test_items_above_threshold_fail():
    failures = gate_failures([_item("a"), _item("b")], [0.9, 0.1], 0.4)
    assert [f["id"] for f in failures] == ["a"]


def test_ungated_items_never_fail():
    failures = gate_failures([_item("a", gate=False), _item("b")], [0.99, 0.5], 0.4)
    assert [f["id"] for f in failures] == ["b"]
