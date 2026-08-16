"""Tests for the sanity-probe hard gate in train_grpo.py."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import train_grpo  # noqa: E402


def _write_classifier(base: Path) -> Path:
    classifier_dir = base / "best"
    classifier_dir.mkdir(parents=True)
    (classifier_dir / "model.pt").write_bytes(b"fake-weights")
    return classifier_dir


def _write_report(base: Path, *, passed: bool) -> Path:
    report_path = base / "sanity_probe.json"
    report_path.write_text(
        json.dumps({"passed": passed, "failures": [] if passed else [{"id": "x"}]}),
        encoding="utf-8",
    )
    return report_path


def test_skip_flag_bypasses_gate_without_touching_disk(tmp_path):
    train_grpo.check_probe_gate(str(tmp_path / "nonexistent"), skip=True)


def test_missing_report_exits_nonzero(tmp_path):
    classifier_dir = _write_classifier(tmp_path)
    with pytest.raises(SystemExit):
        train_grpo.check_probe_gate(str(classifier_dir), skip=False)


def test_failed_probe_exits_nonzero(tmp_path):
    classifier_dir = _write_classifier(tmp_path)
    _write_report(tmp_path, passed=False)
    with pytest.raises(SystemExit):
        train_grpo.check_probe_gate(str(classifier_dir), skip=False)


def test_stale_report_exits_nonzero(tmp_path):
    classifier_dir = _write_classifier(tmp_path)
    _write_report(tmp_path, passed=True)
    # Model checkpoint retrained after the probe report was written.
    time.sleep(0.01)
    (classifier_dir / "model.pt").write_bytes(b"retrained-weights")
    with pytest.raises(SystemExit):
        train_grpo.check_probe_gate(str(classifier_dir), skip=False)


def test_passing_fresh_report_does_not_exit(tmp_path):
    classifier_dir = _write_classifier(tmp_path)
    time.sleep(0.01)
    _write_report(tmp_path, passed=True)
    train_grpo.check_probe_gate(str(classifier_dir), skip=False)
