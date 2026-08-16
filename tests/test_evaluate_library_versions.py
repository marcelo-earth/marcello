"""Tests for the resolved library versions recorded in evaluate.py's run metadata."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import evaluate  # noqa: E402


def test_library_versions_reports_the_installed_fast_movers():
    versions = evaluate._library_versions()

    assert set(versions) == {"torch", "transformers", "trl", "peft"}
    for package, resolved in versions.items():
        assert resolved is not None, f"{package} should be installed in the test environment"


def test_library_versions_reports_none_for_a_missing_package(monkeypatch):
    def fake_version(name):
        if name == "trl":
            raise evaluate.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(evaluate, "version", fake_version)

    versions = evaluate._library_versions()

    assert versions["trl"] is None
    assert versions["torch"] == "1.2.3"
