"""Tests for the pure GRPO arg resolver (issue #20).

These tests import only ``marcello.grpo.argresolver`` — a TRL-free module —
so they run with plain Python and no ML stack. Run with:

    python -m pytest tests/test_argresolver.py

or directly: ``python tests/test_argresolver.py`` (falls back to pytest-free
asserts via ``main()``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from marcello.grpo.argresolver import record, resolve


def test_maps_kl_coef_to_beta_when_beta_supported():
    out = resolve({"kl_coef": 0.1, "learning_rate": 5e-7}, supported={"beta", "learning_rate"})
    assert out == {"beta": 0.1, "learning_rate": 5e-7}


def test_falls_back_to_kl_coef_when_beta_missing():
    out = resolve({"kl_coef": 0.1}, supported={"kl_coef"})
    assert out == {"kl_coef": 0.1}


def test_maps_clip_range_to_epsilon_when_supported():
    out = resolve({"clip_range": 0.2}, supported={"epsilon"})
    assert out == {"epsilon": 0.2}


def test_falls_back_to_clip_range_when_epsilon_missing():
    out = resolve({"clip_range": 0.2}, supported={"clip_range"})
    assert out == {"clip_range": 0.2}


def test_passes_through_plain_supported_keys():
    out = resolve(
        {"output_dir": "outputs/grpo", "num_generations": 8, "temperature": 0.8},
        supported={"output_dir", "num_generations", "temperature"},
    )
    assert out == {"output_dir": "outputs/grpo", "num_generations": 8, "temperature": 0.8}


def test_raises_when_neither_beta_nor_kl_coef_supported():
    try:
        resolve({"kl_coef": 0.1}, supported={"learning_rate"})
    except RuntimeError as exc:
        assert "kl_coef" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unmapped kl_coef")


def test_raises_when_neither_epsilon_nor_clip_range_supported():
    try:
        resolve({"clip_range": 0.2}, supported={"learning_rate"})
    except RuntimeError as exc:
        assert "clip_range" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unmapped clip_range")


def test_warns_and_drops_optional_fields_without_raising():
    import logging

    warnings = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    logger = logging.getLogger("marcello.grpo.argresolver")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        out = resolve(
            {"temperature": 0.8, "top_p": 0.95, "learning_rate": 5e-7},
            supported={"learning_rate"},
        )
    finally:
        logger.removeHandler(handler)
    assert out == {"learning_rate": 5e-7}
    assert any("temperature" in w for w in warnings)
    assert any("top_p" in w for w in warnings)


def test_raises_when_a_plain_key_is_unsupported():
    try:
        resolve({"output_dir": "outputs/grpo"}, supported={"learning_rate"})
    except RuntimeError as exc:
        assert "output_dir" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unmapped output_dir")


def test_record_writes_requested_and_resolved(tmp_path: Path):
    requested = {"kl_coef": 0.1, "clip_range": 0.2}
    resolved = {"beta": 0.1, "epsilon": 0.2}
    path = record(requested, resolved, tmp_path / "grpo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["requested"] == requested
    assert payload["resolved_for_trl"] == resolved


def test_record_serializes_non_json_values_with_str(tmp_path: Path):
    requested = {"path": Path("outputs/grpo")}
    resolved = {"obj": {"x": Path("a/b")}}
    path = record(requested, resolved, tmp_path / "grpo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Path objects are coerced via str(); compare platform-agnostically.
    assert payload["requested"] == {"path": str(Path("outputs/grpo"))}
    assert payload["resolved_for_trl"] == {"obj": {"x": str(Path("a/b"))}}


if __name__ == "__main__":
    # Minimal pytest-free runner so the suite runs on plain Python.
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        checks = [
            test_maps_kl_coef_to_beta_when_beta_supported,
            test_falls_back_to_kl_coef_when_beta_missing,
            test_maps_clip_range_to_epsilon_when_supported,
            test_falls_back_to_clip_range_when_epsilon_missing,
            test_passes_through_plain_supported_keys,
            test_raises_when_neither_beta_nor_kl_coef_supported,
            test_raises_when_neither_epsilon_nor_clip_range_supported,
            test_warns_and_drops_optional_fields_without_raising,
            test_raises_when_a_plain_key_is_unsupported,
            lambda: test_record_writes_requested_and_resolved(tmp / "t"),
            lambda: test_record_serializes_non_json_values_with_str(tmp / "t"),
        ]
        for fn in checks:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL: {fn.__name__}: {exc}")
                sys.exit(1)
            print(f"ok: {fn.__name__}")
    print("All argresolver checks passed")
