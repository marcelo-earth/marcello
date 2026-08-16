"""Tests that _build_grpo_args never silently drops a configured hyperparameter."""

from __future__ import annotations

import json

import pytest

import marcello.grpo.trainer as trainer_module
from marcello.grpo.trainer import MarceLLoGRPOConfig, MarceLLoGRPOTrainer

BASE_PARAMS = [
    "output_dir",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_grad_norm",
    "num_generations",
    "max_completion_length",
    "temperature",
    "top_p",
    "log_completions",
    "report_to",
]


def _fake_grpo_config(param_names):
    """Build a fake GRPOConfig whose __init__ only accepts the given names.

    Mimics inspecting a specific installed TRL version: only params present
    in `param_names` are "supported" by inspect.signature.
    """
    params = ", ".join(f"{name}=None" for name in param_names)
    body = ", ".join(f"{name!r}: {name}" for name in param_names)
    namespace: dict = {}
    exec(f"def __init__(self, {params}):\n    self.kwargs = {{{body}}}\n", namespace)  # noqa: S102
    return type("FakeGRPOConfig", (), {"__init__": namespace["__init__"]})


def _trainer(tmp_path, **overrides):
    config = MarceLLoGRPOConfig(output_dir=str(tmp_path), **overrides)
    return MarceLLoGRPOTrainer(config)


def test_maps_modern_field_names_without_warning(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(
        trainer_module, "GRPOConfig", _fake_grpo_config(BASE_PARAMS + ["beta", "epsilon"])
    )
    t = _trainer(tmp_path, kl_coef=0.1, clip_range=0.2)

    resolved = t._build_grpo_args()

    assert resolved.kwargs["beta"] == 0.1
    assert resolved.kwargs["epsilon"] == 0.2
    assert len(recwarn) == 0


def test_falls_back_to_legacy_field_names_without_warning(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(
        trainer_module, "GRPOConfig", _fake_grpo_config(BASE_PARAMS + ["kl_coef", "clip_range"])
    )
    t = _trainer(tmp_path, kl_coef=0.1, clip_range=0.2)

    resolved = t._build_grpo_args()

    assert resolved.kwargs["kl_coef"] == 0.1
    assert resolved.kwargs["clip_range"] == 0.2
    assert len(recwarn) == 0


def test_warns_and_records_when_kl_coef_has_no_supported_name(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "GRPOConfig", _fake_grpo_config(BASE_PARAMS))
    t = _trainer(tmp_path)

    with pytest.warns(UserWarning, match="kl_coef"):
        t._build_grpo_args()

    record = json.loads((tmp_path / "resolved_grpo_config.json").read_text())
    assert any("kl_coef" in item for item in record["dropped"])
    assert "beta" not in record["resolved_kwargs"]
    assert "kl_coef" not in record["resolved_kwargs"]


def test_warns_when_a_core_field_is_renamed_upstream(tmp_path, monkeypatch):
    renamed = [p for p in BASE_PARAMS if p != "temperature"] + ["beta", "epsilon"]
    monkeypatch.setattr(trainer_module, "GRPOConfig", _fake_grpo_config(renamed))
    t = _trainer(tmp_path)

    with pytest.warns(UserWarning, match="temperature"):
        t._build_grpo_args()

    record = json.loads((tmp_path / "resolved_grpo_config.json").read_text())
    assert "temperature" in record["dropped"]
    assert "temperature" not in record["resolved_kwargs"]


def test_resolved_config_is_saved_next_to_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer_module, "GRPOConfig", _fake_grpo_config(BASE_PARAMS + ["beta", "epsilon"])
    )
    t = _trainer(tmp_path, learning_rate=5e-7)

    t._build_grpo_args()

    record = json.loads((tmp_path / "resolved_grpo_config.json").read_text())
    assert record["resolved_kwargs"]["learning_rate"] == 5e-7
    assert record["dropped"] == []
