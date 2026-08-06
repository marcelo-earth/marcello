"""Loud version-tolerance resolution for TRL GRPOConfig fields.

Pure module: it must never import TRL (or torch/transformers), so the
mapping can be unit-tested without the heavy ML stack.

The problem this solves (issue #20): `_build_grpo_args` used to filter every
kwarg by the installed TRL's supported fields, so an upstream rename turned a
configured hyperparameter into a silently absent one — training ran with TRL's
default instead of the configured value, printing nothing and failing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

# MarceLLo config name -> TRL GRPOConfig field fallback chain (first match wins).
_KL_COEF_CHAIN = ("beta", "kl_coef")
_CLIP_RANGE_CHAIN = ("epsilon", "clip_range")


def resolve(config_kwargs: dict, supported: set[str]) -> dict:
    """Map MarceLLo kwarg names onto fields supported by the installed TRL.

    Raises ``RuntimeError`` if an explicitly configured hyperparameter cannot
    be mapped to any supported field, so an upstream rename can never make a
    configured value silently disappear.

    Args:
        config_kwargs: kwargs keyed by MarceLLo names (``kl_coef``,
            ``clip_range``, ...). Keys with no fallback chain (``output_dir``,
            ``learning_rate``, ...) must exist verbatim in ``supported``.
        supported: the set of ``GRPOConfig.__init__`` parameter names.

    Returns:
        A kwargs dict keyed by TRL field names, ready for ``GRPOConfig(**out)``.
    """
    resolved: dict = {}
    unmapped: list[str] = []

    for key, value in config_kwargs.items():
        if key == "kl_coef":
            target = next((name for name in _KL_COEF_CHAIN if name in supported), None)
        elif key == "clip_range":
            target = next((name for name in _CLIP_RANGE_CHAIN if name in supported), None)
        else:
            target = key if key in supported else None

        if target is None:
            unmapped.append(key)
            continue
        resolved[target] = value

    if unmapped:
        raise RuntimeError(
            "Configured GRPO hyperparameters are not accepted by the installed "
            f"TRL and were silently dropped before: {sorted(unmapped)}. "
            f"TRL exposes: {sorted(supported)}. Pin trl to a known version or "
            "extend the mapping in marcello.grpo.argresolver."
        )
    return resolved


def record(config_kwargs: dict, resolved: dict, output_dir: Path) -> Path:
    """Persist the resolved TRL config next to the run output.

    Writes ``resolved_grpo_config.json`` under ``output_dir`` so the run record
    states what actually ran rather than what the YAML asked for, and returns
    the written path. Never raises: recording must not block training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "resolved_grpo_config.json"
    payload = {
        "requested": config_kwargs,
        "resolved_for_trl": resolved,
    }
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record_path
