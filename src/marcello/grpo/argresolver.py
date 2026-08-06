"""Loud version-tolerance resolution for TRL GRPOConfig fields.

Pure module: it must never import TRL (or torch/transformers), so the
mapping can be unit-tested without the heavy ML stack.

The problem this solves (issue #20): `_build_grpo_args` used to filter every
kwarg by the installed TRL's supported fields, so an upstream rename turned a
configured hyperparameter into a silently absent one — training ran with TRL's
default instead of the configured value, printing nothing and failing nothing.

The fix keeps the fallback chains but makes every silent drop loud:

- Safety-critical RL hyperparameters (``kl_coef``, ``clip_range``) raise
  ``RuntimeError`` when they map to no supported field: silently running with
  TRL's default KL/clip changes training behavior.
- Optional sampling/observability fields (``temperature``, ``top_p``,
  ``log_completions``) emit a loud warning instead, preserving version
  tolerance (some TRL versions simply do not expose them as GRPOConfig args).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# MarceLLo config name -> TRL GRPOConfig field fallback chain (first match wins).
_KL_COEF_CHAIN = ("beta", "kl_coef")
_CLIP_RANGE_CHAIN = ("epsilon", "clip_range")

# Fields whose silent absence changes training behavior. They must map or fail.
_CRITICAL = frozenset({"kl_coef", "clip_range"})

# Fields TRL versions may legitimately not expose as GRPOConfig args. These
# drop with a loud warning, not an error, so training stays version-tolerant.
_OPTIONAL = frozenset({"temperature", "top_p", "log_completions"})


def _target_for(key: str, supported: set[str]) -> str | None:
    if key == "kl_coef":
        return next((name for name in _KL_COEF_CHAIN if name in supported), None)
    if key == "clip_range":
        return next((name for name in _CLIP_RANGE_CHAIN if name in supported), None)
    return key if key in supported else None


def resolve(config_kwargs: dict, supported: set[str]) -> dict:
    """Map MarceLLo kwarg names onto fields supported by the installed TRL.

    Raises ``RuntimeError`` for safety-critical configured hyperparameters
    that map to no supported field, and logs a loud warning for optional ones,
    so an upstream rename can never make a configured value silently disappear.

    Args:
        config_kwargs: kwargs keyed by MarceLLo names (``kl_coef``,
            ``clip_range``, ...). Keys with no fallback chain (``output_dir``,
            ``learning_rate``, ...) must exist verbatim in ``supported``.
        supported: the set of ``GRPOConfig.__init__`` parameter names.

    Returns:
        A kwargs dict keyed by TRL field names, ready for ``GRPOConfig(**out)``.
    """
    resolved: dict = {}
    unmapped_critical: list[str] = []
    unmapped_optional: list[str] = []

    for key, value in config_kwargs.items():
        target = _target_for(key, supported)
        if target is None:
            (unmapped_optional if key in _OPTIONAL else unmapped_critical).append(key)
            continue
        resolved[target] = value

    if unmapped_optional:
        logger.warning(
            "Configured GRPO fields not accepted by the installed TRL and "
            "dropped (training continues with TRL defaults): %s. TRL exposes: "
            "%s. Pin trl to a known version or extend the mapping in "
            "marcello.grpo.argresolver.",
            sorted(unmapped_optional),
            sorted(supported),
        )

    if unmapped_critical:
        raise RuntimeError(
            "Configured GRPO hyperparameters are not accepted by the installed "
            f"TRL and would run with TRL defaults instead: "
            f"{sorted(unmapped_critical)}. TRL exposes: {sorted(supported)}. "
            "Pin trl to a known version or extend the mapping in "
            "marcello.grpo.argresolver."
        )
    return resolved


def record(requested: dict, resolved: dict, output_dir: Path) -> Path:
    """Persist the resolved TRL config next to the run output.

    Writes ``resolved_grpo_config.json`` under ``output_dir`` so the run record
    states what actually ran rather than what the YAML asked for, and returns
    the written path. Recording must never block training, so values that are
    not JSON-serializable are coerced with ``str``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "resolved_grpo_config.json"
    payload = {"requested": requested, "resolved_for_trl": resolved}
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return record_path
