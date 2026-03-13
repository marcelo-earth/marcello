"""GRPO utilities with lazy imports for lightweight prompt helpers."""

from __future__ import annotations

__all__ = ["StyleReward", "MarceLLoGRPOTrainer", "build_control_prompt", "ensure_control_prompt"]


def __getattr__(name: str):
    if name == "StyleReward":
        from marcello.grpo.reward import StyleReward

        return StyleReward
    if name == "MarceLLoGRPOTrainer":
        from marcello.grpo.trainer import MarceLLoGRPOTrainer

        return MarceLLoGRPOTrainer
    if name == "build_control_prompt":
        from marcello.grpo.prompting import build_control_prompt

        return build_control_prompt
    if name == "ensure_control_prompt":
        from marcello.grpo.prompting import ensure_control_prompt

        return ensure_control_prompt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
