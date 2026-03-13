"""Tests for prompt formatting helpers that do not depend on torch."""

from __future__ import annotations

from marcello.grpo.prompting import ensure_control_prompt, is_control_prompt


def test_ensure_control_prompt_wraps_raw_prompt():
    prompt = ensure_control_prompt(
        "La noche se hizo más grande que la avenida.",
        style="poetic",
        language="es",
    )

    assert is_control_prompt(prompt)
    assert prompt.startswith("<style:poetic> <lang:es> <task:continue>")


def test_ensure_control_prompt_preserves_existing_control_prompt():
    prompt = "<style:standard> <lang:en> <task:continue>\nContinue this text in Marcelo's voice. Keep it clear, warm, and precise.\n\nSeed:\nThe city waited for an answer.\n\nCompletion:"

    assert ensure_control_prompt(prompt, style="poetic", language="es") == prompt
