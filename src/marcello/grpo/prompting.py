"""Prompt formatting utilities for style-aligned GRPO training."""

from __future__ import annotations

import re

from datasets import Dataset

SPANISH_HINTS = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "los",
    "para",
    "con",
    "como",
    "pero",
    "porque",
    "una",
    "del",
}
ENGLISH_HINTS = {
    "the",
    "and",
    "that",
    "with",
    "for",
    "you",
    "this",
    "have",
    "from",
    "they",
    "your",
    "into",
    "about",
    "because",
}


def infer_language(text: str) -> str:
    """Infer whether a text is primarily Spanish or English."""
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Záéíóúñü]+", lowered)
    if not tokens:
        return "es"

    english_hits = sum(1 for token in tokens if token in ENGLISH_HINTS)
    spanish_hits = sum(1 for token in tokens if token in SPANISH_HINTS)
    if any(ch in lowered for ch in "áéíóúñ¿¡"):
        spanish_hits += 2

    if english_hits > spanish_hits:
        return "en"
    return "es"


def extract_seed_text(prompt: str) -> str:
    """Extract the raw seed from a formatted GRPO prompt."""
    match = re.search(r"Seed:\s*(.+?)\n\nCompletion:", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected = [sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip()]
    return " ".join(selected)


def build_control_prompt(seed_text: str, style: str, language: str) -> str:
    """Build a prompt with explicit control tags for style and language."""
    normalized_style = style if style in {"poetic", "standard"} else "standard"
    normalized_language = language if language in {"es", "en"} else "es"

    if normalized_language == "es":
        instruction = (
            "Continua este texto con la voz de Marcelo. Mantén claridad, calidez y precisión."
        )
    else:
        instruction = "Continue this text in Marcelo's voice. Keep it clear, warm, and precise."

    return (
        f"<style:{normalized_style}> <lang:{normalized_language}> <task:continue>\n"
        f"{instruction}\n\n"
        f"Seed:\n{seed_text.strip()}\n\n"
        "Completion:"
    )


def is_control_prompt(prompt: str) -> bool:
    """Return whether a prompt already uses the control-token format."""
    stripped = prompt.lstrip()
    return stripped.startswith("<style:") and "Completion:" in stripped


def ensure_control_prompt(
    prompt: str,
    style: str = "standard",
    language: str | None = None,
) -> str:
    """Wrap a raw prompt with control tokens unless it is already formatted."""
    if is_control_prompt(prompt):
        return prompt.strip()

    resolved_language = language or infer_language(prompt)
    return build_control_prompt(prompt, style=style, language=resolved_language)


def extract_prompts_from_positive_dataset(dataset: Dataset, max_prompts: int = 500) -> Dataset:
    """Create control-tagged prompts from positive writing samples."""
    prompts: list[str] = []
    raw_prompts: list[str] = []
    styles: list[str] = []
    languages: list[str] = []

    for row in dataset:
        if row.get("label") != 1:
            continue

        text = row["text"].strip()
        seed_text = _first_sentences(text)
        if len(seed_text) <= 10:
            continue

        style = row.get("style", "standard")
        language = row.get("language") or infer_language(text)

        prompts.append(build_control_prompt(seed_text, style=style, language=language))
        raw_prompts.append(seed_text)
        styles.append(style)
        languages.append(language)

        if len(prompts) >= max_prompts:
            break

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "raw_prompt": raw_prompts,
            "style": styles,
            "language": languages,
        }
    )
