"""Clean raw base-model generations before they become negatives (issue #12).

Two things in the raw output let a classifier separate them without reading:

1. Markup. About half echo control tags (`<task:continue>`, `<lang:de>`) or
   HTML. "Has tags means negative" is a rule the policy escapes for free by not
   emitting tags, so the tags have to go before training, not after.
2. Prompt scaffolding. Most echo `Seed:` / `Completion:` labels, which no
   positive contains, so those are cut the same way.
3. Length. The generations run 138 words at the median against the positives'
   34, so the length-matched balancer threw away most of them.

After the first tag that follows real content, the model nearly always derails:
it switches language, repeats the seed, or loops on tags. So the text is cut
there, not just stripped, and then trimmed at a line or sentence boundary to a
length drawn from the positives. Cutting mid-sentence would add a new tell.
"""

from __future__ import annotations

import random
import re

# any tag-like token: <task:continue>, </seed>, <br>, <span class="x">, <\/p>
TAG = re.compile(r"<\\?/?[A-Za-z][^<>\n]{0,60}>")
# a line that opens with a prompt-template label, in either language
SCAFFOLD = re.compile(
    r"^[ \t]*(seeds?|complet\w*|continu\w*|task|translation|description|date|theme"
    r"|location|(the )?end|final)[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def strip_leading_tags(text: str) -> str:
    """Drop tags and whitespace before the first word of content."""
    while True:
        stripped = text.lstrip()
        match = TAG.match(stripped)
        if not match:
            return stripped
        text = stripped[match.end() :]


def cut_at_first_tag(text: str) -> str:
    """Keep only what comes before the first tag or scaffold label."""
    text = strip_leading_tags(text)
    cuts = [m.start() for m in (TAG.search(text), SCAFFOLD.search(text)) if m]
    if cuts:
        text = text[: min(cuts)]
    return text.strip()


def _units(text: str) -> tuple[list[str], str]:
    """Split into lines for verse, sentences for prose, and the joiner to use."""
    lines = [line.rstrip() for line in text.splitlines()]
    if sum(1 for line in lines if line.strip()) >= 3:
        return lines, "\n"
    return [s for s in SENTENCE_END.split(" ".join(text.split())) if s], " "


def trim_to_words(text: str, target: int) -> str:
    """Longest prefix of whole lines or sentences that stays within `target` words.

    If even the first unit is longer, it is kept whole: a cut mid-sentence is
    easier to spot than a long one.
    """
    units, joiner = _units(text)
    kept: list[str] = []
    words = 0
    for unit in units:
        count = len(unit.split())
        if kept and words + count > target:
            break
        kept.append(unit)
        words += count
        if words >= target:
            break
    return joiner.join(kept).strip()


def clean_generation(text: str, target_words: int) -> str:
    return trim_to_words(cut_at_first_tag(text), target_words)


def clean_generations(
    texts: list[str],
    positive_word_counts: list[int],
    seed: int = 42,
    min_chars: int = 50,
) -> list[str | None]:
    """Clean each text to a length drawn from the positives' word counts.

    Returns one entry per input, None where nothing usable is left, so callers
    can keep the metadata of each row aligned.
    """
    rng = random.Random(seed)
    cleaned: list[str | None] = []
    for text in texts:
        result = clean_generation(text, rng.choice(positive_word_counts))
        cleaned.append(result if len(result) >= min_chars else None)
    return cleaned
