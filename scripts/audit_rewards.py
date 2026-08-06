"""Reward-hacking audit for top-k completions (issue #15).

Prints the highest- and lowest-reward completions next to their per-component
breakdown, flags the degenerate pattern FEEDBACK.md warns about (total rising
while raw_style_prob stays flat and length_bonus does the moving), and reports
n-grams shared across the top-k — a phrase in eight of ten top completions is
a hack, not a style.

Usage:
    python scripts/audit_rewards.py --generations outputs/eval/<run>.json --top-k 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

console = Console()

COMPONENT_KEYS = [
    "total",
    "raw_style_prob",
    "style_score",
    "length_bonus",
    "prompt_relevance",
    "repetition_penalty",
    "prompt_echo_penalty",
    "reference_copy_penalty",
]


def _load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data)
    if "per_prompt" not in results:
        raise SystemExit(
            f"{path}: no per_prompt entries with *_reward_breakdown. "
            "Re-run scripts/evaluate.py with --reward-config (and --reference-texts)."
        )
    # reward breakdowns were written per label (base/grpo); collect whichever exist
    labels = [label for label in ("base", "grpo") if f"{label}_reward_breakdown" in results["per_prompt"][0]]
    entries = []
    for entry in results["per_prompt"]:
        for label in labels:
            bd = entry.get(f"{label}_reward_breakdown")
            if bd is None:
                continue
            entries.append(
                {
                    "label": label,
                    "prompt": entry.get("prompt", ""),
                    "completion": entry.get(f"{label}_completion", ""),
                    **{k: bd.get(k, 0.0) for k in COMPONENT_KEYS},
                }
            )
    if not entries:
        raise SystemExit(f"{path}: no reward breakdowns found")
    return entries


def _content_tokens(text: str) -> list[str]:
    import re

    tokens = re.findall(r"[a-zA-Záéíóúñü']+", text.lower())
    return [t for t in tokens if len(t) > 2]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _flag_components(row: dict) -> bool:
    """Degenerate pattern: total up while raw style flat and length_bonus moving."""
    return (
        row["raw_style_prob"] < 0.3
        and row["length_bonus"] > 0.05
        and row["length_bonus"] > row["style_score"]
    )


def _print_rows(rows: list[dict], title: str) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("label", style="cyan")
    table.add_column("total")
    table.add_column("raw_style")
    table.add_column("style")
    table.add_column("len_bonus")
    table.add_column("rel")
    table.add_column("rep_pen")
    table.add_column("echo_pen")
    table.add_column("ref_pen")
    table.add_column("completion", overflow="fold", max_width=70)
    for row in rows:
        flag = "⚠" if _flag_components(row) else ""
        table.add_row(
            row["label"],
            f"{row['total']:.4f}",
            f"{row['raw_style_prob']:.3f}",
            f"{row['style_score']:.3f}",
            f"{row['length_bonus']:.3f}",
            f"{row['prompt_relevance']:.3f}",
            f"{row['repetition_penalty']:.3f}",
            f"{row['prompt_echo_penalty']:.3f}",
            f"{row['reference_copy_penalty']:.3f}",
            f"{flag} {row['completion'][:280]}",
        )
    console.print(table)


def _report_shared_ngrams(rows: list[dict], ngram_size: int, top: int) -> None:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows[:top]:
        counts.update(_ngrams(_content_tokens(row["completion"]), ngram_size))
    shared = [(gram, count) for gram, count in counts.items() if count >= max(2, top // 2)]
    console.print(f"\n[bold]n-grams shared across >= {max(2, top // 2)}/{top} top completions[/]")
    if not shared:
        console.print("  none — no obvious lexical hack")
        return
    for gram, count in sorted(shared, key=lambda item: -item[1])[:15]:
        console.print(f"  {count}x  {' '.join(gram)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit top-k completions for reward hacking")
    parser.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="Path to evaluate.py JSON output (must include per-prompt reward breakdowns)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reward-config", type=Path, default=None, help="Optional grpo.yaml for ngram size")
    parser.add_argument("--ngram-size", type=int, default=3)
    args = parser.parse_args()

    if args.reward_config is not None:
        with open(args.reward_config, encoding="utf-8") as f:
            rcfg = yaml.safe_load(f)
        args.ngram_size = rcfg.get("reward", rcfg).get("reference_ngram_size", args.ngram_size)

    entries = _load_entries(args.generations)
    console.print(f"Loaded {len(entries)} scored completions from {args.generations}")

    ranked = sorted(entries, key=lambda row: row["total"], reverse=True)
    _print_rows(ranked[: args.top_k], f"Top {args.top_k} by total reward")
    _print_rows(ranked[-args.top_k :], f"Bottom {args.top_k} by total reward")

    flagged = [row for row in entries if _flag_components(row)]
    console.print(
        f"\n[bold]{len(flagged)}/{len(entries)} completions[/] show the degenerate pattern "
        "(raw style flat, length_bonus carrying the total)"
    )
    if flagged:
        console.print("  → inspect these: the length bonus is being gamed, not style learned")

    _report_shared_ngrams(ranked, args.ngram_size, args.top_k)


if __name__ == "__main__":
    main()
