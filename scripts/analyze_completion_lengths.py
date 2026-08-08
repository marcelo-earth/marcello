"""Analyze completion token lengths against the issue #30 checklist.

Author: RawNuke
Copyright (c) 2026 RawNuke. All rights reserved.

This script answers three questions from the issue:
  1. Is the completion length mass pinned at max_completion_length?
  2. Do the generations sit near the corpus reference (median 51, p90 85)?
  3. Does reward_length_bonus_mean show within-group variance?

It works with:
  - evaluate.py JSON output (--eval-json)
  - raw completions file, one per line (--completions)
  - GRPO training log files (not yet implemented)

Usage:
  python scripts/analyze_completion_lengths.py --eval-json outputs/eval/latest.json
  python scripts/analyze_completion_lengths.py --completions completions.txt
  python scripts/analyze_completion_lengths.py --completions completions.txt --target-length 60
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

console = Console()

# Reference values from scripts/measure_length_units.py (corpus positives)
CORPUS_MEDIAN = 51
CORPUS_P90 = 85
MAX_COMPLETION_LENGTH = 256


def tokenize(texts: list[str], model_name: str = "Qwen/Qwen2.5-1.5B") -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def length_bonus(length: int, target: int) -> float:
    diff = abs(length - target) / target
    return max(0.0, 1.0 - diff)


def histogram_string(
    values: list[int],
    bins: int = 20,
    width: int = 50,
    cap: int = MAX_COMPLETION_LENGTH,
) -> str:
    if not values:
        return "no data"

    bin_width = cap / bins
    counts = [0] * bins
    for value in values:
        idx = min(int(value / bin_width), bins - 1) if bin_width > 0 else 0
        counts[idx] = counts[idx] + 1

    max_count = max(counts) if counts else 1
    lines = []
    for i in range(bins):
        left = int(i * bin_width)
        right = int((i + 1) * bin_width) if i < bins - 1 else cap
        bar_len = int((counts[i] / max_count) * width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"  {left:>3}-{right:<3} [{counts[i]:>4}] {bar}")
    return "\n".join(lines)


def analyze(
    completions: list[str],
    target_length: int = 60,
    model_name: str = "Qwen/Qwen2.5-1.5B",
    max_completion_length: int = MAX_COMPLETION_LENGTH,
) -> dict:
    token_lengths = tokenize(completions, model_name)
    bonuses = [length_bonus(length, target_length) for length in token_lengths]

    n = len(token_lengths)
    cap_threshold = max_completion_length - 2
    at_cap = sum(1 for length in token_lengths if length >= cap_threshold)
    near_cap = sum(
        1 for length in token_lengths if length >= max_completion_length * 0.9
    )
    near_target = sum(
        1 for length in token_lengths if abs(length - target_length) <= target_length * 0.2
    )

    mean_len = statistics.mean(token_lengths) if token_lengths else 0
    std_len = statistics.pstdev(token_lengths) if n > 1 else 0
    median_len = statistics.median(token_lengths) if token_lengths else 0
    p90_len = percentile(token_lengths, 0.9) if token_lengths else 0
    p95_len = percentile(token_lengths, 0.95) if token_lengths else 0
    bonus_mean = statistics.mean(bonuses) if bonuses else 0
    bonus_std = statistics.pstdev(bonuses) if n > 1 else 0
    bonus_at_ceiling = sum(1 for bonus in bonuses if bonus >= 0.95)

    cap_threshold_pct = (cap_threshold / max_completion_length * 100) if max_completion_length > 0 else 0
    cap_mass = (at_cap / n * 100) if n > 0 else 0
    bonus_ceiling_fraction = (bonus_at_ceiling / n * 100) if n > 0 else 0

    concern_cap_pinned = cap_mass > 20
    concern_near_cap = (near_cap / n * 100) > 30 if n > 0 else False
    concern_no_variance = bonus_std < 0.01 and n > 0
    concern_median_offset = median_len > CORPUS_MEDIAN * 3 if CORPUS_MEDIAN > 0 else False
    concern_p90_offset = p90_len > CORPUS_P90 * 3 if CORPUS_P90 > 0 else False
    concern_bonus_ceiling = bonus_ceiling_fraction > 30

    flags = []
    if concern_cap_pinned:
        flags.append(f"CAP-PINNED: {cap_mass:.1f}% mass at cap ({cap_threshold_pct:.0f}%+ of {max_completion_length})")
    if concern_near_cap and not concern_cap_pinned:
        flags.append(f"NEAR-CAP: {near_cap / n * 100:.1f}% mass near cap (90%+ of {max_completion_length})")
    if concern_no_variance:
        flags.append(f"NO-VARIANCE: length bonus std is {bonus_std:.4f}")
    if concern_median_offset:
        flags.append(f"MEDIAN-OFFSET: median is {median_len:.0f}, corpus median is {CORPUS_MEDIAN}")
    if concern_p90_offset:
        flags.append(f"P90-OFFSET: p90 is {p90_len:.0f}, corpus p90 is {CORPUS_P90}")
    if concern_bonus_ceiling:
        flags.append(f"BONUS-CEILING: {bonus_ceiling_fraction:.1f}% of bonuses at ceiling (>0.95)")

    decision = "PASS" if not flags else "FAIL: " + "; ".join(flags)

    return {
        "n": n,
        "mean": mean_len,
        "std": std_len,
        "median": median_len,
        "p90": p90_len,
        "p95": p95_len,
        "min": min(token_lengths) if token_lengths else 0,
        "max": max(token_lengths) if token_lengths else 0,
        "at_cap_n": at_cap,
        "at_cap_pct": cap_mass,
        "near_cap_n": near_cap,
        "near_target_n": near_target,
        "cap_threshold": cap_threshold,
        "reward_length_bonus_mean": bonus_mean,
        "reward_length_bonus_std": bonus_std,
        "bonus_at_ceiling_n": bonus_at_ceiling,
        "bonus_at_ceiling_pct": bonus_ceiling_fraction,
        "corpus_median_reference": CORPUS_MEDIAN,
        "corpus_p90_reference": CORPUS_P90,
        "target_length": target_length,
        "max_completion_length": max_completion_length,
        "flags": flags,
        "decision": decision,
        "histogram": histogram_string(token_lengths, cap=max_completion_length),
        "token_lengths": token_lengths,
        "bonuses": bonuses,
    }


def load_completions_from_eval_json(path: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", {})
    per_prompt = results.get("per_prompt", [])
    completions = []
    for entry in per_prompt:
        for key in ("grpo_completion", "completion"):
            text = entry.get(key)
            if text:
                completions.append(text)
                break
    return completions


def main():
    parser = argparse.ArgumentParser(
        description="Analyze completion token lengths against issue #30 checklist"
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        help="Path to evaluate.py JSON output",
    )
    parser.add_argument(
        "--completions",
        type=str,
        help="Path to file with one completion per line",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=60,
        help="target_length from grpo.yaml (default: 60)",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=MAX_COMPLETION_LENGTH,
        help="max_completion_length from grpo config (default: 256)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Model for the tokenizer",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    completions: list[str] = []
    source_label = ""

    if args.eval_json:
        completions = load_completions_from_eval_json(args.eval_json)
        source_label = f"eval JSON ({args.eval_json})"
    elif args.completions:
        with open(args.completions) as f:
            completions = [line.strip() for line in f if line.strip()]
        source_label = f"completions file ({args.completions})"
    else:
        console.print("[red]Provide --eval-json or --completions[/]")
        return

    if not completions:
        console.print("[yellow]No completions found[/]")
        return

    result = analyze(
        completions,
        target_length=args.target_length,
        model_name=args.model,
        max_completion_length=args.max_completion_length,
    )

    console.print(f"\n[bold]Completion Length Analysis[/]")
    console.print(f"Source: {source_label}")
    console.print(f"Completions: {result['n']}")
    console.print()

    console.print("[bold]Length Histogram (tokens)[/]")
    console.print(result["histogram"])
    console.print()

    table = Table(title="Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Reference", justify="right")

    rows = [
        ("n", str(result["n"]), ""),
        ("mean", f"{result['mean']:.1f}", ""),
        ("std", f"{result['std']:.1f}", ""),
        ("median", f"{result['median']:.0f}", f"{CORPUS_MEDIAN} (corpus)"),
        ("p90", f"{result['p90']:.0f}", f"{CORPUS_P90} (corpus)"),
        ("p95", f"{result['p95']:.0f}", ""),
        ("min", str(result["min"]), ""),
        ("max", str(result["max"]), ""),
        (f"at cap (>{result['cap_threshold']})", f"{result['at_cap_n']} ({result['at_cap_pct']:.1f}%)", "0 (target)"),
        ("near cap (90%+)", str(result["near_cap_n"]), ""),
        ("near target (+-20%)", str(result["near_target_n"]), ""),
        ("", "", ""),
        ("reward_length_bonus_mean", f"{result['reward_length_bonus_mean']:.4f}", ""),
        ("reward_length_bonus_std", f"{result['reward_length_bonus_std']:.4f}", "> 0.01"),
        (
            "bonus at ceiling (>0.95)",
            f"{result['bonus_at_ceiling_n']} ({result['bonus_at_ceiling_pct']:.1f}%)",
            "0 (target)",
        ),
    ]
    for metric, value, reference in rows:
        table.add_row(metric, value, reference)

    console.print(table)
    console.print()

    if result["flags"]:
        console.print("[bold red]Issues found:[/]")
        for flag in result["flags"]:
            console.print(f"  [red]\u2717[/] {flag}")
    else:
        console.print("[bold green]No issues found[/]")

    console.print(f"\n[bold]Decision: {result['decision']}[/]")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: value for key, value in result.items() if key != "histogram"}
        serializable["histogram_text"] = result["histogram"]
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        console.print(f"\n[green]Saved to {output_path}[/]")


if __name__ == "__main__":
    main()
