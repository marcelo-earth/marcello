"""Measure whether two reward components move together across completions.

Issue #16 asked for this measurement before trusting that prompt_relevance and
prompt_echo_penalty grade different things. They are disjoint by construction now
(a token counted by the echo penalty is removed from relevance), but construction
is an argument, not evidence: this reads the per-completion breakdowns an eval run
already writes and reports what the components actually did on real generations.

A strong correlation between two components means one of them is redundant and the
weights are double-counting one behaviour. Near zero means they are measuring
different things and both earn their place.

Usage:
    python scripts/analyze_reward_independence.py outputs/eval/run.json
    python scripts/analyze_reward_independence.py outputs/eval/run.json --label base
    python scripts/analyze_reward_independence.py run.json --output independence.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# |r| above this means the pair is redundant enough that one component should go
REDUNDANT_ABOVE = 0.7
# ... and below this they are independent enough to keep both
INDEPENDENT_BELOW = 0.3

DEFAULT_PAIRS = [
    ("prompt_relevance", "prompt_echo_penalty"),
    ("prompt_relevance", "reference_copy_penalty"),
    ("style_score", "length_bonus"),
    ("prompt_echo_penalty", "repetition_penalty"),
]


def load_breakdowns(path: Path, label: str) -> list[dict]:
    """Pull the per-completion reward breakdowns an eval run wrote for one model."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        data = data["results"]

    key = f"{label}_reward_breakdown"
    breakdowns = [entry[key] for entry in data.get("per_prompt", []) if key in entry]
    if not breakdowns:
        raise SystemExit(
            f"No '{key}' entries in {path}. Re-run scripts/evaluate.py with "
            "--reward-config so it writes the per-component breakdown."
        )
    return breakdowns


def correlate(breakdowns: list[dict], left: str, right: str) -> dict | None:
    """Pearson r between two components, or None when one of them never varies."""
    xs = [b[left] for b in breakdowns if left in b and right in b]
    ys = [b[right] for b in breakdowns if left in b and right in b]
    if len(xs) < 3:
        return None

    row = {
        "left": left,
        "right": right,
        "n": len(xs),
        "left_mean": statistics.fmean(xs),
        "right_mean": statistics.fmean(ys),
        "left_stdev": statistics.pstdev(xs),
        "right_stdev": statistics.pstdev(ys),
        "both_active": sum(1 for x, y in zip(xs, ys) if x > 0 and y > 0),
    }

    # a component pinned at one value has no correlation to report, and that is
    # itself worth seeing: a weight that never moves is a weight doing no work
    if row["left_stdev"] == 0 or row["right_stdev"] == 0:
        row["r"] = None
        row["verdict"] = "flat"
        return row

    row["r"] = statistics.correlation(xs, ys)
    magnitude = abs(row["r"])
    if magnitude >= REDUNDANT_ABOVE:
        row["verdict"] = "redundant"
    elif magnitude <= INDEPENDENT_BELOW:
        row["verdict"] = "independent"
    else:
        row["verdict"] = "entangled"
    return row


VERDICT_STYLE = {
    "independent": "green",
    "entangled": "yellow",
    "redundant": "red",
    "flat": "dim",
}


def print_report(rows: list[dict], label: str) -> None:
    table = Table(title=f"Reward component independence ({label})")
    table.add_column("Pair")
    table.add_column("n", justify="right")
    table.add_column("both > 0", justify="right")
    table.add_column("means", justify="right")
    table.add_column("r", justify="right")
    table.add_column("verdict")

    for row in rows:
        r_text = "n/a" if row["r"] is None else f"{row['r']:+.3f}"
        table.add_row(
            f"{row['left']} vs {row['right']}",
            str(row["n"]),
            str(row["both_active"]),
            f"{row['left_mean']:.3f} / {row['right_mean']:.3f}",
            r_text,
            f"[{VERDICT_STYLE[row['verdict']]}]{row['verdict']}[/]",
        )

    console.print(table)
    console.print(
        f"[dim]|r| >= {REDUNDANT_ABOVE} is redundant, "
        f"<= {INDEPENDENT_BELOW} is independent, 'flat' means one side never varied.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Eval run JSON from scripts/evaluate.py")
    parser.add_argument(
        "--label",
        default="grpo",
        choices=["grpo", "base"],
        help="Which model's completions to read (default: grpo)",
    )
    parser.add_argument("--output", type=Path, help="Write the rows to JSON")
    args = parser.parse_args()

    breakdowns = load_breakdowns(args.run, args.label)
    rows = [
        row
        for left, right in DEFAULT_PAIRS
        if (row := correlate(breakdowns, left, right)) is not None
    ]
    print_report(rows, args.label)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"run": str(args.run), "label": args.label, "pairs": rows}, indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote {args.output}[/]")


if __name__ == "__main__":
    main()
