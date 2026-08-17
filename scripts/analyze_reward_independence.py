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
# co-activation is reported, not judged: for two components that are always on, such as
# style_score and length_bonus, it is 1.0 by construction and says nothing. It is worth
# reading for the sparse ones, the penalties that fire on some completions and not others


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


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks, ties averaged, so Spearman handles the many zeros here."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def correlate(breakdowns: list[dict], left: str, right: str) -> dict | None:
    """How much two components move together, or None when there is too little data.

    Pearson alone is not enough to settle this question. The way two reward components
    entangle is usually not linear: one can be flat over most of the other's range and
    jump at a threshold, and Pearson reads that as near zero. The verdict is therefore
    driven by whichever of Pearson and Spearman is larger.

    Neither coefficient can see a coupling that is not monotone, and reward components
    do produce those: relevance can rise and then fall again as a completion copies more
    of the seed. A clean report is evidence that the pair is not obviously redundant, not
    proof that the two are independent. The co-activation rate is printed for the same
    reason: for the sparse components it shows how often both fired on one completion,
    which is worth reading even when both coefficients sit at zero.
    """
    xs = [b[left] for b in breakdowns if left in b and right in b]
    ys = [b[right] for b in breakdowns if left in b and right in b]
    if len(xs) < 3:
        return None

    active = [(x > 0, y > 0) for x, y in zip(xs, ys)]
    either_active = sum(1 for a, b in active if a or b)
    row = {
        "left": left,
        "right": right,
        "n": len(xs),
        "left_mean": statistics.fmean(xs),
        "right_mean": statistics.fmean(ys),
        "left_stdev": statistics.pstdev(xs),
        "right_stdev": statistics.pstdev(ys),
        "both_active": sum(1 for a, b in active if a and b),
        # share of the completions where one fired on which both did
        "co_activation": (
            sum(1 for a, b in active if a and b) / either_active if either_active else 0.0
        ),
    }

    # a component pinned at one value has no correlation to report, and that is itself
    # worth seeing: either its weight is zero, or it never fired on this run. Both mean
    # the component contributed no spread for GRPO to rank completions on.
    if row["left_stdev"] == 0 or row["right_stdev"] == 0:
        row["pearson"] = None
        row["spearman"] = None
        row["verdict"] = "flat"
        return row

    row["pearson"] = statistics.correlation(xs, ys)
    x_ranks, y_ranks = _ranks(xs), _ranks(ys)
    row["spearman"] = (
        statistics.correlation(x_ranks, y_ranks)
        if statistics.pstdev(x_ranks) and statistics.pstdev(y_ranks)
        else None
    )

    magnitude = max(abs(row["pearson"]), abs(row["spearman"] or 0.0))
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
    table.add_column("co-act", justify="right")
    table.add_column("means", justify="right")
    table.add_column("pearson", justify="right")
    table.add_column("spearman", justify="right")
    table.add_column("verdict")

    for row in rows:
        table.add_row(
            f"{row['left']} vs {row['right']}",
            str(row["n"]),
            str(row["both_active"]),
            f"{row['co_activation']:.2f}",
            f"{row['left_mean']:.3f} / {row['right_mean']:.3f}",
            "n/a" if row["pearson"] is None else f"{row['pearson']:+.3f}",
            "n/a" if row["spearman"] is None else f"{row['spearman']:+.3f}",
            f"[{VERDICT_STYLE[row['verdict']]}]{row['verdict']}[/]",
        )

    console.print(table)
    console.print(
        f"[dim]Verdict takes the larger of |pearson| and |spearman|: "
        f">= {REDUNDANT_ABOVE} is redundant, <= {INDEPENDENT_BELOW} is independent. "
        f"'flat' means one side never varied: zero weight, or it never fired. "
        f"Co-activation is informative only for components that are not always on.[/]"
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any pair comes back redundant, so this can gate a run",
    )
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

    redundant = [
        f"{row['left']} vs {row['right']}" for row in rows if row["verdict"] == "redundant"
    ]
    if redundant and args.strict:
        raise SystemExit(
            "Redundant reward components, one of each pair is not earning its "
            f"weight: {', '.join(redundant)}"
        )


if __name__ == "__main__":
    main()
