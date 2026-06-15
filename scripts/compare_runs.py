"""Compare two saved evaluation runs and print a diff table.

Usage:
    python scripts/compare_runs.py outputs/eval/run_a.json outputs/eval/run_b.json
    python scripts/compare_runs.py run_a.json run_b.json --top-n 10
    python scripts/compare_runs.py run_a.json run_b.json --output diff.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# metrics where higher is better (used to pick delta color)
HIGHER_IS_BETTER = {
    "style_score_mean",
    "style_score_min",
    "style_score_max",
    "distinct_1",
    "distinct_2",
    "avg_words",
}


def load_run(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # support both flat and nested (results key) formats
    if "results" in data:
        data["_meta"] = {k: v for k, v in data.items() if k != "results"}
        data.update(data.pop("results"))
    return data


def _delta_style(metric: str, delta: float) -> str:
    """Return rich color markup based on whether the delta is good or bad."""
    if abs(delta) < 1e-6:
        return "dim"
    good = (metric in HIGHER_IS_BETTER and delta > 0) or (
        metric not in HIGHER_IS_BETTER and delta < 0
    )
    return "green" if good else "red"


def compare_aggregate_metrics(run_a: dict, run_b: dict, label_a: str, label_b: str) -> list[dict]:
    """Return per-metric comparison rows."""
    base_a = run_a.get("base_metrics", {})
    grpo_a = run_a.get("grpo_metrics", {})
    base_b = run_b.get("base_metrics", {})
    grpo_b = run_b.get("grpo_metrics", {})

    rows = []
    all_keys = sorted(set(grpo_a) | set(grpo_b))
    for key in all_keys:
        val_a = grpo_a.get(key)
        val_b = grpo_b.get(key)
        if val_a is None or val_b is None:
            continue
        delta = val_b - val_a
        rows.append(
            {
                "metric": key,
                "base_a": base_a.get(key),
                "grpo_a": val_a,
                "base_b": base_b.get(key),
                "grpo_b": val_b,
                "delta": delta,
            }
        )
    return rows


def compare_per_prompt(run_a: dict, run_b: dict) -> list[dict]:
    """Return per-prompt style score deltas, sorted by change (worst regressions first)."""
    per_a = {item["prompt"]: item for item in run_a.get("per_prompt", [])}
    per_b = {item["prompt"]: item for item in run_b.get("per_prompt", [])}
    shared = sorted(set(per_a) & set(per_b))

    rows = []
    for prompt in shared:
        a = per_a[prompt]
        b = per_b[prompt]
        score_a = a.get("grpo_style_score")
        score_b = b.get("grpo_style_score")
        if score_a is None or score_b is None:
            continue
        rows.append(
            {
                "prompt": prompt,
                "score_a": score_a,
                "score_b": score_b,
                "delta": score_b - score_a,
            }
        )

    rows.sort(key=lambda r: r["delta"])
    return rows


def print_run_header(run: dict, label: str) -> None:
    meta = run.get("_meta", {})
    run_id = meta.get("run_id", "unknown")
    git_hash = meta.get("git_hash") or "unknown"
    model = meta.get("model", run.get("model", "unknown"))
    prompt_count = meta.get("prompt_count", len(run.get("prompts", [])))
    console.print(
        f"  [bold]{label}[/]: run_id=[cyan]{run_id}[/]  git=[dim]{git_hash}[/]"
        f"  model=[dim]{model}[/]  prompts=[dim]{prompt_count}[/]"
    )


def print_aggregate_table(rows: list[dict], label_a: str, label_b: str) -> None:
    table = Table(title="Aggregate Metrics", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column(f"GRPO ({label_a})", style="yellow", justify="right")
    table.add_column(f"GRPO ({label_b})", style="yellow", justify="right")
    table.add_column("Delta", justify="right")

    for row in rows:
        delta = row["delta"]
        color = _delta_style(row["metric"], delta)
        sign = "+" if delta > 0 else ""
        table.add_row(
            row["metric"],
            f"{row['grpo_a']:.4f}",
            f"{row['grpo_b']:.4f}",
            f"[{color}]{sign}{delta:.4f}[/]",
        )

    console.print(table)


def print_per_prompt_table(rows: list[dict], label_a: str, label_b: str, top_n: int) -> None:
    if not rows:
        return

    table = Table(title=f"Per-Prompt Style Score (worst {top_n} regressions first)")
    table.add_column("Prompt", no_wrap=False, max_width=60)
    table.add_column(label_a, justify="right")
    table.add_column(label_b, justify="right")
    table.add_column("Delta", justify="right")

    for row in rows[:top_n]:
        delta = row["delta"]
        color = "green" if delta >= 0 else "red"
        sign = "+" if delta > 0 else ""
        short_prompt = row["prompt"][:80] + "..." if len(row["prompt"]) > 80 else row["prompt"]
        table.add_row(
            short_prompt,
            f"{row['score_a']:.4f}",
            f"{row['score_b']:.4f}",
            f"[{color}]{sign}{delta:.4f}[/]",
        )

    console.print(table)


def _print_completions(
    run_a: dict,
    run_b: dict,
    prompt_rows: list[dict],
    label_a: str,
    label_b: str,
    top_n: int,
) -> None:
    """Print side-by-side completions for the top-N most regressed prompts."""
    per_a = {item["prompt"]: item for item in run_a.get("per_prompt", [])}
    per_b = {item["prompt"]: item for item in run_b.get("per_prompt", [])}

    # worst regressions are at the front of prompt_rows
    regressions = [r for r in prompt_rows if r["delta"] < 0][:top_n]
    if not regressions:
        return

    console.print(f"\n[bold]Completions for {len(regressions)} most regressed prompt(s)[/]\n")
    for row in regressions:
        prompt = row["prompt"]
        short = prompt[:80] + "..." if len(prompt) > 80 else prompt
        console.print(f"[dim]Prompt:[/] {short}")
        console.print(
            f"  [yellow]{label_a}[/] score={row['score_a']:.4f}  "
            f"[yellow]{label_b}[/] score={row['score_b']:.4f}  "
            f"delta=[red]{row['delta']:+.4f}[/]\n"
        )
        a_item = per_a.get(prompt, {})
        b_item = per_b.get(prompt, {})
        if a_item.get("grpo_completion"):
            console.print(Panel(a_item["grpo_completion"], title=label_a, border_style="yellow"))
        if b_item.get("grpo_completion"):
            console.print(Panel(b_item["grpo_completion"], title=label_b, border_style="red"))
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MarceLLo evaluation runs")
    parser.add_argument("run_a", help="Path to first eval JSON")
    parser.add_argument("run_b", help="Path to second eval JSON")
    parser.add_argument("--label-a", default="A", help="Label for run A")
    parser.add_argument("--label-b", default="B", help="Label for run B")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of per-prompt rows to show (sorted by delta, worst first)",
    )
    parser.add_argument("--output", default=None, help="Save diff summary to a JSON file")
    parser.add_argument(
        "--show-completions",
        action="store_true",
        help="Print the actual completions for the top-N most regressed prompts",
    )
    parser.add_argument(
        "--only-regressions",
        action="store_true",
        help="Only show prompts where run B scored lower than run A",
    )
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)

    console.print("\n[bold]MarceLLo Run Comparison[/]\n")
    print_run_header(run_a, args.label_a)
    print_run_header(run_b, args.label_b)
    console.print()

    agg_rows = compare_aggregate_metrics(run_a, run_b, args.label_a, args.label_b)
    print_aggregate_table(agg_rows, args.label_a, args.label_b)
    console.print()

    prompt_rows = compare_per_prompt(run_a, run_b)
    if args.only_regressions:
        prompt_rows = [r for r in prompt_rows if r["delta"] < 0]
    if prompt_rows:
        print_per_prompt_table(prompt_rows, args.label_a, args.label_b, args.top_n)
        improved = sum(1 for r in prompt_rows if r["delta"] > 0)
        regressed = sum(1 for r in prompt_rows if r["delta"] < 0)
        console.print(
            f"\n[green]{improved} prompts improved[/]  "
            f"[red]{regressed} regressed[/]  "
            f"{len(prompt_rows) - improved - regressed} unchanged"
        )

        if args.show_completions:
            _print_completions(run_a, run_b, prompt_rows, args.label_a, args.label_b, args.top_n)

    if args.output:
        out = {
            "run_a": args.run_a,
            "run_b": args.run_b,
            "aggregate": agg_rows,
            "per_prompt": prompt_rows,
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        console.print(f"\n[green]Saved diff to {args.output}[/]")


if __name__ == "__main__":
    main()
