"""Sanity probe for the style classifier.

Scores out-of-distribution texts (encyclopedic prose, public-domain poetry,
generic LLM output, English tech blogs, languages Marcelo does not write in)
with the trained classifier. None of them should score high.

If any OOD text scores above the fail threshold, the classifier has learned a
spurious signal and any GRPO run using it as reward is meaningless.

Usage:
    python scripts/sanity_probe.py
    python scripts/sanity_probe.py --classifier outputs/classifier/best --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from datasets import load_from_disk
from rich.console import Console
from rich.table import Table

from marcello.classifier.model import StyleClassifier
from marcello.classifier.train import _get_device

console = Console()


def load_probe_texts(path: str) -> list[dict]:
    """Load the probe set from a JSONL file."""
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_positive_controls(val_path: str, limit: int) -> list[dict]:
    """Held-out real samples, used as the positive reference point."""
    dataset = load_from_disk(val_path)
    positives = [row for row in dataset if row["label"] == 1]
    return [
        {
            "id": f"val_positive_{i}",
            "category": "held_out_marcelo",
            "lang": "-",
            "text": row["text"],
        }
        for i, row in enumerate(positives[:limit])
    ]


def main():
    parser = argparse.ArgumentParser(description="Sanity-probe the style classifier")
    parser.add_argument("--classifier", type=str, default="outputs/classifier/best")
    parser.add_argument("--probe-texts", type=str, default="data/probe_texts.jsonl")
    parser.add_argument("--val-path", type=str, default="data/processed/val")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="An OOD text scoring above this fails the probe (default: 0.4)",
    )
    parser.add_argument("--max-controls", type=int, default=6)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to sanity_probe.json beside the classifier being probed",
    )
    args = parser.parse_args()

    # One fixed path meant probing the judge silently overwrote the reward
    # model's report, leaving no record of which model the gate was opened on.
    if args.output is None:
        args.output = str(Path(args.classifier).parent / "sanity_probe.json")

    console.print("\n[bold]MarceLLo Classifier Sanity Probe[/]\n")

    probes = load_probe_texts(args.probe_texts)
    controls = load_positive_controls(args.val_path, args.max_controls)

    model = StyleClassifier.from_pretrained(args.classifier).to(_get_device())
    scores = model.predict([item["text"] for item in probes + controls])

    ood_scores = scores[: len(probes)]
    control_scores = scores[len(probes) :]

    table = Table(title=f"Probe scores — {args.classifier}")
    table.add_column("id")
    table.add_column("category")
    table.add_column("lang")
    table.add_column("P(Marcelo)", justify="right")
    table.add_column("verdict")

    for item, score in zip(probes, ood_scores):
        failed = score > args.threshold
        table.add_row(
            item["id"],
            item["category"],
            item["lang"],
            f"{score:.4f}",
            "[red]FAIL[/]" if failed else "[green]ok[/]",
        )

    for item, score in zip(controls, control_scores):
        low = score < 0.5
        table.add_row(
            item["id"],
            item["category"],
            item["lang"],
            f"{score:.4f}",
            "[yellow]low[/]" if low else "[green]ok[/]",
        )

    console.print(table)

    failures = [
        {"id": item["id"], "category": item["category"], "score": score}
        for item, score in zip(probes, ood_scores)
        if score > args.threshold
    ]
    max_ood = max(ood_scores)
    mean_ood = statistics.mean(ood_scores)
    mean_control = statistics.mean(control_scores) if control_scores else float("nan")
    margin = mean_control - mean_ood

    console.print(f"\nMean OOD score:      {mean_ood:.4f}")
    console.print(f"Max OOD score:       {max_ood:.4f}")
    console.print(f"Mean control score:  {mean_control:.4f}")
    console.print(f"Separation margin:   {margin:.4f}")

    if failures:
        console.print(
            f"\n[bold red]PROBE FAILED[/] — {len(failures)} out-of-distribution "
            f"text(s) scored above {args.threshold}. The classifier has a spurious "
            "signal; rebuild the corpus before running GRPO."
        )
    else:
        console.print(f"\n[bold green]PROBE PASSED[/] — no OOD text scored above {args.threshold}.")

    report = {
        "classifier": args.classifier,
        "threshold": args.threshold,
        "passed": not failures,
        "mean_ood_score": mean_ood,
        "max_ood_score": max_ood,
        "mean_control_score": mean_control,
        "separation_margin": margin,
        "failures": failures,
        "scores": [
            {"id": item["id"], "category": item["category"], "lang": item["lang"], "score": score}
            for item, score in zip(probes + controls, scores)
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nReport written to {output_path}\n")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
