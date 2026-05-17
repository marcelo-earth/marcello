"""Evaluate the GRPO-trained model against the base model."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from marcello.classifier.model import StyleClassifier
from marcello.eval.compare import compare_models
from marcello.eval.metrics import perplexity

console = Console()


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate MarceLLo model")
    parser.add_argument("--model", type=str, default="outputs/grpo/final")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--classifier", type=str, default="outputs/classifier/best")
    parser.add_argument(
        "--prompts", type=str, required=True, help="Path to prompts file (one per line)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--format-prompts",
        action="store_true",
        help="Wrap raw prompts with the GRPO control-token template before generation",
    )
    parser.add_argument("--style", type=str, default="standard")
    parser.add_argument("--language", type=str, choices=["es", "en"], default=None)
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Compute perplexity using the base model (slow — loads model twice)",
    )
    args = parser.parse_args()

    console.print("\n[bold]MarceLLo Evaluation[/]\n")

    # load prompts
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]

    console.print(f"Loaded {len(prompts)} evaluation prompts")

    # load classifier
    classifier = StyleClassifier.from_pretrained(args.classifier)
    classifier.eval()

    results = compare_models(
        base_model=args.base_model,
        grpo_model_path=args.model,
        prompts=prompts,
        classifier=classifier,
        max_new_tokens=args.max_new_tokens,
        format_prompts=args.format_prompts,
        prompt_style=args.style,
        prompt_language=args.language,
    )

    if args.perplexity:
        console.print("\n[bold]Computing perplexity (base model)...[/]")
        base_ppl = perplexity(results["base_completions"], model_name=args.base_model)
        grpo_ppl = perplexity(results["grpo_completions"], model_name=args.base_model)
        results["base_metrics"]["perplexity"] = base_ppl
        results["grpo_metrics"]["perplexity"] = grpo_ppl
        console.print(f"  Base perplexity:  {base_ppl:.2f}")
        console.print(f"  GRPO perplexity:  {grpo_ppl:.2f}  ({grpo_ppl - base_ppl:+.2f})")

    style_improvement = (
        results["grpo_metrics"]["style_score_mean"] - results["base_metrics"]["style_score_mean"]
    )
    console.print(f"\n[bold]Style score improvement: {style_improvement:+.4f}[/]")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        output_path.write_text(
            json.dumps(
                {
                    "run_id": now.strftime("%Y%m%d-%H%M%S"),
                    "timestamp": now.isoformat(),
                    "git_hash": _git_hash(),
                    "model": args.model,
                    "base_model": args.base_model,
                    "classifier": args.classifier,
                    "format_prompts": args.format_prompts,
                    "style": args.style,
                    "language": args.language,
                    "prompt_count": len(prompts),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]Saved evaluation results to {output_path}[/]")


if __name__ == "__main__":
    main()
