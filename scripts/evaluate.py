"""Evaluate the GRPO-trained model against the base model."""

from __future__ import annotations

import argparse

from rich.console import Console

from marcello.classifier.model import StyleClassifier
from marcello.eval.compare import compare_models

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MarceLLo model")
    parser.add_argument("--model", type=str, default="outputs/grpo/final")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--classifier", type=str, default="outputs/classifier/best")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (one per line)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
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
    )

    style_improvement = (
        results["grpo_metrics"]["style_score_mean"] - results["base_metrics"]["style_score_mean"]
    )
    console.print(f"\n[bold]Style score improvement: {style_improvement:+.4f}[/]")


if __name__ == "__main__":
    main()
