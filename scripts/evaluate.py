"""Evaluate the GRPO-trained model against the base model."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
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


def _print_summary(base_metrics: dict, grpo_metrics: dict) -> None:
    from rich.table import Table

    table = Table(title="Evaluation Summary", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Base", justify="right")
    table.add_column("GRPO", justify="right")
    table.add_column("Delta", justify="right", style="bold")

    display = [
        ("style_score_mean", "Style score (mean)"),
        ("style_score_std", "Style score (std)"),
        ("judge_score_mean", "Judge score (mean)"),
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
        ("avg_words", "Avg words"),
        ("perplexity", "Perplexity"),
        ("reward_style_score_mean", "Reward: style"),
        ("reward_length_bonus_mean", "Reward: length bonus"),
        ("reward_prompt_relevance_mean", "Reward: prompt relevance"),
        ("reward_repetition_penalty_mean", "Reward: repetition pen."),
        ("reward_prompt_echo_penalty_mean", "Reward: echo pen."),
        ("reward_reference_copy_penalty_mean", "Reward: ref-copy pen."),
        ("reward_total_mean", "Reward: total"),
    ]
    for key, label in display:
        base_val = base_metrics.get(key)
        grpo_val = grpo_metrics.get(key)
        if base_val is None or grpo_val is None:
            continue
        delta = grpo_val - base_val
        sign = "+" if delta > 0 else ""
        table.add_row(label, f"{base_val:.4f}", f"{grpo_val:.4f}", f"{sign}{delta:.4f}")

    console.print()
    console.print(table)


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
    parser.add_argument(
        "--judge-classifier",
        type=str,
        default=None,
        help="Path to a second independent classifier used only for evaluation (never for training)",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="Path to grpo.yaml; enables per-component reward breakdown in output",
    )
    parser.add_argument(
        "--reference-texts",
        type=str,
        default=None,
        help="Path to the training dataset (HuggingFace disk format) for reference-copy penalty",
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

    if args.judge_classifier:
        console.print("\n[bold]Scoring with independent judge classifier...[/]")
        judge = StyleClassifier.from_pretrained(args.judge_classifier)
        judge.eval()
        base_judge = judge.predict(results["base_completions"])
        grpo_judge = judge.predict(results["grpo_completions"])
        results["base_metrics"]["judge_score_mean"] = sum(base_judge) / len(base_judge)
        results["grpo_metrics"]["judge_score_mean"] = sum(grpo_judge) / len(grpo_judge)
        results["base_metrics"]["judge_score_std"] = (sum((s - results["base_metrics"]["judge_score_mean"]) ** 2 for s in base_judge) / len(base_judge)) ** 0.5
        results["grpo_metrics"]["judge_score_std"] = (sum((s - results["grpo_metrics"]["judge_score_mean"]) ** 2 for s in grpo_judge) / len(grpo_judge)) ** 0.5
        for i, entry in enumerate(results["per_prompt"]):
            entry["base_judge_score"] = base_judge[i]
            entry["grpo_judge_score"] = grpo_judge[i]
        console.print(f"  Base judge score: {results['base_metrics']['judge_score_mean']:.4f}")
        console.print(f"  GRPO judge score: {results['grpo_metrics']['judge_score_mean']:.4f}  ({results['grpo_metrics']['judge_score_mean'] - results['base_metrics']['judge_score_mean']:+.4f})")

    if args.reward_config:
        console.print("\n[bold]Computing per-component reward breakdown...[/]")
        from marcello.grpo.reward import StyleReward
        with open(args.reward_config) as f:
            rcfg = yaml.safe_load(f)
        rw_cfg = rcfg.get("reward", rcfg)
        reward_fn = StyleReward(
            classifier_path=args.classifier,
            temperature=rw_cfg.get("temperature", 1.0),
            style_weight=rw_cfg.get("style_weight", 0.65),
            length_bonus_weight=rw_cfg.get("length_bonus_weight", 0.1),
            prompt_relevance_weight=rw_cfg.get("prompt_relevance_weight", 0.2),
            repetition_penalty_weight=rw_cfg.get("repetition_penalty_weight", 0.15),
            prompt_echo_penalty_weight=rw_cfg.get("prompt_echo_penalty_weight", 0.1),
            reference_copy_penalty_weight=rw_cfg.get("reference_copy_penalty_weight", 0.15),
            target_length=rw_cfg.get("target_length", 180),
            reference_texts_path=args.reference_texts,
            reference_ngram_size=rw_cfg.get("reference_ngram_size", 8),
        )
        prompts_for_reward = results["prompts"]
        for label, completions, metrics_dict in [
            ("base", results["base_completions"], results["base_metrics"]),
            ("grpo", results["grpo_completions"], results["grpo_metrics"]),
        ]:
            breakdowns = reward_fn.score(completions, prompts=prompts_for_reward, return_breakdown=True)
            component_keys = ["total", "raw_style_prob", "style_score", "length_bonus", "prompt_relevance", "repetition_penalty", "prompt_echo_penalty", "reference_copy_penalty"]
            for key in component_keys:
                vals = [b[key] for b in breakdowns]
                metrics_dict[f"reward_{key}_mean"] = sum(vals) / len(vals)
            for i, entry in enumerate(results["per_prompt"]):
                entry[f"{label}_reward_breakdown"] = breakdowns[i]

    _print_summary(results["base_metrics"], results["grpo_metrics"])

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
