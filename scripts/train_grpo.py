"""Run GRPO training to align the base LLM with Marcelo's writing style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_from_disk
from rich.console import Console

from marcello.grpo.prompting import extract_prompts_from_positive_dataset
from marcello.grpo.trainer import MarceLLoGRPOConfig, MarceLLoGRPOTrainer

console = Console()


def check_probe_gate(classifier_path: str, skip: bool) -> None:
    """Refuse to train against an unprobed, failing, or stale classifier.

    Mirrors the default report location `sanity_probe.py` writes to: a
    `sanity_probe.json` beside the classifier directory being probed.
    """
    if skip:
        console.print(
            "[bold yellow]--skip-probe-gate set: training against an unverified classifier.[/]"
        )
        return

    classifier_dir = Path(classifier_path)
    report_path = classifier_dir.parent / "sanity_probe.json"
    model_path = classifier_dir / "model.pt"

    if not report_path.exists():
        raise SystemExit(
            f"[probe gate] No sanity probe report at {report_path}. "
            f"Run `make probe` (or `python scripts/sanity_probe.py --classifier "
            f"{classifier_path}`) before training, or pass --skip-probe-gate."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    if not report.get("passed", False):
        raise SystemExit(
            f"[probe gate] {report_path} reports a failed probe "
            f"({len(report.get('failures', []))} failure(s)). Rebuild the corpus "
            "or classifier and re-run `make probe` before training."
        )

    if model_path.exists() and report_path.stat().st_mtime < model_path.stat().st_mtime:
        raise SystemExit(
            f"[probe gate] {report_path} is older than {model_path} — the probe "
            "was run against a different classifier checkpoint. Re-run `make probe` "
            "before training."
        )

    console.print(f"[green]Probe gate passed[/] ({report_path})")


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument(
        "--skip-probe-gate",
        action="store_true",
        help="Skip the sanity-probe gate (escape hatch; prints a loud warning)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print("\n[bold]MarceLLo GRPO Training[/]\n")

    check_probe_gate(config["reward"]["classifier_path"], skip=args.skip_probe_gate)

    # build prompt dataset
    prompts_path = config.get("prompts", {}).get("path")
    if prompts_path:
        try:
            prompt_dataset = load_from_disk(prompts_path)
            console.print(f"Loaded {len(prompt_dataset)} prompts from {prompts_path}")
        except Exception:
            console.print("[yellow]Prompt dataset not found, extracting from training data[/]")
            train_ds = load_from_disk("data/processed/train")
            prompt_dataset = extract_prompts_from_positive_dataset(train_ds)
    else:
        train_ds = load_from_disk("data/processed/train")
        prompt_dataset = extract_prompts_from_positive_dataset(train_ds)

    console.print(f"Using {len(prompt_dataset)} prompts for GRPO training\n")

    grpo_config = MarceLLoGRPOConfig(
        model_name=config["model"]["name"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        num_generations=config["grpo"]["num_generations"],
        max_new_tokens=config["grpo"]["max_new_tokens"],
        temperature=config["grpo"]["temperature"],
        top_p=config["grpo"].get("top_p", 0.95),
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        kl_coef=config["training"].get("kl_coef", 0.1),
        clip_range=config["training"].get("clip_range", 0.2),
        classifier_path=config["reward"]["classifier_path"],
        reward_temperature=config["reward"]["temperature"],
        reward_style_weight=config["reward"].get("style_weight", 0.65),
        reward_length_bonus_weight=config["reward"].get("length_bonus_weight", 0.1),
        reward_prompt_relevance_weight=config["reward"].get("prompt_relevance_weight", 0.2),
        reward_repetition_penalty_weight=config["reward"].get("repetition_penalty_weight", 0.15),
        reward_prompt_echo_penalty_weight=config["reward"].get("prompt_echo_penalty_weight", 0.1),
        reward_reference_copy_penalty_weight=config["reward"].get(
            "reference_copy_penalty_weight", 0.15
        ),
        reward_target_length=config["reward"].get("target_length", 60),
        reward_reference_texts_path=config["reward"].get(
            "reference_texts_path", "data/processed/train"
        ),
        reward_reference_ngram_size=config["reward"].get("reference_ngram_size", 8),
        reward_echo_ngram_size=config["reward"].get("echo_ngram_size", 4),
        reward_echo_floor=config["reward"].get("echo_floor", 0.35),
        reward_relevance_target_tokens=config["reward"].get("relevance_target_tokens", 8),
        output_dir=config["output"]["dir"],
        use_wandb=config["training"].get("use_wandb", False),
    )

    trainer = MarceLLoGRPOTrainer(grpo_config)
    trainer.train(prompt_dataset)

    console.print("\n[bold green]GRPO training complete![/]")
    console.print(f"Model saved to {config['output']['dir']}/final")


if __name__ == "__main__":
    main()
