"""Run GRPO training to align the base LLM with Marcelo's writing style."""

from __future__ import annotations

import argparse
import re

import yaml
from datasets import Dataset, load_from_disk
from rich.console import Console

from marcello.grpo.trainer import MarceLLoGRPOConfig, MarceLLoGRPOTrainer

console = Console()


def extract_prompts_from_samples(train_path: str, max_prompts: int = 500) -> Dataset:
    """Extract prompts from the training data.

    Strategy: use the first sentence of each positive (Marcelo's) sample
    as a prompt. The model learns to complete it in Marcelo's style.
    """
    dataset = load_from_disk(train_path)

    # only use positive samples (Marcelo's writing)
    positive = dataset.filter(lambda x: x["label"] == 1)
    prompts = []

    for text in positive["text"]:
        # grab the first sentence as a prompt
        match = re.match(r"^(.+?[.!?])\s", text)
        if match:
            prompt = match.group(1).strip()
            if len(prompt) > 10:
                prompts.append(prompt)

    prompts = prompts[:max_prompts]
    return Dataset.from_dict({"prompt": prompts})


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print("\n[bold]MarceLLo GRPO Training[/]\n")

    # build prompt dataset
    prompts_path = config.get("prompts", {}).get("path")
    if prompts_path:
        try:
            prompt_dataset = load_from_disk(prompts_path)
            console.print(f"Loaded {len(prompt_dataset)} prompts from {prompts_path}")
        except Exception:
            console.print("[yellow]Prompt dataset not found, extracting from training data[/]")
            prompt_dataset = extract_prompts_from_samples("data/processed/train")
    else:
        prompt_dataset = extract_prompts_from_samples("data/processed/train")

    console.print(f"Using {len(prompt_dataset)} prompts for GRPO training\n")

    grpo_config = MarceLLoGRPOConfig(
        model_name=config["model"]["name"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        num_generations=config["grpo"]["num_generations"],
        max_new_tokens=config["grpo"]["max_new_tokens"],
        temperature=config["grpo"]["temperature"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        classifier_path=config["reward"]["classifier_path"],
        reward_temperature=config["reward"]["temperature"],
        output_dir=config["output"]["dir"],
        use_wandb=config["training"].get("use_wandb", False),
    )

    trainer = MarceLLoGRPOTrainer(grpo_config)
    trainer.train(prompt_dataset)

    console.print("\n[bold green]GRPO training complete![/]")
    console.print(f"Model saved to {config['output']['dir']}/final")


if __name__ == "__main__":
    main()
