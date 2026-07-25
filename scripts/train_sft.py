"""Train the SFT baseline.

Usage:
    python scripts/train_sft.py --config configs/sft.yaml
"""

from __future__ import annotations

import argparse

import yaml
from datasets import concatenate_datasets, load_from_disk
from rich.console import Console

from marcello.sft.train import SFTConfig, build_sft_dataset, train_sft

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train the SFT baseline")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also train on the validation split (only for a final run)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print("\n[bold]MarceLLo SFT Baseline[/]\n")

    dataset = load_from_disk(config["data"]["train_path"])
    if args.include_val:
        dataset = concatenate_datasets([dataset, load_from_disk(config["data"]["val_path"])])

    sft_dataset = build_sft_dataset(dataset)
    console.print(f"Training pairs: {len(sft_dataset)}\n")
    if len(sft_dataset) == 0:
        raise SystemExit("No usable prompt/completion pairs — check the positive samples.")

    sft_config = SFTConfig(
        model_name=config["model"]["name"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        learning_rate=float(config["training"]["learning_rate"]),
        batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        weight_decay=config["training"].get("weight_decay", 0.01),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        max_length=config["training"].get("max_length", 512),
        output_dir=config["output"]["dir"],
        use_wandb=config["training"].get("use_wandb", False),
    )

    train_sft(sft_dataset, sft_config)
    console.print(f"\nAdapter saved to {sft_config.output_dir}/final\n")


if __name__ == "__main__":
    main()
