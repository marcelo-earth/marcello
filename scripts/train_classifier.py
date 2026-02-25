"""Train the style classifier on the contrastive dataset."""

from __future__ import annotations

import argparse

import yaml
from datasets import load_from_disk
from rich.console import Console

from marcello.classifier.train import ClassifierTrainingConfig, train_classifier
from marcello.classifier.evaluate import evaluate_classifier, print_evaluation_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train the style classifier")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print("\n[bold]MarceLLo Style Classifier Training[/]\n")

    train_dataset = load_from_disk(config["data"]["train_path"])
    val_dataset = load_from_disk(config["data"]["val_path"])

    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Val samples:   {len(val_dataset)}\n")

    training_config = ClassifierTrainingConfig(
        model_name=config["model"]["name"],
        dropout=config["model"].get("dropout", 0.1),
        freeze_encoder_layers=config["model"].get("freeze_encoder_layers", 0),
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        weight_decay=config["training"].get("weight_decay", 0.01),
        max_length=config["training"].get("max_length", 512),
        output_dir=config["output"]["dir"],
        use_wandb=config["training"].get("use_wandb", False),
    )

    model = train_classifier(train_dataset, val_dataset, training_config)

    console.print("\n[bold]Final evaluation on validation set:[/]")
    metrics = evaluate_classifier(model, val_dataset)
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()
