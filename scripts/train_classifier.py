"""Train the style classifier on the contrastive dataset."""

from __future__ import annotations

import argparse
import tempfile

import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk
from rich.console import Console
from rich.table import Table

from marcello.classifier.evaluate import evaluate_classifier, print_evaluation_report
from marcello.classifier.train import ClassifierTrainingConfig, train_classifier

console = Console()


def _run_cross_validation(
    full_dataset: Dataset,
    training_config: ClassifierTrainingConfig,
    k: int = 5,
) -> None:
    """Run stratified k-fold cross-validation and report per-fold + aggregate accuracy."""
    from sklearn.model_selection import StratifiedKFold

    labels = full_dataset["label"]
    indices = list(range(len(full_dataset)))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    console.print(f"\n[bold]Running {k}-fold stratified cross-validation ({len(full_dataset)} samples)[/]\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        fold_train = full_dataset.select(train_idx)
        fold_val = full_dataset.select(val_idx)

        pos = sum(fold_val["label"])
        console.print(f"Fold {fold}/{k}: train={len(fold_train)}, val={len(fold_val)} ({pos} pos / {len(fold_val) - pos} neg)")

        with tempfile.TemporaryDirectory() as tmp_dir:
            fold_config = ClassifierTrainingConfig(
                model_name=training_config.model_name,
                dropout=training_config.dropout,
                freeze_encoder_layers=training_config.freeze_encoder_layers,
                learning_rate=training_config.learning_rate,
                batch_size=training_config.batch_size,
                epochs=training_config.epochs,
                warmup_ratio=training_config.warmup_ratio,
                weight_decay=training_config.weight_decay,
                max_length=training_config.max_length,
                output_dir=tmp_dir,
                use_wandb=False,
            )
            model = train_classifier(fold_train, fold_val, fold_config)
            metrics = evaluate_classifier(model, fold_val)

        fold_accuracies.append(metrics["accuracy"])
        console.print(f"  Fold {fold} accuracy: {metrics['accuracy']:.4f}\n")

    mean_acc = sum(fold_accuracies) / k
    variance = sum((a - mean_acc) ** 2 for a in fold_accuracies) / k
    std_acc = variance ** 0.5

    table = Table(title=f"{k}-Fold CV Results", show_lines=False)
    table.add_column("Fold", style="cyan")
    table.add_column("Accuracy", justify="right")
    for i, acc in enumerate(fold_accuracies, start=1):
        table.add_row(str(i), f"{acc:.4f}")
    table.add_row("[bold]Mean[/]", f"[bold]{mean_acc:.4f}[/]")
    table.add_row("[bold]Std[/]", f"[bold]{std_acc:.4f}[/]")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Train the style classifier")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run 5-fold stratified cross-validation instead of a single train run",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print("\n[bold]MarceLLo Style Classifier Training[/]\n")

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

    if args.cross_validate:
        train_dataset = load_from_disk(config["data"]["train_path"])
        val_dataset = load_from_disk(config["data"]["val_path"])
        full_dataset = concatenate_datasets([train_dataset, val_dataset])
        console.print(f"Combined dataset: {len(full_dataset)} samples")
        _run_cross_validation(full_dataset, training_config, k=args.cv_folds)
        return

    train_dataset = load_from_disk(config["data"]["train_path"])
    val_dataset = load_from_disk(config["data"]["val_path"])

    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Val samples:   {len(val_dataset)}\n")

    model = train_classifier(train_dataset, val_dataset, training_config)

    console.print("\n[bold]Final evaluation on validation set:[/]")
    metrics = evaluate_classifier(model, val_dataset)
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()
