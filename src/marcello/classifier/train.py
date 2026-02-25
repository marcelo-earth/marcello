"""Training loop for the style classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from marcello.classifier.model import StyleClassifier


@dataclass
class ClassifierTrainingConfig:
    model_name: str = "microsoft/deberta-v3-small"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512
    dropout: float = 0.1
    freeze_encoder_layers: int = 0
    output_dir: str = "outputs/classifier"
    use_wandb: bool = False


def collate_fn(batch, tokenizer, max_length: int = 512):
    """Tokenize and collate a batch of text samples."""
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {**encoded, "labels": labels}


def train_classifier(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: ClassifierTrainingConfig,
) -> StyleClassifier:
    """Train the style classifier and return the best model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleClassifier(
        model_name=config.model_name,
        dropout=config.dropout,
        freeze_encoder_layers=config.freeze_encoder_layers,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model.tokenizer, config.max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=lambda b: collate_fn(b, model.tokenizer, config.max_length),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    if config.use_wandb:
        import wandb

        wandb.init(project="marcello-classifier", config=vars(config))

    best_val_loss = float("inf")
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        for epoch in range(config.epochs):
            # --- Train ---
            model.train()
            train_loss = 0.0
            task = progress.add_task(f"Epoch {epoch + 1}/{config.epochs}", total=len(train_loader))

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                loss = output["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                progress.advance(task)

            avg_train_loss = train_loss / len(train_loader)

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = model(**batch)
                    val_loss += output["loss"].item()
                    preds = (output["probs"] > 0.5).long()
                    correct += (preds == batch["labels"].long()).sum().item()
                    total += len(batch["labels"])

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total

            progress.console.print(
                f"  Epoch {epoch + 1} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_acc={accuracy:.4f}"
            )

            if config.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_accuracy": accuracy,
                    }
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(str(output_path / "best"))
                progress.console.print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    model.save_pretrained(str(output_path / "final"))
    return model
