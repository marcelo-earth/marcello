"""Training loop for the style classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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
    head_norm: bool = True
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


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS skipped: DeBERTa-v3 backward through embeddings produces NaN
    # gradients on MPS (PyTorch MPS backend bug). CPU is correct here.
    return torch.device("cpu")


def _cache_features(model: StyleClassifier, loader: DataLoader, device: torch.device):
    """Run the frozen encoder once and keep the pooled features.

    A frozen encoder produces the same features every epoch, so re-running it
    20 times is the entire cost of training a head that takes seconds.
    """
    model.encoder.eval()
    features, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            features.append(model.mean_pool(outputs.last_hidden_state, batch["attention_mask"]))
            labels.append(batch["labels"])

    return torch.cat(features), torch.cat(labels)


def _iter_batches(source, batch_size: int, device: torch.device, shuffle: bool):
    """Yield (inputs, labels) from either cached features or a tokenizing loader."""
    if isinstance(source, tuple):
        features, labels = source
        order = torch.randperm(len(labels)) if shuffle else torch.arange(len(labels))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            yield features[index], labels[index]
    else:
        for batch in source:
            batch = {k: v.to(device) for k, v in batch.items()}
            yield batch, batch["labels"]


def _forward(model: StyleClassifier, inputs) -> dict[str, torch.Tensor]:
    """Run the head on cached features, or the whole model on tokenized inputs."""
    if isinstance(inputs, dict):
        return model(**inputs)

    logits = model.head(inputs)
    return {"logits": logits, "probs": torch.sigmoid(logits)}


def _batch_count(source, batch_size: int) -> int:
    if isinstance(source, tuple):
        return (len(source[1]) + batch_size - 1) // batch_size
    return len(source)


def train_classifier(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: ClassifierTrainingConfig,
) -> StyleClassifier:
    """Train the style classifier and return the best model."""
    device = _get_device()

    model = StyleClassifier(
        model_name=config.model_name,
        dropout=config.dropout,
        freeze_encoder_layers=config.freeze_encoder_layers,
        head_norm=config.head_norm,
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

    encoder_is_frozen = not any(p.requires_grad for p in model.encoder.parameters())

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
        foreach=False,
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

    loss_fn = nn.BCEWithLogitsLoss()
    train_source, val_source = train_loader, val_loader
    if encoder_is_frozen:
        train_source = _cache_features(model, train_loader, device)
        val_source = _cache_features(model, val_loader, device)
        # Fit on train only. Standardising with statistics that saw the
        # validation set would leak, and the buffers ship inside the checkpoint
        # so inference applies exactly the scaling training used.
        with torch.no_grad():
            model.set_feature_stats(train_source[0])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        for epoch in range(config.epochs):
            # --- Train ---
            model.train()
            if encoder_is_frozen:
                # A fully frozen encoder is a feature extractor: leaving it in
                # train mode only adds dropout noise to features that never
                # get updated, which the head then has to average out.
                model.encoder.eval()
            train_loss = 0.0
            train_steps = 0
            task = progress.add_task(
                f"Epoch {epoch + 1}/{config.epochs}",
                total=_batch_count(train_source, config.batch_size),
            )

            for inputs, labels in _iter_batches(
                train_source, config.batch_size, device, shuffle=True
            ):
                output = _forward(model, inputs)
                loss = loss_fn(output["logits"], labels.float())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                progress.advance(task)

            avg_train_loss = train_loss / train_steps

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            val_steps = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in _iter_batches(
                    val_source, config.batch_size, device, shuffle=False
                ):
                    output = _forward(model, inputs)
                    val_loss += loss_fn(output["logits"], labels.float()).item()
                    val_steps += 1
                    preds = (output["probs"] > 0.5).long()
                    correct += (preds == labels.long()).sum().item()
                    total += len(labels)

            avg_val_loss = val_loss / val_steps
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
