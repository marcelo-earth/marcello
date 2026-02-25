"""Evaluation utilities for the style classifier."""

from __future__ import annotations

import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from rich.console import Console
from rich.table import Table

from marcello.classifier.model import StyleClassifier

console = Console()


def evaluate_classifier(
    model: StyleClassifier,
    dataset: Dataset,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> dict:
    """Run full evaluation on a dataset. Returns metrics dict."""
    device = next(model.parameters()).device
    model.eval()

    all_probs = []
    all_labels = []

    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset["text"][i : i + batch_size]
        batch_labels = dataset["label"][i : i + batch_size]

        encoded = model.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        all_probs.extend(output["probs"].cpu().tolist())
        all_labels.extend(batch_labels)

    preds = [1 if p > threshold else 0 for p in all_probs]

    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary"
    )
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "threshold": threshold,
        "num_samples": len(dataset),
    }

    return metrics


def print_evaluation_report(metrics: dict):
    """Pretty-print evaluation metrics."""
    table = Table(title="Classifier Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1']:.4f}")
    table.add_row("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    table.add_row("Threshold", f"{metrics['threshold']:.2f}")
    table.add_row("Samples", str(metrics["num_samples"]))

    console.print(table)
