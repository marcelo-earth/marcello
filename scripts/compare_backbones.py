"""Compare encoder backbones as feature extractors for the style classifier.

With ~180 samples, fine-tuning a large encoder overfits and, as the sanity
probe showed, an English-only backbone (deberta-v3-small) cannot separate a
mostly-Spanish corpus at all. This script embeds the whole corpus with each
candidate backbone (frozen, mean-pooled) and runs stratified k-fold logistic
regression on top, so the backbone choice is made with numbers.

Usage:
    python scripts/compare_backbones.py
    python scripts/compare_backbones.py --models microsoft/deberta-v3-small intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

console = Console()

DEFAULT_MODELS = [
    "microsoft/deberta-v3-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "FacebookAI/xlm-roberta-base",
]


def embed(texts: list[str], model_name: str, max_length: int, batch_size: int) -> np.ndarray:
    """Mean-pooled frozen embeddings for a list of texts."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    vectors = []
    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vectors.append(pooled.numpy())

    return np.concatenate(vectors, axis=0)


def cross_validate(features: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> dict:
    """Stratified k-fold logistic regression on frozen features."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracies, aucs = [], []

    for train_idx, val_idx in splitter.split(features, labels):
        head = LogisticRegression(max_iter=2000, C=1.0)
        head.fit(features[train_idx], labels[train_idx])
        probs = head.predict_proba(features[val_idx])[:, 1]
        accuracies.append(accuracy_score(labels[val_idx], probs > 0.5))
        aucs.append(roc_auc_score(labels[val_idx], probs))

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare classifier backbones")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--train-path", type=str, default="data/processed/train")
    parser.add_argument("--val-path", type=str, default="data/processed/val")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs/classifier/backbone_comparison.json")
    args = parser.parse_args()

    dataset = concatenate_datasets([load_from_disk(args.train_path), load_from_disk(args.val_path)])
    texts = list(dataset["text"])
    labels = np.array(dataset["label"])
    console.print(f"\n[bold]Backbone comparison[/] — {len(texts)} samples, {args.folds}-fold CV\n")

    results = {}
    for model_name in args.models:
        console.print(f"Embedding with [cyan]{model_name}[/] ...")
        features = embed(texts, model_name, args.max_length, args.batch_size)
        results[model_name] = cross_validate(features, labels, args.folds, args.seed)
        results[model_name]["dim"] = int(features.shape[1])

    table = Table(title="Frozen backbone + logistic regression")
    table.add_column("backbone")
    table.add_column("dim", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("AUC", justify="right")

    for model_name, metrics in sorted(results.items(), key=lambda kv: -kv[1]["auc_mean"]):
        table.add_row(
            model_name,
            str(metrics["dim"]),
            f"{metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}",
            f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}",
        )

    console.print(table)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"\nResults written to {output_path}\n")


if __name__ == "__main__":
    main()
