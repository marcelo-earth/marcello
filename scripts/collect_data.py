"""Collect, process, and prepare contrastive dataset for classifier training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from marcello.data.balance import length_matched_indices, surface_cell_counts, undersample_indices
from marcello.data.collector import WritingSampleCollector
from marcello.data.manifest import corpus_hash
from marcello.data.negative_sampler import NegativeSampler, NegativeStrategy
from marcello.data.processor import TextProcessor

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Collect and process writing samples")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Collect ---
    console.print("\n[bold blue]Phase 1:[/] Collecting writing samples...")
    proc_cfg = config.get("processing", {})
    collector = WritingSampleCollector(
        min_length=proc_cfg.get("min_length", 50),
        max_length=proc_cfg.get("max_length", 2048),
    )

    for source in config["sources"]:
        path = Path(source["path"])
        style = source.get("style", "standard")
        collector.set_style(style)

        if not path.exists():
            console.print(f"  [yellow]Skipping {path} (not found)[/]")
            continue

        if source["type"] == "text_files":
            n = collector.collect_from_directory(path)
        elif source["type"] == "jsonl":
            n = collector.collect_from_jsonl(path, text_field=source.get("text_field", "text"))
        else:
            console.print(f"  [red]Unknown source type: {source['type']}[/]")
            continue

        console.print(f"  Collected [green]{n}[/] samples from {path} [dim]({style})[/]")

    console.print(f"  Total: [bold green]{len(collector)}[/] samples\n")

    if len(collector) == 0:
        console.print("[red]No samples collected. Add writing samples to data/raw/[/]")
        return

    # --- Process ---
    console.print("[bold blue]Phase 2:[/] Processing and cleaning...")
    processor = TextProcessor(
        remove_urls=proc_cfg.get("remove_urls", True),
        remove_emails=proc_cfg.get("remove_emails", True),
        normalize_whitespace=proc_cfg.get("normalize_whitespace", True),
        max_length=proc_cfg.get("max_length", 2048),
        deduplicate=proc_cfg.get("deduplicate", True),
    )

    positive_dataset = collector.to_dataset()
    positive_dataset = processor.process_dataset(positive_dataset)
    console.print(f"  After cleaning: [green]{len(positive_dataset)}[/] samples\n")

    # --- Negative Sampling ---
    neg_cfg = config.get("negative_sampling", {})
    strategy = neg_cfg.get("strategy", "prewritten")
    console.print("[bold blue]Phase 3:[/] Generating contrastive negatives...")
    console.print(f"  Strategy: {strategy}")

    sampler = NegativeSampler(
        strategy=NegativeStrategy(strategy),
        num_negatives_per_positive=neg_cfg.get("num_negatives_per_positive", 2),
        model_name=neg_cfg.get("model", "Qwen/Qwen2.5-1.5B"),
        prewritten_path=neg_cfg.get("path", "data/raw/negative_samples"),
        seed=neg_cfg.get("seed", 42),
    )

    contrastive_dataset = sampler.build_contrastive_dataset(positive_dataset)
    for source, count in sorted(sampler.source_counts.items()):
        console.print(f"    negatives from {source}: [green]{count}[/]")
    console.print(f"  Contrastive dataset: [green]{len(contrastive_dataset)}[/] total samples\n")

    # --- Balance classes ---
    out_cfg = config.get("output", {})
    surface_cells: dict[str, int] = {}
    kept_sources: dict[str, int] = {}
    if out_cfg.get("balance_classes", True):
        texts = list(contrastive_dataset["text"])
        labels = list(contrastive_dataset["label"])
        seed = out_cfg.get("seed", 42)

        if out_cfg.get("length_match", True):
            num_bins = out_cfg.get("length_bins", 8)
            match_form = out_cfg.get("match_form", True)
            groups = None
            if out_cfg.get("spread_sources", True) and "source" in contrastive_dataset.column_names:
                groups = list(contrastive_dataset["source"])
            balanced_indices = length_matched_indices(
                texts, labels, seed=seed, num_bins=num_bins, match_form=match_form, groups=groups
            )
            surface_cells = surface_cell_counts(
                texts, labels, num_bins=num_bins, match_form=match_form
            )
            how = "balanced per surface stratum"
        else:
            balanced_indices = undersample_indices(labels, seed=seed)
            how = "balanced"

        # Languages Marcelo never writes are a true signal, not a shortcut: no
        # positive exists there to balance against, so those negatives would
        # otherwise compete for the few Spanish-prose slots and mostly drop out.
        always_keep = set(out_cfg.get("always_keep_sources", []))
        if always_keep and "source" in contrastive_dataset.column_names:
            kept_set = set(balanced_indices)
            extra = [
                i
                for i, source in enumerate(contrastive_dataset["source"])
                if source in always_keep and i not in kept_set
            ]
            balanced_indices = sorted(balanced_indices + extra)
            console.print(f"  Kept {len(extra)} extra negatives from {sorted(always_keep)}")

        contrastive_dataset = contrastive_dataset.select(balanced_indices)
        kept_sources: dict[str, int] = {}
        if "source" in contrastive_dataset.column_names:
            for source, label in zip(contrastive_dataset["source"], contrastive_dataset["label"]):
                if label == 0:
                    kept_sources[source] = kept_sources.get(source, 0) + 1
            console.print(f"  Negatives kept per source: {dict(sorted(kept_sources.items()))}")
        kept_pos = sum(1 for label in contrastive_dataset["label"] if label == 1)
        kept_neg = len(contrastive_dataset) - kept_pos
        console.print(
            f"  {how.capitalize()} to [green]{len(contrastive_dataset)}[/] samples "
            f"({kept_pos} pos + {kept_neg} neg)\n"
        )

    # --- Split and Save ---
    output_path = Path(out_cfg.get("path", "data/processed/"))
    output_path.mkdir(parents=True, exist_ok=True)

    # the processed format stays text + label; sources go to the manifest
    if "source" in contrastive_dataset.column_names:
        contrastive_dataset = contrastive_dataset.remove_columns("source")

    # cast label to ClassLabel so train_test_split can stratify
    from datasets import ClassLabel

    contrastive_dataset = contrastive_dataset.cast_column(
        "label", ClassLabel(names=["generic", "marcelo"])
    )

    split = contrastive_dataset.train_test_split(
        test_size=out_cfg.get("val_split", 0.15),
        seed=out_cfg.get("seed", 42),
        stratify_by_column="label",
    )

    split["train"].save_to_disk(output_path / "train")
    split["test"].save_to_disk(output_path / "val")

    # --- Manifest ---
    # A rebuild that silently drifts (one fewer negative, a shifted balance
    # cell) is otherwise invisible: data/processed/ is gitignored, so nothing
    # here is checked into git except this file.
    manifest = {
        "counts_per_class": {
            name: {
                "positive": sum(1 for lbl in ds["label"] if lbl == 1),
                "negative": sum(1 for lbl in ds["label"] if lbl == 0),
            }
            for name, ds in [("train", split["train"]), ("val", split["test"])]
        },
        "negative_source_counts": dict(sorted(sampler.source_counts.items())),
        "kept_negative_source_counts": kept_sources,
        "surface_cell_counts": surface_cells,
        "text_hash": corpus_hash(list(split["train"]["text"]) + list(split["test"]["text"])),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"Manifest written to [bold]{output_path / 'manifest.json'}[/]\n")

    # --- Summary ---
    table = Table(title="Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Positive (Marcelo)", style="green")
    table.add_column("Negative (Generic)", style="red")

    for name, ds in [("train", split["train"]), ("val", split["test"])]:
        pos = sum(1 for lbl in ds["label"] if lbl == 1)
        neg = sum(1 for lbl in ds["label"] if lbl == 0)
        table.add_row(name, str(len(ds)), str(pos), str(neg))

    console.print(table)
    console.print(f"\nSaved to [bold]{output_path}[/]")


if __name__ == "__main__":
    main()
