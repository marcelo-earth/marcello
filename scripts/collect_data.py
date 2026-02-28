"""Collect, process, and prepare contrastive dataset for classifier training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from marcello.data.collector import WritingSampleCollector
from marcello.data.processor import TextProcessor
from marcello.data.negative_sampler import NegativeSampler, NegativeStrategy

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
    console.print("[bold blue]Phase 3:[/] Generating contrastive negatives...")
    console.print(f"  Strategy: {neg_cfg.get('strategy', 'llm_rephrase')}")
    console.print(f"  Negatives per positive: {neg_cfg.get('num_negatives_per_positive', 2)}")

    sampler = NegativeSampler(
        strategy=NegativeStrategy(neg_cfg.get("strategy", "llm_rephrase")),
        num_negatives_per_positive=neg_cfg.get("num_negatives_per_positive", 2),
        model_name=neg_cfg.get("model", "Qwen/Qwen2.5-1.5B"),
        seed=neg_cfg.get("seed", 42),
    )

    contrastive_dataset = sampler.build_contrastive_dataset(positive_dataset)
    console.print(f"  Contrastive dataset: [green]{len(contrastive_dataset)}[/] total samples\n")

    # --- Split and Save ---
    out_cfg = config.get("output", {})
    output_path = Path(out_cfg.get("path", "data/processed/"))
    output_path.mkdir(parents=True, exist_ok=True)

    split = contrastive_dataset.train_test_split(
        test_size=out_cfg.get("val_split", 0.15),
        seed=out_cfg.get("seed", 42),
        stratify_by_column="label",
    )

    split["train"].save_to_disk(output_path / "train")
    split["test"].save_to_disk(output_path / "val")

    # --- Summary ---
    table = Table(title="Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Positive (Marcelo)", style="green")
    table.add_column("Negative (Generic)", style="red")

    for name, ds in [("train", split["train"]), ("val", split["test"])]:
        pos = sum(1 for l in ds["label"] if l == 1)
        neg = sum(1 for l in ds["label"] if l == 0)
        table.add_row(name, str(len(ds)), str(pos), str(neg))

    console.print(table)
    console.print(f"\nSaved to [bold]{output_path}[/]")


if __name__ == "__main__":
    main()
