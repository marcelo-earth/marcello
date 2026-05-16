"""Push trained MarceLLo artifacts to the Hugging Face Hub.

Usage:
    # Push everything
    python scripts/push_to_hub.py --all

    # Push individual artifacts
    python scripts/push_to_hub.py --classifier
    python scripts/push_to_hub.py --model
    python scripts/push_to_hub.py --dataset

    # Dry run (show what would be pushed, no network calls)
    python scripts/push_to_hub.py --all --dry-run
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, whoami
from rich.console import Console
from rich.panel import Panel

console = Console()

# Default Hub repo IDs — override with --org if you forked the project
DEFAULT_ORG = "marcelo-earth"
CLASSIFIER_REPO = "marcello-style-classifier"
MODEL_REPO = "marcello-qwen2.5-1.5b-grpo"
DATASET_REPO = "marcello-writing-samples"


def check_auth(token: str | None) -> str | None:
    """Verify HF credentials and return the resolved token.

    Returns None in dry-run mode (no auth needed).
    Exits with an error message if credentials are missing or invalid.
    """
    try:
        info = whoami(token=token)
        console.print(f"[green]Logged in as:[/] {info['name']}")
        return token
    except Exception:
        console.print(
            "[red]Not logged in to Hugging Face.[/]\n"
            "Run [bold]huggingface-cli login[/] or pass [bold]--token <HF_TOKEN>[/].",
            highlight=False,
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push MarceLLo artifacts to the Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Push classifier, model, and dataset")
    parser.add_argument("--classifier", action="store_true", help="Push the style classifier")
    parser.add_argument("--model", action="store_true", help="Push the GRPO fine-tuned LLM adapter")
    parser.add_argument("--dataset", action="store_true", help="Push the writing samples dataset")
    parser.add_argument(
        "--classifier-path",
        type=str,
        default="outputs/classifier/best",
        help="Local path to the trained classifier (default: outputs/classifier/best)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/grpo/final",
        help="Local path to the GRPO adapter (default: outputs/grpo/final)",
    )
    parser.add_argument(
        "--samples-path",
        type=str,
        default="data/raw/writing_samples",
        help="Local path to positive writing samples (default: data/raw/writing_samples)",
    )
    parser.add_argument(
        "--samples-blog-path",
        type=str,
        default="data/raw/writing_samples_blog",
        help="Local path to blog writing samples (default: data/raw/writing_samples_blog)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=DEFAULT_ORG,
        help=f"Hugging Face org/user namespace (default: {DEFAULT_ORG})",
    )
    parser.add_argument(
        "--merge-weights",
        action="store_true",
        help="Merge LoRA into base model before pushing (larger upload, produces standalone model)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (falls back to huggingface-cli login cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pushed without making any network calls",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    push_classifier = args.all or args.classifier
    push_model = args.all or args.model
    push_dataset = args.all or args.dataset

    if not any([push_classifier, push_model, push_dataset]):
        parser.print_help()
        sys.exit(0)

    console.print(Panel("[bold]MarceLLo — Push to Hub[/]", border_style="cyan"))

    if args.dry_run:
        console.print("[yellow]DRY RUN — no files will be uploaded[/]\n")
        token = None
    else:
        token = check_auth(args.token)

    api = HfApi(token=token)

    if push_classifier:
        push_classifier_artifact(api, args, token)
    if push_model:
        push_model_artifact(api, args, token)
    if push_dataset:
        push_dataset_artifact(api, args, token)

    console.print("\n[bold green]Done![/]")


def push_classifier_artifact(api: HfApi, args: argparse.Namespace, token: str | None):
    pass  # implemented in next commit


def push_model_artifact(api: HfApi, args: argparse.Namespace, token: str | None):
    pass  # implemented in next commit


def push_dataset_artifact(api: HfApi, args: argparse.Namespace, token: str | None):
    pass  # implemented in next commit


if __name__ == "__main__":
    main()
