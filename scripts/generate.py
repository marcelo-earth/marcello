"""Generate text with the base or GRPO-adapted MarceLLo model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from marcello.eval.compare import generate_completions
from marcello.grpo.prompting import ensure_control_prompt

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Generate text with MarceLLo")
    parser.add_argument("--model", type=str, default="outputs/grpo/final")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompt", type=str, help="Single prompt to complete")
    parser.add_argument("--prompts-file", type=str, help="Path to prompt file (one per line)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--style", type=str, default="standard")
    parser.add_argument("--language", type=str, choices=["es", "en"], default=None)
    parser.add_argument(
        "--format-prompts",
        action="store_true",
        help="Wrap raw prompts with the GRPO control-token template before generation",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Generate with the base model instead of loading the LoRA adapter",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON output")
    args = parser.parse_args()

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt.strip())
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if not prompts:
        raise SystemExit("Provide --prompt or --prompts-file")

    if args.format_prompts:
        prompts = [
            ensure_control_prompt(prompt, style=args.style, language=args.language)
            for prompt in prompts
        ]

    generations = generate_completions(
        args.base_model if args.base_only else args.model,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        is_lora=not args.base_only,
        base_model=args.base_model,
    )

    for prompt, generation in zip(prompts, generations):
        console.print(Panel(prompt, title="Prompt", border_style="cyan"))
        console.print(Panel(generation, title="Completion", border_style="green"))
        console.print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model": args.base_model if args.base_only else args.model,
                    "base_model": args.base_model,
                    "format_prompts": args.format_prompts,
                    "style": args.style,
                    "language": args.language,
                    "generations": [
                        {"prompt": prompt, "completion": generation}
                        for prompt, generation in zip(prompts, generations)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]Saved generations to {output_path}[/]")


if __name__ == "__main__":
    main()
