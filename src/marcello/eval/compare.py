"""Side-by-side comparison of base model vs GRPO-trained model."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from marcello.eval.metrics import compute_style_metrics

console = Console()


def generate_completions(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    is_lora: bool = False,
    base_model: str = "Qwen/Qwen2.5-1.5B",
) -> list[str]:
    """Generate completions from a model (base or LoRA-adapted)."""
    if is_lora:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
    )

    completions = []
    for prompt in prompts:
        result = pipe(prompt, return_full_text=False)
        completions.append(result[0]["generated_text"].strip())

    return completions


def compare_models(
    base_model: str,
    grpo_model_path: str,
    prompts: list[str],
    classifier=None,
    max_new_tokens: int = 200,
) -> dict:
    """Generate from both models and compare metrics side-by-side."""
    console.print("\n[bold]Generating from base model...[/]")
    base_completions = generate_completions(base_model, prompts, max_new_tokens)

    console.print("[bold]Generating from GRPO model...[/]")
    grpo_completions = generate_completions(
        grpo_model_path, prompts, max_new_tokens, is_lora=True, base_model=base_model
    )

    base_metrics = compute_style_metrics(base_completions, classifier)
    grpo_metrics = compute_style_metrics(grpo_completions, classifier)

    # print comparison table
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Base", style="yellow")
    table.add_column("GRPO (MarceLLo)", style="green")
    table.add_column("Delta", style="bold")

    for key in base_metrics:
        base_val = base_metrics[key]
        grpo_val = grpo_metrics[key]
        delta = grpo_val - base_val
        sign = "+" if delta > 0 else ""
        table.add_row(key, f"{base_val:.4f}", f"{grpo_val:.4f}", f"{sign}{delta:.4f}")

    console.print(table)

    # show a few example completions
    console.print("\n[bold]Example Completions[/]\n")
    for i, prompt in enumerate(prompts[:3]):
        console.print(f"[dim]Prompt: {prompt}[/]\n")
        console.print(Panel(base_completions[i], title="Base", border_style="yellow"))
        console.print(Panel(grpo_completions[i], title="MarceLLo (GRPO)", border_style="green"))
        console.print()

    return {
        "base_metrics": base_metrics,
        "grpo_metrics": grpo_metrics,
        "base_completions": base_completions,
        "grpo_completions": grpo_completions,
    }
