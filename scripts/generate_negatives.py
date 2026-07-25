"""Generate content-matched negative samples by rephrasing Marcelo's own texts.

The prewritten negatives (Wikipedia, encyclopedic prose) differ from the
positives in topic, register and length, so a classifier trained on them
learns topic detection instead of voice. Rephrasing each positive into a
neutral voice keeps the content fixed and leaves style as the only signal.

Output is JSONL so the exact text (line breaks included) survives, which
matters for poems.

Usage:
    python scripts/generate_negatives.py
    python scripts/generate_negatives.py --model Qwen/Qwen2.5-1.5B-Instruct --per-positive 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

SYSTEM_PROMPTS = {
    # neutral: strips the voice but keeps the content, so style is the only signal
    "neutral": (
        "Eres un editor que reescribe textos en una voz neutra y genérica. "
        "Conservas el idioma, el tema y la longitud aproximada del original, "
        "pero eliminas toda voz personal: metáforas propias, ritmo, saltos de línea "
        "expresivos, giros idiosincráticos. El resultado debe sonar a texto escrito "
        "por cualquiera. Responde solo con el texto reescrito."
    ),
    # poetic: another poetic voice on the same theme, so the classifier cannot
    # settle for "poetry equals Marcelo" — the probe scores poems it never saw
    "poetic": (
        "Eres un poeta clásico. Reescribes el texto sobre el mismo tema pero con "
        "otra voz poética: registro formal y solemne, imágenes tradicionales, "
        "vocabulario elevado. Conservas el idioma y la longitud aproximada. "
        "No imites el estilo del original. Responde solo con el texto reescrito."
    ),
}

USER_PROMPTS = {
    "neutral": "Reescribe este texto en voz neutra y genérica:\n\n{text}",
    "poetic": "Reescribe este texto con otra voz poética, formal y clásica:\n\n{text}",
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_positives(train_path: str, val_path: str) -> list[str]:
    dataset = concatenate_datasets([load_from_disk(train_path), load_from_disk(val_path)])
    return [row["text"] for row in dataset if row["label"] == 1]


def rephrase(
    texts: list[str],
    model_name: str,
    variant: str,
    per_positive: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    device = _get_device()
    console.print(f"Loading [cyan]{model_name}[/] on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    torch.manual_seed(seed)

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Rephrasing", total=len(texts) * per_positive)

        for index, text in enumerate(texts):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPTS[variant]},
                    {"role": "user", "content": USER_PROMPTS[variant].format(text=text)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(device)

            for repeat in range(per_positive):
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()

                if generated:
                    results.append(
                        {
                            "text": generated,
                            "source": f"llm_rephrase_{variant}",
                            "model": model_name,
                            "positive_index": index,
                            "variant": variant,
                            "repeat": repeat,
                        }
                    )
                progress.advance(task)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate rephrased negative samples")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--variant",
        type=str,
        default="neutral",
        choices=sorted(SYSTEM_PROMPTS),
        help="neutral strips the voice; poetic writes another poetic voice on the same theme",
    )
    parser.add_argument("--train-path", type=str, default="data/processed/train")
    parser.add_argument("--val-path", type=str, default="data/processed/val")
    parser.add_argument("--per-positive", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N positives")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to data/raw/negative_samples/rephrased/llm_rephrase_<variant>.jsonl",
    )
    args = parser.parse_args()

    output = args.output or (
        f"data/raw/negative_samples/rephrased/llm_rephrase_{args.variant}.jsonl"
    )

    positives = load_positives(args.train_path, args.val_path)
    if args.limit:
        positives = positives[: args.limit]
    console.print(
        f"\n[bold]Rephrasing {len(positives)} positive samples[/] (variant: {args.variant})\n"
    )

    results = rephrase(
        positives,
        args.model,
        args.variant,
        args.per_positive,
        args.temperature,
        args.max_new_tokens,
        args.seed,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"\nWrote {len(results)} negatives to {output_path}\n")


if __name__ == "__main__":
    main()
