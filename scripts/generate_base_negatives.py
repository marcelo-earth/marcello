"""Generate negatives from the untrained base model on the GRPO seed prompts (issue #12).

The classifier scores free continuations from Qwen2.5-1.5B during GRPO, so it has to
have seen some. Seeds come from the training positives only, through the same
extraction GRPO uses, and the sampling settings mirror configs/grpo.yaml.

Output is JSONL under data/raw/negative_samples/, which NegativeSampler reads.
The run resumes from what already exists, like generate_negatives.py.

Usage:
    python scripts/generate_base_negatives.py --limit 8     # smoke test
    python scripts/generate_base_negatives.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from marcello.grpo.prompting import extract_prompts_from_positive_dataset

console = Console()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--output", default="data/raw/negative_samples/base_generations.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N seeds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prompts = extract_prompts_from_positive_dataset(load_from_disk(args.train_dir))
    rows = list(prompts)
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out.exists():
        done = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = rows[done:]
    console.print(f"{done} already written, {len(rows)} seeds left")
    if not rows:
        return

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp16 overflows to NaN on MPS with Qwen, so only CUDA gets reduced precision
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # eager attention: SDPA with left padding yields NaN rows on MPS
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    )
    model.to(device).eval()

    with out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            enc = tokenizer([r["prompt"] for r in batch], return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            for row, tokens in zip(batch, new_tokens):
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if not text:
                    continue
                record = {
                    "text": text,
                    "source": "base_generation",
                    "model": args.model,
                    "language": row["language"],
                    "style": row["style"],
                    "seed": row["raw_prompt"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            console.print(f"{done + start + len(batch)} seeds done")


if __name__ == "__main__":
    main()
