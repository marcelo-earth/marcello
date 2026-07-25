"""Supervised fine-tuning baseline.

GRPO only means something if it beats plain SFT on the same corpus, prompts
and evaluation. This trains a LoRA adapter on (control prompt -> continuation)
pairs built from the positive samples, using the exact prompt format from
`marcello.grpo.prompting`, so the two runs are comparable and the same
`evaluate.py` invocation works on either checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from marcello.grpo.prompting import (
    build_control_prompt,
    infer_language,
    split_seed_and_continuation,
)

LABEL_IGNORE_INDEX = -100


@dataclass
class SFTConfig:
    """Configuration for the SFT baseline."""

    model_name: str = "Qwen/Qwen2.5-1.5B"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    learning_rate: float = 1e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 512

    output_dir: str = "outputs/sft"
    use_wandb: bool = False


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sft_dataset(dataset: Dataset, min_completion_words: int = 5) -> Dataset:
    """Split each positive sample into a control prompt and its continuation.

    The seed sentences become the prompt (identical to what GRPO sees) and the
    rest of the text becomes the target, so SFT learns the same task GRPO is
    rewarded for rather than plain language modelling over the corpus.
    """
    prompts: list[str] = []
    completions: list[str] = []

    for row in dataset:
        if row.get("label") != 1:
            continue

        text = row["text"].strip()
        seed_text, completion = split_seed_and_continuation(text)
        if len(seed_text.split()) < 5 or len(completion.split()) < min_completion_words:
            continue

        style = row.get("style", "standard")
        language = row.get("language") or infer_language(text)

        prompts.append(build_control_prompt(seed_text, style=style, language=language))
        completions.append(completion)

    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def collate_fn(batch, tokenizer, max_length: int):
    """Tokenize prompt+completion pairs, masking prompt tokens out of the loss."""
    input_ids_batch = []
    labels_batch = []

    for item in batch:
        prompt_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            " " + item["completion"] + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = (prompt_ids + completion_ids)[:max_length]
        labels = ([LABEL_IGNORE_INDEX] * len(prompt_ids) + completion_ids)[:max_length]

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)

    longest = max(len(ids) for ids in input_ids_batch)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = [ids + [pad_id] * (longest - len(ids)) for ids in input_ids_batch]
    attention_mask = [[1] * len(ids) + [0] * (longest - len(ids)) for ids in input_ids_batch]
    padded_labels = [
        lbl + [LABEL_IGNORE_INDEX] * (longest - len(lbl)) for lbl in labels_batch
    ]

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(padded_labels),
    }


def train_sft(train_dataset: Dataset, config: SFTConfig):
    """Train the LoRA adapter and save it to `config.output_dir`/final."""
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float32 if device.type == "cpu" else torch.float16,
    ).to(device)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, config.max_length),
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(1, len(loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    if config.use_wandb:
        import wandb

        wandb.init(project="marcello-sft", config=vars(config))

    model.train()
    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / config.gradient_accumulation_steps).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{config.num_train_epochs} | train_loss={avg_loss:.4f}")

        if config.use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

    output_path = Path(config.output_dir) / "final"
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return model
