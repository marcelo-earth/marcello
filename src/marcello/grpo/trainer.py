"""GRPO trainer for style-aligned generation.

Implements Group Relative Policy Optimization (DeepSeek-Math, 2024) with
TRL's GRPOTrainer as the backbone. The key idea: instead of a learned value
function, GRPO uses the group mean reward as baseline.

Algorithm per training step:
  1. Sample a batch of prompts
  2. For each prompt, generate G completions (group)
  3. Score all completions with the reward function
  4. Compute group-relative advantages: A_i = (r_i - mean(r_group)) / std(r_group)
  5. Policy gradient update with clipped surrogate objective
  6. KL penalty to stay close to the reference model
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from marcello.grpo.reward import StyleReward


@dataclass
class MarceLLoGRPOConfig:
    """Configuration for MarceLLo GRPO training."""

    # model
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # LoRA (keeps memory manageable on free GPUs)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # None = auto-detect

    # GRPO
    num_generations: int = 8  # G: completions per prompt
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95

    # training
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    kl_coef: float = 0.1
    clip_range: float = 0.2
    max_grad_norm: float = 1.0

    # reward
    classifier_path: str = "outputs/classifier/best"
    reward_temperature: float = 1.0
    reward_style_weight: float = 0.65
    reward_length_bonus_weight: float = 0.1
    reward_prompt_relevance_weight: float = 0.2
    reward_repetition_penalty_weight: float = 0.15
    reward_prompt_echo_penalty_weight: float = 0.1
    reward_reference_copy_penalty_weight: float = 0.15
    reward_target_length: int = 200
    reward_reference_texts_path: str | None = "data/processed/train"
    reward_reference_ngram_size: int = 8

    # output
    output_dir: str = "outputs/grpo"
    use_wandb: bool = False


class MarceLLoGRPOTrainer:
    """Wraps TRL's GRPOTrainer with MarceLLo-specific reward and config.

    Usage:
        trainer = MarceLLoGRPOTrainer(config)
        trainer.train(prompt_dataset)
    """

    def __init__(self, config: MarceLLoGRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.reward_fn = None
        self._trainer = None

    def _load_model(self):
        """Load the base model with LoRA applied."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _load_reward(self):
        """Load the style classifier as reward function."""
        self.reward_fn = StyleReward(
            classifier_path=self.config.classifier_path,
            temperature=self.config.reward_temperature,
            style_weight=self.config.reward_style_weight,
            length_bonus_weight=self.config.reward_length_bonus_weight,
            prompt_relevance_weight=self.config.reward_prompt_relevance_weight,
            repetition_penalty_weight=self.config.reward_repetition_penalty_weight,
            prompt_echo_penalty_weight=self.config.reward_prompt_echo_penalty_weight,
            reference_copy_penalty_weight=self.config.reward_reference_copy_penalty_weight,
            target_length=self.config.reward_target_length,
            reference_texts_path=self.config.reward_reference_texts_path,
            reference_ngram_size=self.config.reward_reference_ngram_size,
        )

    def _normalize_prompts(self, prompts, expected: int) -> list[str] | None:
        """Normalize prompt payloads from TRL to match the completions batch."""
        if prompts is None:
            return None

        normalized = []
        for prompt in prompts:
            if isinstance(prompt, list):
                normalized.append(prompt[0]["content"])
            else:
                normalized.append(prompt)

        if len(normalized) == expected:
            return normalized
        if len(normalized) == 1:
            return normalized * expected
        if expected % len(normalized) == 0:
            repeat = expected // len(normalized)
            return [prompt for prompt in normalized for _ in range(repeat)]
        return normalized[:expected]

    def _build_grpo_args(self) -> GRPOConfig:
        """Build TRL config using only fields supported by the installed version."""
        supported = set(inspect.signature(GRPOConfig.__init__).parameters)
        config_kwargs = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "max_grad_norm": self.config.max_grad_norm,
            "num_generations": self.config.num_generations,
            "max_completion_length": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "log_completions": True,
            "report_to": "wandb" if self.config.use_wandb else "none",
        }

        if "beta" in supported:
            config_kwargs["beta"] = self.config.kl_coef
        elif "kl_coef" in supported:
            config_kwargs["kl_coef"] = self.config.kl_coef

        if "epsilon" in supported:
            config_kwargs["epsilon"] = self.config.clip_range
        elif "clip_range" in supported:
            config_kwargs["clip_range"] = self.config.clip_range

        filtered_kwargs = {key: value for key, value in config_kwargs.items() if key in supported}
        return GRPOConfig(**filtered_kwargs)

    def _build_reward_function(self):
        """Build a reward function compatible with TRL's GRPOTrainer.

        TRL expects: reward_fn(completions: list[str]) -> list[float]
        """
        reward = self.reward_fn

        def reward_function(completions, **kwargs):
            texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
            prompts = kwargs.get("prompts") or kwargs.get("prompt")
            normalized_prompts = self._normalize_prompts(prompts, expected=len(texts))
            scores = reward.score(texts, prompts=normalized_prompts)
            return scores

        return reward_function

    def train(self, prompt_dataset: Dataset):
        """Run GRPO training on a dataset of prompts.

        The prompt_dataset should have a 'prompt' column with text prompts
        that the model will complete. The reward function scores completions.
        """
        self._load_model()
        self._load_reward()

        grpo_config = self._build_grpo_args()

        self._trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=prompt_dataset,
            reward_funcs=self._build_reward_function(),
            processing_class=self.tokenizer,
        )

        self._trainer.train()
        self.save(Path(self.config.output_dir) / "final")

    def save(self, path: Path):
        """Save the trained LoRA adapter and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
