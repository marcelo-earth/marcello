"""Tests for GRPO prompt formatting and reward shaping."""

from __future__ import annotations

import torch
from datasets import Dataset

from marcello.grpo.prompting import (
    build_control_prompt,
    extract_prompts_from_positive_dataset,
    extract_seed_text,
    infer_language,
)
from marcello.grpo.reward import StyleReward


class FakeClassifier:
    def eval(self):
        return self

    def cuda(self):
        return self

    def predict(self, texts):
        return [0.9 for _ in texts]


def test_infer_language_prefers_spanish_with_accents():
    assert infer_language("La educación también cambia cómo vivimos hoy.") == "es"


def test_extract_prompts_from_positive_dataset_adds_control_tags():
    dataset = Dataset.from_dict(
        {
            "text": [
                "This is a reflective opening sentence. This is the next sentence for context.",
                "Texto corto.",
            ],
            "label": [1, 1],
            "style": ["standard", "poetic"],
        }
    )

    prompts = extract_prompts_from_positive_dataset(dataset, max_prompts=5)

    assert len(prompts) == 1
    assert prompts[0]["prompt"].startswith("<style:standard> <lang:en> <task:continue>")
    assert extract_seed_text(prompts[0]["prompt"]).startswith("This is a reflective opening")


def test_reward_penalizes_echo_repetition_and_reference_copy(tmp_path, monkeypatch):
    reference_path = tmp_path / "train_ds"
    Dataset.from_dict(
        {
            "text": [
                "The stars were waiting quietly above us while the city forgot to look up."
            ],
            "label": [1],
        }
    ).save_to_disk(reference_path)

    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )

    reward = StyleReward(
        classifier_path="unused",
        prompt_relevance_weight=0.3,
        repetition_penalty_weight=0.5,
        prompt_echo_penalty_weight=0.4,
        reference_copy_penalty_weight=0.6,
        length_bonus_weight=0.0,
        reference_texts_path=str(reference_path),
        reference_ngram_size=4,
    )

    prompt = build_control_prompt(
        "The stars were waiting quietly above us.",
        style="standard",
        language="en",
    )
    good = "They made the night feel less empty, and that changed how I walked home."
    bad = (
        "The stars were waiting quietly above us. "
        "The stars were waiting quietly above us. "
        "The city forgot to look up."
    )

    scores = reward.score([good, bad], prompts=[prompt, prompt])

    assert scores[0] > scores[1]


def test_classifier_from_pretrained_reads_saved_config(tmp_path, monkeypatch):
    from marcello.classifier.model import StyleClassifier

    captured = {}

    def fake_init(self, model_name="microsoft/deberta-v3-small", dropout=0.1, freeze_encoder_layers=0):
        torch.nn.Module.__init__(self)
        captured["args"] = (model_name, dropout, freeze_encoder_layers)
        self.model_name = model_name
        self.dropout = dropout
        self.freeze_encoder_layers = freeze_encoder_layers
        self.encoder = type("Encoder", (), {"config": type("Cfg", (), {"hidden_size": 4})()})()
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(4, 1))
        self.tokenizer = object()

    monkeypatch.setattr(StyleClassifier, "__init__", fake_init)

    path = tmp_path / "classifier"
    path.mkdir()
    (path / "config.json").write_text(
        '{"model_name": "tiny-test-model", "dropout": 0.25, "freeze_encoder_layers": 2}',
        encoding="utf-8",
    )

    template = StyleClassifier("tiny-test-model", dropout=0.25, freeze_encoder_layers=2)
    torch.save(template.state_dict(), path / "model.pt")

    StyleClassifier.from_pretrained(str(path))

    assert captured["args"] == ("tiny-test-model", 0.25, 2)
