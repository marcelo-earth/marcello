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
            "text": ["The stars were waiting quietly above us while the city forgot to look up."],
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

    def fake_init(
        self,
        model_name="microsoft/deberta-v3-small",
        dropout=0.1,
        freeze_encoder_layers=0,
        head_norm=True,
    ):
        torch.nn.Module.__init__(self)
        captured["args"] = (model_name, dropout, freeze_encoder_layers, head_norm)
        self.model_name = model_name
        self.dropout = dropout
        self.freeze_encoder_layers = freeze_encoder_layers
        self.head_norm = head_norm
        self.encoder = type("Encoder", (), {"config": type("Cfg", (), {"hidden_size": 4})()})()
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(4, 1))
        self.register_buffer("feature_mean", torch.zeros(4))
        self.register_buffer("feature_std", torch.ones(4))
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

    # checkpoints saved before head_norm existed must load with it disabled
    assert captured["args"] == ("tiny-test-model", 0.25, 2, False)


def test_classifier_from_pretrained_without_feature_stats_is_identity(tmp_path):
    """Old checkpoints lack the standardiser and must score exactly as before."""
    from marcello.classifier.model import StyleClassifier

    model = StyleClassifier.__new__(StyleClassifier)
    torch.nn.Module.__init__(model)
    model.register_buffer("feature_mean", torch.zeros(4))
    model.register_buffer("feature_std", torch.ones(4))
    model.classifier = torch.nn.Sequential(torch.nn.Linear(4, 1))

    pooled = torch.randn(3, 4)
    expected = model.classifier(pooled).squeeze(-1)

    assert torch.allclose(model.head(pooled), expected)


def test_set_feature_stats_standardises_the_training_features():
    from marcello.classifier.model import StyleClassifier

    model = StyleClassifier.__new__(StyleClassifier)
    torch.nn.Module.__init__(model)
    model.register_buffer("feature_mean", torch.zeros(4))
    model.register_buffer("feature_std", torch.ones(4))

    torch.manual_seed(0)
    features = torch.randn(64, 4) * torch.tensor([100.0, 1.0, 0.01, 5.0]) + 7.0
    with torch.no_grad():
        model.set_feature_stats(features)

    scaled = (features - model.feature_mean) / model.feature_std

    # The 0.01-scale dimension sits on an offset of 7, so subtracting the mean
    # cancels away most of its float32 precision and dividing by 0.01 puts the
    # remainder back. Residual mean lands around 1e-4; that is the arithmetic,
    # not the standardiser. The point of the test is the spread.
    assert torch.allclose(scaled.mean(dim=0), torch.zeros(4), atol=1e-3)
    assert torch.allclose(scaled.std(dim=0), torch.ones(4), atol=1e-5)


def test_set_feature_stats_survives_a_constant_dimension():
    """A dimension that never varies must not be amplified into noise."""
    from marcello.classifier.model import StyleClassifier

    model = StyleClassifier.__new__(StyleClassifier)
    torch.nn.Module.__init__(model)
    model.register_buffer("feature_mean", torch.zeros(2))
    model.register_buffer("feature_std", torch.ones(2))

    features = torch.stack([torch.randn(32), torch.full((32,), 3.0)], dim=1)
    with torch.no_grad():
        model.set_feature_stats(features)

    scaled = (features - model.feature_mean) / model.feature_std
    assert torch.isfinite(scaled).all()


def test_style_reward_defaults_match_grpo_yaml(monkeypatch):
    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )

    reward = StyleReward(classifier_path="unused")

    assert reward.target_length == 180
    assert reward.length_bonus_weight == 0.1
    assert reward.style_weight == 0.65
    assert reward.prompt_relevance_weight == 0.2
    assert reward.repetition_penalty_weight == 0.15
    assert reward.reference_copy_penalty_weight == 0.15
    assert reward.prompt_echo_penalty_weight == 0.1
