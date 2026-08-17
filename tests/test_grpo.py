"""Tests for GRPO prompt formatting and reward shaping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
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


class FakeTokenizer:
    """One token per 4 characters, so token counts never equal word counts."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))


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


def _reward_with_fake_classifier(monkeypatch, **kwargs):
    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )
    kwargs.setdefault("length_bonus_weight", 0.0)
    return StyleReward(classifier_path="unused", **kwargs)


def test_prompt_relevance_and_echo_penalty_never_grade_the_same_tokens(monkeypatch):
    """Issue #16: the two components must measure disjoint things.

    Copying the seed is charged once, by the echo penalty, and earns no relevance.
    Carrying the seed's vocabulary into new phrasing earns relevance and is charged
    nothing. No completion can be paid and charged for one reuse.
    """
    reward = _reward_with_fake_classifier(monkeypatch)
    prompt = build_control_prompt(
        "The stars were waiting quietly above us.",
        style="standard",
        language="en",
    )

    copied = "The stars were waiting quietly above us, and nothing moved."
    reworded = "I kept waiting under them, quietly, until the stars felt closer than the city."

    assert reward._prompt_echo_penalty(prompt, copied) > 0
    assert reward._prompt_relevance(prompt, copied) == 0.0

    assert reward._prompt_echo_penalty(prompt, reworded) == 0.0
    assert reward._prompt_relevance(prompt, reworded) > 0


def test_prompt_relevance_ceiling_does_not_depend_on_seed_length(monkeypatch):
    """Issue #16: relevance was recall over the seed, so its scale moved with the seed.

    The same carried vocabulary now scores the same whether the seed is one line or a
    whole poem, and the ceiling stays reachable on the long seed.
    """
    reward = _reward_with_fake_classifier(monkeypatch, relevance_target_tokens=4)

    short_seed = build_control_prompt("Rivers remember mountains.", "poetic", "en")
    long_seed = build_control_prompt(
        "Rivers remember mountains.\n"
        "Harbors forget every vessel that ever wintered against their stones.\n"
        "Lanterns keep burning through arguments no one finished.\n"
        "Bridges outlast whichever quarrel first funded them.",
        "poetic",
        "en",
    )
    completion = "Mountains stay in how the rivers move, long after nobody remembers why."

    assert reward._prompt_relevance(short_seed, completion) == pytest.approx(
        reward._prompt_relevance(long_seed, completion)
    )

    carries_the_budget = "Rivers, mountains, harbors and lanterns all keep their own accounting."
    assert reward._prompt_relevance(long_seed, carries_the_budget) == pytest.approx(1.0)


def test_echo_penalty_charges_short_seeds(monkeypatch):
    """A seed shorter than the n-gram window used to be uncharged by construction.

    Seed and completion n-grams were sized independently, so a 3-token seed produced
    3-grams against the completion's 4-grams and the intersection was always empty.
    """
    reward = _reward_with_fake_classifier(monkeypatch, echo_floor=0.0)
    prompt = build_control_prompt("Rivers remember mountains.", "poetic", "en")

    assert reward._prompt_echo_penalty(prompt, "Rivers remember mountains, always.") > 0


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

    reward = StyleReward(classifier_path="unused", tokenizer=FakeTokenizer())

    config = yaml.safe_load(Path("configs/grpo.yaml").read_text(encoding="utf-8"))["reward"]
    assert reward.target_length == config["target_length"]
    assert reward.length_bonus_weight == config["length_bonus_weight"]
    assert reward.echo_ngram_size == config["echo_ngram_size"]
    assert reward.echo_floor == config["echo_floor"]
    assert reward.relevance_target_tokens == config["relevance_target_tokens"]

    assert reward.target_length == 60
    assert reward.length_bonus_weight == 0.1
    assert reward.style_weight == 0.65
    assert reward.prompt_relevance_weight == 0.2
    assert reward.repetition_penalty_weight == 0.15
    assert reward.reference_copy_penalty_weight == 0.15
    assert reward.prompt_echo_penalty_weight == 0.1


def test_length_bonus_measures_tokens_not_words(monkeypatch):
    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )

    reward = StyleReward(classifier_path="unused", target_length=10, tokenizer=FakeTokenizer())

    # 4 words, 40 characters, so 10 tokens under FakeTokenizer: the peak of the bonus
    text = "aaaaaaa bbbbbbbbb ccccccccc ddddddddd ee"
    assert len(text) // 4 == 10
    assert len(text.split()) == 5

    assert reward._length_bonus(text) == 1.0
    # a word count of 5 against a target of 10 would have scored 0.5, not 1.0
    assert reward._length_bonus(text) != 0.5


def test_length_bonus_is_exercised_by_score_at_the_default_weight(monkeypatch):
    """Regression for #17: the bonus used to raise on every call and no test caught it,
    because the only test reaching score() switched the bonus off."""
    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )

    reward = StyleReward(classifier_path="unused", tokenizer=FakeTokenizer())
    assert reward.length_bonus_weight > 0

    breakdown = reward.score(["Una frase corta sobre la noche."], return_breakdown=True)

    assert breakdown[0]["length_bonus"] > 0
    assert isinstance(breakdown[0]["total"], float)


def test_style_reward_refuses_a_live_length_bonus_without_a_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "marcello.grpo.reward.StyleClassifier.from_pretrained",
        lambda _: FakeClassifier(),
    )

    with pytest.raises(ValueError, match="requires a tokenizer"):
        StyleReward(classifier_path="unused", length_bonus_weight=0.1)

    # switching the bonus off is still a valid way to build it
    StyleReward(classifier_path="unused", length_bonus_weight=0.0)


def test_target_length_peak_is_reachable_within_the_generation_budget():
    """The bonus is zero at 2 * target_length. If that sits past max_new_tokens the
    curve is monotone inside the budget and rewards padding to the cap."""
    config = yaml.safe_load(Path("configs/grpo.yaml").read_text(encoding="utf-8"))

    target = config["reward"]["target_length"]
    budget = config["grpo"]["max_new_tokens"]

    assert 2 * target <= budget
