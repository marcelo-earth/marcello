"""Tests for the group reward variance diagnostics (issue #18)."""

from __future__ import annotations

import json

from marcello.grpo.diagnostics import (
    DEAD_GROUP_STD,
    RewardVarianceLog,
    group_spans,
    summarize_group,
)


def _breakdown(total: float, **components) -> dict:
    row = {
        "total": total,
        "raw_style_prob": 0.9,
        "style_score": total,
        "length_bonus": 0.1,
        "prompt_relevance": 0.0,
    }
    row.update(components)
    return row


def test_group_spans_follow_the_prompt_not_the_batch_layout():
    """Groups are defined by the shared prompt, so a short last group stays a group."""
    prompts = ["a", "a", "a", "b", "b"]

    spans = group_spans(prompts, batch_size=5, group_size=3)

    assert [list(span) for span in spans] == [[0, 1, 2], [3, 4]]


def test_group_spans_fall_back_to_fixed_chunks_without_prompts():
    """TRL does not always hand the prompts over; the diagnostic still has to group."""
    spans = group_spans(None, batch_size=5, group_size=2)

    assert [list(span) for span in spans] == [[0, 1], [2, 3], [4]]


def test_summarize_group_reports_spread_per_component():
    breakdowns = [
        _breakdown(0.2, style_score=0.2, length_bonus=0.1),
        _breakdown(0.6, style_score=0.6, length_bonus=0.1),
    ]

    stats = summarize_group(breakdowns)

    assert stats.size == 2
    assert stats.minimum == 0.2
    assert stats.maximum == 0.6
    assert stats.std > 0
    assert stats.component_std["style_score"] > 0
    # a component identical across the group moved nothing, and has to say so
    assert stats.component_std["length_bonus"] == 0.0
    assert "raw_style_prob" not in stats.component_std


def test_a_group_whose_completions_all_score_the_same_is_flagged_dead():
    """The point of the whole diagnostic: equal rewards mean zero advantage."""
    stats = summarize_group([_breakdown(0.42) for _ in range(8)])

    assert stats.std == 0.0
    assert stats.is_dead


def test_a_group_with_real_spread_is_not_flagged():
    stats = summarize_group([_breakdown(t) for t in [0.1, 0.4, 0.5, 0.9]])

    assert stats.std > DEAD_GROUP_STD
    assert not stats.is_dead


def test_summary_names_components_that_never_varied_within_a_group():
    """A component constant inside every group is weight the objective never spends."""
    log = RewardVarianceLog(group_size=2)
    log.record(
        [
            _breakdown(0.2, style_score=0.2),
            _breakdown(0.6, style_score=0.6),
            _breakdown(0.3, style_score=0.3),
            _breakdown(0.8, style_score=0.8),
        ],
        prompts=["a", "a", "b", "b"],
    )

    summary = log.summary()

    assert summary["groups"] == 2
    assert summary["dead_groups"] == 0
    assert "length_bonus" in summary["components_without_spread"]
    assert "style_score" not in summary["components_without_spread"]


def test_summary_counts_flat_groups_across_the_run():
    log = RewardVarianceLog(group_size=2)
    log.record([_breakdown(0.5), _breakdown(0.5)], prompts=["a", "a"])
    log.record([_breakdown(0.1), _breakdown(0.9)], prompts=["b", "b"])

    summary = log.summary()

    assert summary["dead_groups"] == 1
    assert summary["dead_group_fraction"] == 0.5
    assert summary["group_std_min"] == 0.0
    assert summary["group_std_max"] > 0


def test_format_summary_calls_out_a_run_that_cannot_learn():
    log = RewardVarianceLog(group_size=2)
    for _ in range(3):
        log.record([_breakdown(0.5), _breakdown(0.5)], prompts=["a", "a"])

    text = log.format_summary()

    assert "flat groups" in text
    assert "change the reward, not the optimizer" in text


def test_format_batch_reports_the_spread_of_one_scored_batch():
    log = RewardVarianceLog(group_size=2)
    groups = log.record([_breakdown(0.1), _breakdown(0.9)], prompts=["a", "a"])

    line = RewardVarianceLog.format_batch(groups)

    assert "reward groups: 1" in line
    assert "std mean" in line


def test_write_persists_every_group_next_to_the_run(tmp_path):
    log = RewardVarianceLog(group_size=2, output_dir=tmp_path / "run")
    log.record([_breakdown(0.1), _breakdown(0.9)], prompts=["a", "a"])

    path = log.write()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["summary"]["groups"] == 1
    assert len(record["groups"]) == 1
    assert record["groups"][0]["size"] == 2


def test_nothing_recorded_says_so_instead_of_crashing():
    log = RewardVarianceLog(group_size=8)

    assert log.summary() == {"groups": 0}
    assert "nothing can be said" in log.format_summary()


def test_the_trainer_reward_function_records_every_scored_batch(monkeypatch):
    """The diagnostic is only worth anything if it sits on the path TRL actually calls."""
    from marcello.grpo.trainer import MarceLLoGRPOConfig, MarceLLoGRPOTrainer

    trainer = MarceLLoGRPOTrainer(MarceLLoGRPOConfig(num_generations=2))

    class FakeReward:
        def score(self, texts, prompts=None, return_breakdown=False):
            assert return_breakdown, "the trainer needs the breakdown to log component spread"
            return [_breakdown(0.1 * i, style_score=0.1 * i) for i in range(len(texts))]

    trainer.reward_fn = FakeReward()
    reward_function = trainer._build_reward_function()

    scores = reward_function(["one", "two"], prompts=["a", "a"])

    assert scores == [0.0, 0.1]
    assert len(trainer.variance_log.groups) == 1
    assert trainer.variance_log.groups[0].size == 2
