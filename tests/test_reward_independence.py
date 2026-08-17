"""Tests for the reward-component independence analysis (issue #16)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so analyze_reward_independence can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import analyze_reward_independence as ari  # noqa: E402


def _write_run(path: Path, breakdowns: list[dict], label: str = "grpo") -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "test",
                "results": {
                    "per_prompt": [
                        {"prompt": f"p{i}", f"{label}_reward_breakdown": b}
                        for i, b in enumerate(breakdowns)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_correlate_flags_a_pair_that_moves_together():
    breakdowns = [{"prompt_relevance": x / 10, "prompt_echo_penalty": x / 10} for x in range(10)]

    row = ari.correlate(breakdowns, "prompt_relevance", "prompt_echo_penalty")

    assert row["r"] == pytest.approx(1.0)
    assert row["verdict"] == "redundant"


def test_correlate_clears_a_pair_that_does_not():
    breakdowns = [
        {"prompt_relevance": r, "prompt_echo_penalty": e}
        for r, e in [(0.0, 0.3), (0.2, 0.0), (0.4, 0.1), (0.6, 0.1), (0.8, 0.0), (1.0, 0.3)]
    ]

    row = ari.correlate(breakdowns, "prompt_relevance", "prompt_echo_penalty")

    assert abs(row["r"]) <= ari.INDEPENDENT_BELOW
    assert row["verdict"] == "independent"


def test_correlate_reports_a_component_that_never_varies():
    """A weight pinned at one value contributes nothing and should be visible as such."""
    breakdowns = [{"prompt_relevance": x / 10, "prompt_echo_penalty": 0.0} for x in range(6)]

    row = ari.correlate(breakdowns, "prompt_relevance", "prompt_echo_penalty")

    assert row["r"] is None
    assert row["verdict"] == "flat"


def test_load_breakdowns_reads_an_eval_run(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [{"prompt_relevance": 0.1, "prompt_echo_penalty": 0.0}])

    assert ari.load_breakdowns(run, "grpo") == [
        {"prompt_relevance": 0.1, "prompt_echo_penalty": 0.0}
    ]


def test_load_breakdowns_explains_a_run_without_the_breakdown(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [{"prompt_relevance": 0.1}], label="grpo")
    data = json.loads(run.read_text(encoding="utf-8"))
    data["results"]["per_prompt"] = [{"prompt": "p0"}]
    run.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SystemExit, match="--reward-config"):
        ari.load_breakdowns(run, "grpo")
