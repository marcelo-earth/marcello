"""Tests for the compare_runs script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add scripts/ to path so compare_runs can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import compare_runs  # noqa: E402


def _write_run(
    path: Path, grpo_metrics: dict, per_prompt: list[dict], run_id: str = "test"
) -> None:
    data = {
        "run_id": run_id,
        "timestamp": "2026-05-16T00:00:00+00:00",
        "git_hash": "abc1234",
        "model": "outputs/grpo/final",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "prompt_count": len(per_prompt),
        "results": {
            "prompts": [p["prompt"] for p in per_prompt],
            "base_metrics": {k: 0.0 for k in grpo_metrics},
            "grpo_metrics": grpo_metrics,
            "base_completions": ["base"] * len(per_prompt),
            "grpo_completions": ["grpo"] * len(per_prompt),
            "per_prompt": per_prompt,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_run_flattens_results_key(tmp_path):
    run_path = tmp_path / "run.json"
    _write_run(
        run_path,
        grpo_metrics={"style_score_mean": 0.7},
        per_prompt=[
            {
                "prompt": "p",
                "base_style_score": 0.2,
                "grpo_style_score": 0.7,
                "base_completion": "b",
                "grpo_completion": "g",
            }
        ],
    )
    run = compare_runs.load_run(str(run_path))
    assert "grpo_metrics" in run
    assert "_meta" in run
    assert run["_meta"]["run_id"] == "test"


def test_compare_aggregate_metrics_delta(tmp_path):
    run_a_path = tmp_path / "run_a.json"
    run_b_path = tmp_path / "run_b.json"
    _write_run(run_a_path, grpo_metrics={"style_score_mean": 0.6}, per_prompt=[])
    _write_run(run_b_path, grpo_metrics={"style_score_mean": 0.75}, per_prompt=[])

    run_a = compare_runs.load_run(str(run_a_path))
    run_b = compare_runs.load_run(str(run_b_path))
    rows = compare_runs.compare_aggregate_metrics(run_a, run_b, "A", "B")

    assert len(rows) == 1
    assert rows[0]["metric"] == "style_score_mean"
    assert abs(rows[0]["delta"] - 0.15) < 1e-9


def test_compare_per_prompt_sorted_worst_first(tmp_path):
    prompt_a = {
        "prompt": "p_a",
        "base_style_score": 0.3,
        "grpo_style_score": 0.8,
        "base_completion": "b",
        "grpo_completion": "g",
    }
    prompt_b = {
        "prompt": "p_b",
        "base_style_score": 0.3,
        "grpo_style_score": 0.4,
        "base_completion": "b",
        "grpo_completion": "g",
    }

    run_a_path = tmp_path / "run_a.json"
    run_b_path = tmp_path / "run_b.json"

    # run A has same scores as seed (0.8 and 0.4)
    _write_run(
        run_a_path,
        grpo_metrics={},
        per_prompt=[
            {**prompt_a, "grpo_style_score": 0.8},
            {**prompt_b, "grpo_style_score": 0.4},
        ],
    )
    # run B drops p_a from 0.8 → 0.5 (regression) and improves p_b from 0.4 → 0.7
    _write_run(
        run_b_path,
        grpo_metrics={},
        per_prompt=[
            {**prompt_a, "grpo_style_score": 0.5},
            {**prompt_b, "grpo_style_score": 0.7},
        ],
    )

    run_a = compare_runs.load_run(str(run_a_path))
    run_b = compare_runs.load_run(str(run_b_path))
    rows = compare_runs.compare_per_prompt(run_a, run_b)

    # worst regression first → p_a (delta=-0.3) before p_b (delta=+0.3)
    assert rows[0]["prompt"] == "p_a"
    assert rows[0]["delta"] < 0
    assert rows[1]["prompt"] == "p_b"
    assert rows[1]["delta"] > 0


def test_compare_per_prompt_skips_prompts_missing_from_one_run(tmp_path):
    shared = {
        "prompt": "shared",
        "base_style_score": 0.3,
        "grpo_style_score": 0.6,
        "base_completion": "b",
        "grpo_completion": "g",
    }
    only_in_a = {
        "prompt": "only_a",
        "base_style_score": 0.3,
        "grpo_style_score": 0.5,
        "base_completion": "b",
        "grpo_completion": "g",
    }

    run_a_path = tmp_path / "run_a.json"
    run_b_path = tmp_path / "run_b.json"
    _write_run(run_a_path, grpo_metrics={}, per_prompt=[shared, only_in_a])
    _write_run(run_b_path, grpo_metrics={}, per_prompt=[shared])

    run_a = compare_runs.load_run(str(run_a_path))
    run_b = compare_runs.load_run(str(run_b_path))
    rows = compare_runs.compare_per_prompt(run_a, run_b)

    assert len(rows) == 1
    assert rows[0]["prompt"] == "shared"


def test_delta_style_higher_is_better():
    assert compare_runs._delta_style("style_score_mean", 0.1) == "green"
    assert compare_runs._delta_style("style_score_mean", -0.1) == "red"


def test_delta_style_lower_is_better():
    # style_score_std is not in HIGHER_IS_BETTER → lower is better
    assert compare_runs._delta_style("style_score_std", -0.05) == "green"
    assert compare_runs._delta_style("style_score_std", 0.05) == "red"


def test_main_prints_output(tmp_path, capsys):
    per_prompt = [
        {
            "prompt": "Write about stars.",
            "base_style_score": 0.2,
            "grpo_style_score": 0.7,
            "base_completion": "base text",
            "grpo_completion": "grpo text",
        },
    ]
    run_a_path = tmp_path / "run_a.json"
    run_b_path = tmp_path / "run_b.json"
    _write_run(
        run_a_path,
        grpo_metrics={"style_score_mean": 0.6, "distinct_2": 0.5},
        per_prompt=per_prompt,
        run_id="run-a",
    )
    _write_run(
        run_b_path,
        grpo_metrics={"style_score_mean": 0.75, "distinct_2": 0.6},
        per_prompt=[{**per_prompt[0], "grpo_style_score": 0.8}],
        run_id="run-b",
    )

    sys.argv = [
        "compare_runs.py",
        str(run_a_path),
        str(run_b_path),
        "--label-a",
        "A",
        "--label-b",
        "B",
    ]  # noqa: E501
    compare_runs.main()
