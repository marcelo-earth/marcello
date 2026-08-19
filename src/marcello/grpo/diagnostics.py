"""Group-level reward diagnostics for GRPO.

GRPO turns rewards into advantages as `A_i = (r_i - mean(r)) / std(r)`, computed
inside each group of completions that share one prompt. What moves the policy is
the spread inside a group, never the absolute reward. If all G completions land in
a narrow band, the advantages are noise divided by a tiny standard deviation and
the step carries almost nothing, at any learning rate. A run can hold a healthy
mean reward for hours and be learning nothing at all, and nothing in the training
log would say so. Issue #18.

The same argument applies per component: a component that is near-constant across
the completions of a group contributes no spread, so its weight is doing no work in
the objective however sensible the number looks in the config.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# below this, a group's completions are close enough that the normalized advantages
# are dominated by whatever noise is in the reward rather than by real differences
DEAD_GROUP_STD = 0.01


@dataclass
class GroupReward:
    """What one group of completions contributed to the gradient."""

    size: int
    mean: float
    std: float
    minimum: float
    maximum: float
    component_std: dict[str, float] = field(default_factory=dict)

    @property
    def is_dead(self) -> bool:
        """Whether this group is too flat to have taught the policy anything."""
        return self.std < DEAD_GROUP_STD

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "is_dead": self.is_dead,
            "component_std": self.component_std,
        }


def group_spans(prompts: list[str] | None, batch_size: int, group_size: int) -> list[range]:
    """Split a scored batch into the groups GRPO will compute advantages over.

    Grouping follows the prompts when they are available, since that is what
    actually defines a group, and falls back to fixed chunks of `group_size` when
    TRL did not hand any over. A trailing partial chunk is kept: it is still a group
    the trainer will normalize over, and dropping it would hide a flat one.
    """
    if not prompts or len(prompts) != batch_size:
        return [
            range(start, min(start + group_size, batch_size))
            for start in range(0, batch_size, max(1, group_size))
        ]

    spans: list[range] = []
    start = 0
    for index in range(1, batch_size + 1):
        if index == batch_size or prompts[index] != prompts[start]:
            spans.append(range(start, index))
            start = index
    return spans


def _spread(values: list[float]) -> float:
    """Sample standard deviation, matching what GRPO divides the advantages by."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_group(breakdowns: list[dict]) -> GroupReward:
    """Reduce one group's per-completion reward breakdowns to its group statistics."""
    totals = [b["total"] for b in breakdowns]
    components = [key for key in breakdowns[0] if key not in {"total", "raw_style_prob"}]
    return GroupReward(
        size=len(totals),
        mean=statistics.fmean(totals),
        std=_spread(totals),
        minimum=min(totals),
        maximum=max(totals),
        component_std={key: _spread([b[key] for b in breakdowns]) for key in components},
    )


class RewardVarianceLog:
    """Accumulates group reward spread across a run and reports what it found."""

    def __init__(self, group_size: int, output_dir: str | Path | None = None):
        self.group_size = group_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.groups: list[GroupReward] = []

    def record(self, breakdowns: list[dict], prompts: list[str] | None = None) -> list[GroupReward]:
        """Split one scored batch into groups and keep each group's statistics."""
        batch = [
            summarize_group([breakdowns[i] for i in span])
            for span in group_spans(prompts, len(breakdowns), self.group_size)
            if len(span) > 0
        ]
        self.groups.extend(batch)
        return batch

    @staticmethod
    def format_batch(groups: list[GroupReward]) -> str:
        """A one-line report for a scored batch, meant for the training log."""
        if not groups:
            return "reward groups: none"
        stds = [group.std for group in groups]
        dead = sum(1 for group in groups if group.is_dead)
        line = (
            f"reward groups: {len(groups)} | "
            f"std mean {statistics.fmean(stds):.4f} min {min(stds):.4f} max {max(stds):.4f} | "
            f"reward mean {statistics.fmean([g.mean for g in groups]):.4f}"
        )
        if dead:
            line += f" | {dead} flat (std < {DEAD_GROUP_STD}), no gradient from those"
        return line

    def summary(self) -> dict:
        """Run-level summary, including which components never varied within a group."""
        if not self.groups:
            return {"groups": 0}

        stds = [group.std for group in self.groups]
        dead = [group for group in self.groups if group.is_dead]
        component_keys = sorted({key for group in self.groups for key in group.component_std})
        component_std_mean = {
            key: statistics.fmean([group.component_std.get(key, 0.0) for group in self.groups])
            for key in component_keys
        }
        return {
            "groups": len(self.groups),
            "group_std_mean": statistics.fmean(stds),
            "group_std_min": min(stds),
            "group_std_max": max(stds),
            "group_reward_mean": statistics.fmean([group.mean for group in self.groups]),
            "dead_groups": len(dead),
            "dead_group_fraction": len(dead) / len(self.groups),
            "dead_group_std_threshold": DEAD_GROUP_STD,
            "component_std_mean": component_std_mean,
            # a component whose within-group spread is always zero is weight the
            # objective never spends: it shifts every completion in the group equally
            "components_without_spread": [
                key for key, value in component_std_mean.items() if value == 0.0
            ],
        }

    def format_summary(self) -> str:
        """A readable verdict for the end of a run."""
        summary = self.summary()
        if not summary.get("groups"):
            return "No reward groups were recorded, so nothing can be said about the spread."

        lines = [
            f"Reward groups scored: {summary['groups']}",
            f"  group reward std: mean {summary['group_std_mean']:.4f}, "
            f"min {summary['group_std_min']:.4f}, max {summary['group_std_max']:.4f}",
            f"  group reward mean: {summary['group_reward_mean']:.4f}",
            f"  flat groups (std < {DEAD_GROUP_STD}): {summary['dead_groups']} "
            f"({summary['dead_group_fraction']:.1%})",
        ]
        for key, value in sorted(summary["component_std_mean"].items(), key=lambda item: -item[1]):
            lines.append(f"    within-group std of {key}: {value:.4f}")
        if summary["components_without_spread"]:
            lines.append(
                "  components that never varied within a group, so their weights did "
                f"no work: {', '.join(summary['components_without_spread'])}"
            )
        if summary["dead_group_fraction"] > 0.5:
            lines.append(
                "  Most groups were too flat to produce a gradient. The reward is not "
                "separating completions of the same prompt, so no learning rate will "
                "fix this: change the reward, not the optimizer."
            )
        return "\n".join(lines)

    def write(self, filename: str = "reward_variance.json") -> Path | None:
        """Persist the per-group record next to the run, or return None without one."""
        if not self.output_dir:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        path.write_text(
            json.dumps(
                {
                    "summary": self.summary(),
                    "groups": [group.as_dict() for group in self.groups],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
