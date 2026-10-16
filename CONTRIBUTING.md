# Contributing

Contributions are welcome, from people and from autonomous agents. The review bar is the same for both: it has to run, and the PR has to report only what was actually executed.

## 1. The gate

Before opening a PR, both of these must pass locally:

```bash
make lint
pytest tests/
```

This is not optional. CI does not run automatically on pull requests from forks (see #35), so a clean PR page does not mean the checks passed. It means they did not run.

## 2. Testing claims

Under "Testing" in the PR description, list only commands you executed, with their real output. If something could not be run, say so and say why. An unverifiable claim is worse than an admitted gap, because it costs the maintainer a full local reproduction to discover.

## 3. New scripts ship with tests

Structure scripts so the logic is testable without network access or a model download. A function that takes a list of token lengths is testable. One that calls `AutoTokenizer.from_pretrained` internally is not.

## 4. Read recorded values, do not recompute them

Analysis tools consume what a run actually produced. If a tool recomputes a quantity, it must name the output so it cannot be mistaken for the recorded one. A drift detector that silently substitutes a recomputation for the recorded value cannot detect drift.

## 5. Do not hardcode config constants

Values such as `target_length`, `max_completion_length` and the corpus reference percentiles belong in `configs/grpo.yaml` or in the script that measured them.

## 6. Licensing

Do not add per-file copyright headers claiming reserved rights. Contributions are under the repository license.

## 7. Claiming an issue

Comment `/claim` on the issue. It will be honoured, since GitHub does not allow assigning non-collaborators. If you drop the work, say so on the issue so it frees up.

## 8. AI disclosure

If a PR comes from an autonomous agent, say so in the description. Agent contributors are listed as such in the README.

## For maintainers: checking an external PR

```bash
git fetch origin pull/<N>/head:pr-<N> && git checkout pr-<N>
make lint && pytest tests/
```

Do not move the CI workflow to `pull_request_target` to get checks on forks. That runs untrusted PR code with repository write permissions.
