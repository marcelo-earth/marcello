.PHONY: install lint format test data verify-corpus negatives classifier probe judge sft train eval human-eval push clean

install:
	pip install -e ".[dev]"

# mirrors the CI lint job: a green `make lint` has to mean a green CI
lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest tests/ -v

data:
	python scripts/collect_data.py --config configs/data.yaml

verify-corpus:
	python scripts/verify_corpus.py

negatives:
	python scripts/generate_negatives.py --model Qwen/Qwen2.5-1.5B-Instruct

classifier:
	python scripts/train_classifier.py --config configs/classifier.yaml

# gate: must pass before spending compute on GRPO
probe:
	python scripts/sanity_probe.py --classifier outputs/classifier/best

judge:
	python scripts/train_classifier.py --config configs/classifier.yaml \
		--resplit-seed 1337 \
		--output-dir outputs/classifier/judge

sft:
	python scripts/train_sft.py --config configs/sft.yaml

train:
	python scripts/train_grpo.py --config configs/grpo.yaml

eval:
	python scripts/evaluate.py \
		--model outputs/grpo/final \
		--prompts data/eval_prompts.txt \
		--format-prompts \
		--output outputs/eval/latest.json

human-eval:
	python scripts/build_human_eval.py --generations outputs/eval/latest.json

push:
	python scripts/push_to_hub.py --all

push-dry:
	python scripts/push_to_hub.py --all --dry-run

clean:
	rm -rf outputs/grpo outputs/sft outputs/classifier outputs/eval __pycache__ .pytest_cache
