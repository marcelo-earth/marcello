.PHONY: install lint format test data classifier train eval push clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/ -v

data:
	python scripts/collect_data.py --config configs/data.yaml

classifier:
	python scripts/train_classifier.py --config configs/classifier.yaml

train:
	python scripts/train_grpo.py --config configs/grpo.yaml

eval:
	python scripts/evaluate.py \
		--model outputs/grpo/final \
		--prompts data/eval_prompts.txt \
		--format-prompts \
		--output outputs/eval/latest.json

push:
	python scripts/push_to_hub.py --all

push-dry:
	python scripts/push_to_hub.py --all --dry-run

clean:
	rm -rf outputs/grpo outputs/classifier outputs/eval __pycache__ .pytest_cache
