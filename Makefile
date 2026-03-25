.PHONY: setup train-trial train-full eval summarize lint format clean

setup:
	uv venv .venv
	uv pip install -e ".[dev]"

train-trial:
	python scripts/train.py --config configs/trial.yaml

train-full:
	python scripts/train.py --config configs/full.yaml

eval:
	python scripts/evaluate.py --config configs/full.yaml

summarize:
	python scripts/summarize.py --config configs/full.yaml

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	rm -rf outputs/ __pycache__ .pytest_cache .ruff_cache dist/ *.egg-info/
