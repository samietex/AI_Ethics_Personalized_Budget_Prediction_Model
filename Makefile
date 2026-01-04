.PHONY: install lint test train

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

train:
	python -m budget_fairness.train --data-path ./data/udacity_ai_ethics_project_data.csv --model logreg
