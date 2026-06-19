.PHONY: install dev train clean-graph serve lint format test

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

# ML Pipeline
clean-graph:
	python -m app.scripts.clean_graph

train:
	python -m app.scripts.train_model

pipeline: clean-graph train
	@echo "✅ Full ML pipeline complete"

# Quality
lint:
	ruff check app/ tests/
	ruff format --check app/ tests/

format:
	ruff format app/ tests/

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -k "not API"

# Docker
docker-up:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker-compose.dev.yml down
