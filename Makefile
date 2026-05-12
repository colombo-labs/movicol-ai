.PHONY: install dev train serve lint format test

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	python training/train.py

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

lint:
	ruff check app/ training/ tests/
	ruff format --check app/ training/ tests/

format:
	ruff format app/ training/ tests/

test:
	pytest tests/ -v

docker-up:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker-compose.dev.yml down
