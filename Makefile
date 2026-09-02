.PHONY: up down logs migrate revision test test-unit test-integration lint format typecheck seed loop

COMPOSE := docker compose
UV := uv run

.env:
	@cp .env.example .env
	@echo ".env created from .env.example — fill in your provider API keys."

up: .env ## postgres + api, migrations applied
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

migrate: .env
	$(COMPOSE) run --rm migrate alembic upgrade head

revision: .env
	@test -n "$(m)" || (echo 'usage: make revision m="describe the change"'; exit 1)
	$(COMPOSE) run --rm migrate alembic revision --autogenerate -m "$(m)"

test:
	$(UV) pytest

test-unit:
	$(UV) pytest tests/unit

test-integration:
	$(UV) pytest tests/integration

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

format:
	$(UV) ruff format .
	$(UV) ruff check --fix .

typecheck:
	$(UV) mypy

seed:
	@echo "seed: not implemented until phase 1 (corpus, canaries, seed attacks)"

loop:
	@echo "loop: not implemented until phase 6 (round orchestration)"
