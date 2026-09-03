.PHONY: up down logs migrate revision test test-unit test-integration lint format typecheck seed corpus archive-stats loop

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
	$(UV) python -m crucible.cli.seed

corpus:
	$(UV) python -m crucible.target.reference.corpus_gen
	$(UV) python -m crucible.target.clinic.corpus_gen

findings: .env ## regenerate every generated block in docs/findings.md
	$(UV) crucible findings regenerate

findings-check: .env ## fail if a committed number differs from the stored data
	$(UV) crucible findings regenerate --check

experiments: ## list the committed experiments with time estimates
	$(UV) crucible experiment list

archive-stats:
	$(UV) python -m crucible.cli.archive

loop:
	@echo "loop: not implemented until phase 6 (round orchestration)"
