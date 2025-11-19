# ==== Shell ====
SHELL := bash
-include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
# ==== Environment variables ====
APP ?= app.main:app
PY := ./.venv/Scripts/python.exe
DC := docker compose
# 0 — do not start Docker automatically; 1 — start  Docker automatically and wait
AUTO_START_DOCKER ?= 1

# ==== Phony targets ====
.PHONY: help init start stop ci pull run-image tag push open-docs prune \
        install dev-install dev lint lint-fix test build run logs down open \
        run-open restart clean doctor start-docker wait-docker docker-up

# ==== Help ====
help:
	@echo "make init          - bootstrap: venv + deps + docker compose pull"
	@echo "make start         - up container(s) and open UI"
	@echo "make stop          - docker compose down"
	@echo "make ci            - local CI (ruff + pytest)"
	@echo "make pull          - docker pull ${IMAGE}"
	@echo "make run-image     - docker run -d -p ${PORT:-8000}:8000 ${IMAGE}"
	@echo "make tag           - tag local image as ${IMAGE}"
	@echo "make push          - docker push ${IMAGE}"
	@echo "make open-docs     - open http://localhost:${PORT:-8000}/docs"
	@echo "make prune         - docker system prune -f"
	@echo "make install       - install runtime deps from requirements.txt"
	@echo "make dev-install   - install dev deps (pytest, ruff)"
	@echo "make dev           - uvicorn --reload (local, no Docker)"
	@echo "make lint          - ruff check ."
	@echo "make lint-fix      - ruff check . --fix"
	@echo "make test          - pytest -q"
	@echo "make build         - docker build -t heart-risk-api:latest ."
	@echo "make run           - docker compose up -d"
	@echo "make logs          - docker compose logs -f"
	@echo "make down          - docker compose down"
	@echo "make open          - wait for port and open http://localhost:${PORT:-8000}/"
	@echo "make run-open      - run + open"
	@echo "make restart       - docker compose restart"
	@echo "make clean         - remove __pycache__ and .pytest_cache"
	@echo "make doctor        - check docker and ensure .venv exists"

# ==== Bootstrap ====
doctor:
	@command -v docker >/dev/null 2>&1 || { echo "Docker not found. Install Docker Desktop and restart terminal."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Docker Engine is not running. Start Docker Desktop and try again."; exit 1; }
	@[ -d .venv ] || python -m venv .venv

init: docker-up
	"$(PY)" -m pip install -U pip
	@if [ -f requirements.txt ]; then "$(PY)" -m pip install -r requirements.txt; fi
	@if [ -f requirements-dev.txt ]; then "$(PY)" -m pip install -r requirements-dev.txt; fi
	-$(DC) pull

# ==== One-click start/stop ====
start: run open

stop:
	$(DC) down

# ==== Local CI ====
ci:
	"$(PY)" -m ruff check .
	"$(PY)" -m pytest -q

# ==== Image ops ====
pull:
	docker pull ${IMAGE}

run-image:
	docker run -d --name heart_api -p ${PORT:-8000}:8000 --rm ${IMAGE}

tag:
	docker tag heart-risk-api:latest ${IMAGE}

push:
	docker push ${IMAGE}

open-docs:
	@start "" "http://localhost:$${PORT:-8000}/docs"

prune:
	docker system prune -f

# ==== Python deps / dev run ====
install:
	@if [ -f requirements.txt ]; then "$(PY)" -m pip install -r requirements.txt; else echo "requirements.txt not found"; fi

dev-install:
	@if [ -f requirements-dev.txt ]; then "$(PY)" -m pip install -r requirements-dev.txt; else echo "requirements-dev.txt not found"; fi

dev:
	"$(PY)" -m uvicorn $(APP) --host 127.0.0.1 --port $(PORT) --reload

# ==== Quality ====
lint:
	"$(PY)" -m ruff check .

lint-fix:
	"$(PY)" -m ruff check . --fix

test:
	"$(PY)" -m pytest -q

# ==== Docker compose ====
build:
	docker build -t heart-risk-api:latest .

run:
	$(DC) up -d

logs:
	$(DC) logs -f

down:
	$(DC) down

# Wait for HTTP then open default browser (works from Git Bash on Windows)
open:
	@until curl -sSf "http://localhost:$${PORT:-8000}/docs" >/dev/null; do sleep 1; done
	@start "" "http://localhost:$${PORT:-8000}/"

run-open: run open

restart:
	$(DC) restart

# Clean
clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# Start Docker Desktop
start-docker:
	@start "" "$${DOCKER_DESKTOP_EXE:-C:\Program Files\Docker\Docker\Docker Desktop.exe}"

# Wait until demon ready
wait-docker:
	@until docker info >/dev/null 2>&1; do echo "Waiting for Docker Engine..."; sleep 2; done

# Check and auto-start Docker if possible
docker-up:
	@if docker info >/dev/null 2>&1; then \
	  echo "Docker Engine is running."; \
	else \
	  if [ "$(AUTO_START_DOCKER)" = "1" ]; then \
	    echo "Starting Docker Desktop..."; \
	    start "" "$${DOCKER_DESKTOP_EXE:-C:\Program Files\Docker\Docker\Docker Desktop.exe}"; \
	    "$(MAKE)" wait-docker; \
	  else \
	    echo "Docker Engine is not running. Start Docker Desktop or run: make start-docker"; \
	    exit 1; \
	  fi; \
	fi