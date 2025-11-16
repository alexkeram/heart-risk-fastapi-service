# ==== Environment variables ====
PORT ?= 8000
APP ?= app.main:app
IMAGE ?= heart-api
PY := .venv/Scripts/python.exe
DC := docker compose

.PHONY: help init install dev-install dev lint lint-fix test build run logs down open run-open restart clean

# Goals
help:
	@echo "make init          - preparing: venv + deps + docker compose pull"
	@echo "make start         - start a container and open a page"
	@echo "make stop          - docker compose down"
	@echo "make ci            - local CI (ruff + pytest)"
	@echo "make pull          - docker pull $(IMAGE)"
	@echo "make run-image     - docker run -d -p $(PORT):8000 $(IMAGE)"
	@echo "make open-docs     - open http://localhost:$(PORT)/docs"
	@echo "make prune         - docker system prune (cleaning)"
	@echo "make install       - install runtime-dependencies from requirements.txt"
	@echo "make dev-install   - install dev-dependencies (pytest, ruff)"
	@echo "make dev           - local start uvicorn --reload"
	@echo "make lint          - check code with ruff"
	@echo "make lint-fix      - fix code with ruff"
	@echo "make test          - pytest"
	@echo "make build         - docker build image heart-risk-api:latest"
	@echo "make run           - docker compose up -d"
	@echo "make logs          - docker compose logs -f"
	@echo "make down          - docker compose down"
	@echo "make open          - open page http://localhost:$(PORT)/"
	@echo "make run-open      - run + open"
	@echo "make restart       - docker compose restart"
	@echo "make clean         - clear cache __pycache__ and .pytest_cache"

# Full bootstrap: venv + dependencies + image (if in compose specified image:)
init: doctor
	$(PY) -m pip install -U pip
	@if exist requirements.txt $(PY) -m pip install -r requirements.txt
	@if exist requirements-dev.txt $(PY) -m pip install -r requirements-dev.txt
	-$(DC) pull

# Start container and open UI
start:
	$(DC) up -d
	powershell -NoProfile -Command "while (-not (Test-NetConnection localhost -Port $(PORT) -InformationLevel Quiet)) { Start-Sleep 1 }; Start-Process 'http://localhost:$(PORT)/'"

# Stop container
stop:
	$(DC) down

# Local CI
ci:
	$(PY) -m ruff check .
	$(PY) -m pytest -q

# Download image from dockerhub
pull:
	docker pull $(IMAGE)

# Start image bypassing compose
run-image:
	docker run -d --name heart_api -p $(PORT):8000 --rm $(IMAGE)

# Mark local created image with tag
# Example: make tag IMAGE=docker.io/you/heart-risk-api:0.1
tag:
	docker tag heart-risk-api:latest $(IMAGE)

# Push image to dockerhub
push:
	docker push $(IMAGE)


open-docs:
	powershell -NoProfile -Command "Start-Process 'http://localhost:$(PORT)/docs'"

prune:
	docker system prune -f


install:
	$(PY) -m pip install -r requirements.txt

dev-install:
	$(PY) -m pip install -r requirements-dev.txt

dev:
	$(PY) -m uvicorn $(APP) --host 127.0.0.1 --port $(PORT) --reload

lint:
	$(PY) -m ruff check .

lint-fix:
	$(PY) -m ruff check . --fix

test:
	$(PY) -m pytest -q

build:
	docker build -t heart-risk-api:latest .

run:
	$(DC) up -d

logs:
	$(DC) logs -f

down:
	$(DC) down

open:
	powershell -NoProfile -Command "while (-not (Test-NetConnection localhost -Port $(PORT) -InformationLevel Quiet)) { Start-Sleep 1 }; Start-Process 'http://localhost:$(PORT)/'"

run-open: run open

restart:
	$(DC) restart

clean:
	-powershell -NoProfile -Command "if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache }"
	-powershell -NoProfile -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"

doctor:
	-powershell -NoProfile -Command "if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Write-Error 'Docker not found. Install Docker Desktop and restart terminal.'; exit 1 }"
	-powershell -NoProfile -Command "if (-not (Test-Path .venv)) { Write-Host 'Creating .venv'; python -m venv .venv } else { Write-Host '.venv already exists' }"
