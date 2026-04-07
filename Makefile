SHELL := /bin/bash

# Disable BuildKit Bake warning on systems without buildx.
COMPOSE := COMPOSE_BAKE=false docker compose

.PHONY: all up build down restart logs ps backend-logs frontend-logs clean deep-clean

# Default target: build + run everything in background, then print status.
all: up ps

up:
	$(COMPOSE) up --build -d
	@echo ""
	@echo "PneumoScan is starting..."
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000"
	@echo "Docs:     http://localhost:8000/docs"

build:
	$(COMPOSE) build --no-cache

down:
	$(COMPOSE) down --remove-orphans

restart: down up

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

backend-logs:
	$(COMPOSE) logs -f backend

frontend-logs:
	$(COMPOSE) logs -f frontend

# Safe cleanup for project resources.
clean:
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) rm -f

# Aggressive Docker cleanup to recover disk space.
deep-clean: clean
	docker image prune -a -f
	docker container prune -f
	docker builder prune -a -f
	docker volume prune -f
