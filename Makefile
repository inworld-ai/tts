.PHONY:
.DEFAULT_GOAL := help
# print help annotations for each command in list of entries
define PRINT_HELP_PYSCRIPT
import re
import sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_.-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

SRC_DIR := tts

.PHONY: help
help:
	@echo "===== All tasks ====="
	@cat $(MAKEFILE_LIST) | $(PYTHON) -c "$$PRINT_HELP_PYSCRIPT"


install: ## Install the project python environment
	./setup/setup_python.sh


version: ## Print the project version or bump it (usage: make version [major|minor|patch])
	$(eval VERSION_ARGS := $(filter-out version,$(MAKECMDGOALS)))
	$(if $(VERSION_ARGS),uv run --extra dev hatch version $(VERSION_ARGS),uv run --extra dev hatch version)

# Prevent Make from trying to build major/minor/patch as separate targets
major minor patch:
	@:

test: ## Run unit test with coverage
	uv run --extra dev pytest --cov=tts --cov-report=xml tests/unit


lint: ## Run the linter
	uv run ruff check $(SRC_DIR)


lint-fix: ## Run the linter with fix
	uv run ruff check $(SRC_DIR) --fix
