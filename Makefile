include makefiles/help.mk
include makefiles/uv-multirepo.mk

install:  ## install software
	@echo ============= install required ASDF plugins
	asdf plugin add python
	asdf plugin add uv
	asdf plugin add pre-commit
	@echo "\033[0;32m✓ ASDF plugins installed\033[0m"

	@echo "\n============= install tool from .tool-versions"
	asdf install || true
	@echo "\033[0;32m✓ ASDF tools installed\033[0m"

	@echo "\n============= install precommit hooks"
	pre-commit install
	@echo "\033[0;32m✓ Pre-commit hooks installed\033[0m"


lint:  ## Run linter
	@echo Lint code with pre-commit checks
	pre-commit run --all-files

requirements.txt:  ## generate requirements.txt
	uv venv
	uv export > requirements.txt
