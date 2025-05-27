.PHONY: build test publish clean

build: clean ## Build packages
	uv build

test: ## Run tests
	uv run --extra dev pytest

publish: build ## Publish packages to Internal PyPI
	@if [ -z "$$PYPI_URL" ]; then \
		echo "Error: PYPI_URL environment variable not set"; \
		exit 1; \
	fi
	@echo "Publishing packages to Internal PyPI at $$PYPI_URL"
	uv run --extra=dev twine upload --verbose --repository-url ${PYPI_URL} ./dist/*.whl

clean: ## Clean build artifacts
	rm -rf ./dist ./build ./*.egg-info


hatch-version: ## Hatch version: make hatch [patch|minor|major]
	@echo "Updating version"
	@uv run --extra=dev hatch version $(filter-out hatch-version,$(MAKECMDGOALS))
	@echo "Version updated successfully"
