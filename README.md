# Inworld TTS Repo

## Development Commands

This project uses Make commands for common development tasks. Here are the available commands:

### Building and Testing
- `make build` - Build project packages
- `make test` - Run test suite
- `make clean` - Clean build artifacts
- `make publish` - Publish packages to Internal PyPI

### Development Setup
- `make install` - Install project dependencies
- `make requirements.txt` - Generate requirements.txt file

### Code Quality
- `make lint` - Run code linter

### Version Management
- `make hatch-version` - Update version using Hatch (options: patch|minor|major)
