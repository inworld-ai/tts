# Contributing to TTS

Thank you for your interest in contributing to TTS! We welcome contributions from the community and appreciate your help in making this project better.

## Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/inworld-ai/tts.git
   cd tts
   ```
3. **Set up development environment**:
   ```bash
   make install
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
5. **Make your changes** and commit them
6. **Run tests and linting**:
   ```bash
   make test
   make lint-fix
   ```
7. **Push to your fork** and submit a pull request

## Development Guidelines

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting:

```bash
# Check code style
make lint

# Auto-fix style issues
make lint-fix
```

**Key style guidelines:**
- Follow PEP 8 conventions
- Line length: 88 characters
- Use double quotes for strings
- Use type hints where appropriate
- Add docstrings for public functions and classes

### Testing

We require tests for all new features and bug fixes:

```bash
# Run all tests
make test
```

**Testing guidelines:**
- Write unit tests for new functions/classes
- Test both CUDA and CPU environments when applicable
- Test edge cases and error conditions
- Maintain or improve test coverage

### Environment Setup

**Supported environments:**
- **Python**: 3.10+
- **CUDA**: 12.4 or 12.8 (Linux only)
- **Platforms**: Linux (recommended), macOS

**Development setup:**
```bash
# Default setup (CUDA 12.8 on Linux, CPU on macOS)
make install

# Specific CUDA version
make install CUDA_VERSION=12.4
make install CUDA_VERSION=12.8
```

## Reporting Issues

When reporting issues, please include:

1. **Environment details**:
   - Python version (`python --version`)
   - CUDA version (`nvcc --version` or `nvidia-smi`)
   - Platform (Linux/macOS)
   - TTS version

2. **Clear description** of the problem
3. **Steps to reproduce** the issue
4. **Expected vs actual behavior**
5. **Error messages** (if any)
6. **Minimal code example** that reproduces the issue

## Contributing Features

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss large features before implementation
3. **Follow the project roadmap** and priorities

### Feature Development Process

1. **Design discussion**: For significant features, open an issue first
2. **Implementation**: Follow coding standards and write tests
3. **Documentation**: Update relevant docs and add examples
4. **Testing**: Test on multiple environments (CUDA versions, platforms)
5. **Pull request**: Use our PR template and provide thorough description

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines (`make lint` passes)
- [ ] All tests pass (`make test` passes)
- [ ] Documentation is updated (if applicable)
- [ ] Changes are tested on relevant environments
- [ ] Commit messages are clear and descriptive

### PR Requirements

1. **Fill out the PR template** completely
2. **Link related issues** (use "Closes #123" syntax)
3. **Provide clear description** of changes and motivation
4. **Include test results** from your environment
5. **Update documentation** if needed

### Review Process

1. **Automated checks** must pass (linting, tests)
2. **Code review** by maintainers
3. **Testing** on different environments (if needed)
4. **Approval** and merge by maintainers

## Documentation

We use several types of documentation:

- **README.md**: Project overview and quick start
- **Code comments**: For complex logic
- **Docstrings**: For public APIs
- **Examples**: In `/examples` directory (if applicable)

**Documentation guidelines:**
- Use clear, concise language
- Provide code examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## Performance Considerations

When contributing performance-related changes:

1. **Benchmark** your changes
2. **Profile** memory usage (especially CUDA memory)
3. **Test** on different hardware configurations
4. **Document** performance improvements/impacts
5. **Consider** backward compatibility

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for new audio codec
fix: resolve CUDA memory leak in inference
docs: update installation instructions
test: add tests for multi-GPU training
```

**Format:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community chat
- **Documentation**: Check existing docs first

### Recognition

Contributors are recognized in:
- Git commit history
- Release notes for significant contributions
- README acknowledgments

## Advanced Development

### Working with CUDA

When developing CUDA-related features:

```bash
# Test on both supported CUDA versions
make install CUDA_VERSION=12.4
make test

make install CUDA_VERSION=12.8
make test
```

### Memory Management

- Monitor CUDA memory usage
- Use appropriate batch sizes for testing
- Consider memory efficiency in implementations

### Release Process

For maintainers:

```bash
# Bump version
make version patch  # or minor/major

# Build and test
make test
make lint

# Tag and release (maintainers only)
```

## Contact
- **Email**: opensource@inworld.ai

---

**Happy contributing!**
