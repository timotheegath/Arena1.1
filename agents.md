# Agents

This repository is a personal implementation of the clean Transformer from Arena Chapter 1, "Transformer Interpretability", following the guide here:

<https://learn.arena.education/chapter1_transformer_interp/01_transformers/2-clean-transformer-implementation/>

The goal is to reproduce and understand a minimal Transformer implementation, step by step, rather than to build a production-ready library.

## Tools and workflow

This project is intentionally lightweight and uses a small set of tools to keep the feedback loop fast and the setup simple.

### Python and environment management with `uv`

`uv` is used as the Python package and environment manager for this repository. It provides:

- Fast installation of Python dependencies
- Lightweight, reproducible virtual environments
- Simple commands for running scripts inside the managed environment

Basic usage in this repo:

- Create / sync the environment (from `pyproject.toml`):
  - `uv sync`
- Run a script within the environment, for example the main training / demo script:
  - `uv run python main.py`

You can also install additional packages during experimentation with:

- `uv add <package-name>`

### Linting and formatting with `ruff`

`ruff` is used as the main code quality tool. It is a very fast linter (and optional formatter) for Python that helps keep the implementation clean and consistent as it evolves.

Typical `ruff` usage for this repo:

- Lint the whole codebase:
  - `ruff check .`
- (If enabled in your local config) auto-fix simple issues:
  - `ruff check . --fix`

Running `ruff` regularly while working through the Arena exercises makes it easier to focus on the Transformer logic instead of style issues.

## Project structure (high level)

- `main.py`: entry point for running experiments or demos with the clean Transformer
- `custom_transformer.py`: core model implementation, closely following the Arena clean Transformer design
- `pyproject.toml`: project metadata and dependency specification for `uv`

As you work through the Arena chapter, these files may grow with additional experiments, helpers, or visualizations, but the core idea remains: keep things small, readable, and aligned with the tutorial.
