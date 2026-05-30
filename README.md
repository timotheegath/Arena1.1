# Arena 1.1 – Clean Transformer implementation

This repository is a personal implementation of the clean Transformer from Arena Chapter 1, "Transformer Interpretability", closely following the tutorial here:

<https://learn.arena.education/chapter1_transformer_interp/01_transformers/2-clean-transformer-implementation/>

The aim is to understand and re‑implement the core components of a GPT‑2–style Transformer, with simple tests around each layer to check that the implementation behaves as expected.

## Getting started

This project uses `uv` for Python environment and dependency management.

1. Install `uv` (see the `uv` docs for your platform).
2. From the repo root, create/sync the environment:
   - `uv sync`
3. Run the main script:
   - `uv run python main.py`

You can add extra packages for experiments with:

- `uv add <package-name>`

## Code structure

- `custom_transformer.py`: core implementation of the custom Transformer building blocks (LayerNorm, embedding layers, etc.)
- `main.py`: entry point for running experiments or quick demos with the model
- `agents.md`: notes on tools (`uv`, `ruff`) and overall project context
- `pyproject.toml`: project metadata and dependencies for `uv`

## Layer-level test functions

Each custom layer in `custom_transformer.py` is written to be easy to test in isolation.

- A `Tests` helper class provides simple utilities like:
  - `rand_float_test` and `rand_int_test` to check shapes and basic behaviour of a layer given random inputs
  - `load_gpt2_test` to compare a custom layer’s output against the corresponding pretrained GPT‑2 layer
- Individual layers define small static test helpers that call into `Tests`, for example:
  - `LayerNorm.test(sentence: str)` runs the custom `LayerNorm` on cached GPT‑2 activations and checks that the outputs match GPT‑2’s final layer norm
  - `Embed.test(sentence: str)` checks that the custom embedding layer matches GPT‑2’s embedding on a tokenized sentence

The idea is that you can iteratively implement or modify a layer, then quickly run its test function to confirm that:

- Input and output shapes look sensible
- The layer does not modify inputs in place
- The output is numerically close to GPT‑2’s behaviour where applicable

These tests keep the feedback loop tight while working through the Arena chapter.

## Tooling – `uv` and `ruff`

The tooling is intentionally minimal:

- `uv` manages the Python environment and dependencies, keeping setup fast and reproducible.
- `ruff` is used as a linter (and optionally formatter) to maintain a consistent, clean code style while iterating on the implementation. Typical commands are:
  - `ruff check .`
  - `ruff check . --fix`

For more detail on the tools and workflow, see `agents.md`.
