# Contributing

Create a Python 3.9–3.11 environment and install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
black --check .
isort --check-only .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest --cov=naim_simple
python -m build
python -m pip check
```

Keep tests small and deterministic. Changes to saved-model metadata must update the
schema version or include a backward-compatible migration.

Releases use `v*` tags. GitHub Actions runs the supported Python matrix before
building and publishing the distributions.
