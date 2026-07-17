# NAIM Simple

NAIM Simple is a work-in-progress scikit-learn-like wrapper for the
[NAIM model](https://github.com/cosbidev/NAIM) for tabular classification with
missing values.

## Installation

NAIM Simple supports Python 3.9 through 3.11.

```bash
git clone https://github.com/Emilio-Gordillo-Esparragoza/naim-simple.git
cd naim-simple
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Python API

```python
import pandas as pd

from naim_simple import NAIM

train = pd.read_csv("train.csv")
cat_features = ["region", "crop_type"]
num_features = ["temperature", "humidity", "ph"]
features = cat_features + num_features

model = NAIM(
    cat_features=cat_features,
    num_features=num_features,
    embedding_dim=32,
    n_layers=4,
    n_heads=4,
    batch_size=128,
    epochs=200,
    random_state=42,
)
model.fit(train[features], train["yield"])

predictions = model.predict(train[features])
probabilities = model.predict_proba(train[features])

# The model format is a directory containing JSON metadata and safe tensor weights.
model.save("yield-model.naim")
loaded = NAIM.load("yield-model.naim")
```

Categorical mappings are fitted once and stored with the model. Missing and unseen
categorical values use the model's padding path; numerical NaNs remain NaN until
the model constructs its attention mask. Prediction accepts feature columns in any
order but rejects missing or unexpected columns.

Early stopping is active only when both validation inputs are supplied:

```python
model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
```

## YAML configuration

`embedding_dim` must be divisible by `n_heads`. Unknown configuration keys are
rejected.

```yaml
cat_features: ["region", "crop_type"]
num_features: ["temperature", "humidity", "ph"]
embedding_dim: 64
n_layers: 6
n_heads: 8
learning_rate: 0.001
batch_size: 128
epochs: 200
early_stopping_patience: 20
missing_simulation: true
device: auto
random_state: 42
```

```python
model = NAIM.from_yaml("config.yaml")
model.fit(X_train, y_train)
```

## Command line

The installed commands invoke training and prediction directly:

```bash
naim-train \
  --data train.csv \
  --target yield \
  --cat region,crop_type \
  --num temperature,humidity,ph \
  --output yield-model.naim

naim-predict \
  --model yield-model.naim \
  --test test.csv \
  --out predictions.csv
```

Use `--validation-data valid.csv` during training to activate early stopping.
`--validation-target` defaults to the training target name. Add `--probs` during
prediction to write one probability column per class.

## Model format and legacy pickle migration

Normal `save` and `load` use a versioned directory bundle:

- `manifest.json` contains validated metadata, feature mappings, and a weights checksum.
- `weights.safetensors` contains tensors without executable pickle payloads.

Do not load pickle files from untrusted sources. A model produced by version 0.1.0
can be migrated only after verifying its origin:

```python
legacy = NAIM.load_legacy_pickle("old-model.pkl", trusted=True)
legacy.save("migrated-model.naim")
```

The legacy loader is deliberately separate, emits a deprecation warning, and can
execute code embedded in a malicious pickle.

## Requirements

Core dependency floors are declared in `pyproject.toml`: PyTorch 2.0, pandas 1.5,
NumPy 1.23, scikit-learn 1.2, PyYAML 6, Click 8, Pydantic 2.5, and Safetensors
0.4. CI tests Python 3.9, 3.10, and 3.11 independently.

## Development and releases

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks. Tagged releases are built
only after the full Python test matrix passes. User-visible changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
