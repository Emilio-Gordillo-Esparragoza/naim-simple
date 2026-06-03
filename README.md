# NAIM Simple
[WIP]
A simplified scikit-learn-like wrapper for the NAIM (Not Another Imputation Method) model that hides all Hydra complexity.

## Installation

```bash
pip install naim_simple
```

Or install from source:

```bash
git clone https://github.com/yourusername/naim_simple.git
cd naim_simple
pip install -e .
```

## Usage

### Basic Example

```python
from naim_simple import NAIM
import pandas as pd

# Load your data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Define feature types
cat_features = ['region', 'crop_type']
num_features = ['temperature', 'humidity', 'ph']

# Initialize model
model = NAIM(
    cat_features=cat_features,
    num_features=num_features,
    embedding_dim=32,
    n_layers=4,
    n_heads=4,
    learning_rate=1e-3,
    batch_size=128,
    epochs=200,
    early_stopping_patience=20,
    missing_simulation=True,
    device='auto'
)

# Train model
X_train = train_df[cat_features + num_features]
y_train = train_df['yield']
model.fit(X_train, y_train)

# Make predictions
X_test = test_df[cat_features + num_features]
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save and load model
model.save('naim_model.pkl')
loaded_model = NAIM.load('naim_model.pkl')
```

### From YAML Configuration

Create a `config.yaml` file:

```yaml
embedding_dim: 64
n_layers: 6
n_heads: 6
learning_rate: 0.001
batch_size: 128
epochs: 200
early_stopping_patience: 20
missing_simulation: true
```

Then use it:

```python
from naim_simple import NAIM

model = NAIM.from_yaml('config.yaml')
# You still need to provide cat_features and num_features when fitting
model.fit(X_train, y_train, cat_features=['region', 'crop_type'], 
          num_features=['temperature', 'humidity', 'ph'])
```

### Command-Line Interface

Train a model:

```bash
naim-train --data train.csv --target yield --cat region,crop_type --num temp,hum,ph --output model.pkl
```

Make predictions:

```bash
naim-predict --model model.pkl --test test.csv --out predictions.csv
```

To output class probabilities instead of labels:

```bash
naim-predict --model model.pkl --test test.csv --out probabilities.csv --probs
```

## Features

- Scikit-learn-like API (`fit`, `predict`, `predict_proba`)
- Automatic handling of missing values (creates missing mask internally)
- Support for categorical and numerical features
- GPU acceleration when available
- Model saving/loading with pickle
- YAML configuration support
- Simple CLI for training and prediction
- Early stopping to prevent overfitting
- Missing simulation regularization as in the original paper

## Requirements

- Python >= 3.7
- torch >= 1.9.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- pyyaml >= 5.4.0
- click >= 8.0.0
- pydantic >= 1.8.0

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This wrapper is based on the original NAIM implementation by Cosbidev:
https://github.com/cosbidev/NAIM