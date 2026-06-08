"""
Test YAML configuration loading with features included.
"""

from naim_simple import NAIM
import pandas as pd
import numpy as np

# Load sample data
df = pd.read_csv("tests/sample_data.csv")
X = df[["region", "crop_type", "temperature", "humidity", "ph"]]
y = df["yield"]

# Test from_yaml with features in config
print("Testing from_yaml with features in config...")
model = NAIM.from_yaml("config.yaml")
print(
    f"Model created with embedding_dim={model.embedding_dim}, n_layers={model.n_layers}"
)
print(f"Cat features: {model.cat_features}")
print(f"Num features: {model.num_features}")

# Train the model
model.fit(X, y)
print("Model trained successfully!")

# Make predictions
preds = model.predict(X)
print(f"Predictions shape: {preds.shape}")
print(f"First 5 predictions: {preds[:5]}")

# Test predict_proba
probs = model.predict_proba(X)
print(f"Probabilities shape: {probs.shape}")
print(f"First 5 probabilities:\n{probs[:5]}")
