"""
Test YAML configuration loading.
"""
from naim_simple import NAIM
import pandas as pd
import numpy as np

# Load sample data
df = pd.read_csv('sample_data.csv')
X = df[['region', 'crop_type', 'temperature', 'humidity', 'ph']]
y = df['yield']

# Test from_yaml
print("Testing from_yaml...")
model = NAIM.from_yaml('config.yaml')
# Set the features (since they're not in the YAML)
model.cat_features = ['region', 'crop_type']
model.num_features = ['temperature', 'humidity', 'ph']
print(f"Model created with embedding_dim={model.embedding_dim}, n_layers={model.n_layers}")

# Train the model
model.fit(X, y)
print("Model trained successfully!")

# Make predictions
preds = model.predict(X)
print(f"Predictions shape: {preds.shape}")
print(f"First 5 predictions: {preds[:5]}")