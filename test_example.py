"""
Test script for naim_simple package.
"""
import pandas as pd
import numpy as np
from naim_simple import NAIM

# Create sample data with missing values
np.random.seed(42)
n_samples = 100

# Generate categorical data
regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
crop_types = np.random.choice(['Wheat', 'Corn', 'Soybean', 'Rice'], n_samples)

# Generate numerical data
temperature = np.random.normal(20, 5, n_samples)
humidity = np.random.normal(60, 10, n_samples)
ph = np.random.normal(6.5, 0.5, n_samples)

# Introduce missing values
mask = np.random.random(n_samples) < 0.1  # 10% missing
temperature[mask] = np.nan
humidity[mask] = np.nan
ph[mask] = np.nan

# Generate target (binary classification)
yield_cat = (temperature + humidity * 0.1) > 25
yield_cat = yield_cat.astype(int)

# Create DataFrame
train_df = pd.DataFrame({
    'region': regions,
    'crop_type': crop_types,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'yield': yield_cat
})

# Test the NAIM wrapper
print("Testing NAIM Simple wrapper...")
print(f"Training data shape: {train_df.shape}")
print(f"Missing values:\n{train_df.isna().sum()}")

# Initialize model
model = NAIM(
    cat_features=['region', 'crop_type'],
    num_features=['temperature', 'humidity', 'ph'],
    embedding_dim=16,
    n_layers=2,
    n_heads=2,
    learning_rate=1e-3,
    batch_size=16,
    epochs=5,  # Small number for testing
    early_stopping_patience=3,
    missing_simulation=True,
    device='cpu'
)

# Train model
X_train = train_df[['region', 'crop_type', 'temperature', 'humidity', 'ph']]
y_train = train_df['yield']

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# Make predictions
print("\nMaking predictions...")
preds = model.predict(X_train)
probs = model.predict_proba(X_train)

print(f"Predictions shape: {preds.shape}")
print(f"Probabilities shape: {probs.shape}")
print(f"First 5 predictions: {preds[:5]}")
print(f"First 5 probabilities:\n{probs[:5]}")

# Test saving and loading
print("\nTesting save/load...")
model.save('test_model.pkl')
loaded_model = NAIM.load('test_model.pkl')

# Verify loaded model works
loaded_preds = loaded_model.predict(X_train)
print(f"Loaded model predictions match: {np.array_equal(preds, loaded_preds)}")

print("\nAll tests passed!")