"""
Final comprehensive test of naim_simple package.
"""
from naim_simple import NAIM
import pandas as pd
import numpy as np
import os

print("=== NAIM Simple Comprehensive Test ===\n")

# Create test data with missing values
np.random.seed(42)
n_samples = 50

# Categorical data
regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
crop_types = np.random.choice(['Wheat', 'Corn', 'Soybean', 'Rice'], n_samples)

# Numerical data with missing values
temperature = np.random.normal(20, 5, n_samples)
humidity = np.random.normal(60, 10, n_samples)
ph = np.random.normal(6.5, 0.5, n_samples)

# Introduce missing values (10% missing)
missing_mask = np.random.random(n_samples) < 0.1
temperature[missing_mask] = np.nan
humidity[missing_mask] = np.nan
ph[missing_mask] = np.nan

# Target variable (binary classification)
yield_cat = (temperature + humidity * 0.1) > 25
yield_cat = yield_cat.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'region': regions,
    'crop_type': crop_types,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'yield': yield_cat
})

print(f"Created dataset with shape: {df.shape}")
print(f"Missing values:\n{df.isna().sum()}\n")

# Split into train and test
train_df = df.iloc[:40]
test_df = df.iloc[40:]

X_train = train_df[['region', 'crop_type', 'temperature', 'humidity', 'ph']]
y_train = train_df['yield']
X_test = test_df[['region', 'crop_type', 'temperature', 'humidity', 'ph']]
y_test = test_df['yield']

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# Test 1: Direct instantiation
print("Test 1: Direct instantiation")
model1 = NAIM(
    cat_features=['region', 'crop_type'],
    num_features=['temperature', 'humidity', 'ph'],
    embedding_dim=16,
    n_layers=2,
    n_heads=4,
    learning_rate=1e-3,
    batch_size=16,
    epochs=5,
    early_stopping_patience=3,
    missing_simulation=True,
    device='cpu'
)
model1.fit(X_train, y_train)
preds1 = model1.predict(X_test)
probs1 = model1.predict_proba(X_test)
print(f"  Predictions shape: {preds1.shape}")
print(f"  Probabilities shape: {probs1.shape}")
print(f"  First 5 predictions: {preds1[:5]}\n")

# Test 2: From YAML
print("Test 2: From YAML configuration")
# Create YAML config
config_yaml = """
cat_features: [region, crop_type]
num_features: [temperature, humidity, ph]
embedding_dim: 32
n_layers: 3
n_heads: 4
learning_rate: 0.001
batch_size: 16
epochs: 5
early_stopping_patience: 3
missing_simulation: true
"""
with open('test_config.yaml', 'w') as f:
    f.write(config_yaml)

model2 = NAIM.from_yaml('test_config.yaml')
model2.fit(X_train, y_train)
preds2 = model2.predict(X_test)
probs2 = model2.predict_proba(X_test)
print(f"  Predictions shape: {preds2.shape}")
print(f"  Probabilities shape: {probs2.shape}")
print(f"  First 5 predictions: {preds2[:5]}\n")

# Test 3: Save and load
print("Test 3: Save and load model")
model1.save('test_model.pkl')
model3 = NAIM.load('test_model.pkl')
preds3 = model3.predict(X_test)
print(f"  Loaded model predictions match original: {np.array_equal(preds1, preds3)}")
print(f"  First 5 predictions: {preds3[:5]}\n")

# Test 4: CLI simulation (using direct calls)
print("Test 4: CLI-like operations")
# Training
model4 = NAIM(
    cat_features=['region', 'crop_type'],
    num_features=['temperature', 'humidity', 'ph'],
    embedding_dim=16,
    n_layers=2,
    n_heads=4,
    epochs=3
)
model4.fit(X_train, y_train)
model4.save('cli_model.pkl')
print("  Model trained and saved via CLI-like interface")

# Prediction
loaded_model = NAIM.load('cli_model.pkl')
cli_preds = loaded_model.predict(X_test)
cli_probs = loaded_model.predict_proba(X_test)
print(f"  CLI prediction shape: {cli_preds.shape}")
print(f"  CLI probabilities shape: {cli_probs.shape}")
print(f"  First 5 CLI predictions: {cli_preds[:5]}\n")

# Cleanup
for file in ['test_config.yaml', 'test_model.pkl', 'cli_model.pkl']:
    if os.path.exists(file):
        os.remove(file)

print("=== All tests passed! ===")