"""
Utility functions for data conversion.
"""

from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

CategoricalEncoders = Dict[str, Dict[str, int]]


def fit_categorical_encoders(df: pd.DataFrame, cat_features: List[str]) -> CategoricalEncoders:
    """Fit deterministic string-to-index mappings for categorical columns."""
    encoders: CategoricalEncoders = {}
    for feature in cat_features:
        values = sorted({str(value) for value in df[feature].dropna().tolist()})
        encoders[feature] = {value: index for index, value in enumerate(values)}
    return encoders


def dataframe_to_tensors(
    df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    device: torch.device,
    categorical_encoders: Mapping[str, Mapping[str, int]],
) -> torch.Tensor:
    """
    Convert pandas DataFrame to tensors with missing mask.

    Parameters
    ----------
    df : pandas DataFrame
        Input data.
    cat_features : list of str
        Names of categorical columns.
    num_features : list of str
        Names of numerical columns.
    device : torch.device
        Target device.

    Returns
    -------
    tensor_data : torch.Tensor
        Data tensor. Missing and unknown values remain NaN so the model can mask them.
    """
    # Ensure we have all required columns
    required_cols = cat_features + num_features
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Select only required columns in the specified order
    df = df[required_cols].copy()

    # Process each column
    processed_arrays = []

    for feat in cat_features:
        if feat not in categorical_encoders:
            raise ValueError(f"Missing categorical encoder for feature: {feat}")
        mapping = categorical_encoders[feat]
        encoded = df[feat].map(
            lambda value: np.nan if pd.isna(value) else mapping.get(str(value), np.nan)
        )
        processed_arrays.append(encoded.to_numpy().reshape(-1, 1).astype(np.float32))

    for feat in num_features:
        numeric = pd.to_numeric(df[feat], errors="raise")
        processed_arrays.append(numeric.to_numpy().reshape(-1, 1).astype(np.float32))

    # Combine all columns horizontally
    if processed_arrays:
        processed_array = np.hstack(processed_arrays)
    else:
        # If no features, create empty array with correct number of rows
        processed_array = np.empty((len(df), 0), dtype=np.float32)

    tensor_data = torch.from_numpy(processed_array).to(device)
    return tensor_data
