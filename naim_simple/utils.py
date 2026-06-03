"""
Utility functions for data conversion.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder


def dataframe_to_tensors(
    df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Data tensor with NaN replaced by appropriate values.
    missing_mask : torch.Tensor
        Binary mask indicating missing values (1 = missing, 0 = present).
    """
    # Ensure we have all required columns
    required_cols = cat_features + num_features
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Select only required columns in the specified order
    df = df[required_cols].copy()

    # Create missing mask (1 = missing, 0 = present) from original data
    missing_mask = df.isna().astype(np.float32).values

    # Process each column
    processed_arrays = []

    # Handle categorical columns
    for feat in cat_features:
        series = df[feat].copy()
        # Fill NaN with a special string that we'll encode as -1
        filled = series.fillna('__NAIM_MISSING__')
        # Use LabelEncoder to convert to integers
        le = LabelEncoder()
        encoded = le.fit_transform(filled.astype(str))
        # Find the encoding for our missing placeholder and change it to -1
        missing_encoding = np.where(le.classes_ == '__NAIM_MISSING__')[0]
        if len(missing_encoding) > 0:
            encoded[encoded == missing_encoding[0]] = -1
        processed_arrays.append(encoded.reshape(-1, 1).astype(np.float32))

    # Handle numerical columns
    for feat in num_features:
        series = df[feat].fillna(0)  # Fill NaN with 0
        processed_arrays.append(series.values.reshape(-1, 1).astype(np.float32))

    # Combine all columns horizontally
    if processed_arrays:
        processed_array = np.hstack(processed_arrays)
    else:
        # If no features, create empty array with correct number of rows
        processed_array = np.empty((len(df), 0), dtype=np.float32)

    # Convert to tensor
    tensor_data = torch.from_numpy(processed_array).to(device)
    missing_mask = torch.from_numpy(missing_mask).to(device)

    return tensor_data, missing_mask