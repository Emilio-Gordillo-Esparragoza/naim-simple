"""
Main NAIM wrapper class with scikit-learn-like API.
"""
import pickle
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from ._compat import NAIM as _NAIM
from .utils import dataframe_to_tensors


class NAIM:
    """
    NAIM (Not Another Imputation Method) wrapper for classification tasks.
    Provides a scikit-learn-like interface while hiding Hydra complexity.

    Parameters
    ----------
    cat_features : list of str, optional
        Names of categorical columns.
    num_features : list of str, optional
        Names of numerical columns.
    embedding_dim : int, default=32
        Dimension of feature embeddings.
    n_layers : int, default=4
        Number of transformer encoder layers.
    n_heads : int, default=4
        Number of attention heads.
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=200
        Maximum number of training epochs.
    early_stopping_patience : int, default=20
        Patience for early stopping.
    missing_simulation : bool, default=True
        Whether to use missing simulation regularization.
    device : str, default='auto'
        Device to use: 'cpu', 'cuda', or 'auto'.
    """
    def __init__(
        self,
        cat_features: Optional[List[str]] = None,
        num_features: Optional[List[str]] = None,
        embedding_dim: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        missing_simulation: bool = True,
        device: str = 'auto'
    ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.missing_simulation = missing_simulation
        self.device = device

        # Will be set during fit
        self.model_ = None
        self.feature_names_ = None
        self.cat_idxs_ = None
        self.cat_dims_ = None
        self.classes_ = None
        self.class_to_idx_ = None
        self.idx_to_class_ = None

    def _get_device(self):
        """Get actual device based on availability."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray, list]] = None
    ) -> 'NAIM':
        """
        Fit the NAIM model.

        Parameters
        ----------
        X : pandas DataFrame
            Training data with NaN values.
        y : array-like
            Target values.
        X_val : pandas DataFrame, optional
            Validation data.
        y_val : array-like, optional
            Validation targets.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.cat_features is None or self.num_features is None:
            raise ValueError("cat_features and num_features must be set before calling fit")

        # Store feature information
        self.feature_names_ = list(X.columns)
        self.cat_idxs_ = [self.feature_names_.index(f) for f in self.cat_features]
        # For numerical features, we don't need cardinalities in the same way
        # We'll determine categorical dimensions from data
        self.cat_dims_ = [int(X[f].nunique()) for f in self.cat_features]

        # Convert targets to numpy array
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)
        self.classes_ = np.unique(y)
        # Map target values to indices for classification
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class_ = {idx: cls for idx, cls in enumerate(self.classes_)}
        y_mapped = np.array([self.class_to_idx_[val] for val in y])
        num_classes = len(self.classes_)

        # Convert data to tensors
        train_tensors = dataframe_to_tensors(
            X, self.cat_features, self.num_features, self._get_device()
        )
        X_tensor, missing_mask = train_tensors

        if X_val is not None:
            val_tensors = dataframe_to_tensors(
                X_val, self.cat_features, self.num_features, self._get_device()
            )
            X_val_tensor, missing_mask_val = val_tensors
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            elif isinstance(y_val, list):
                y_val = np.array(y_val)
            # Map validation target values to indices
            y_val_mapped = np.array([self.class_to_idx_[val] for val in y_val])
            y_val_tensor = torch.tensor(y_val_mapped, dtype=torch.long, device=self._get_device())
        else:
            X_val_tensor, missing_mask_val, y_val_tensor = None, None, None

        y_tensor = torch.tensor(y_mapped, dtype=torch.long, device=self._get_device())

        # Initialize the underlying NAIM model
        input_size = len(self.feature_names_)
        self.model_ = _NAIM(
            input_size=input_size,
            output_size=num_classes,
            cat_idxs=self.cat_idxs_,
            cat_dims=self.cat_dims_,
            d_token=self.embedding_dim,
            embedder_initialization='uniform',
            bias=False,
            mask_type=0 if self.missing_simulation else 2,  # 0 for missing simulation, 2 for no masking
            missing_value='-inf',
            num_heads=self.n_heads,
            feedforward_dim=1000,  # Default from original implementation
            dropout_rate=0.1,
            activation='relu',
            num_layers=self.n_layers,
            extractor=False
        ).to(self._get_device())

        # Training loop (simplified version - in practice we'd use the original training utilities)
        # For now, we'll implement a basic training loop
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            if X_val_tensor is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        self.best_model_state_ = self.model_.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Load best model if validation was used
        if hasattr(self, 'best_model_state_'):
            self.model_.load_state_dict(self.best_model_state_)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pandas DataFrame
            Input data.

        Returns
        -------
        preds : numpy array
            Predicted class labels.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict")
        self.model_.eval()
        tensors = dataframe_to_tensors(X, self.cat_features, self.num_features, self._get_device())
        X_tensor, _ = tensors

        with torch.no_grad():
            logits = self.model_(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Map back to original class labels
        return np.array([self.idx_to_class_[int(p)] for p in preds])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pandas DataFrame
            Input data.

        Returns
        -------
        probs : numpy array
            Predicted class probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        self.model_.eval()
        tensors = dataframe_to_tensors(X, self.cat_features, self.num_features, self._get_device())
        X_tensor, _ = tensors

        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            File path to save the model.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before saving")
        model_dict = {
            'model_state': self.model_.state_dict(),
            'config': {
                'cat_features': self.cat_features,
                'num_features': self.num_features,
                'embedding_dim': self.embedding_dim,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'missing_simulation': self.missing_simulation,
                'device': str(self.device)
            },
            'feature_names': self.feature_names_,
            'cat_idxs': self.cat_idxs_,
            'cat_dims': self.cat_dims_,
            'classes': self.classes_,
            'class_to_idx': self.class_to_idx_,
            'idx_to_class': self.idx_to_class_
        }
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)

    @classmethod
    def load(cls, path: str) -> 'NAIM':
        """
        Load a model from a file.

        Parameters
        ----------
        path : str
            File path to load the model from.

        Returns
        -------
        model : NAIM
            Loaded model instance.
        """
        with open(path, 'rb') as f:
            model_dict = pickle.load(f) # nosec B301

        # Create new instance
        model = cls(
            cat_features=model_dict['config']['cat_features'],
            num_features=model_dict['config']['num_features'],
            embedding_dim=model_dict['config']['embedding_dim'],
            n_layers=model_dict['config']['n_layers'],
            n_heads=model_dict['config']['n_heads'],
            learning_rate=model_dict['config']['learning_rate'],
            batch_size=model_dict['config']['batch_size'],
            epochs=model_dict['config']['epochs'],
            early_stopping_patience=model_dict['config']['early_stopping_patience'],
            missing_simulation=model_dict['config']['missing_simulation'],
            device=model_dict['config']['device']
        )

        # Restore attributes
        model.feature_names_ = model_dict['feature_names']
        model.cat_idxs_ = model_dict['cat_idxs']
        model.cat_dims_ = model_dict['cat_dims']
        model.classes_ = model_dict['classes']
        model.class_to_idx_ = model_dict['class_to_idx']
        model.idx_to_class_ = model_dict['idx_to_class']

        # Initialize and load model state
        input_size = len(model.feature_names_)
        num_classes = len(model.classes_)
        model.model_ = _NAIM(
            input_size=input_size,
            output_size=num_classes,
            cat_idxs=model.cat_idxs_,
            cat_dims=model.cat_dims_,
            d_token=model.embedding_dim,
            embedder_initialization='uniform',
            bias=False,
            mask_type=0 if model.missing_simulation else 2,
            missing_value='-inf',
            num_heads=model.n_heads,
            feedforward_dim=1000,
            dropout_rate=0.1,
            activation='relu',
            num_layers=model.n_layers,
            extractor=False
        ).to(model._get_device())
        model.model_.load_state_dict(model_dict['model_state'])

        return model

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NAIM':
        """
        Create a NAIM instance from a YAML configuration file.

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file.

        Returns
        -------
        model : NAIM
            Configured NAIM instance (cat_features and num_features must be set before fitting).
        """
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract cat_features and num_features from config if present
        cat_features = config.pop('cat_features', None)
        num_features = config.pop('num_features', None)
        # Create instance with remaining config
        instance = cls(**config)
        # Set cat_features and num_features if they were in the YAML
        if cat_features is not None:
            instance.cat_features = cat_features
        if num_features is not None:
            instance.num_features = num_features

        return instance
