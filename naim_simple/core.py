"""Scikit-learn-like wrapper around the NAIM PyTorch model."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from ._compat import NAIM as _NAIM
from .utils import (
    CategoricalEncoders,
    dataframe_to_tensors,
    fit_categorical_encoders,
)


class NAIM:
    """NAIM classifier with a small, explicit training and persistence API."""

    SCHEMA_VERSION = 1
    MANIFEST_NAME = "manifest.json"
    WEIGHTS_NAME = "weights.safetensors"

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
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        self.cat_features = list(cat_features) if cat_features is not None else None
        self.num_features = list(num_features) if num_features is not None else None
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.missing_simulation = missing_simulation
        self.device = device
        self.random_state = random_state
        self._validate_parameters()

        self.model_: Optional[_NAIM] = None
        self.feature_names_: Optional[List[str]] = None
        self.cat_idxs_: Optional[List[int]] = None
        self.cat_dims_: Optional[List[int]] = None
        self.categorical_encoders_: CategoricalEncoders = {}
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx_: Optional[Dict[Any, int]] = None
        self.idx_to_class_: Optional[Dict[int, Any]] = None
        self.best_model_state_: Optional[Dict[str, torch.Tensor]] = None
        self._legacy_dynamic_encoding = False

    def _validate_parameters(self) -> None:
        positive_ints = {
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError("n_heads must divide embedding_dim")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    def _get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(self.device)

    def _validate_features(self, X: pd.DataFrame) -> None:
        if self.cat_features is None or self.num_features is None:
            raise ValueError("cat_features and num_features must be set before fitting")
        configured = self.cat_features + self.num_features
        if len(configured) != len(set(configured)):
            raise ValueError("Feature names must be unique across cat_features and num_features")
        missing = set(configured) - set(X.columns)
        extra = set(X.columns) - set(configured)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing columns: {sorted(missing)}")
            if extra:
                details.append(f"unexpected columns: {sorted(extra)}")
            raise ValueError("; ".join(details))

    def _transform(self, X: pd.DataFrame) -> torch.Tensor:
        self._validate_features(X)
        encoders = self.categorical_encoders_
        if self._legacy_dynamic_encoding:
            encoders = fit_categorical_encoders(X, self.cat_features or [])
        return dataframe_to_tensors(
            X,
            self.cat_features or [],
            self.num_features or [],
            self._get_device(),
            encoders,
        )

    def _new_model(self) -> _NAIM:
        if self.feature_names_ is None or self.classes_ is None:
            raise RuntimeError("Model metadata is incomplete")
        return _NAIM(
            input_size=len(self.feature_names_),
            output_size=len(self.classes_),
            cat_idxs=self.cat_idxs_ or [],
            cat_dims=self.cat_dims_ or [],
            d_token=self.embedding_dim,
            embedder_initialization="uniform",
            bias=False,
            mask_type=0 if self.missing_simulation else 2,
            missing_value="-inf",
            num_heads=self.n_heads,
            feedforward_dim=1000,
            dropout_rate=0.1,
            activation="relu",
            num_layers=self.n_layers,
            extractor=False,
        ).to(self._get_device())

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray, list]] = None,
    ) -> "NAIM":
        """Fit the classifier."""
        self._validate_features(X)
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if len(X) == 0:
            raise ValueError("Training data must contain at least one row")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self.feature_names_ = (self.cat_features or []) + (self.num_features or [])
        self.cat_idxs_ = list(range(len(self.cat_features or [])))
        self.categorical_encoders_ = fit_categorical_encoders(X, self.cat_features or [])
        self.cat_dims_ = [
            len(self.categorical_encoders_[feature]) for feature in self.cat_features or []
        ]

        target = np.asarray(y)
        self.classes_ = np.unique(target)
        self.class_to_idx_ = {value: index for index, value in enumerate(self.classes_)}
        self.idx_to_class_ = {index: value for value, index in self.class_to_idx_.items()}
        y_mapped = np.array([self.class_to_idx_[value] for value in target])

        X_tensor = self._transform(X)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long, device=self._get_device())
        dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            self._validate_features(X_val)
            unknown_labels = set(np.asarray(y_val)) - set(self.classes_)
            if unknown_labels:
                raise ValueError(f"Validation data contains unknown labels: {unknown_labels}")
            X_val_tensor = self._transform(X_val)
            y_val_mapped = [self.class_to_idx_[value] for value in np.asarray(y_val)]
            y_val_tensor = torch.tensor(y_val_mapped, dtype=torch.long, device=self._get_device())

        self.model_ = self._new_model()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience_counter = 0
        self.best_model_state_ = None

        for _ in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            if X_val_tensor is not None and y_val_tensor is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(X_val_tensor), y_val_tensor).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state_ = copy.deepcopy(self.model_.state_dict())
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if self.best_model_state_ is not None:
            self.model_.load_state_dict(self.best_model_state_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.model_ is None or self.idx_to_class_ is None:
            raise RuntimeError("Model must be fitted before calling predict")
        self.model_.eval()
        with torch.no_grad():
            predictions = torch.argmax(self.model_(self._transform(X)), dim=1).cpu().numpy()
        return np.array([self.idx_to_class_[int(index)] for index in predictions])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._transform(X))
            return torch.softmax(logits, dim=1).cpu().numpy()

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Class label {value!r} is not JSON serializable")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _package_version() -> str:
        try:
            return version("naim-simple")
        except PackageNotFoundError:
            return "0.1.0"

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save a model as a safe, versioned directory bundle."""
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("Model must be fitted before saving")
        destination = Path(path)
        if destination.exists() and not destination.is_dir():
            raise ValueError("Model path must be a directory, not an existing file")
        destination.mkdir(parents=True, exist_ok=True)

        weights_path = destination / self.WEIGHTS_NAME
        temporary_weights = destination / f".{self.WEIGHTS_NAME}.tmp"
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.model_.state_dict().items()
        }
        save_file(state, temporary_weights)
        os.replace(temporary_weights, weights_path)

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "naim_simple_version": self._package_version(),
            "weights_file": self.WEIGHTS_NAME,
            "weights_sha256": self._sha256(weights_path),
            "config": {
                "cat_features": self.cat_features,
                "num_features": self.num_features,
                "embedding_dim": self.embedding_dim,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "missing_simulation": self.missing_simulation,
                "device": "auto",
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names_,
            "cat_idxs": self.cat_idxs_,
            "cat_dims": self.cat_dims_,
            "categorical_encoders": self.categorical_encoders_,
            "classes": [self._json_value(value) for value in self.classes_],
        }
        manifest_path = destination / self.MANIFEST_NAME
        temporary_manifest = destination / f".{self.MANIFEST_NAME}.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike[str]],
        device: str = "auto",
    ) -> "NAIM":
        """Load a model from a safe directory bundle."""
        source = Path(path)
        manifest_path = source / cls.MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"Missing model manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Model manifest is unreadable or invalid") from exc
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported model schema version: {manifest.get('schema_version')}")

        if manifest.get("weights_file") != cls.WEIGHTS_NAME:
            raise ValueError("Model manifest references an unsupported weights file")
        weights_path = source / cls.WEIGHTS_NAME
        if not weights_path.is_file():
            raise ValueError(f"Missing model weights: {weights_path}")
        if cls._sha256(weights_path) != manifest.get("weights_sha256"):
            raise ValueError("Model weights checksum does not match the manifest")

        config = dict(manifest["config"])
        config["device"] = device
        model = cls(**config)
        model.feature_names_ = list(manifest["feature_names"])
        model.cat_idxs_ = list(manifest["cat_idxs"])
        model.cat_dims_ = list(manifest["cat_dims"])
        model.categorical_encoders_ = {
            feature: {value: int(index) for value, index in mapping.items()}
            for feature, mapping in manifest["categorical_encoders"].items()
        }
        model.classes_ = np.asarray(manifest["classes"])
        model.class_to_idx_ = {value: index for index, value in enumerate(model.classes_)}
        model.idx_to_class_ = {index: value for value, index in model.class_to_idx_.items()}
        model.model_ = model._new_model()
        model.model_.load_state_dict(load_file(weights_path, device=str(model._get_device())))
        return model

    @classmethod
    def load_legacy_pickle(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        trusted: bool = False,
        device: str = "auto",
    ) -> "NAIM":
        """Load the old pickle format only after an explicit trust decision."""
        if not trusted:
            raise ValueError(
                "Legacy pickle loading can execute arbitrary code; pass trusted=True "
                "only for an artifact you created and control"
            )
        warnings.warn(
            "Legacy pickle models are unsafe and deprecated; save the loaded model "
            "immediately in the new directory format.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Isolated behind the explicit trusted=True guard for legacy migration.
        import pickle  # nosec B403

        with Path(path).open("rb") as stream:
            # The caller has explicitly asserted that this legacy artifact is trusted.
            payload = pickle.load(stream)  # nosec B301
        config = dict(payload["config"])
        config["device"] = device
        model = cls(**config)
        model.feature_names_ = list(payload["feature_names"])
        model.cat_idxs_ = list(payload["cat_idxs"])
        model.cat_dims_ = list(payload["cat_dims"])
        model.classes_ = np.asarray(payload["classes"])
        model.class_to_idx_ = {value: index for index, value in enumerate(model.classes_)}
        model.idx_to_class_ = {index: value for value, index in model.class_to_idx_.items()}
        model._legacy_dynamic_encoding = True
        model.model_ = model._new_model()
        model.model_.load_state_dict(payload["model_state"])
        return model

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, os.PathLike[str]]) -> "NAIM":
        """Create a validated model instance from YAML."""
        import yaml

        from .config_schema import NAIMConfig

        with Path(yaml_path).open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
        if not isinstance(raw_config, dict):
            raise ValueError("YAML configuration must contain a mapping")
        config = NAIMConfig.model_validate(raw_config)
        return cls(**config.model_dump())
