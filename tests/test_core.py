import json

import numpy as np
import pandas as pd
import pytest

from naim_simple import NAIM


def test_fit_predict_and_probability_shapes(fitted_model, sample_frame):
    features = sample_frame[["category", "value"]]

    predictions = fitted_model.predict(features)
    probabilities = fitted_model.predict_proba(features)

    assert predictions.shape == (len(features),)
    assert probabilities.shape == (len(features), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_feature_order_is_canonical_and_input_is_validated(fitted_model, sample_frame):
    assert fitted_model.feature_names_ == ["category", "value"]
    assert fitted_model.cat_idxs_ == [0]

    shuffled = sample_frame[["value", "category"]]
    np.testing.assert_array_equal(
        fitted_model.predict(shuffled),
        fitted_model.predict(sample_frame[["category", "value"]]),
    )

    with pytest.raises(ValueError, match="unexpected columns"):
        fitted_model.predict(sample_frame[["category", "value", "target"]])


def test_unknown_categories_and_missing_values_use_missing_path(fitted_model):
    frame = pd.DataFrame({"category": ["never-seen", None], "value": [None, 0.25]})

    probabilities = fitted_model.predict_proba(frame)

    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()


def test_save_load_round_trip_uses_versioned_safe_bundle(fitted_model, sample_frame, tmp_path):
    destination = tmp_path / "model.naim"
    expected = fitted_model.predict_proba(sample_frame[["category", "value"]])

    fitted_model.save(destination)
    loaded = NAIM.load(destination, device="cpu")

    assert (destination / "manifest.json").is_file()
    assert (destination / "weights.safetensors").is_file()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    np.testing.assert_allclose(
        loaded.predict_proba(sample_frame[["category", "value"]]),
        expected,
        atol=1e-7,
    )


def test_load_rejects_corrupted_weights(fitted_model, tmp_path):
    destination = tmp_path / "model.naim"
    fitted_model.save(destination)
    weights = destination / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        NAIM.load(destination)


def test_load_rejects_unsupported_schema(fitted_model, tmp_path):
    destination = tmp_path / "model.naim"
    fitted_model.save(destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported model schema"):
        NAIM.load(destination)


def test_validation_labels_must_be_known(sample_frame):
    model = NAIM(
        cat_features=["category"],
        num_features=["value"],
        embedding_dim=4,
        n_heads=1,
        n_layers=1,
        epochs=1,
        device="cpu",
    )
    X = sample_frame[["category", "value"]]
    with pytest.raises(ValueError, match="unknown labels"):
        model.fit(X, sample_frame["target"], X_val=X.iloc[:1], y_val=[99])
