import pickle

import pytest

from naim_simple import NAIM


def test_legacy_pickle_requires_explicit_trust(fitted_model, tmp_path):
    legacy_path = tmp_path / "legacy.pkl"
    payload = {
        "model_state": fitted_model.model_.state_dict(),
        "config": {
            "cat_features": fitted_model.cat_features,
            "num_features": fitted_model.num_features,
            "embedding_dim": fitted_model.embedding_dim,
            "n_layers": fitted_model.n_layers,
            "n_heads": fitted_model.n_heads,
            "learning_rate": fitted_model.learning_rate,
            "batch_size": fitted_model.batch_size,
            "epochs": fitted_model.epochs,
            "early_stopping_patience": fitted_model.early_stopping_patience,
            "missing_simulation": fitted_model.missing_simulation,
            "device": "cpu",
        },
        "feature_names": fitted_model.feature_names_,
        "cat_idxs": fitted_model.cat_idxs_,
        "cat_dims": fitted_model.cat_dims_,
        "classes": fitted_model.classes_,
    }
    with legacy_path.open("wb") as stream:
        pickle.dump(payload, stream)

    with pytest.raises(ValueError, match="arbitrary code"):
        NAIM.load_legacy_pickle(legacy_path)
    with pytest.warns(DeprecationWarning, match="unsafe"):
        loaded = NAIM.load_legacy_pickle(legacy_path, trusted=True, device="cpu")
    assert loaded.model_ is not None
