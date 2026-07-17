import pandas as pd
import pytest

from naim_simple import NAIM


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "category": ["a", "b", "a", "b", None, "a", "b", "a"],
            "value": [0.1, 0.9, 0.2, 1.1, 0.4, None, 1.2, 0.3],
            "target": [0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def fitted_model(sample_frame):
    model = NAIM(
        cat_features=["category"],
        num_features=["value"],
        embedding_dim=4,
        n_heads=1,
        n_layers=1,
        batch_size=3,
        epochs=1,
        device="cpu",
        random_state=7,
    )
    model.fit(sample_frame[["category", "value"]], sample_frame["target"])
    return model
