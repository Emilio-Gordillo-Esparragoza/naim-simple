import pytest
from pydantic import ValidationError

from naim_simple import NAIM


def test_yaml_configuration_is_validated(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
cat_features: [category]
num_features: [value]
embedding_dim: 8
n_heads: 2
n_layers: 1
epochs: 1
random_state: 3
""".strip(),
        encoding="utf-8",
    )

    model = NAIM.from_yaml(config)

    assert model.cat_features == ["category"]
    assert model.embedding_dim == 8
    assert model.random_state == 3


def test_yaml_rejects_unknown_keys_and_invalid_attention_shape(tmp_path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("unknown_setting: true", encoding="utf-8")
    with pytest.raises(ValidationError):
        NAIM.from_yaml(unknown)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("embedding_dim: 64\nn_heads: 6\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="must divide"):
        NAIM.from_yaml(invalid)
