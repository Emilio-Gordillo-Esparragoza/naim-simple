"""Validated YAML configuration for NAIM Simple."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NAIMConfig(BaseModel):
    """Configuration schema for a NAIM classifier."""

    model_config = ConfigDict(extra="forbid")

    cat_features: Optional[List[str]] = None
    num_features: Optional[List[str]] = None
    embedding_dim: int = Field(32, gt=0)
    n_layers: int = Field(4, gt=0)
    n_heads: int = Field(4, gt=0)
    learning_rate: float = Field(1e-3, gt=0)
    batch_size: int = Field(128, gt=0)
    epochs: int = Field(200, gt=0)
    early_stopping_patience: int = Field(20, gt=0)
    missing_simulation: bool = True
    device: str = "auto"
    random_state: Optional[int] = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        allowed = {"cpu", "cuda", "auto"}
        if value not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_attention_shape(self) -> "NAIMConfig":
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide embedding_dim " f"({self.embedding_dim})"
            )
        return self
