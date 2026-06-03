"""
Configuration schema for NAIM Simple.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class NAIMConfig(BaseModel):
    """Configuration schema for NAIM model."""
    embedding_dim: int = Field(32, gt=0, description="Dimension of feature embeddings")
    n_layers: int = Field(4, gt=0, description="Number of transformer encoder layers")
    n_heads: int = Field(4, gt=0, description="Number of attention heads")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate for optimizer")
    batch_size: int = Field(128, gt=0, description="Batch size for training")
    epochs: int = Field(200, gt=0, description="Maximum number of training epochs")
    early_stopping_patience: int = Field(20, gt=0, description="Patience for early stopping")
    missing_simulation: bool = Field(True, description="Whether to use missing simulation regularization")
    device: str = Field('auto', description="Device to use: 'cpu', 'cuda', or 'auto'")
    
    @validator('device')
    def validate_device(cls, v):
        allowed = ['cpu', 'cuda', 'auto']
        if v not in allowed:
            raise ValueError(f'device must be one of {allowed}')
        return v
    
    @validator('n_heads')
    def validate_n_heads(cls, v, values):
        if 'embedding_dim' in values and v > 0:
            if values['embedding_dim'] % v != 0:
                raise ValueError(f'n_heads ({v}) must divide embedding_dim ({values["embedding_dim"]})')
        return v