"""
Compatibility module for importing NAIM model components.
"""
from .naim.naim import NAIM
from .naim.tabular_tokenizer import CategoricalFeatureTokenizer

__all__ = ["NAIM", "CategoricalFeatureTokenizer"]