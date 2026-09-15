"""Small, offline scikit-learn workloads for the ML batch pack."""

from .char_wb_tfidf_logistic_score import (
    execute_char_wb_tfidf_logistic_score as execute_char_wb_tfidf_logistic_score,
)
from .pack import FakeEncoder as FakeEncoder
from .pack import MLPack as MLPack

__all__ = ["FakeEncoder", "MLPack", "execute_char_wb_tfidf_logistic_score"]
