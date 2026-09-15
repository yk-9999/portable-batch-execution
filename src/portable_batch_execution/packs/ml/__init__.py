"""Small, offline scikit-learn workloads for the ML batch pack."""

from .pack import FakeEncoder as FakeEncoder
from .pack import MLPack as MLPack

__all__ = ["FakeEncoder", "MLPack"]
