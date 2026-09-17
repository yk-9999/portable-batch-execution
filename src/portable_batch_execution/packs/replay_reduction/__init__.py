"""Generic replay reduction primitives (structural canonicalize, event windows)."""

from .canonicalize import StructuralCanonicalizeError, execute_structural_canonicalize
from .event_window import execute_event_window_extract
from .pack import ReplayReductionPack

__all__ = [
    "ReplayReductionPack",
    "StructuralCanonicalizeError",
    "execute_event_window_extract",
    "execute_structural_canonicalize",
]
