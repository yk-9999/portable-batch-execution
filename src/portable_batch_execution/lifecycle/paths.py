"""Filesystem layout for lifecycle metadata under a PBE state root."""

from __future__ import annotations

from pathlib import Path

_LIFECYCLE_DIR = "lifecycle"


def lifecycle_root(state_root: Path) -> Path:
    return state_root.resolve() / _LIFECYCLE_DIR


def writer_lock_path(state_root: Path) -> Path:
    return lifecycle_root(state_root) / ".writer-lock"
