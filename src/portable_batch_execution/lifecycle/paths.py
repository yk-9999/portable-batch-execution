"""Filesystem layout for lifecycle metadata under a PBE state root."""

from __future__ import annotations

from pathlib import Path

_POLICY_FILENAME = "lifecycle-policy.json"
_LIFECYCLE_DIR = "lifecycle"


def lifecycle_policy_path(state_root: Path) -> Path:
    return state_root.resolve() / _POLICY_FILENAME


def lifecycle_root(state_root: Path) -> Path:
    return state_root.resolve() / _LIFECYCLE_DIR


def deliveries_dir(state_root: Path) -> Path:
    return lifecycle_root(state_root) / "deliveries"


def holds_dir(state_root: Path) -> Path:
    return lifecycle_root(state_root) / "holds"


def deletion_receipts_dir(state_root: Path) -> Path:
    return lifecycle_root(state_root) / "deletion-receipts"


def deletion_intents_dir(state_root: Path) -> Path:
    return lifecycle_root(state_root) / "deletion-intents"


def writer_lock_path(state_root: Path) -> Path:
    return lifecycle_root(state_root) / ".writer-lock"


def artifacts_dir(state_root: Path) -> Path:
    return state_root.resolve() / "artifacts"
