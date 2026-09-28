"""Atomic durable persistence with fsync and post-write verification."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any


def persist_verified_bytes(
    data: bytes,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    if len(data) != expected_size_bytes:
        raise ValueError("size mismatch before persist")
    digest = f"sha256:{sha256(data).hexdigest()}"
    if digest != expected_sha256:
        raise ValueError("digest mismatch before persist")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    _fsync_parent(destination)
    verified = destination.read_bytes()
    if len(verified) != expected_size_bytes:
        raise ValueError("post-persist size verification failed")
    if f"sha256:{sha256(verified).hexdigest()}" != expected_sha256:
        raise ValueError("post-persist digest verification failed")


def verify_file(destination: Path, *, expected_sha256: str, expected_size_bytes: int) -> None:
    data = destination.read_bytes()
    if len(data) != expected_size_bytes:
        raise ValueError("size verification failed")
    if f"sha256:{sha256(data).hexdigest()}" != expected_sha256:
        raise ValueError("digest verification failed")


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically with flush, fsync, replace, and parent fsync."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_parent(path)


def _fsync_parent(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        return
    try:
        fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
