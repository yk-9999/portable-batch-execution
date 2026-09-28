"""Lifecycle policy loaded from the PBE state root (not project code)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portable_batch_execution.lifecycle.paths import lifecycle_policy_path

_POLICY_SCHEMA = "pbe.lifecycle-policy.v1"


def _string_list_field(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = payload.get(key, [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise TypeError(f"{key} must be an array of strings")
    return tuple(raw)


@dataclass(frozen=True)
class LifecyclePolicy:
    schema_version: str
    delivery_grace_seconds: int
    legacy_retention_seconds: int
    authoritative_reference_files: tuple[str, ...] = ()
    authoritative_reference_roots: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> LifecyclePolicy:
        if payload.get("schema_version") != _POLICY_SCHEMA:
            raise ValueError("unsupported lifecycle policy schema_version")
        grace = int(payload["delivery_grace_seconds"])
        legacy = int(payload["legacy_retention_seconds"])
        if grace < 0 or legacy < 0:
            raise ValueError("retention intervals must be non-negative")
        return cls(
            schema_version=_POLICY_SCHEMA,
            delivery_grace_seconds=grace,
            legacy_retention_seconds=legacy,
            authoritative_reference_files=_string_list_field(
                payload, "authoritative_reference_files"
            ),
            authoritative_reference_roots=_string_list_field(
                payload, "authoritative_reference_roots"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "delivery_grace_seconds": self.delivery_grace_seconds,
            "legacy_retention_seconds": self.legacy_retention_seconds,
            "authoritative_reference_files": list(self.authoritative_reference_files),
            "authoritative_reference_roots": list(self.authoritative_reference_roots),
        }


def load_lifecycle_policy(state_root: Path) -> LifecyclePolicy:
    path = lifecycle_policy_path(state_root)
    if not path.is_file():
        raise FileNotFoundError(f"lifecycle policy missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("lifecycle policy must be a JSON object")
    return LifecyclePolicy.from_json(payload)


def validate_lifecycle_policy_file(path: Path) -> LifecyclePolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("lifecycle policy must be a JSON object")
    return LifecyclePolicy.from_json(payload)
