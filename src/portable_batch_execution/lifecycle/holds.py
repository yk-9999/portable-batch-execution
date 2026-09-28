"""Explicit preserve/hold records that block GC for referenced digests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from portable_batch_execution.lifecycle.durable import (
    fsync_directory,
    write_json_atomically,
)
from portable_batch_execution.lifecycle.lock import lifecycle_state_lock
from portable_batch_execution.lifecycle.paths import holds_dir

_HOLD_SCHEMA = "pbe.lifecycle.hold.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class HoldRecord:
    hold_id: str
    kind: Literal["preserve", "hold"]
    artifact_digests: tuple[str, ...]
    reason: str
    created_at: datetime

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": _HOLD_SCHEMA,
            "hold_id": self.hold_id,
            "kind": self.kind,
            "artifact_digests": list(self.artifact_digests),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> HoldRecord:
        if payload.get("schema_version") != _HOLD_SCHEMA:
            raise ValueError("unsupported hold schema")
        digests = tuple(str(item) for item in payload.get("artifact_digests", ()))
        for digest in digests:
            if not _DIGEST.fullmatch(digest):
                raise ValueError("invalid artifact digest in hold")
        kind = str(payload["kind"])
        if kind not in {"preserve", "hold"}:
            raise ValueError("invalid hold kind")
        return cls(
            hold_id=str(payload["hold_id"]),
            kind=kind,
            artifact_digests=digests,
            reason=str(payload.get("reason", "")),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
        )


class HoldStore:
    def __init__(self, state_root: Path):
        self._state_root = state_root.resolve()
        self._root = holds_dir(state_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, hold_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", hold_id)
        return self._root / f"{safe}.json"

    def put(self, record: HoldRecord) -> None:
        path = self._path(record.hold_id)
        with self._lock, lifecycle_state_lock(self._state_root):
            write_json_atomically(path, record.to_json())

    def remove(self, hold_id: str) -> bool:
        path = self._path(hold_id)
        with self._lock, lifecycle_state_lock(self._state_root):
            if path.is_file():
                path.unlink()
                fsync_directory(self._root)
                return True
            return False

    def list_all(self) -> tuple[HoldRecord, ...]:
        records: list[HoldRecord] = []
        for path in sorted(self._root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(HoldRecord.from_json(payload))
        return tuple(records)
