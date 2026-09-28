"""Deletion receipts for crash-safe, idempotent artifact payload removal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from portable_batch_execution.lifecycle.paths import deletion_receipts_dir

_RECEIPT_SCHEMA = "pbe.lifecycle.deletion-receipt.v1"


@dataclass(frozen=True)
class DeletionReceipt:
    artifact_digest: str
    artifact_size_bytes: int
    logical_run_ids: tuple[str, ...]
    producer_uid: int | None
    consumer_uid: int | None
    provenance: dict[str, Any] | None
    deleted_at: datetime
    mode: Literal["normal", "legacy"]
    policy_identity: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": _RECEIPT_SCHEMA,
            "artifact_digest": self.artifact_digest,
            "artifact_size_bytes": self.artifact_size_bytes,
            "logical_run_ids": list(self.logical_run_ids),
            "producer_uid": self.producer_uid,
            "consumer_uid": self.consumer_uid,
            "provenance": self.provenance,
            "deleted_at": self.deleted_at.isoformat(),
            "mode": self.mode,
            "policy_identity": self.policy_identity,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DeletionReceipt:
        if payload.get("schema_version") != _RECEIPT_SCHEMA:
            raise ValueError("unsupported deletion receipt schema")
        return cls(
            artifact_digest=str(payload["artifact_digest"]),
            artifact_size_bytes=int(payload["artifact_size_bytes"]),
            logical_run_ids=tuple(str(item) for item in payload.get("logical_run_ids", ())),
            producer_uid=payload.get("producer_uid"),
            consumer_uid=payload.get("consumer_uid"),
            provenance=payload.get("provenance"),
            deleted_at=datetime.fromisoformat(str(payload["deleted_at"])),
            mode=str(payload["mode"]),
            policy_identity=str(payload["policy_identity"]),
        )


def digest_hex(artifact_digest: str) -> str:
    if artifact_digest.startswith("sha256:"):
        return artifact_digest.removeprefix("sha256:")
    return artifact_digest


class DeletionReceiptStore:
    def __init__(self, state_root: Path):
        self._root = deletion_receipts_dir(state_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, artifact_digest: str) -> Path:
        return self._root / f"{digest_hex(artifact_digest)}.json"

    def load(self, artifact_digest: str) -> DeletionReceipt | None:
        path = self._path(artifact_digest)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid deletion receipt")
        return DeletionReceipt.from_json(payload)

    def save(self, receipt: DeletionReceipt) -> None:
        path = self._path(receipt.artifact_digest)
        with self._lock:
            if path.is_file():
                existing = DeletionReceipt.from_json(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if existing.to_json() != receipt.to_json():
                    raise ValueError("deletion receipt conflict")
                return
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt.to_json()) + "\n", encoding="utf-8")
            temporary.replace(path)

    def list_all(self) -> tuple[DeletionReceipt, ...]:
        records: list[DeletionReceipt] = []
        for path in sorted(self._root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(DeletionReceipt.from_json(payload))
        return tuple(records)
