"""Deletion intents and receipts for crash-safe, idempotent artifact payload removal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from portable_batch_execution.lifecycle.durable import fsync_directory, write_json_atomically
from portable_batch_execution.lifecycle.paths import deletion_intents_dir, deletion_receipts_dir

_RECEIPT_SCHEMA = "pbe.lifecycle.deletion-receipt.v1"
_INTENT_SCHEMA = "pbe.lifecycle.deletion-intent.v1"


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
    created_at: datetime | None = None
    delivered_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        if self.delivered_at is not None:
            payload["delivered_at"] = self.delivered_at.isoformat()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DeletionReceipt:
        if payload.get("schema_version") != _RECEIPT_SCHEMA:
            raise ValueError("unsupported deletion receipt schema")
        created_raw = payload.get("created_at")
        delivered_raw = payload.get("delivered_at")
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
            created_at=(
                datetime.fromisoformat(created_raw)
                if isinstance(created_raw, str)
                else None
            ),
            delivered_at=(
                datetime.fromisoformat(delivered_raw)
                if isinstance(delivered_raw, str)
                else None
            ),
        )


@dataclass(frozen=True)
class DeletionIntent:
    artifact_digest: str
    artifact_size_bytes: int
    logical_run_ids: tuple[str, ...]
    producer_uid: int | None
    consumer_uid: int | None
    provenance: dict[str, Any] | None
    created_at: datetime | None
    delivered_at: datetime | None
    intended_at: datetime
    mode: Literal["normal", "legacy"]
    policy_identity: str

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": _INTENT_SCHEMA,
            "artifact_digest": self.artifact_digest,
            "artifact_size_bytes": self.artifact_size_bytes,
            "logical_run_ids": list(self.logical_run_ids),
            "producer_uid": self.producer_uid,
            "consumer_uid": self.consumer_uid,
            "provenance": self.provenance,
            "intended_at": self.intended_at.isoformat(),
            "mode": self.mode,
            "policy_identity": self.policy_identity,
        }
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        if self.delivered_at is not None:
            payload["delivered_at"] = self.delivered_at.isoformat()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DeletionIntent:
        if payload.get("schema_version") != _INTENT_SCHEMA:
            raise ValueError("unsupported deletion intent schema")
        created_raw = payload.get("created_at")
        delivered_raw = payload.get("delivered_at")
        return cls(
            artifact_digest=str(payload["artifact_digest"]),
            artifact_size_bytes=int(payload["artifact_size_bytes"]),
            logical_run_ids=tuple(str(item) for item in payload.get("logical_run_ids", ())),
            producer_uid=payload.get("producer_uid"),
            consumer_uid=payload.get("consumer_uid"),
            provenance=payload.get("provenance"),
            created_at=(
                datetime.fromisoformat(created_raw)
                if isinstance(created_raw, str)
                else None
            ),
            delivered_at=(
                datetime.fromisoformat(delivered_raw)
                if isinstance(delivered_raw, str)
                else None
            ),
            intended_at=datetime.fromisoformat(str(payload["intended_at"])),
            mode=str(payload["mode"]),
            policy_identity=str(payload["policy_identity"]),
        )

    def to_receipt(self, deleted_at: datetime) -> DeletionReceipt:
        return DeletionReceipt(
            artifact_digest=self.artifact_digest,
            artifact_size_bytes=self.artifact_size_bytes,
            logical_run_ids=self.logical_run_ids,
            producer_uid=self.producer_uid,
            consumer_uid=self.consumer_uid,
            provenance=self.provenance,
            created_at=self.created_at,
            delivered_at=self.delivered_at,
            deleted_at=deleted_at,
            mode=self.mode,
            policy_identity=self.policy_identity,
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
            write_json_atomically(path, receipt.to_json())

    def list_all(self) -> tuple[DeletionReceipt, ...]:
        records: list[DeletionReceipt] = []
        for path in sorted(self._root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(DeletionReceipt.from_json(payload))
        return tuple(records)


class DeletionIntentStore:
    def __init__(self, state_root: Path):
        self._root = deletion_intents_dir(state_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, artifact_digest: str) -> Path:
        return self._root / f"{digest_hex(artifact_digest)}.json"

    def load(self, artifact_digest: str) -> DeletionIntent | None:
        path = self._path(artifact_digest)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid deletion intent")
        return DeletionIntent.from_json(payload)

    def save(self, intent: DeletionIntent) -> None:
        path = self._path(intent.artifact_digest)
        with self._lock:
            if path.is_file():
                existing = DeletionIntent.from_json(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if existing.to_json() != intent.to_json():
                    raise ValueError("deletion intent conflict")
                return
            write_json_atomically(path, intent.to_json())

    def finalize(self, artifact_digest: str) -> None:
        path = self._path(artifact_digest)
        with self._lock:
            if path.is_file():
                path.unlink()
                fsync_directory(self._root)

    def list_all(self) -> tuple[DeletionIntent, ...]:
        records: list[DeletionIntent] = []
        for path in sorted(self._root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(DeletionIntent.from_json(payload))
        return tuple(records)
