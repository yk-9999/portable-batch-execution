"""Immutable delivery records for broker transport handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from portable_batch_execution.broker.config import opaque_request_id
from portable_batch_execution.lifecycle.durable import write_json_atomically
from portable_batch_execution.lifecycle.lock import lifecycle_state_lock
from portable_batch_execution.lifecycle.paths import deliveries_dir

_DELIVERY_SCHEMA = "pbe.lifecycle.delivery-record.v1"


@dataclass(frozen=True)
class DeliveryRecord:
    request_id: str
    logical_run_id: str
    artifact_digest: str
    artifact_size_bytes: int
    producer_uid: int | None
    consumer_uid: int
    created_at: datetime | None
    delivered_at: datetime
    provenance: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": _DELIVERY_SCHEMA,
            "request_id": self.request_id,
            "logical_run_id": self.logical_run_id,
            "artifact_digest": self.artifact_digest,
            "artifact_size_bytes": self.artifact_size_bytes,
            "producer_uid": self.producer_uid,
            "consumer_uid": self.consumer_uid,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "delivered_at": self.delivered_at.isoformat(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DeliveryRecord:
        if payload.get("schema_version") != _DELIVERY_SCHEMA:
            raise ValueError("unsupported delivery record schema")
        created_raw = payload.get("created_at")
        created = (
            datetime.fromisoformat(created_raw)
            if isinstance(created_raw, str)
            else None
        )
        delivered_raw = payload["delivered_at"]
        if not isinstance(delivered_raw, str):
            raise TypeError("delivered_at required")
        return cls(
            request_id=str(payload["request_id"]),
            logical_run_id=str(payload["logical_run_id"]),
            artifact_digest=str(payload["artifact_digest"]),
            artifact_size_bytes=int(payload["artifact_size_bytes"]),
            producer_uid=payload.get("producer_uid"),
            consumer_uid=int(payload["consumer_uid"]),
            created_at=created,
            delivered_at=datetime.fromisoformat(delivered_raw),
            provenance=payload.get("provenance"),
        )


class DeliveryRecordStore:
    def __init__(self, state_root: Path):
        self._state_root = state_root.resolve()
        self._root = deliveries_dir(state_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, request_id: str) -> Path:
        return self._root / f"{opaque_request_id(request_id)}.json"

    def load(self, request_id: str) -> DeliveryRecord | None:
        path = self._path(request_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid delivery record")
        return DeliveryRecord.from_json(payload)

    @staticmethod
    def _transport_identity(record: DeliveryRecord) -> tuple:
        created = (
            record.created_at.isoformat() if record.created_at is not None else None
        )
        return (
            record.request_id,
            record.logical_run_id,
            record.artifact_digest,
            record.artifact_size_bytes,
            record.producer_uid,
            record.consumer_uid,
            created,
            json.dumps(record.provenance, sort_keys=True) if record.provenance else None,
        )

    def save_new(self, record: DeliveryRecord) -> None:
        """Persist only if no record exists; conflicting payloads fail closed."""
        path = self._path(record.request_id)
        with self._lock:
            with lifecycle_state_lock(self._state_root):
                if path.is_file():
                    existing = DeliveryRecord.from_json(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                    if self._transport_identity(existing) != self._transport_identity(
                        record
                    ):
                        raise ValueError("delivery_commit_conflict")
                    return
                write_json_atomically(path, record.to_json())

    def list_all(self) -> tuple[DeliveryRecord, ...]:
        records: list[DeliveryRecord] = []
        for path in sorted(self._root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(DeliveryRecord.from_json(payload))
        return tuple(records)


def utc_now() -> datetime:
    return datetime.now(UTC)
