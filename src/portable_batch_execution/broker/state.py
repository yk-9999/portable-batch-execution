"""Durable broker request state under the controller state root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from portable_batch_execution.broker.config import opaque_request_id

_STATE_SCHEMA = "pbe.a1-unix-broker.request-state.v1"


@dataclass(frozen=True)
class RequestBinding:
    input_digest: str
    pack: str
    operation: str
    operation_params: dict[str, Any]
    execution_fingerprint: str
    public_sha: str

    def matches(self, other: RequestBinding) -> bool:
        return (
            self.input_digest == other.input_digest
            and self.pack == other.pack
            and self.operation == other.operation
            and self.operation_params == other.operation_params
            and self.execution_fingerprint == other.execution_fingerprint
            and self.public_sha == other.public_sha
        )


@dataclass
class BrokerRequestState:
    request_id: str
    binding: RequestBinding
    logical_run_id: str
    wave_id: str
    shard_id: str
    status: str
    execution_id: str | None = None
    output_sha256: str | None = None
    output_media_type: str | None = None
    dispatch_count: int = 0
    last_dispatched_failure_count: int = 0
    last_reconciled_attempt_marker: str | None = None
    result_manifest_object_path: str | None = None
    wave_descriptor_hf_ref: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": _STATE_SCHEMA,
            "request_id": self.request_id,
            "binding": {
                "input_digest": self.binding.input_digest,
                "pack": self.binding.pack,
                "operation": self.binding.operation,
                "operation_params": self.binding.operation_params,
                "execution_fingerprint": self.binding.execution_fingerprint,
                "public_sha": self.binding.public_sha,
            },
            "logical_run_id": self.logical_run_id,
            "wave_id": self.wave_id,
            "shard_id": self.shard_id,
            "status": self.status,
            "execution_id": self.execution_id,
            "output_sha256": self.output_sha256,
            "output_media_type": self.output_media_type,
            "dispatch_count": self.dispatch_count,
            "last_dispatched_failure_count": self.last_dispatched_failure_count,
            "last_reconciled_attempt_marker": self.last_reconciled_attempt_marker,
        }
        if self.result_manifest_object_path is not None:
            payload["result_manifest_object_path"] = self.result_manifest_object_path
        if self.wave_descriptor_hf_ref is not None:
            payload["wave_descriptor_hf_ref"] = self.wave_descriptor_hf_ref
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> BrokerRequestState:
        if payload.get("schema_version") != _STATE_SCHEMA:
            raise ValueError("unsupported broker request state schema")
        binding_payload = payload.get("binding")
        if not isinstance(binding_payload, dict):
            raise TypeError("invalid binding")
        binding = RequestBinding(
            input_digest=str(binding_payload["input_digest"]),
            pack=str(binding_payload["pack"]),
            operation=str(binding_payload["operation"]),
            operation_params=dict(binding_payload.get("operation_params", {})),
            execution_fingerprint=str(binding_payload["execution_fingerprint"]),
            public_sha=str(binding_payload["public_sha"]),
        )
        wave_descriptor = payload.get("wave_descriptor_hf_ref")
        return cls(
            request_id=str(payload["request_id"]),
            binding=binding,
            logical_run_id=str(payload["logical_run_id"]),
            wave_id=str(payload["wave_id"]),
            shard_id=str(payload["shard_id"]),
            status=str(payload["status"]),
            execution_id=payload.get("execution_id"),
            output_sha256=payload.get("output_sha256"),
            output_media_type=payload.get("output_media_type"),
            dispatch_count=int(payload.get("dispatch_count", 0)),
            last_dispatched_failure_count=int(
                payload.get("last_dispatched_failure_count", 0)
            ),
            last_reconciled_attempt_marker=payload.get(
                "last_reconciled_attempt_marker"
            ),
            result_manifest_object_path=payload.get("result_manifest_object_path"),
            wave_descriptor_hf_ref=dict(wave_descriptor)
            if isinstance(wave_descriptor, dict)
            else None,
        )


class BrokerRequestStore:
    def __init__(self, controller_root: Path):
        self._root = controller_root.resolve() / "broker" / "requests"
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, request_id: str) -> Path:
        request_id = opaque_request_id(request_id)
        return self._root / f"{request_id}.json"

    def load(self, request_id: str) -> BrokerRequestState | None:
        path = self._path(request_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid broker request state")
        return BrokerRequestState.from_json(payload)

    def save(self, state: BrokerRequestState) -> None:
        path = self._path(state.request_id)
        temporary = path.with_suffix(".tmp")
        with self._lock:
            temporary.write_text(json.dumps(state.to_json()) + "\n", encoding="utf-8")
            temporary.replace(path)
