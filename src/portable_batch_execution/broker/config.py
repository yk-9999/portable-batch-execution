"""Broker configuration and peer authorization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from portable_batch_execution.contracts import ArtifactRef

_FORBIDDEN_ID_CHARS = "/\\?#"
_MAX_ID_LEN = 128
_DEFAULT_MAX_INPUT_BYTES = 1_048_576
_ABSOLUTE_MAX_INPUT_BYTES = 134_217_728
_DEFAULT_SOCKET_MODE = 0o666
_JSON_FRAME_OVERHEAD_BYTES = 65_536
_MAX_STATIC_INPUT_REFS = 16
_STATIC_BINDING_KEYS = frozenset({"pack", "operation", "input_refs"})


def max_request_frame_bytes(max_input_bytes: int) -> int:
    if max_input_bytes <= 0 or max_input_bytes > _ABSOLUTE_MAX_INPUT_BYTES:
        raise ValueError("max_input_bytes exceeds broker absolute bound")
    encoded = ((max_input_bytes + 2) // 3) * 4
    return encoded + _JSON_FRAME_OVERHEAD_BYTES


def _opaque_identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_ID_LEN
        or any(character in value for character in _FORBIDDEN_ID_CHARS)
    ):
        raise ValueError(f"{name} must be an opaque identifier")
    return value


@dataclass(frozen=True)
class BrokerConfig:
    schema_version: str
    public_sha: str
    max_input_bytes: int
    socket_mode: int
    allowed_operations_by_uid: dict[int, frozenset[tuple[str, str]]]
    static_input_bindings: dict[tuple[str, str], tuple[ArtifactRef, ...]]

    @property
    def max_request_frame_bytes(self) -> int:
        return max_request_frame_bytes(self.max_input_bytes)

    @classmethod
    def load(cls, path: Path) -> BrokerConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("broker config must be an object")
        version = payload.get("schema_version")
        if version != "pbe.a1-unix-broker.config.v1":
            raise ValueError("unsupported broker config schema_version")
        public_sha = payload.get("public_sha")
        if not isinstance(public_sha, str) or not public_sha:
            raise ValueError("public_sha is required")
        max_input_bytes = payload.get("max_input_bytes", _DEFAULT_MAX_INPUT_BYTES)
        if not isinstance(max_input_bytes, int) or max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive integer")
        if max_input_bytes > _ABSOLUTE_MAX_INPUT_BYTES:
            raise ValueError("max_input_bytes exceeds broker absolute bound")
        socket_mode = payload.get("socket_mode", _DEFAULT_SOCKET_MODE)
        if not isinstance(socket_mode, int) or socket_mode < 0 or socket_mode > 0o777:
            raise ValueError("socket_mode must be a unix permission mask")
        raw_allowlist = payload.get("allowed_operations_by_uid")
        if not isinstance(raw_allowlist, dict):
            raise TypeError("allowed_operations_by_uid is required")
        allowed: dict[int, frozenset[tuple[str, str]]] = {}
        for key, value in raw_allowlist.items():
            try:
                uid = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("uid keys must be integers") from exc
            if not isinstance(value, list):
                raise TypeError("allowed operation list must be an array")
            pairs: set[tuple[str, str]] = set()
            for item in value:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(isinstance(part, str) and part for part in item)
                ):
                    raise ValueError("allowed operation entries must be [pack, operation]")
                pairs.add((item[0], item[1]))
            allowed[uid] = frozenset(pairs)
        static_input_bindings = _parse_static_input_bindings(
            payload.get("static_input_bindings", [])
        )
        return cls(
            schema_version=version,
            public_sha=public_sha,
            max_input_bytes=max_input_bytes,
            socket_mode=socket_mode,
            allowed_operations_by_uid=allowed,
            static_input_bindings=static_input_bindings,
        )

    def static_input_refs_for(self, pack: str, operation: str) -> tuple[ArtifactRef, ...]:
        return self.static_input_bindings.get((pack, operation), ())

    def authorize(self, uid: int, pack: str, operation: str) -> bool:
        allowed = self.allowed_operations_by_uid.get(uid)
        if allowed is None:
            return False
        return (pack, operation) in allowed


def load_broker_config_from_environment() -> BrokerConfig:
    import os

    path = os.environ.get("PBE_BROKER_CONFIG")
    if not path:
        raise ValueError("PBE_BROKER_CONFIG is required")
    return BrokerConfig.load(Path(path))


def opaque_request_id(value: str) -> str:
    return _opaque_identifier(value, "request_id")


def _parse_static_input_bindings(
    raw_bindings: object,
) -> dict[tuple[str, str], tuple[ArtifactRef, ...]]:
    if raw_bindings is None:
        raw_bindings = []
    if not isinstance(raw_bindings, list):
        raise TypeError("static_input_bindings must be an array")
    bindings: dict[tuple[str, str], tuple[ArtifactRef, ...]] = {}
    for entry in raw_bindings:
        if not isinstance(entry, dict):
            raise TypeError("static_input_bindings entries must be objects")
        if frozenset(entry.keys()) != _STATIC_BINDING_KEYS:
            raise ValueError("static_input_bindings entry has unexpected fields")
        pack = entry.get("pack")
        operation = entry.get("operation")
        if not isinstance(pack, str) or not pack:
            raise ValueError("static_input_bindings pack must be a nonempty string")
        if not isinstance(operation, str) or not operation:
            raise ValueError("static_input_bindings operation must be a nonempty string")
        key = (pack, operation)
        if key in bindings:
            raise ValueError("duplicate static_input_bindings pack and operation")
        raw_refs = entry.get("input_refs")
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("static_input_bindings input_refs must be a nonempty array")
        if len(raw_refs) > _MAX_STATIC_INPUT_REFS:
            raise ValueError("static_input_bindings input_refs exceeds bound")
        refs = tuple(ArtifactRef.model_validate(item) for item in raw_refs)
        object_ids = [ref.object_id for ref in refs]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("static_input_bindings input_refs contain duplicate object_id")
        bindings[key] = refs
    return bindings
