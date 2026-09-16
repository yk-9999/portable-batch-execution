"""Broker configuration and peer authorization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_FORBIDDEN_ID_CHARS = "/\\?#"
_MAX_ID_LEN = 128


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
    allowed_operations_by_uid: dict[int, frozenset[tuple[str, str]]]

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
        max_input_bytes = payload.get("max_input_bytes", 1_048_576)
        if not isinstance(max_input_bytes, int) or max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive integer")
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
        return cls(
            schema_version=version,
            public_sha=public_sha,
            max_input_bytes=max_input_bytes,
            allowed_operations_by_uid=allowed,
        )

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
