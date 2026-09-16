"""Versioned JSON request/response models for the Unix broker."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from portable_batch_execution.broker.config import opaque_request_id

_REQUEST_SCHEMA = "pbe.a1-unix-broker.request.v1"
_RESPONSE_SCHEMA = "pbe.a1-unix-broker.response.v1"

_RESERVED_PARAM_KEYS = frozenset(
    {
        "shell",
        "command",
        "cmd",
        "python",
        "python_code",
        "script",
        "sql",
        "import_path",
        "entrypoint",
        "executable",
        "url",
        "uri",
        "path",
        "file",
        "token",
        "secret",
        "credential",
        "backend",
        "github",
        "job",
        "shard",
        "wave",
        "run_id",
        "wave_id",
    }
)
_URL_LIKE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_PATH_LIKE = re.compile(r"(^|[\\])\.{0,2}([/\\]|$)|^[a-zA-Z]:[/\\]|^/")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BrokerExecuteRequest(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.request.v1"] = _REQUEST_SCHEMA
    request_id: str
    pack: Literal[
        "tabular-batch",
        "ml-batch",
        "media-batch",
    ]
    operation: str
    operation_params: dict[str, Any] = Field(default_factory=dict)
    input_media_type: str
    input_b64: str

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return opaque_request_id(value)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        if not value or len(value) > 128:
            raise ValueError("operation must be a closed identifier")
        return value

    @field_validator("input_media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        if not value or len(value) > 128 or "/" not in value:
            raise ValueError("input_media_type must be a media type")
        return value

    @field_validator("operation_params")
    @classmethod
    def validate_operation_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        _walk_forbidden(value)
        return value

    def decode_input(self, *, max_bytes: int) -> bytes:
        try:
            payload = base64.b64decode(self.input_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("input_b64 is invalid") from exc
        if len(payload) > max_bytes:
            raise ValueError("input exceeds broker size bound")
        return payload


class BrokerExecuteResponse(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.response.v1"] = _RESPONSE_SCHEMA
    request_id: str
    status: Literal["succeeded", "exhausted", "failed"]
    error_code: str | None = None
    logical_run_id: str | None = None
    wave_id: str | None = None
    execution_id: str | None = None
    input_digest: str | None = None
    execution_fingerprint: str | None = None
    output_media_type: str | None = None
    output_sha256: str | None = None
    output_b64: str | None = None


def _walk_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        if _RESERVED_PARAM_KEYS & {str(key).lower() for key in value}:
            raise ValueError("reserved request field")
        for item in value.values():
            _walk_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden(item)
    elif isinstance(value, str):
        if _URL_LIKE.match(value) or _PATH_LIKE.search(value):
            raise ValueError("path and URL values are not accepted")
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError("JSON compatible values only")


def parse_request(payload: object) -> BrokerExecuteRequest:
    if not isinstance(payload, dict):
        raise TypeError("request must be a JSON object")
    return BrokerExecuteRequest.model_validate(payload)


def response_to_json(response: BrokerExecuteResponse) -> bytes:
    return (response.model_dump_json(exclude_none=True) + "\n").encode("utf-8")
