"""HF-direct Unix broker request/response (metadata-only artifact refs)."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, field_validator

from portable_batch_execution.broker.config import opaque_request_id
from portable_batch_execution.broker.protocol import _StrictModel, _walk_forbidden
from portable_batch_execution.transport.hf_bucket import validate_hf_bucket_ref

_HF_DIRECT_REQUEST_SCHEMA = "pbe.a1-unix-broker.hf-direct-request.v1"
_HF_DIRECT_RESPONSE_SCHEMA = "pbe.a1-unix-broker.hf-direct-response.v1"


class BrokerHfDirectExecuteRequest(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.hf-direct-request.v1"] = (
        _HF_DIRECT_REQUEST_SCHEMA
    )
    request_id: str
    pack: Literal["replay-batch"]
    operation: Literal["replay.trade_path_scenario_evaluate_fixed_set"]
    operation_params: dict[str, Any] = Field(default_factory=dict)
    input_media_type: str
    input_hf_ref: dict[str, Any]

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return opaque_request_id(value)

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

    @field_validator("input_hf_ref")
    @classmethod
    def validate_input_hf_ref(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_hf_bucket_ref(value)


class BrokerHfDirectExecuteResponse(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.hf-direct-response.v1"] = (
        _HF_DIRECT_RESPONSE_SCHEMA
    )
    request_id: str
    status: Literal["succeeded", "exhausted", "failed"]
    error_code: str | None = None
    logical_run_id: str | None = None
    wave_id: str | None = None
    execution_id: str | None = None
    input_digest: str | None = None
    execution_fingerprint: str | None = None
    output_media_type: str | None = None
    output_hf_ref: dict[str, Any] | None = None


def parse_hf_direct_request(payload: object) -> BrokerHfDirectExecuteRequest:
    if not isinstance(payload, dict):
        raise TypeError("request must be a JSON object")
    if payload.get("schema_version") != _HF_DIRECT_REQUEST_SCHEMA:
        raise TypeError("not an HF-direct broker request")
    if "input_b64" in payload or "output_b64" in payload:
        raise ValueError("byte/base64 broker fields are forbidden in HF-direct mode")
    return BrokerHfDirectExecuteRequest.model_validate(payload)


def reject_legacy_byte_fields(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("transport_profile") == "hf_bucket_direct" or payload.get(
        "schema_version"
    ) in {_HF_DIRECT_REQUEST_SCHEMA, _HF_DIRECT_RESPONSE_SCHEMA}:
        for key in ("input_b64", "result_b64", "output_b64"):
            if key in payload:
                raise ValueError(f"{key} forbidden in HF-direct mode")


def hf_direct_response_to_json(response: BrokerHfDirectExecuteResponse) -> bytes:
    return (response.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


def hf_direct_input_digest(input_hf_ref: dict[str, Any]) -> str:
    validated = validate_hf_bucket_ref(input_hf_ref)
    material = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    return sha256(material.encode("utf-8")).hexdigest()
