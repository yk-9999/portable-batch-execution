"""Closed public-runner external API evaluation for replay_eval.external_api_evaluation."""

from __future__ import annotations

import os
from typing import Final

import httpx

NVIDIA_OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL: Final = (
    "https://integrate.api.nvidia.com/v1/chat/completions"
)
EXTERNAL_API_MAX_BODY_BYTES: Final = 16 * 1024 * 1024
EXTERNAL_API_REQUEST_TIMEOUT_SECONDS: Final = 300.0
NVIDIA_API_KEY_ENV_VAR: Final = "PBE_EXTERNAL_API_NVIDIA_API_KEY"
CLOSED_PROVIDER_VALUE: Final = "nvidia-openai-compatible"
CLOSED_OPERATION_PARAM_KEYS: Final = frozenset({"provider"})

FORBIDDEN_OPERATION_PARAM_KEYS: Final = frozenset(
    {
        "url", "uri", "endpoint", "host", "path", "scheme", "headers", "header",
        "authorization", "token", "secret", "api_key", "method", "code", "callback",
        "shell", "command", "cmd", "python", "python_code", "script", "import_path",
        "entrypoint", "executable", "callable", "function", "handler",
    }
)


class ExternalApiEvaluationError(Exception):
    """Sanitized shard failure with a stable public code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_closed_operation_params(params: object) -> None:
    """Accept only the closed provider selector; reject hooks and extra keys."""
    if not isinstance(params, dict):
        raise ValueError("closed operation parameters")  # noqa: TRY004
    keys = frozenset(params)
    if keys != CLOSED_OPERATION_PARAM_KEYS:
        raise ValueError("closed operation parameters")
    if FORBIDDEN_OPERATION_PARAM_KEYS.intersection(keys):
        raise ValueError("closed operation parameters")
    if params.get("provider") != CLOSED_PROVIDER_VALUE:
        raise ValueError("closed operation parameters")


def execute_nvidia_openai_compatible_request(
    request_body_bytes: bytes,
    *,
    client: httpx.Client | None = None,
    api_key: str | None = None,
) -> bytes:
    """POST one bounded provider request and return the exact raw JSON response bytes."""
    if len(request_body_bytes) > EXTERNAL_API_MAX_BODY_BYTES:
        raise ExternalApiEvaluationError("external_api_request_too_large")
    resolved_key = api_key if api_key is not None else os.environ.get(NVIDIA_API_KEY_ENV_VAR, "")
    if not resolved_key:
        raise ExternalApiEvaluationError("external_api_credential_missing")

    owned_client = client is None
    active = client or httpx.Client(
        timeout=httpx.Timeout(EXTERNAL_API_REQUEST_TIMEOUT_SECONDS),
        follow_redirects=False,
    )
    try:
        try:
            with active.stream(
                "POST",
                NVIDIA_OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL,
                content=request_body_bytes,
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise ExternalApiEvaluationError("external_api_provider_failed")
                body = _read_bounded_response_body(response)
        except httpx.HTTPError:
            raise ExternalApiEvaluationError("external_api_transport_failed") from None
    finally:
        if owned_client:
            active.close()
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        raise ExternalApiEvaluationError("external_api_response_invalid") from None
    return body


def _read_bounded_response_body(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > EXTERNAL_API_MAX_BODY_BYTES:
                raise ExternalApiEvaluationError("external_api_response_too_large")
            chunks.append(chunk)
    except httpx.HTTPError:
        raise ExternalApiEvaluationError("external_api_transport_failed") from None
    body = b"".join(chunks)
    if not body:
        raise ExternalApiEvaluationError("external_api_response_invalid")
    return body
