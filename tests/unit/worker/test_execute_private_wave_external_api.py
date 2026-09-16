import json
from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.worker.execute_wave import (
    PrivateWaveExecutionError,
    execute_private_wave,
)
from portable_batch_execution.worker.external_api_evaluation import (
    EXTERNAL_API_MAX_BODY_BYTES,
    NVIDIA_API_KEY_ENV_VAR,
    NVIDIA_OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL,
    ExternalApiEvaluationError,
    execute_nvidia_openai_compatible_request,
    validate_closed_operation_params,
)

PROVIDER_PARAMS = {"provider": "nvidia-openai-compatible"}


def _plane(operation_params=None, security_profile="external-api"):
    payload = json.dumps({"model": "closed", "messages": []}, sort_keys=True).encode()
    ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    now = datetime.now(UTC).isoformat()
    shard = {
        "logical_run_id": "opaque-run",
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [ref.model_dump(mode="json")],
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "replay-eval-batch",
        "operation": "replay_eval.external_api_evaluation",
        "input_manifest_ref": ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
        "security_profile": security_profile,
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": PROVIDER_PARAMS if operation_params is None else operation_params,
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": ["opaque-shard"],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.appended = []
            self.last_written = None

        def resolve_wave(self, run_id, wave_id):
            return {"job": job, "wave": wave, "shards": [shard]}

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, artifact_ref):
            return payload

        def write(self, data, media_type):
            self.last_written = data
            return ArtifactRef(
                object_id="output",
                uri="pbe://private/output",
                sha256="sha256:" + sha256(data).hexdigest(),
                size_bytes=len(data),
            )

        def append_attempt(self, record):
            self.appended.append(record)

    return Plane(), payload


def test_closed_operation_params_reject_extra_keys():
    with pytest.raises(ValueError, match="closed operation parameters"):
        validate_closed_operation_params({"provider": "nvidia-openai-compatible", "url": "x"})


def test_private_external_api_success(monkeypatch):
    response = b'{"ok":true}'
    plane, payload = _plane()

    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == NVIDIA_OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.read() == payload
        return httpx.Response(200, content=response)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    monkeypatch.setattr(
        "portable_batch_execution.worker.execute_wave.execute_nvidia_openai_compatible_request",
        lambda body: execute_nvidia_openai_compatible_request(body, client=client, api_key="test-key"),
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    assert plane.last_written == response
    assert attempts[0].counts == {"input_bytes": len(payload), "output_bytes": len(response)}


def test_missing_credential_is_sanitized(monkeypatch):
    plane, _ = _plane()
    monkeypatch.delenv(NVIDIA_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_credential_missing"


def test_wrong_security_profile_rejected_before_attempt():
    plane, _ = _plane(security_profile="offline")
    with pytest.raises(ValueError, match="external-api"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended == []


def test_oversized_streaming_response_stops_incrementally():
    chunk = b"x" * (256 * 1024)
    emitted = []

    def handler(request):
        def stream():
            while True:
                emitted.append(len(chunk))
                yield chunk
        return httpx.Response(200, content=stream())

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(ExternalApiEvaluationError) as exc:
        execute_nvidia_openai_compatible_request(b"{}", client=client, api_key="test-key")
    assert exc.value.code == "external_api_response_too_large"
    assert sum(emitted) <= EXTERNAL_API_MAX_BODY_BYTES + len(chunk)
