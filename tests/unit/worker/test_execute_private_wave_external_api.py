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

_SENTINEL = "SENTINEL_PRIVATE_LEAK_DO_NOT_LOG"
_PROVIDER_PARAMS = {"provider": "nvidia-openai-compatible"}


def _plane(
    *,
    request_body=None,
    operation_params=None,
    security_profile="external-api",
    input_ref_overrides=None,
    write_ref_overrides=None,
    appended=None,
):
    request_body = request_body if request_body is not None else {"model": "closed", "messages": []}
    payload = json.dumps(request_body, sort_keys=True).encode("utf-8")
    input_ref_fields = {
        "object_id": "input",
        "uri": "pbe://private/input",
        "sha256": "sha256:" + sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    input_ref_fields.update(input_ref_overrides or {})
    input_ref = ArtifactRef(**input_ref_fields)
    now = datetime.now(UTC).isoformat()
    shard = {
        "logical_run_id": "opaque-run",
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [input_ref.model_dump(mode="json")],
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "replay-eval-batch",
        "operation": "replay_eval.external_api_evaluation",
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
        "security_profile": security_profile,
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": (
            _PROVIDER_PARAMS if operation_params is None else operation_params
        ),
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": [shard["shard_id"]],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.appended = list(appended or [])
            self.last_written = b""

        def resolve_wave(self, run_id, wave_id):
            return {"job": job, "wave": wave, "shards": [shard]}

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            return payload

        def write(self, data, media_type):
            self.last_written = data
            output_ref_fields = {
                "object_id": "output",
                "uri": "pbe://private/output",
                "sha256": "sha256:" + sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
            output_ref_fields.update(write_ref_overrides or {})
            return ArtifactRef(**output_ref_fields)

        def append_attempt(self, record):
            self.appended.append(record)

    return Plane(), payload


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _bind_real_external_api_call(monkeypatch, client):
    from portable_batch_execution.worker import (
        external_api_evaluation as external_api_mod,
    )

    monkeypatch.setattr(
        "portable_batch_execution.worker.execute_wave.execute_nvidia_openai_compatible_request",
        lambda body, **kwargs: external_api_mod.execute_nvidia_openai_compatible_request(
            body, client=client, api_key="test-key"
        ),
    )


def test_validate_closed_operation_params_rejects_extra_keys():
    with pytest.raises(ValueError, match="closed operation parameters"):
        validate_closed_operation_params(
            {"provider": "nvidia-openai-compatible", "url": "https://evil.example"}
        )


@pytest.mark.parametrize(
    "operation_params",
    (
        {},
        {"provider": "other"},
        {"provider": "nvidia-openai-compatible", "headers": {"Authorization": "x"}},
        {"provider": "nvidia-openai-compatible", "secret": _SENTINEL},
        {"provider": "nvidia-openai-compatible", "callback": "run"},
    ),
)
def test_rejects_non_closed_operation_params_before_network(operation_params):
    plane, _ = _plane(operation_params=operation_params)
    with pytest.raises(ValueError, match="closed operation parameters"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended == []


def test_rejects_wrong_security_profile_before_network():
    plane, _ = _plane(security_profile="offline")
    with pytest.raises(ValueError, match="external-api"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended == []


def test_external_api_success_writes_exact_response_and_succeeds(monkeypatch):
    response_body = {"id": "resp-1", "choices": [{"message": {"content": "ok"}}]}
    raw_response = json.dumps(response_body, sort_keys=True).encode("utf-8")
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == NVIDIA_OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.read() == json.dumps(
            {"model": "closed", "messages": []}, sort_keys=True
        ).encode("utf-8")
        return httpx.Response(200, content=raw_response)

    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()
    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    record = attempts[0]
    assert record.status == "succeeded"
    assert plane.last_written == raw_response
    assert record.counts == {
        "input_bytes": len(
            json.dumps({"model": "closed", "messages": []}, sort_keys=True).encode()
        ),
        "output_bytes": len(raw_response),
    }
    assert len(calls) == 1
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))


def test_missing_credential_fails_sanitized(monkeypatch):
    monkeypatch.delenv(NVIDIA_API_KEY_ENV_VAR, raising=False)
    plane, _ = _plane()
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.failure == "external_api_credential_missing"
    assert _SENTINEL not in str(error.value)
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))


def test_request_over_limit_with_matching_digest_fails_before_network(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    oversized_payload = json.dumps(
        {"blob": "x" * EXTERNAL_API_MAX_BODY_BYTES}
    ).encode("utf-8")
    plane, _ = _plane(
        request_body={"blob": "x" * EXTERNAL_API_MAX_BODY_BYTES},
        input_ref_overrides={
            "sha256": "sha256:" + sha256(oversized_payload).hexdigest(),
            "size_bytes": len(oversized_payload),
        },
    )
    plane.read = lambda ref: oversized_payload  # type: ignore[method-assign]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_request_too_large"
    assert calls == []


def test_non_2xx_records_sanitized_failure(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()

    def handler(request):
        return httpx.Response(502, text=_SENTINEL)

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.failure == "external_api_provider_failed"
    assert _SENTINEL not in str(error.value)
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))


def test_transport_timeout_records_sanitized_failure(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()

    def handler(request):
        raise httpx.ReadTimeout(_SENTINEL)

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_transport_failed"
    assert _SENTINEL not in str(error.value)


def test_oversized_response_records_sanitized_failure(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()
    chunk = b"a" * (512 * 1024)
    chunks_yielded = []

    def handler(request):
        def stream():
            emitted = 0
            while emitted <= EXTERNAL_API_MAX_BODY_BYTES:
                chunks_yielded.append(len(chunk))
                yield chunk
                emitted += len(chunk)

        return httpx.Response(200, content=stream())

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_response_too_large"
    assert sum(chunks_yielded) <= EXTERNAL_API_MAX_BODY_BYTES + len(chunk)


def test_oversized_streaming_response_enforces_limit_incrementally():
    chunk = b"b" * (256 * 1024)
    chunks_yielded: list[int] = []

    def handler(request):
        def stream():
            total = 0
            while total <= EXTERNAL_API_MAX_BODY_BYTES:
                chunks_yielded.append(len(chunk))
                yield chunk
                total += len(chunk)

        return httpx.Response(200, content=stream())

    client = _mock_client(handler)
    with pytest.raises(ExternalApiEvaluationError) as error:
        execute_nvidia_openai_compatible_request(
            b"{}",
            client=client,
            api_key="test-key",
        )
    assert error.value.code == "external_api_response_too_large"
    assert sum(chunks_yielded) <= EXTERNAL_API_MAX_BODY_BYTES + len(chunk)


def test_invalid_response_json_records_sanitized_failure(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()

    def handler(request):
        return httpx.Response(200, content=b"not-json " + _SENTINEL.encode())

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_response_invalid"
    assert _SENTINEL not in str(error.value)


def test_non_object_response_json_records_sanitized_failure(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()

    def handler(request):
        return httpx.Response(200, content=b"[1,2,3]")

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "external_api_response_invalid"


def test_exactly_one_provider_request_per_attempt(monkeypatch):
    monkeypatch.setenv(NVIDIA_API_KEY_ENV_VAR, "test-key")
    plane, _ = _plane()
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={"ok": True})

    client = _mock_client(handler)
    _bind_real_external_api_call(monkeypatch, client)
    execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert calls == ["POST"]


def test_existing_tabular_private_wave_still_succeeds():
    from tests.unit.worker.test_execute_private_wave import _plane as tabular_plane

    plane = tabular_plane()
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
