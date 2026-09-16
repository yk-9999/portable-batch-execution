import base64
import json
from datetime import UTC, datetime
from hashlib import sha256
from unittest.mock import patch

import httpx

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    broker_execution_fingerprint,
    broker_input_digest,
    canonical_operation_params,
    opaque_run_id,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.contracts import ShardAttemptRecord
from portable_batch_execution.controller.a1_controller import A1Controller

_SENTINEL = "SENTINEL_BROKER_LEAK_DO_NOT_LOG"
_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000


def _config(tmp_path, *, max_input_bytes: int = 1_048_576) -> BrokerConfig:
    path = tmp_path / "broker-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": max_input_bytes,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        ["tabular-batch", "tabular.sort"],
                        ["ml-batch", "ml.cosine_similarity_matrix"],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def _mock_backend(handler):
    return GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(handler)
        ),
    )


def _request(
    *,
    request_id: str = "req-1",
    pack: str = "tabular-batch",
    operation: str = "tabular.sort",
    operation_params: dict | None = None,
    payload: bytes | None = None,
    input_media_type: str = "application/json",
) -> dict:
    body = payload if payload is not None else json.dumps([{"id": 2}, {"id": 1}]).encode()
    return {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": request_id,
        "pack": pack,
        "operation": operation,
        "operation_params": operation_params or {"by": [{"column": "id"}]},
        "input_media_type": input_media_type,
        "input_b64": base64.b64encode(body).decode("ascii"),
    }


def _service(tmp_path, config: BrokerConfig, handler) -> UnixBrokerService:
    controller = A1Controller(tmp_path, backend=_mock_backend(handler))
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
        sleeper=lambda _seconds: None,
    )


def _append_success_attempt(controller: A1Controller, request_id: str, output: bytes):
    run_id = opaque_run_id(request_id)
    wave_id = f"wave-{sha256(request_id.encode('utf-8')).hexdigest()[:12]}"
    payload = controller.registry.resolve_wave(run_id, wave_id)
    shard = payload["shards"][0]
    output_ref = controller.data_plane.write(output, "application/json")
    now = datetime.now(UTC)
    controller.data_plane.append_attempt(
        ShardAttemptRecord(
            logical_run_id=run_id,
            shard_id=shard["shard_id"],
            attempt_id="attempt-1",
            status="succeeded",
            input_digest=shard["input_digest"],
            execution_fingerprint=shard["execution_fingerprint"],
            started_at=now,
            finished_at=now,
            wave_id=wave_id,
            output_refs=(output_ref,),
            output_digest=sha256(output).hexdigest(),
        )
    )


def _append_failed_attempt(controller: A1Controller, request_id: str, attempt_id: str):
    run_id = opaque_run_id(request_id)
    wave_id = f"wave-{sha256(request_id.encode('utf-8')).hexdigest()[:12]}"
    payload = controller.registry.resolve_wave(run_id, wave_id)
    shard = payload["shards"][0]
    now = datetime.now(UTC)
    controller.data_plane.append_attempt(
        ShardAttemptRecord(
            logical_run_id=run_id,
            shard_id=shard["shard_id"],
            attempt_id=attempt_id,
            status="failed",
            input_digest=shard["input_digest"],
            execution_fingerprint=shard["execution_fingerprint"],
            started_at=now,
            finished_at=now,
            wave_id=wave_id,
            failure="shard_pack_execution_failed",
        )
    )


def test_peer_not_authorized(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    response = service.handle_payload(9999, _request())
    assert response.status == "failed"
    assert response.error_code == "peer_not_authorized"


def test_operation_not_in_allowlist(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    response = service.handle_payload(
        _UID, _request(operation="tabular.rolling", operation_params={"column": "id", "window_size": 2, "output_column": "x"})
    )
    assert response.status == "failed"
    assert response.error_code == "peer_not_authorized"


def test_malformed_base64_and_payload_bound(tmp_path):
    config = _config(tmp_path, max_input_bytes=8)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    bad = _request()
    bad["input_b64"] = "%%%"
    assert service.handle_payload(_UID, bad).error_code == "input_invalid"
    large = _request(payload=b"x" * 16)
    assert service.handle_payload(_UID, large).error_code == "input_invalid"


def test_request_id_conflict(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 1, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def wrapped_dispatch(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-1", b"[{\"id\":1}]")
        return ref

    with patch.object(service.controller, "dispatch_private_wave", wrapped_dispatch):
        first = service.handle_payload(_UID, _request())
    assert first.status == "succeeded"
    conflict = _request(operation_params={"by": [{"column": "value"}]})
    conflict["request_id"] = "req-1"
    second = service.handle_payload(_UID, conflict)
    assert second.status == "failed"
    assert second.error_code == "request_id_conflict"


def test_success_and_reuse_without_redispatch(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 9, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)

    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-2", b"[{\"id\":1}]")
        return ref

    with patch.object(service.controller, "dispatch_private_wave", dispatch_with_success):
        first = service.handle_payload(_UID, _request(request_id="req-2"))
        second = service.handle_payload(_UID, _request(request_id="req-2"))
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert dispatch_calls == 1
    assert _SENTINEL not in first.model_dump_json()
    assert first.output_b64 is not None


def test_active_dispatch_reuses_existing_execution(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 11, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_once(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-3", b"[{\"id\":1}]")
        return ref

    with patch.object(service.controller, "dispatch_private_wave", dispatch_once):
        service.handle_payload(_UID, _request(request_id="req-3"))
    assert dispatch_calls == 1
    with patch.object(
        service.controller,
        "dispatch_private_wave",
        side_effect=AssertionError("must not redispatch while active"),
    ):
        response = service.handle_payload(_UID, _request(request_id="req-3"))
    assert response.status == "succeeded"
    assert dispatch_calls == 1


def test_stale_attempts_do_not_consume_budget(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 3, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave
    run_id = opaque_run_id("req-stale")
    now = datetime.now(UTC)

    def dispatch_with_stale_and_success(run_id_arg, wave_id):
        for index in range(9):
            service.controller.data_plane.append_attempt(
                ShardAttemptRecord(
                    logical_run_id=run_id,
                    shard_id="shard-000000",
                    attempt_id=f"stale-{index}",
                    status="failed",
                    input_digest="stale",
                    execution_fingerprint="stale",
                    started_at=now,
                    finished_at=now,
                    failure="shard_execution_failed",
                )
            )
        ref = real_dispatch(run_id_arg, wave_id)
        _append_success_attempt(service.controller, "req-stale", b"[{\"id\":1}]")
        return ref

    with patch.object(
        service.controller, "dispatch_private_wave", dispatch_with_stale_and_success
    ):
        response = service.handle_payload(_UID, _request(request_id="req-stale"))
    assert response.status == "succeeded"
    assert dispatch_calls == 1


def test_exactly_four_failures_exhausted_without_fifth_dispatch(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 4, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_fail_four(run_id, wave_id):
        for index in range(4):
            _append_failed_attempt(service.controller, "req-exhaust", f"fail-{index}")
        return real_dispatch(run_id, wave_id)

    with patch.object(service.controller, "dispatch_private_wave", dispatch_fail_four):
        response = service.handle_payload(_UID, _request(request_id="req-exhaust"))
    assert response.status == "exhausted"
    assert response.error_code == "attempt_budget_exhausted"
    assert dispatch_calls == 1
    with patch.object(
        service.controller,
        "dispatch_private_wave",
        side_effect=AssertionError("must not dispatch after exhaustion"),
    ):
        again = service.handle_payload(_UID, _request(request_id="req-exhaust"))
    assert again.status == "exhausted"
    assert dispatch_calls == 1


def test_response_does_not_leak_private_payload_or_secrets(tmp_path):
    config = _config(tmp_path)

    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                201,
                json={"workflow_run_id": 1, "html_url": f"https://run/{_SENTINEL}"},
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-leak", b"[{\"id\":1}]")
        return ref

    with patch.object(service.controller, "dispatch_private_wave", dispatch_with_success):
        response = service.handle_payload(_UID, _request(request_id="req-leak"))
    blob = json.dumps(response.model_dump(mode="json"))
    assert _SENTINEL not in blob
    assert "plane-token" not in blob


def test_execution_fingerprint_binds_public_sha(tmp_path):
    rows = json.dumps([{"id": 1}]).encode()
    params = canonical_operation_params(
        "tabular-batch", "tabular.sort", {"by": [{"column": "id"}]}
    )
    digest = broker_input_digest(rows)
    fingerprint = broker_execution_fingerprint(
        request_id="req-bind",
        input_digest=digest,
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params=params,
        public_sha=_PUBLIC_SHA,
    )
    other = broker_execution_fingerprint(
        request_id="req-bind",
        input_digest=digest,
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params=params,
        public_sha="different-sha",
    )
    assert fingerprint != other
