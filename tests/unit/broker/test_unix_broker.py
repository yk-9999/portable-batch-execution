import base64
import json
import os
import socket
import threading
from datetime import UTC, datetime
from hashlib import sha256
from unittest.mock import patch

import httpx
import pytest

from portable_batch_execution.backends.github_actions import (
    GitHubActionsAPIError,
    GitHubActionsBackend,
)
from portable_batch_execution.broker.config import BrokerConfig, max_request_frame_bytes
from portable_batch_execution.broker.planning import (
    broker_execution_fingerprint,
    broker_input_digest,
    canonical_operation_params,
    opaque_run_id,
    register_broker_private_run,
)
from portable_batch_execution.broker.server import _handle_connection, serve_unix_broker
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.broker.state import (
    BrokerRequestState,
    BrokerRequestStore,
    RequestBinding,
)
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


def _service(
    tmp_path,
    config: BrokerConfig,
    handler,
    *,
    sleeper=None,
) -> UnixBrokerService:
    controller = A1Controller(tmp_path, backend=_mock_backend(handler))
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
        sleeper=sleeper or (lambda _seconds: None),
    )


def _failure_progress_sleeper(service: UnixBrokerService, request_id: str):
    store = BrokerRequestStore(service.state_root / "controller")

    def sleeper(_seconds: float) -> None:
        state = store.load(request_id)
        if state is None:
            return
        payload = service.controller.registry.resolve_wave(
            state.logical_run_id, state.wave_id
        )
        shard = payload["shards"][0]
        attempts = service.controller.data_plane.read_attempts(state.logical_run_id)
        terminal = [
            item
            for item in attempts
            if item.shard_id == shard["shard_id"]
            and item.input_digest == shard["input_digest"]
            and item.execution_fingerprint == shard["execution_fingerprint"]
            and item.status in {"failed", "cancelled"}
        ]
        if state.dispatch_count > len(terminal) and len(terminal) < 4:
            _append_failed_attempt(service.controller, request_id, f"fail-{len(terminal)}")

    return sleeper


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


def test_sequential_four_dispatches_then_exhausted_without_fifth(tmp_path):
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

    service = UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=A1Controller(tmp_path, backend=_mock_backend(handler)),
        poll_interval_seconds=0.0,
        sleeper=lambda _seconds: None,
    )
    service._sleep = _failure_progress_sleeper(service, "req-exhaust")
    response = service.handle_payload(_UID, _request(request_id="req-exhaust"))
    assert response.status == "exhausted"
    assert response.error_code == "attempt_budget_exhausted"
    assert dispatch_calls == 4
    with patch.object(
        service.controller,
        "dispatch_private_wave",
        side_effect=AssertionError("must not dispatch after exhaustion"),
    ):
        again = service.handle_payload(_UID, _request(request_id="req-exhaust"))
    assert again.status == "exhausted"
    assert dispatch_calls == 4


def test_terminal_backend_without_attempt_triggers_retry(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(
                201, json={"workflow_run_id": dispatch_calls, "html_url": "https://run"}
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "failure", "updated_at": "t"},
        )

    class _PollLimit(RuntimeError):
        pass

    def sleeper(_seconds: float) -> None:
        if dispatch_calls >= 2:
            raise _PollLimit()

    service = _service(tmp_path, config, handler, sleeper=sleeper)
    with pytest.raises(_PollLimit):
        service.handle_payload(_UID, _request(request_id="req-zero-attempt-retry"))
    assert dispatch_calls == 2


def test_four_terminal_zero_attempt_executions_exhaust_without_fifth_dispatch(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(
                201, json={"workflow_run_id": dispatch_calls, "html_url": "https://run"}
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "cancelled", "updated_at": "t"},
        )

    service = _service(
        tmp_path,
        config,
        handler,
        sleeper=lambda _seconds: None,
    )
    response = service.handle_payload(
        _UID, _request(request_id="req-zero-attempt-exhaust")
    )
    assert response.status == "exhausted"
    assert response.error_code == "attempt_budget_exhausted"
    assert dispatch_calls == 4
    with patch.object(
        service.controller,
        "dispatch_private_wave",
        side_effect=AssertionError("must not dispatch after exhaustion"),
    ):
        again = service.handle_payload(
            _UID, _request(request_id="req-zero-attempt-exhaust")
        )
    assert again.status == "exhausted"
    assert dispatch_calls == 4


def test_mixed_zero_attempt_then_failure_attempt_still_retries(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(
                201, json={"workflow_run_id": dispatch_calls, "html_url": "https://run"}
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "failure", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler, sleeper=lambda _seconds: None)
    store = BrokerRequestStore(service.state_root / "controller")

    def sleeper(_seconds: float) -> None:
        state = store.load("req-mixed")
        if state is None or state.dispatch_count < 2:
            return
        if state.dispatch_count == 2:
            _append_failed_attempt(service.controller, "req-mixed", "fail-1")

    service._sleep = sleeper
    response = service.handle_payload(_UID, _request(request_id="req-mixed"))
    assert response.status == "exhausted"
    assert dispatch_calls == 4


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


def test_backend_transient_error_preserves_durable_state(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 6, "html_url": "https://run"})
        return httpx.Response(500, json={"message": "rate limited"})

    service = _service(tmp_path, config, handler)
    response = service.handle_payload(_UID, _request(request_id="req-transient"))
    assert response.status == "failed"
    assert response.error_code == "backend_transient"
    assert dispatch_calls == 1
    state = BrokerRequestStore(service.state_root / "controller").load("req-transient")
    assert state is not None and state.dispatch_count == 1


@pytest.mark.skipif(not hasattr(socket, "socketpair"), reason="socketpair required")
def test_server_survives_transient_backend_lookup(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 6, "html_url": "https://run"})
        return httpx.Response(500, json={"message": "rate limited"})

    service = _service(tmp_path, config, handler)
    client_sock, server_sock = socket.socketpair()
    request = (json.dumps(_request(request_id="req-srv")) + "\n").encode()
    client_sock.sendall(request)
    with (
        patch(
            "portable_batch_execution.broker.server.read_peer_credentials",
            return_value=(1, _UID, 1),
        ),
        patch.object(
            service.controller.backend,
            "get_run",
            side_effect=GitHubActionsAPIError(
                "get run",
                httpx.Response(500, request=httpx.Request("GET", "https://api.github.com")),
            ),
        ),
    ):
        _handle_connection(server_sock, service)
    response_line = client_sock.recv(65536)
    client_sock.close()
    server_sock.close()
    payload = json.loads(response_line.decode())
    assert payload["error_code"] == "backend_transient"


def test_reconcile_updates_manifest_on_success(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 8, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-reconcile", b"[{\"id\":1}]")
        return ref

    with (
        patch.object(service.controller, "dispatch_private_wave", dispatch_with_success),
        patch.object(
            service.controller, "reconcile_run", wraps=service.controller.reconcile_run
        ) as reconcile,
    ):
        response = service.handle_payload(_UID, _request(request_id="req-reconcile"))
    assert response.status == "succeeded"
    assert reconcile.call_count >= 1
    manifest = service.controller.data_plane.read_manifest(opaque_run_id("req-reconcile"))
    assert manifest is not None and manifest.revision >= 1


def test_registration_recovery_after_state_file_loss(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 12, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    service = _service(tmp_path, config, handler)
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_success_attempt(service.controller, "req-recover", b"[{\"id\":1}]")
        return ref

    with patch.object(service.controller, "dispatch_private_wave", dispatch_with_success):
        first = service.handle_payload(_UID, _request(request_id="req-recover"))
    assert first.status == "succeeded"
    state_path = BrokerRequestStore(service.state_root / "controller")._path("req-recover")
    state_path.unlink()
    second = service.handle_payload(_UID, _request(request_id="req-recover"))
    assert second.status == "succeeded"


def test_cancelled_attempts_count_toward_exhaustion(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    request = _request(request_id="req-cancel")
    body = base64.b64decode(request["input_b64"])
    params = canonical_operation_params("tabular-batch", "tabular.sort", request["operation_params"])
    binding = RequestBinding(
        input_digest=broker_input_digest(body),
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params=params,
        execution_fingerprint=broker_execution_fingerprint(
            request_id="req-cancel",
            input_digest=broker_input_digest(body),
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params=params,
            public_sha=_PUBLIC_SHA,
        ),
        public_sha=_PUBLIC_SHA,
    )
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-cancel",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params=request["operation_params"],
        input_bytes=body,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    BrokerRequestStore(service.state_root / "controller").save(
        BrokerRequestState(
            request_id="req-cancel",
            binding=binding,
            logical_run_id=opaque_run_id("req-cancel"),
            wave_id=f"wave-{sha256(b'req-cancel').hexdigest()[:12]}",
            shard_id="shard-000000",
            status="active",
        )
    )
    run_id = opaque_run_id("req-cancel")
    wave_id = f"wave-{sha256(b'req-cancel').hexdigest()[:12]}"
    payload = service.controller.registry.resolve_wave(run_id, wave_id)
    shard = payload["shards"][0]
    now = datetime.now(UTC)
    for index in range(4):
        service.controller.data_plane.append_attempt(
            ShardAttemptRecord(
                logical_run_id=run_id,
                shard_id=shard["shard_id"],
                attempt_id=f"cancel-{index}",
                status="cancelled",
                input_digest=shard["input_digest"],
                execution_fingerprint=shard["execution_fingerprint"],
                started_at=now,
                finished_at=now,
                failure="cancelled",
            )
        )
    response = service.handle_payload(_UID, _request(request_id="req-cancel"))
    assert response.status == "exhausted"


def test_large_input_within_config_frame_bound(tmp_path):
    max_bytes = 12 * 1024 * 1024
    config = _config(tmp_path, max_input_bytes=max_bytes)
    assert max_request_frame_bytes(max_bytes) > 8 * 1024 * 1024
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    payload = b"x" * (9 * 1024 * 1024)
    assert (
        service.handle_payload(_UID, _request(payload=payload)).error_code
        != "input_invalid"
    )
    over = _request(payload=b"x" * (max_bytes + 1))
    assert service.handle_payload(_UID, over).error_code == "input_invalid"


def test_unauthorized_uid_rejected_before_registration(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    response = service.handle_payload(917, _request(request_id="req-deny"))
    assert response.error_code == "peer_not_authorized"
    run_dir = tmp_path / "controller" / "closed_waves" / opaque_run_id("req-deny")
    assert not run_dir.exists()


@pytest.mark.skipif(os.name != "posix", reason="AF_UNIX chmod test requires posix")
def test_socket_mode_set_after_bind(tmp_path):  # pragma: no cover - posix only
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    socket_path = tmp_path / "broker.sock"
    with patch("portable_batch_execution.broker.server.os.chmod") as chmod:
        thread = threading.Thread(
            target=serve_unix_broker,
            kwargs={"socket_path": socket_path, "service": service},
            daemon=True,
        )
        thread.start()
        thread.join(timeout=0.5)
    chmod.assert_called_with(socket_path.resolve(), config.socket_mode)


def test_reconcile_skips_unchanged_attempts_during_polling(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 20, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "in_progress", "conclusion": None, "updated_at": "t"},
        )

    class _PollLimit(RuntimeError):
        pass

    polls = 0

    def sleeper(_seconds: float) -> None:
        nonlocal polls
        polls += 1
        if polls >= 12:
            raise _PollLimit()

    service = _service(tmp_path, config, handler, sleeper=sleeper)
    rows = json.dumps([{"id": 1}]).encode()
    params = canonical_operation_params(
        "tabular-batch", "tabular.sort", {"by": [{"column": "id"}]}
    )
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-rev",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=rows,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    run_id = opaque_run_id("req-rev")
    wave_id = f"wave-{sha256(b'req-rev').hexdigest()[:12]}"
    service.controller.dispatch_private_wave(run_id, wave_id)
    binding = RequestBinding(
        input_digest=broker_input_digest(rows),
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params=params,
        execution_fingerprint=broker_execution_fingerprint(
            request_id="req-rev",
            input_digest=broker_input_digest(rows),
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params=params,
            public_sha=_PUBLIC_SHA,
        ),
        public_sha=_PUBLIC_SHA,
    )
    BrokerRequestStore(service.state_root / "controller").save(
        BrokerRequestState(
            request_id="req-rev",
            binding=binding,
            logical_run_id=run_id,
            wave_id=wave_id,
            shard_id="shard-000000",
            status="active",
            execution_id="20",
            dispatch_count=1,
        )
    )
    revision_before = service.controller.data_plane.read_manifest(run_id).revision
    with (
        patch.object(service.controller, "reconcile_run") as reconcile,
        pytest.raises(_PollLimit),
    ):
        service.handle_payload(_UID, _request(request_id="req-rev", payload=rows))
    reconcile.assert_not_called()
    assert service.controller.data_plane.read_manifest(run_id).revision == revision_before
    _append_success_attempt(service.controller, "req-rev", b"[{\"id\":1}]")
    with patch.object(service.controller, "reconcile_run", wraps=service.controller.reconcile_run) as reconcile:
        response = service.handle_payload(_UID, _request(request_id="req-rev", payload=rows))
    assert response.status == "succeeded"
    revision_after = service.controller.data_plane.read_manifest(run_id).revision
    assert revision_after == revision_before + 1
    assert reconcile.call_count == 1
    again = service.handle_payload(_UID, _request(request_id="req-rev", payload=rows))
    assert again.status == "succeeded"
    assert service.controller.data_plane.read_manifest(run_id).revision == revision_after


def test_dispatch_history_sync_prevents_duplicate_submit_after_crash(tmp_path):
    config = _config(tmp_path)
    dispatch_calls = 0

    def handler(request):
        nonlocal dispatch_calls
        if request.method == "POST":
            dispatch_calls += 1
            return httpx.Response(201, json={"workflow_run_id": 21, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "in_progress", "conclusion": None, "updated_at": "t"},
        )

    class _PollLimit(RuntimeError):
        pass

    polls = 0

    def sleeper(_seconds: float) -> None:
        nonlocal polls
        polls += 1
        if polls >= 6:
            raise _PollLimit()

    service = _service(tmp_path, config, handler, sleeper=sleeper)
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-dispatch-sync",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=json.dumps([{"id": 1}]).encode(),
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    run_id = opaque_run_id("req-dispatch-sync")
    wave_id = f"wave-{sha256(b'req-dispatch-sync').hexdigest()[:12]}"
    service.controller.dispatch_private_wave(run_id, wave_id)
    rows = json.dumps([{"id": 1}]).encode()
    with patch.object(
        service.controller,
        "dispatch_private_wave",
        side_effect=AssertionError("must not duplicate dispatch"),
    ), pytest.raises(_PollLimit):
        service.handle_payload(
            _UID, _request(request_id="req-dispatch-sync", payload=rows)
        )
    assert dispatch_calls == 1


def test_recovery_rejects_conflicting_binding(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-conflict-recover",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=json.dumps([{"id": 1}]).encode(),
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    conflict = _request(
        request_id="req-conflict-recover",
        operation_params={"by": [{"column": "value"}]},
    )
    response = service.handle_payload(_UID, conflict)
    assert response.status == "failed"
    assert response.error_code == "request_id_conflict"


def test_registry_without_manifest_self_heals_on_exact_retry(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config, lambda request: httpx.Response(500))
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-manifest-heal",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=json.dumps([{"id": 1}]).encode(),
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    run_id = opaque_run_id("req-manifest-heal")
    run_manifest_dir = tmp_path / "runs" / run_id
    for path in run_manifest_dir.glob("**/*"):
        if path.is_file():
            path.unlink()
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-manifest-heal",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=json.dumps([{"id": 1}]).encode(),
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    assert service.controller.data_plane.read_manifest(run_id) is not None


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
