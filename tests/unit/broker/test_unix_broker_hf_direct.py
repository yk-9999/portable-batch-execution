import json
from datetime import UTC, datetime
from hashlib import sha256
from unittest.mock import patch

import httpx

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.hf_direct import BrokerHfDirectExecuteResponse
from portable_batch_execution.broker.planning import opaque_run_id, register_broker_hf_direct_run
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.broker.state import BrokerRequestStore
from portable_batch_execution.contracts import ShardAttemptRecord
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.transport.hf_bucket import (
    HF_BUCKET_REF_SCHEMA,
    artifact_ref_from_hf_object,
)

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000


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


def _service(tmp_path, config, handler):
    controller = A1Controller(tmp_path, backend=_mock_backend(handler))
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
        sleeper=lambda _seconds: None,
    )

_FIXED_SET_OPERATION = "replay.trade_path_scenario_evaluate_fixed_set"
_FIXED_SET_JOB_SCHEMA = "pbe.replay.trade-path-scenario-evaluate-fixed-set-job.v1"
_REQUEST_ID = sha256(b"hf-direct-broker-test").hexdigest()


def _hf_config(tmp_path):
    path = tmp_path / "broker-config-hf.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        [
                            "replay-batch",
                            _FIXED_SET_OPERATION,
                        ],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    from portable_batch_execution.broker.config import BrokerConfig

    return BrokerConfig.load(path)


def _input_hf_ref() -> dict:
    payload = b'{"schema_version":"pbe.replay.trade-path-scenario-evaluate.v1","records":[]}'
    digest = sha256(payload).hexdigest()
    return {
        "schema_version": HF_BUCKET_REF_SCHEMA,
        "bucket_id": "yamauchiJP/system-trading-data",
        "object_path": "pair-trading/v1/public-eval/normalized/digest-1/batch-00000-0123456789abcdef.json",
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": "application/json",
    }


def _hf_request(
    *,
    request_id: str = _REQUEST_ID,
    output_object_path: str = "pair-trading/v1/public-eval/results/digest-1/batch-00000-out.json",
) -> dict:
    return {
        "schema_version": "pbe.a1-unix-broker.hf-direct-request.v1",
        "request_id": request_id,
        "pack": "replay-batch",
        "operation": _FIXED_SET_OPERATION,
        "operation_params": {
            "schema_version": _FIXED_SET_JOB_SCHEMA,
            "transport_profile": "hf_bucket_direct",
            "output_object_path": output_object_path,
        },
        "input_media_type": "application/json",
        "input_hf_ref": _input_hf_ref(),
    }


def _append_hf_success_attempt(
    controller: A1Controller,
    request_id: str,
    *,
    output_object_path: str,
) -> None:
    run_id = opaque_run_id(request_id)
    wave_id = f"wave-{sha256(request_id.encode('utf-8')).hexdigest()[:12]}"
    payload = controller.registry.resolve_wave(run_id, wave_id)
    shard = payload["shards"][0]
    output_bytes = (
        b'{"schema_version":"pbe.replay.trade-path-scenario-evaluate-fixed-set-result.v1"}'
    )
    digest = sha256(output_bytes).hexdigest()
    output_ref = artifact_ref_from_hf_object(
        object_path=output_object_path,
        sha256_hex=digest,
        size_bytes=len(output_bytes),
        media_type="application/json",
    )
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
            output_digest=digest,
        )
    )


def _github_handler():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json={"workflow_run_id": 1, "html_url": "https://run"})
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    return handler


def test_hf_direct_success_metadata_only_without_artifact_reader(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    request = _hf_request()
    output_path = request["operation_params"]["output_object_path"]
    registration_calls = 0
    real_register = register_broker_hf_direct_run

    def spy_register(**kwargs):
        nonlocal registration_calls
        registration_calls += 1
        return real_register(**kwargs)

    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_hf_success_attempt(
            service.controller, _REQUEST_ID, output_object_path=output_path
        )
        return ref

    def fail_read(ref):
        raise AssertionError("HF-direct broker must not read Private Data Plane artifacts")

    service._artifact_reader.read = fail_read  # type: ignore[method-assign]

    with (
        patch(
            "portable_batch_execution.broker.service.register_broker_hf_direct_run",
            side_effect=spy_register,
        ),
        patch.object(service.controller, "dispatch_private_wave", dispatch_with_success),
    ):
        response = service.handle_payload(_UID, request)

    assert isinstance(response, BrokerHfDirectExecuteResponse)
    assert response.status == "succeeded"
    assert response.output_hf_ref is not None
    assert response.output_hf_ref["object_path"] == output_path
    assert registration_calls == 1
    dumped = response.model_dump(exclude_none=True)
    assert "output_b64" not in dumped
    assert dumped["schema_version"] == "pbe.a1-unix-broker.hf-direct-response.v1"


def test_hf_direct_rejects_byte_fields(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    bad = _hf_request()
    bad["input_b64"] = "abcd"
    response = service.handle_payload(_UID, bad)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"


def test_hf_direct_recovery_after_state_loss(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    request = _hf_request(request_id=sha256(b"recover-hf").hexdigest())
    output_path = request["operation_params"]["output_object_path"]
    real_dispatch = service.controller.dispatch_private_wave

    def dispatch_with_success(run_id, wave_id):
        ref = real_dispatch(run_id, wave_id)
        _append_hf_success_attempt(
            service.controller, request["request_id"], output_object_path=output_path
        )
        return ref

    with patch.object(service.controller, "dispatch_private_wave", dispatch_with_success):
        first = service.handle_payload(_UID, request)
    assert first.status == "succeeded"
    store = BrokerRequestStore(service.state_root / "controller")
    store._path(request["request_id"]).unlink()
    second = service.handle_payload(_UID, request)
    assert second.status == "succeeded"
    assert second.output_hf_ref == first.output_hf_ref


def test_legacy_fixed_set_with_input_b64_rejected(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    legacy = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": sha256(b"legacy-fixed").hexdigest(),
        "pack": "replay-batch",
        "operation": _FIXED_SET_OPERATION,
        "operation_params": {
            "schema_version": _FIXED_SET_JOB_SCHEMA,
            "transport_profile": "hf_bucket_direct",
            "output_object_path": "pair-trading/v1/public-eval/results/x.json",
        },
        "input_media_type": "application/json",
        "input_b64": "e30=",
    }
    response = service.handle_payload(_UID, legacy)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"
