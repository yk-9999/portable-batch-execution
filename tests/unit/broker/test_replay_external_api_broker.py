import base64
import json

import httpx
import pytest

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    canonical_operation_params,
    register_broker_private_run,
)
from portable_batch_execution.broker.protocol import parse_request
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.controller.a1_controller import A1Controller

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000
_OPERATION = "replay_eval.external_api_evaluation"
_PARAMS = {"provider": "nvidia-openai-compatible"}


def _config(tmp_path) -> BrokerConfig:
    path = tmp_path / "broker-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        ["replay-eval-batch", _OPERATION],
                        ["tabular-batch", "tabular.sort"],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def _request(**overrides):
    body = json.dumps({"model": "closed", "messages": []}).encode()
    payload = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": "req-replay-1",
        "pack": "replay-eval-batch",
        "operation": _OPERATION,
        "operation_params": _PARAMS,
        "input_media_type": "application/json",
        "input_b64": base64.b64encode(body).decode("ascii"),
    }
    payload.update(overrides)
    return payload


def _service(tmp_path, config: BrokerConfig) -> UnixBrokerService:
    controller = A1Controller(
        tmp_path,
        backend=GitHubActionsBackend(
            "owner",
            "repo",
            "execute-wave.yml",
            private_data_plane=True,
            client=httpx.Client(
                base_url="https://api.github.com",
                transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            ),
        ),
    )
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )


def test_protocol_accepts_replay_eval_batch_pack():
    request = parse_request(_request())
    assert request.pack == "replay-eval-batch"
    assert request.operation == _OPERATION


def test_canonical_operation_params_accepts_closed_provider():
    assert (
        canonical_operation_params("replay-eval-batch", _OPERATION, _PARAMS)
        == _PARAMS
    )


@pytest.mark.parametrize(
    "operation_params",
    (
        {},
        {"provider": "other"},
        {"provider": "nvidia-openai-compatible", "url": "https://evil.example"},
    ),
)
def test_canonical_operation_params_rejects_non_closed_params(operation_params):
    with pytest.raises(ValueError):
        canonical_operation_params(
            "replay-eval-batch", _OPERATION, operation_params
        )


def test_canonical_operation_params_rejects_other_replay_operations():
    with pytest.raises(ValueError, match="unsupported broker operation"):
        canonical_operation_params("replay-eval-batch", "replay_eval.replay", _PARAMS)


def test_register_broker_run_uses_external_api_security_profile(tmp_path):
    body = b'{"messages":[]}'
    job, _, _, _ = register_broker_private_run(
        state_root=tmp_path,
        request_id="req-replay-security",
        pack="replay-eval-batch",
        operation=_OPERATION,
        operation_params=_PARAMS,
        input_bytes=body,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    assert job.security_profile == "external-api"
    assert job.operation_params == _PARAMS


def test_register_broker_run_keeps_tabular_offline_security_profile(tmp_path):
    body = b"[{\"id\":1}]"
    job, _, _, _ = register_broker_private_run(
        state_root=tmp_path,
        request_id="req-tabular-security",
        pack="tabular-batch",
        operation="tabular.sort",
        operation_params={"by": [{"column": "id"}]},
        input_bytes=body,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    assert job.security_profile == "offline"


def test_service_rejects_wrong_replay_operation(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    response = service.handle_payload(
        _UID,
        _request(operation="replay_eval.replay"),
    )
    assert response.status == "failed"
    assert response.error_code == "operation_not_allowed"


def test_service_rejects_invalid_replay_operation_params(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    response = service.handle_payload(
        _UID,
        _request(operation_params={"provider": "nvidia-openai-compatible", "extra": "x"}),
    )
    assert response.status == "failed"
    assert response.error_code == "operation_params_invalid"
