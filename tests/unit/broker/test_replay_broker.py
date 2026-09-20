import base64
import json
from unittest.mock import patch

import httpx
import pytest

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    canonical_operation_params,
    opaque_run_id,
    register_broker_private_run,
)
from portable_batch_execution.broker.protocol import BrokerExecuteResponse, parse_request
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate import (
    REQUEST_SCHEMA_VERSION,
)

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000
_JOB_PARAMS = {
    "schema_version": "pbe.replay.trade-path-scenario-evaluate-job.v1",
}


def _closed_record(**overrides):
    base = {
        "record_id": "r1",
        "model": "M1",
        "window": "6m",
        "group_label": "G",
        "month": "2010-01",
        "status": "closed",
        "cancellation_reason": None,
        "reference_notional": 1.0,
        "price_pnl": 10.0,
        "dividend_pnl": 1.0,
        "gross_pnl": 11.0,
        "entry_notional": 100.0,
        "exit_notional": 100.0,
        "traded_notional": 200.0,
        "short_notional": 50.0,
        "holding_days": 10.0,
        "holding_sessions": 10,
        "mae": -2.0,
        "mfe": 3.0,
        "adv_ratio_20d": 0.01,
        "adv_ratio_60d": 0.02,
    }
    base.update(overrides)
    return base


def _batch_payload() -> bytes:
    body = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "batch_id": "batch-1",
        "reference_notional": 1.0,
        "scenario": {
            "commission_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "borrow_bps_per_year": 100.0,
            "short_available": True,
        },
        "records": [_closed_record()],
    }
    return json.dumps(body).encode()


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
                        [
                            "replay-batch",
                            "replay.trade_path_scenario_evaluate",
                        ],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def _service(tmp_path, config: BrokerConfig) -> UnixBrokerService:
    backend = GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        ),
    )
    controller = A1Controller(tmp_path, backend=backend)
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )


def _request(
    *,
    request_id: str = "req-replay-1",
    operation_params: dict | None = None,
    payload: bytes | None = None,
) -> dict:
    body = payload if payload is not None else _batch_payload()
    return {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": request_id,
        "pack": "replay-batch",
        "operation": "replay.trade_path_scenario_evaluate",
        "operation_params": operation_params
        if operation_params is not None
        else dict(_JOB_PARAMS),
        "input_media_type": "application/json",
        "input_b64": base64.b64encode(body).decode("ascii"),
    }


def test_replay_canonical_operation_params_match_pack():
    validated = canonical_operation_params(
        "replay-batch", "replay.trade_path_scenario_evaluate", _JOB_PARAMS
    )
    assert validated == _JOB_PARAMS


def test_replay_operation_params_validation_enforced(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    bad = _request(operation_params={"schema_version": "wrong"})
    response = service.handle_payload(_UID, bad)
    assert response.status == "failed"
    assert response.error_code == "operation_params_invalid"


def test_replay_unauthorized_peer_rejected(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    response = service.handle_payload(4242, _request())
    assert response.status == "failed"
    assert response.error_code == "peer_not_authorized"
    run_dir = tmp_path / "controller" / "closed_waves" / opaque_run_id("req-replay-1")
    assert not run_dir.exists()


def test_replay_allowlisted_request_passes_validation(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    stub = BrokerExecuteResponse(request_id="req-replay-1", status="succeeded")
    with patch.object(UnixBrokerService, "_drive_to_terminal", return_value=stub):
        response = service.handle_payload(_UID, _request())
    assert response.error_code != "operation_params_invalid"
    assert response.status == "succeeded"


def test_replay_request_id_binding_conflict(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    body = _batch_payload()
    register_broker_private_run(
        state_root=tmp_path,
        request_id="req-replay-conflict",
        pack="replay-batch",
        operation="replay.trade_path_scenario_evaluate",
        operation_params=_JOB_PARAMS,
        input_bytes=body,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    conflict = _request(
        request_id="req-replay-conflict",
        payload=body,
        operation_params={
            "schema_version": "pbe.replay.trade-path-scenario-evaluate-job.v1",
        },
    )
    conflict["input_b64"] = base64.b64encode(b'{"different": true}').decode("ascii")
    response = service.handle_payload(_UID, conflict)
    assert response.status == "failed"
    assert response.error_code == "request_id_conflict"


def test_replay_forbidden_executable_in_operation_params():
    payload = _request(operation_params={"executable": "rm -rf /"})
    with pytest.raises(ValueError, match="reserved request field"):
        parse_request(payload)


def test_replay_forbidden_path_like_string_in_operation_params():
    payload = _request(
        operation_params={
            **_JOB_PARAMS,
            "note": "../secrets",
        }
    )
    with pytest.raises(ValueError, match="path and URL values are not accepted"):
        parse_request(payload)


def test_replay_eval_batch_not_broker_pack(tmp_path):
    config = _config(tmp_path)
    service = _service(tmp_path, config)
    invalid = _request()
    invalid["pack"] = "replay-eval-batch"
    invalid["operation"] = "replay_eval.replay"
    invalid["operation_params"] = {}
    with pytest.raises(ValueError):
        parse_request(invalid)
    response = service.handle_payload(_UID, invalid)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"


def test_register_broker_private_run_replay_batch(tmp_path):
    body = _batch_payload()
    job, wave, shard, manifest = register_broker_private_run(
        state_root=tmp_path,
        request_id="req-replay-reg",
        pack="replay-batch",
        operation="replay.trade_path_scenario_evaluate",
        operation_params=_JOB_PARAMS,
        input_bytes=body,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    assert job.pack == "replay-batch"
    assert job.operation == "replay.trade_path_scenario_evaluate"
    assert job.operation_params == _JOB_PARAMS
    assert shard.input_refs[0].media_type == "application/json"
    assert manifest.logical_run_id == job.logical_run_id
    assert wave.wave_id.startswith("wave-")
