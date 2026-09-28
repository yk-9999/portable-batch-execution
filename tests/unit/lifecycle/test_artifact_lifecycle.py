import base64
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from portable_batch_execution.broker.client import UnixBrokerClient
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    broker_execution_fingerprint,
    broker_shard_input_digest,
    canonical_operation_params,
    opaque_run_id,
    opaque_wave_id,
    register_broker_private_run,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.broker.state import BrokerRequestState, BrokerRequestStore, RequestBinding
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.lifecycle.delivery import DeliveryRecordStore
from portable_batch_execution.lifecycle.deletion import DeletionReceiptStore
from portable_batch_execution.lifecycle.gc import apply_gc, plan_gc
from portable_batch_execution.lifecycle.holds import HoldStore, HoldRecord
from portable_batch_execution.lifecycle.paths import lifecycle_policy_path
from portable_batch_execution.lifecycle.policy import LifecyclePolicy
from portable_batch_execution.lifecycle.reachability import artifact_payload_path, build_reachability_index
from tests.unit.broker.test_unix_broker import (
    _UID,
    _append_success_attempt,
    _config,
    _request,
)


def _local_service(tmp_path, config: BrokerConfig) -> UnixBrokerService:
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=A1Controller(tmp_path, backend=None),
        poll_interval_seconds=0.0,
    )

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"


def _write_policy(tmp_path: Path, *, grace: int = 0, legacy: int = 0) -> LifecyclePolicy:
    policy = LifecyclePolicy(
        schema_version="pbe.lifecycle-policy.v1",
        delivery_grace_seconds=grace,
        legacy_retention_seconds=legacy,
    )
    lifecycle_policy_path(tmp_path).write_text(
        json.dumps(policy.to_json()) + "\n", encoding="utf-8"
    )
    return policy


def _register_request(service: UnixBrokerService, request_id: str):
    req = _request(request_id=request_id)
    body = json.dumps([{"id": 2}, {"id": 1}]).encode()
    static = service.config.static_input_refs_for(req["pack"], req["operation"])
    input_digest = broker_shard_input_digest(body, static)
    validated = canonical_operation_params(req["pack"], req["operation"], req["operation_params"])
    register_broker_private_run(
        state_root=service.state_root,
        request_id=request_id,
        pack=req["pack"],
        operation=req["operation"],
        operation_params=req["operation_params"],
        input_bytes=body,
        input_media_type=req["input_media_type"],
        public_sha=service.config.public_sha,
        static_input_refs=static,
    )
    binding = RequestBinding(
        input_digest=input_digest,
        pack=req["pack"],
        operation=req["operation"],
        operation_params=validated,
        execution_fingerprint=broker_execution_fingerprint(
            request_id=request_id,
            input_digest=input_digest,
            pack=req["pack"],
            operation=req["operation"],
            operation_params=validated,
            public_sha=service.config.public_sha,
        ),
        public_sha=service.config.public_sha,
    )
    BrokerRequestStore(service.state_root / "controller").save(
        BrokerRequestState(
            request_id=request_id,
            binding=binding,
            logical_run_id=opaque_run_id(request_id),
            wave_id=opaque_wave_id(request_id),
            shard_id="shard-000000",
            status="active",
        )
    )


def _succeed_broker(service: UnixBrokerService, request_id: str, output: bytes):
    _register_request(service, request_id)
    _append_success_attempt(service.controller, request_id, output)
    state = BrokerRequestStore(service.state_root / "controller").load(request_id)
    assert state is not None
    digest = f"sha256:{sha256(output).hexdigest()}"
    state.status = "succeeded"
    state.output_sha256 = digest
    state.output_size_bytes = len(output)
    state.output_media_type = "application/json"
    BrokerRequestStore(service.state_root / "controller").save(state)


def _delivery_commit(service: UnixBrokerService, request_id: str, digest: str, size: int):
    response = service.handle_delivery_commit(
        _UID,
        {
            "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
            "request_id": request_id,
            "output_sha256": digest,
            "output_size_bytes": size,
        },
    )
    assert response.status == "accepted"


def test_durable_write_failure_does_not_record_delivery(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    request_id = "req-durable-fail"
    output = b"[{\"id\":1}]"
    _succeed_broker(service, request_id, output)
    response = service.handle_payload(_UID, _request(request_id=request_id))
    assert response.status == "succeeded"
    assert response.output_size_bytes == len(output)

    def sender(frame: bytes) -> bytes:
        payload = json.loads(frame.decode("utf-8"))
        if payload.get("schema_version", "").endswith("delivery-commit.v1"):
            return service.handle_message(_UID, payload)
        return service.handle_message(_UID, payload)

    client = UnixBrokerClient(sender=sender)
    destination = tmp_path / "out" / "result.json"
    with patch(
        "portable_batch_execution.broker.client.persist_verified_bytes",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError):
            client.execute_and_durably_persist(
                _request(request_id=request_id), destination
            )
    assert not DeliveryRecordStore(tmp_path).load(request_id)
    retry = service.handle_payload(_UID, _request(request_id=request_id))
    assert retry.status == "succeeded"


def test_durable_success_auto_records_delivery(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    request_id = "req-durable-ok"
    output = b"[{\"id\":1}]"
    _succeed_broker(service, request_id, output)

    client = UnixBrokerClient(
        sender=lambda frame: service.handle_message(_UID, json.loads(frame.decode("utf-8")))
    )
    destination = tmp_path / "out" / "result.json"
    result = client.execute_and_durably_persist(
        _request(request_id=request_id), destination
    )
    assert result.delivery is not None
    assert destination.is_file()
    assert DeliveryRecordStore(tmp_path).load(request_id) is not None


def test_gc_protects_live_broker_and_hold(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=99999)
    active_output = b"active"
    delivered_output = b"delivered"
    _succeed_broker(service, "req-active", active_output)
    _succeed_broker(service, "req-del", delivered_output)
    digest_del = f"sha256:{sha256(delivered_output).hexdigest()}"
    _delivery_commit(service, "req-del", digest_del, len(delivered_output))
    HoldStore(tmp_path).put(
        HoldRecord(
            hold_id="h1",
            kind="hold",
            artifact_digests=(digest_del,),
            reason="test",
            created_at=datetime.now(UTC),
        )
    )
    report = plan_gc(tmp_path, policy, mode="normal")
    assert report.blocked is False
    protected_digests = {item.digest for item in report.protected}
    active_digest = f"sha256:{sha256(active_output).hexdigest()}"
    assert active_digest in protected_digests
    assert digest_del in protected_digests


def test_gc_deletes_after_grace_when_unreachable(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"gone"
    _succeed_broker(service, "req-gc", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-gc", digest, len(output))
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-gc")
    assert state is not None
    state.status = "exhausted"
    store.save(state)
    index = build_reachability_index(tmp_path)
    assert not index.is_protected(digest)
    report = apply_gc(tmp_path, policy, mode="normal")
    assert report.candidate_count == 1
    assert not artifact_payload_path(tmp_path, digest).is_file()
    assert DeletionReceiptStore(tmp_path).load(digest) is not None
    rerun = apply_gc(tmp_path, policy, mode="normal")
    assert rerun.deleted == ()


def test_gc_grace_protects_payload(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=3600, legacy=0)
    output = b"grace"
    _succeed_broker(service, "req-grace", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-grace", digest, len(output))
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-grace")
    state.status = "exhausted"
    store.save(state)
    report = plan_gc(tmp_path, policy, mode="normal", now=datetime.now(UTC))
    assert digest in {item.digest for item in report.protected}


def test_gc_crash_rerun_converges(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"crash"
    _succeed_broker(service, "req-crash", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-crash", digest, len(output))
    state = BrokerRequestStore(service.state_root / "controller").load("req-crash")
    state.status = "exhausted"
    BrokerRequestStore(service.state_root / "controller").save(state)
    with pytest.raises(RuntimeError):
        apply_gc(
            tmp_path,
            policy,
            mode="normal",
            crash_after_unlink={digest},
        )
    assert not artifact_payload_path(tmp_path, digest).is_file()
    assert DeletionReceiptStore(tmp_path).load(digest) is None
    apply_gc(tmp_path, policy, mode="normal")
    assert DeletionReceiptStore(tmp_path).load(digest) is not None


def test_legacy_gc_dry_run_and_unknown_state(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    digest = "sha256:" + "a" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    report = plan_gc(tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=1))
    assert report.artifact_count >= 1
    run_dir = tmp_path / "runs" / "run-bad"
    run_dir.mkdir(parents=True)
    (run_dir / "latest.json").write_text("{not json", encoding="utf-8")
    blocked = plan_gc(tmp_path, policy, mode="legacy")
    assert blocked.blocked is True


def test_broker_success_includes_output_size_bytes(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    request_id = "req-size"
    output = b"[1,2]"
    _succeed_broker(service, request_id, output)
    response = service.handle_payload(_UID, _request(request_id=request_id))
    assert response.output_size_bytes == len(output)
    payload = json.loads(response.model_dump_json())
    assert "output_size_bytes" in payload
