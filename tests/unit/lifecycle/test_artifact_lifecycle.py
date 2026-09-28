import json
import os
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
from portable_batch_execution.broker.state import (
    BROKER_TERMINAL_REQUEST_STATUSES,
    BrokerRequestState,
    BrokerRequestStore,
    RequestBinding,
)
from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    JobSpec,
    Provenance,
    RunManifest,
    ShardAttemptRecord,
    ShardCorrectnessSpec,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.controller.closed_wave_registry import ClosedWaveRegistry
from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.lifecycle import gc as gc_module
from portable_batch_execution.lifecycle.cli import main as lifecycle_cli_main
from portable_batch_execution.lifecycle.deletion import (
    DeletionIntent,
    DeletionIntentStore,
    DeletionReceiptStore,
)
from portable_batch_execution.lifecycle.delivery import (
    DeliveryRecord,
    DeliveryRecordStore,
)
from portable_batch_execution.lifecycle.gc import apply_gc, plan_gc
from portable_batch_execution.lifecycle.holds import HoldRecord, HoldStore
from portable_batch_execution.lifecycle.lock import (
    LifecycleLockError,
    LifecycleStateLock,
)
from portable_batch_execution.lifecycle.paths import lifecycle_policy_path
from portable_batch_execution.lifecycle.policy import LifecyclePolicy
from portable_batch_execution.lifecycle.reachability import (
    artifact_payload_path,
    build_reachability_index,
)
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


def _write_policy(
    tmp_path: Path,
    *,
    grace: int = 0,
    legacy: int = 0,
    authoritative_reference_files: tuple[str, ...] = (),
    authoritative_reference_roots: tuple[str, ...] = (),
) -> LifecyclePolicy:
    policy = LifecyclePolicy(
        schema_version="pbe.lifecycle-policy.v1",
        delivery_grace_seconds=grace,
        legacy_retention_seconds=legacy,
        authoritative_reference_files=authoritative_reference_files,
        authoritative_reference_roots=authoritative_reference_roots,
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
    validated = canonical_operation_params(
        req["pack"], req["operation"], req["operation_params"]
    )
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


def _delivery_commit(
    service: UnixBrokerService, request_id: str, digest: str, size: int
):
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
    output = b'[{"id":1}]'
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
    with (
        patch(
            "portable_batch_execution.broker.client.persist_verified_bytes",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError),
    ):
        client.execute_and_durably_persist(_request(request_id=request_id), destination)
    assert not DeliveryRecordStore(tmp_path).load(request_id)
    retry = service.handle_payload(_UID, _request(request_id=request_id))
    assert retry.status == "succeeded"


def test_durable_success_auto_records_delivery(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    request_id = "req-durable-ok"
    output = b'[{"id":1}]'
    _succeed_broker(service, request_id, output)

    client = UnixBrokerClient(
        sender=lambda frame: service.handle_message(
            _UID, json.loads(frame.decode("utf-8"))
        )
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
    assert state.status == "succeeded"
    index = build_reachability_index(tmp_path, policy)
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
    with pytest.raises(RuntimeError):
        apply_gc(
            tmp_path,
            policy,
            mode="normal",
            crash_after_unlink={digest},
        )
    assert not artifact_payload_path(tmp_path, digest).is_file()
    assert DeletionReceiptStore(tmp_path).load(digest) is None
    assert DeletionIntentStore(tmp_path).load(digest) is not None
    apply_gc(tmp_path, policy, mode="normal")
    assert DeletionReceiptStore(tmp_path).load(digest) is not None
    assert DeletionIntentStore(tmp_path).load(digest) is None
    noop = apply_gc(tmp_path, policy, mode="normal")
    assert noop.deleted == ()


def test_succeeded_undelivered_broker_output_stays_protected(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"awaiting-delivery"
    _succeed_broker(service, "req-await", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    state = BrokerRequestStore(service.state_root / "controller").load("req-await")
    assert state is not None
    assert state.status == "succeeded"
    index = build_reachability_index(tmp_path, policy)
    assert index.is_protected(digest)
    report = plan_gc(tmp_path, policy, mode="normal")
    assert digest in {item.digest for item in report.protected}
    review = report.to_dict()
    assert any(item["reasons"] for item in review["protected"])
    assert any(
        pair["logical_run_id"] and pair["artifact_digest"]
        for pair in review["run_reference_pairs"]
    )
    assert review["protection_reason_aggregates"]


def test_legacy_gc_dry_run_review_json_includes_candidates_and_protected(
    tmp_path, capsys
):
    _write_policy(tmp_path, grace=0, legacy=0)
    digest = "sha256:" + "e" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy-candidate")
    old = datetime.now(UTC) - timedelta(days=30)
    os.utime(path, (old.timestamp(), old.timestamp()))
    rc = lifecycle_cli_main(
        [
            "legacy-gc-dry-run",
            "--state-root",
            str(tmp_path),
            "--policy-file",
            str(lifecycle_policy_path(tmp_path)),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] >= 1
    assert any(item["digest"] == digest for item in payload["candidates"])
    assert payload["candidates"][0]["effective_last_use"] is not None
    assert payload["candidates"][0]["effective_age_seconds"] is not None
    assert "protected" in payload
    assert "unknown_references" in payload
    assert "run_reference_pairs" in payload
    assert isinstance(payload["protection_reason_aggregates"], list)
    assert isinstance(payload["run_reference_pairs"], list)


def test_legacy_gc_crash_after_unlink_converges_receipt(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    digest = "sha256:" + "f" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy-crash")
    old = datetime.now(UTC) - timedelta(days=30)
    os.utime(path, (old.timestamp(), old.timestamp()))
    with pytest.raises(RuntimeError):
        apply_gc(
            tmp_path,
            policy,
            mode="legacy",
            now=datetime.now(UTC),
            crash_after_unlink={digest},
        )
    assert not path.is_file()
    assert DeletionIntentStore(tmp_path).load(digest) is not None
    receipt = apply_gc(tmp_path, policy, mode="legacy", now=datetime.now(UTC))
    assert DeletionReceiptStore(tmp_path).load(digest) is not None
    assert receipt.deleted
    assert (
        apply_gc(tmp_path, policy, mode="legacy", now=datetime.now(UTC)).deleted == ()
    )


def test_legacy_gc_dry_run_and_unknown_state(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    digest = "sha256:" + "a" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    report = plan_gc(
        tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=1)
    )
    assert report.artifact_count >= 1
    run_dir = tmp_path / "runs" / "run-bad"
    run_dir.mkdir(parents=True)
    (run_dir / "latest.json").write_text("{not json", encoding="utf-8")
    blocked = plan_gc(tmp_path, policy, mode="legacy")
    assert blocked.blocked is True


def test_external_reference_file_protects_payload(tmp_path):
    digest = "sha256:" + "b" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"config-bound")
    config_path = tmp_path / "runtime-config.json"
    config_path.write_text(
        json.dumps(
            {
                "static_input_bindings": [
                    {
                        "object_id": "b" * 64,
                        "uri": path.as_uri(),
                        "sha256": digest,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    policy = _write_policy(
        tmp_path,
        grace=0,
        legacy=0,
        authoritative_reference_files=(str(config_path),),
    )
    report = plan_gc(
        tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=30)
    )
    assert digest in {item.digest for item in report.protected}
    assert report.blocked is False


def test_malformed_external_reference_blocks_apply(tmp_path):
    digest = "sha256:" + "c" * 64
    path = artifact_payload_path(tmp_path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"orphan")
    config_path = tmp_path / "broken-config.json"
    config_path.write_text("{not-json", encoding="utf-8")
    policy = _write_policy(
        tmp_path,
        grace=0,
        legacy=0,
        authoritative_reference_files=(str(config_path),),
    )
    report = plan_gc(tmp_path, policy, mode="normal")
    assert report.blocked is True
    apply_report = apply_gc(tmp_path, policy, mode="normal")
    assert apply_report.blocked is True
    assert path.is_file()


def test_legacy_gc_retains_old_payload_with_recent_broker_ref(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=3600)
    output = b"legacy-retained"
    _succeed_broker(service, "req-legacy", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-legacy")
    assert state is not None
    state.status = "exhausted"
    store.save(state)
    payload_path = artifact_payload_path(tmp_path, digest)
    old = datetime.now(UTC) - timedelta(days=30)
    os.utime(payload_path, (old.timestamp(), old.timestamp()))
    now = datetime.now(UTC)
    report = plan_gc(tmp_path, policy, mode="legacy", now=now)
    assert digest in {item.digest for item in report.protected}


def test_normal_gc_waits_for_all_transport_refs_on_shared_digest(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"shared-output"
    _succeed_broker(service, "req-shared-a", output)
    _succeed_broker(service, "req-shared-b", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-shared-a", digest, len(output))
    report = plan_gc(tmp_path, policy, mode="normal", now=datetime.now(UTC))
    assert digest in {item.digest for item in report.protected}
    apply_report = apply_gc(tmp_path, policy, mode="normal")
    assert artifact_payload_path(tmp_path, digest).is_file()
    assert apply_report.deleted == ()


def test_apply_gc_refuses_deletion_when_authoritative_state_unreadable(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"blocked-delete"
    _succeed_broker(service, "req-blocked", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-blocked", digest, len(output))
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-blocked")
    assert state is not None
    run_dir = tmp_path / "runs" / state.logical_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latest.json").write_text("{broken", encoding="utf-8")
    apply_report = apply_gc(tmp_path, policy, mode="normal")
    assert apply_report.blocked is True
    assert artifact_payload_path(tmp_path, digest).is_file()


def _external_reference_candidate(
    tmp_path, request_id, external_path, initial_text="[]"
):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    output = f"external-ref-{request_id}".encode()
    _succeed_broker(service, request_id, output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, request_id, digest, len(output))
    external_path.write_text(initial_text, encoding="utf-8")
    policy = _write_policy(
        tmp_path,
        grace=0,
        authoritative_reference_files=(str(external_path),),
    )
    return policy, digest


def test_apply_gc_rechecks_external_refs_immediately_before_unlink(tmp_path):
    external_path = tmp_path / "external-references.json"
    policy, digest = _external_reference_candidate(
        tmp_path, "req-external-late-ref", external_path
    )
    original = gc_module.build_reachability_index
    calls = 0

    def build_then_publish_ref(state_root, current_policy=None):
        nonlocal calls
        calls += 1
        index = original(state_root, current_policy)
        if calls == 1:
            external_path.write_text(
                json.dumps(
                    {
                        "artifact": {
                            "object_id": digest.removeprefix("sha256:"),
                            "sha256": digest,
                        }
                    }
                ),
                encoding="utf-8",
            )
        return index

    with patch.object(
        gc_module, "build_reachability_index", side_effect=build_then_publish_ref
    ):
        report = apply_gc(tmp_path, policy, mode="normal")

    assert calls == 2
    assert report.deleted == ()
    assert artifact_payload_path(tmp_path, digest).is_file()


def test_apply_gc_blocks_candidate_if_external_refs_turn_unreadable(tmp_path):
    external_path = tmp_path / "external-references.json"
    policy, digest = _external_reference_candidate(
        tmp_path, "req-external-invalid", external_path
    )
    original = gc_module.build_reachability_index
    calls = 0

    def build_then_invalidate_refs(state_root, current_policy=None):
        nonlocal calls
        calls += 1
        index = original(state_root, current_policy)
        if calls == 1:
            external_path.write_text("{invalid", encoding="utf-8")
        return index

    with patch.object(
        gc_module, "build_reachability_index", side_effect=build_then_invalidate_refs
    ):
        report = apply_gc(tmp_path, policy, mode="normal")

    assert calls == 2
    assert report.blocked is True
    assert report.deleted == ()
    assert artifact_payload_path(tmp_path, digest).is_file()


def test_local_data_plane_write_acquires_lifecycle_lock(tmp_path):
    with patch.object(LifecycleStateLock, "acquire", autospec=True) as acquire:
        LocalFilesystemDataPlane(tmp_path).write(
            b"new-artifact", "application/octet-stream"
        )
    acquire.assert_called_once()


def test_delivered_succeeded_zero_grace_becomes_gc_candidate(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"eligible"
    _succeed_broker(service, "req-eligible", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-eligible", digest, len(output))
    index = build_reachability_index(tmp_path, policy)
    assert not index.is_protected(digest)
    report = plan_gc(tmp_path, policy, mode="normal", now=datetime.now(UTC))
    assert digest in {item.digest for item in report.candidates}


def test_terminal_broker_manifest_final_ref_does_not_block_gc(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"broker-final"
    _succeed_broker(service, "req-final", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-final", digest, len(output))
    index = build_reachability_index(tmp_path, policy)
    assert not index.is_protected(digest)
    report = plan_gc(tmp_path, policy, mode="normal")
    assert digest in {item.digest for item in report.candidates}


def _register_closed_wave_input(
    tmp_path: Path,
    *,
    run_id: str,
    input_ref: ArtifactRef,
    wave_id: str = "wave-000001",
) -> None:
    now = datetime.now(UTC)
    job = JobSpec(
        job_id="job",
        logical_run_id=run_id,
        pack="tabular-batch",
        operation="tabular.sort",
        input_manifest_ref=input_ref,
        sharding=ShardCorrectnessSpec(mode="independent"),
        execution=ExecutionPolicy(max_parallel=1, max_attempts_per_shard=4),
        security_profile="offline",
        provenance=Provenance(producer="test", revision="1", created_at=now),
    )
    shard = ShardSpec(
        logical_run_id=run_id,
        shard_id="shard-000000",
        ordinal=0,
        correctness=job.sharding,
        input_refs=(input_ref,),
        input_digest="sha256:" + "1" * 64,
        execution_fingerprint="fp",
    )
    wave = WaveSpec(
        logical_run_id=run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=(shard.shard_id,),
        max_parallel=1,
    )
    ClosedWaveRegistry(tmp_path / "controller").register_closed_wave(
        job, wave, (shard,)
    )


def _historical_closed_wave_job_payload(
    *,
    run_id: str,
    input_manifest_ref: ArtifactRef,
    shard_input_refs: tuple[ArtifactRef, ...],
    created_at: datetime,
) -> dict[str, object]:
    """Job/shard bundle that is reachability-valid but fails current JobSpec semantics."""
    job: dict[str, object] = {
        "schema_version": "1",
        "job_id": "job-historical",
        "logical_run_id": run_id,
        "pack": "tabular-batch",
        "operation": "tabular.text_event_features.v1",
        "input_manifest_ref": input_manifest_ref.model_dump(mode="json"),
        "sharding": {"mode": "independent"},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
        "security_profile": "offline",
        "provenance": {
            "producer": "test",
            "revision": "legacy",
            "created_at": created_at.isoformat(),
        },
        "operation_params": {
            "import_path": "legacy.module:build_features",
            "feature_columns": ["text", "event_id"],
        },
    }
    with pytest.raises(ValueError):
        JobSpec.model_validate(job)
    shard: dict[str, object] = {
        "schema_version": "1",
        "logical_run_id": run_id,
        "shard_id": "shard-000000",
        "ordinal": 0,
        "correctness": {"mode": "independent"},
        "input_refs": [ref.model_dump(mode="json") for ref in shard_input_refs],
        "input_digest": "sha256:" + "1" * 64,
        "execution_fingerprint": "fp",
    }
    return {
        "job": job,
        "wave": {
            "schema_version": "1",
            "logical_run_id": run_id,
            "wave_id": "wave-historical",
            "ordinal": 0,
            "shard_ids": ["shard-000000"],
            "max_parallel": 1,
        },
        "shards": [shard],
    }


def _write_closed_wave_bundle(
    tmp_path: Path,
    *,
    run_id: str,
    bundle: dict[str, object],
    wave_id: str = "wave-historical",
) -> Path:
    run_dir = tmp_path / "controller" / "closed_waves" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    wave_path = run_dir / f"{wave_id}.wave.json"
    wave_path.write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    return wave_path


def _age_payload(path: Path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


def _non_broker_manifest(
    run_id: str,
    *,
    status: str,
    final_output_refs: tuple[ArtifactRef, ...] = (),
) -> RunManifest:
    now = datetime.now(UTC)
    return RunManifest(
        logical_run_id=run_id,
        revision=0,
        job_spec_digest="sha256:" + "0" * 64,
        status=status,
        expected_shard_ids=(),
        created_at=now,
        updated_at=now,
        provenance=Provenance(producer="test", revision="r", created_at=now),
        final_output_refs=final_output_refs,
    )


def test_non_broker_manifest_final_ref_blocks_gc(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=99999)
    data = b"non-broker-ref"
    plane = LocalFilesystemDataPlane(tmp_path)
    ref = plane.write(data, "application/octet-stream")
    run_id = "standalone-run"
    plane.write_next_manifest(
        _non_broker_manifest(
            run_id,
            status="succeeded",
            final_output_refs=(ref,),
        ),
        -1,
    )
    report = plan_gc(tmp_path, policy, mode="normal", now=datetime.now(UTC))
    assert ref.sha256 in {item.digest for item in report.protected}


def test_nonterminal_manifest_still_blocks_gc(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=99999)
    data = b"running-run"
    plane = LocalFilesystemDataPlane(tmp_path)
    ref = plane.write(data, "application/octet-stream")
    plane.write_next_manifest(
        _non_broker_manifest(
            run_id="running-run", status="running", final_output_refs=(ref,)
        ),
        -1,
    )
    index = build_reachability_index(tmp_path, policy)
    assert index.is_protected(ref.sha256)
    report = plan_gc(tmp_path, policy, mode="normal", now=datetime.now(UTC))
    assert ref.sha256 in {item.digest for item in report.protected}


def test_legacy_gc_closed_wave_reachability_tolerates_historical_job_semantics(
    tmp_path,
):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    manifest_input = plane.write(
        b"historical-manifest-input", "application/octet-stream"
    )
    shard_input = plane.write(b"historical-shard-input", "application/octet-stream")
    run_id = "historical-closed-wave-run"
    created_at = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    bundle = _historical_closed_wave_job_payload(
        run_id=run_id,
        input_manifest_ref=manifest_input,
        shard_input_refs=(shard_input,),
        created_at=created_at,
    )
    wave_path = _write_closed_wave_bundle(tmp_path, run_id=run_id, bundle=bundle)
    _age_payload(wave_path, created_at)
    index = build_reachability_index(tmp_path, policy)
    assert not any("closed wave" in message for message in index.unknown_messages)
    assert run_id in index.run_ids_by_digest[manifest_input.sha256]
    assert run_id in index.run_ids_by_digest[shard_input.sha256]
    assert index.digest_last_reference_at[manifest_input.sha256] == created_at
    assert index.digest_last_reference_at[shard_input.sha256] == created_at
    assert not index.protection_reasons(manifest_input.sha256)
    assert not index.protection_reasons(shard_input.sha256)


def test_legacy_gc_closed_wave_file_mtime_advances_last_reference_time(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    manifest_input = plane.write(b"mtime-manifest-input", "application/octet-stream")
    run_id = "historical-closed-wave-mtime"
    created_at = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    observed_at = datetime(2025, 3, 15, 8, 30, tzinfo=UTC)
    bundle = _historical_closed_wave_job_payload(
        run_id=run_id,
        input_manifest_ref=manifest_input,
        shard_input_refs=(),
        created_at=created_at,
    )
    wave_path = _write_closed_wave_bundle(tmp_path, run_id=run_id, bundle=bundle)
    _age_payload(wave_path, observed_at)
    index = build_reachability_index(tmp_path, policy)
    assert index.digest_last_reference_at[manifest_input.sha256] == observed_at


def test_legacy_gc_closed_wave_historical_inputs_hard_protected_for_live_run(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    manifest_input = plane.write(
        b"live-historical-manifest", "application/octet-stream"
    )
    shard_input = plane.write(b"live-historical-shard", "application/octet-stream")
    run_id = "historical-live-run"
    bundle = _historical_closed_wave_job_payload(
        run_id=run_id,
        input_manifest_ref=manifest_input,
        shard_input_refs=(shard_input,),
        created_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
    )
    _write_closed_wave_bundle(tmp_path, run_id=run_id, bundle=bundle)
    plane.write_next_manifest(
        _non_broker_manifest(run_id=run_id, status="running", final_output_refs=()),
        -1,
    )
    index = build_reachability_index(tmp_path, policy)
    assert f"closed_wave_input:{run_id}" in index.protection_reasons(
        manifest_input.sha256
    )
    assert f"closed_wave_input:{run_id}" in index.protection_reasons(shard_input.sha256)


def test_legacy_gc_closed_wave_historical_reachability_fail_closed(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    good_ref = plane.write(b"good-ref", "application/octet-stream")
    run_id = "historical-fail-closed"
    created_at = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    mismatch_bundle = _historical_closed_wave_job_payload(
        run_id="other-run-id",
        input_manifest_ref=good_ref,
        shard_input_refs=(good_ref,),
        created_at=created_at,
    )
    _write_closed_wave_bundle(tmp_path, run_id=run_id, bundle=mismatch_bundle)
    bad_ref_bundle = _historical_closed_wave_job_payload(
        run_id=run_id,
        input_manifest_ref=good_ref,
        shard_input_refs=(good_ref,),
        created_at=created_at,
    )
    bad_ref_bundle["shards"] = [
        {
            "logical_run_id": run_id,
            "input_refs": [{"sha256": "not-a-digest"}],
        }
    ]
    _write_closed_wave_bundle(
        tmp_path,
        run_id=run_id,
        bundle=bad_ref_bundle,
        wave_id="wave-bad-ref",
    )
    index = build_reachability_index(tmp_path, policy)
    assert any(
        "logical_run_id mismatch" in message for message in index.unknown_messages
    )
    assert any(
        "closed wave shard invalid" in message for message in index.unknown_messages
    )


def test_legacy_gc_protects_closed_wave_input_for_nonterminal_broker(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=3600)
    _register_request(service, "req-live-input")
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-live-input")
    assert state is not None
    assert state.status not in BROKER_TERMINAL_REQUEST_STATUSES
    payload = ClosedWaveRegistry(service.state_root / "controller").resolve_wave(
        state.logical_run_id, state.wave_id
    )
    input_digest = payload["job"]["input_manifest_ref"]["sha256"]
    path = artifact_payload_path(tmp_path, input_digest)
    _age_payload(path, datetime.now(UTC) - timedelta(days=30))
    index = build_reachability_index(tmp_path, policy)
    assert f"closed_wave_input:{state.logical_run_id}" in index.protection_reasons(
        input_digest
    )
    report = plan_gc(tmp_path, policy, mode="legacy", now=datetime.now(UTC))
    assert input_digest in {item.digest for item in report.protected}


def test_legacy_gc_shared_closed_wave_input_protected_while_one_run_live(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    _register_request(service, "req-live-shared")
    store = BrokerRequestStore(service.state_root / "controller")
    live_state = store.load("req-live-shared")
    assert live_state is not None
    payload = ClosedWaveRegistry(service.state_root / "controller").resolve_wave(
        live_state.logical_run_id, live_state.wave_id
    )
    shared_digest = payload["job"]["input_manifest_ref"]["sha256"]
    terminal_run = opaque_run_id("req-term-shared")
    terminal_input_ref = ArtifactRef(
        object_id=shared_digest.removeprefix("sha256:"),
        uri=artifact_payload_path(tmp_path, shared_digest).as_uri(),
        sha256=shared_digest,
        media_type="application/octet-stream",
        size_bytes=artifact_payload_path(tmp_path, shared_digest).stat().st_size,
    )
    _register_closed_wave_input(
        tmp_path,
        run_id=terminal_run,
        input_ref=terminal_input_ref,
        wave_id=opaque_wave_id("req-term-shared"),
    )
    terminal_output = b"terminal-run-output"
    terminal_out_digest = f"sha256:{sha256(terminal_output).hexdigest()}"
    store.save(
        BrokerRequestState(
            request_id="req-term-shared",
            binding=live_state.binding,
            logical_run_id=terminal_run,
            wave_id=opaque_wave_id("req-term-shared"),
            shard_id="shard-000000",
            status="succeeded",
            output_sha256=terminal_out_digest,
            output_size_bytes=len(terminal_output),
            output_media_type="application/json",
        )
    )
    _age_payload(
        artifact_payload_path(tmp_path, shared_digest),
        datetime.now(UTC) - timedelta(days=30),
    )
    report = plan_gc(
        tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=30)
    )
    assert shared_digest in {item.digest for item in report.protected}


def test_legacy_gc_protects_closed_wave_input_for_nonterminal_manifest(tmp_path):
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    input_ref = plane.write(b"manifest-closed-wave-input", "application/octet-stream")
    run_id = "standalone-running"
    _register_closed_wave_input(tmp_path, run_id=run_id, input_ref=input_ref)
    plane.write_next_manifest(
        _non_broker_manifest(run_id=run_id, status="running", final_output_refs=()),
        -1,
    )
    _age_payload(
        artifact_payload_path(tmp_path, input_ref.sha256),
        datetime.now(UTC) - timedelta(days=30),
    )
    index = build_reachability_index(tmp_path, policy)
    assert f"closed_wave_input:{run_id}" in index.protection_reasons(input_ref.sha256)
    report = plan_gc(
        tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=30)
    )
    assert input_ref.sha256 in {item.digest for item in report.protected}


def test_legacy_gc_terminal_broker_closed_wave_input_becomes_eligible(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    _register_request(service, "req-terminal-input")
    store = BrokerRequestStore(service.state_root / "controller")
    state = store.load("req-terminal-input")
    assert state is not None
    output = b'[{"id":1}]'
    digest_out = f"sha256:{sha256(output).hexdigest()}"
    _append_success_attempt(service.controller, "req-terminal-input", output)
    state.status = "succeeded"
    state.output_sha256 = digest_out
    state.output_size_bytes = len(output)
    state.output_media_type = "application/json"
    store.save(state)
    _delivery_commit(service, "req-terminal-input", digest_out, len(output))
    payload = ClosedWaveRegistry(service.state_root / "controller").resolve_wave(
        state.logical_run_id, state.wave_id
    )
    input_digest = payload["job"]["input_manifest_ref"]["sha256"]
    input_path = artifact_payload_path(tmp_path, input_digest)
    _age_payload(input_path, datetime.now(UTC) - timedelta(days=30))
    index = build_reachability_index(tmp_path, policy)
    assert f"closed_wave_input:{state.logical_run_id}" not in index.protection_reasons(
        input_digest
    )
    report = plan_gc(
        tmp_path, policy, mode="legacy", now=datetime.now(UTC) + timedelta(days=30)
    )
    assert input_digest in {item.digest for item in report.candidates}


def test_gc_prevents_publishing_ref_to_deleted_payload(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"deleted-before-ref"
    _succeed_broker(service, "req-writer", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-writer", digest, len(output))
    state = BrokerRequestStore(service.state_root / "controller").load("req-writer")
    assert state is not None
    apply_gc(tmp_path, policy, mode="normal")
    plane = LocalFilesystemDataPlane(tmp_path)
    ref = ArtifactRef(
        object_id=digest.removeprefix("sha256:"),
        uri=artifact_payload_path(tmp_path, digest).as_uri(),
        sha256=digest,
        media_type="application/octet-stream",
        size_bytes=len(output),
    )
    now = datetime.now(UTC)
    attempt = ShardAttemptRecord(
        logical_run_id=state.logical_run_id,
        shard_id="shard-000000",
        attempt_id="attempt-gc-order",
        status="succeeded",
        input_digest="sha256:" + "1" * 64,
        execution_fingerprint="fp",
        started_at=now,
        finished_at=now,
        output_refs=(ref,),
    )
    with pytest.raises(FileNotFoundError):
        plane.append_attempt(attempt)


def test_delivery_transport_identity_conflicts_on_created_at(tmp_path):
    now = datetime.now(UTC)
    base = DeliveryRecord(
        request_id="req-created-at",
        logical_run_id="run-1",
        artifact_digest="sha256:" + "a" * 64,
        artifact_size_bytes=4,
        producer_uid=None,
        consumer_uid=1,
        created_at=now,
        delivered_at=now,
        provenance={"pack": "tabular-batch", "operation": "tabular.sort"},
    )
    store = DeliveryRecordStore(tmp_path)
    store.save_new(base)
    conflict = DeliveryRecord(
        request_id=base.request_id,
        logical_run_id=base.logical_run_id,
        artifact_digest=base.artifact_digest,
        artifact_size_bytes=base.artifact_size_bytes,
        producer_uid=base.producer_uid,
        consumer_uid=base.consumer_uid,
        created_at=now + timedelta(seconds=1),
        delivered_at=now + timedelta(days=1),
        provenance=base.provenance,
    )
    with pytest.raises(ValueError, match="delivery_commit_conflict"):
        store.save_new(conflict)


def test_delivery_transport_identity_ignores_delivered_at_retry(tmp_path):
    now = datetime.now(UTC)
    base = DeliveryRecord(
        request_id="req-delivered-at",
        logical_run_id="run-1",
        artifact_digest="sha256:" + "b" * 64,
        artifact_size_bytes=8,
        producer_uid=None,
        consumer_uid=1,
        created_at=now,
        delivered_at=now,
    )
    store = DeliveryRecordStore(tmp_path)
    store.save_new(base)
    retry = DeliveryRecord(
        request_id=base.request_id,
        logical_run_id=base.logical_run_id,
        artifact_digest=base.artifact_digest,
        artifact_size_bytes=base.artifact_size_bytes,
        producer_uid=base.producer_uid,
        consumer_uid=base.consumer_uid,
        created_at=base.created_at,
        delivered_at=now + timedelta(days=2),
    )
    store.save_new(retry)
    assert store.load(base.request_id).delivered_at == now


def test_register_broker_holds_lifecycle_lock_for_input_and_manifest(tmp_path):
    with patch.object(LifecycleStateLock, "acquire", autospec=True) as acquire:
        register_broker_private_run(
            state_root=tmp_path,
            request_id="req-lock-once",
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params={"by": [{"column": "id"}]},
            input_bytes=b"[1,2]",
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
        )
    assert acquire.call_count == 1


def test_register_broker_reuses_gc_eligible_digest_atomically(tmp_path):
    import threading

    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    shared = b"shared-broker-input"
    digest = f"sha256:{sha256(shared).hexdigest()}"
    _succeed_broker(service, "req-old-shared", shared)
    _delivery_commit(service, "req-old-shared", digest, len(shared))
    assert digest in {
        item.digest for item in plan_gc(tmp_path, policy, mode="normal").candidates
    }

    gc_outcome: list[str] = []
    original_register = ClosedWaveRegistry.register_closed_wave

    def register_with_concurrent_gc(self, job, wave, shards, **kwargs):
        def run_gc():
            try:
                apply_gc(tmp_path, policy, mode="normal")
                gc_outcome.append("ran")
            except LifecycleLockError:
                gc_outcome.append("blocked")

        thread = threading.Thread(target=run_gc)
        thread.start()
        thread.join(timeout=10)
        assert gc_outcome == ["blocked"]
        return original_register(self, job, wave, shards, **kwargs)

    with patch.object(
        ClosedWaveRegistry, "register_closed_wave", register_with_concurrent_gc
    ):
        _, _, shard, manifest = register_broker_private_run(
            state_root=tmp_path,
            request_id="req-new-shared",
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params={"by": [{"column": "id"}]},
            input_bytes=shared,
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
        )
    assert artifact_payload_path(tmp_path, digest).is_file()
    assert shard.input_refs[0].sha256 == digest
    assert manifest is not None


def test_gc_deletion_sequence_fsyncs_directories(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    policy = _write_policy(tmp_path, grace=0, legacy=0)
    output = b"fsync-me"
    _succeed_broker(service, "req-fsync", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    _delivery_commit(service, "req-fsync", digest, len(output))
    with patch(
        "portable_batch_execution.lifecycle.gc.fsync_directory"
    ) as fsync_directory:
        apply_gc(tmp_path, policy, mode="normal")
    assert fsync_directory.call_count >= 1


def test_broker_and_closed_wave_writers_obey_lifecycle_lock(tmp_path):
    service = _local_service(tmp_path, _config(tmp_path))
    _register_request(service, "req-lock-writer")
    store = BrokerRequestStore(tmp_path / "controller")
    state = store.load("req-lock-writer")
    assert state is not None
    registry = ClosedWaveRegistry(tmp_path / "controller")
    payload = registry.resolve_wave(state.logical_run_id, state.wave_id)
    job = JobSpec.model_validate(payload["job"])
    wave = WaveSpec.model_validate(payload["wave"])
    shards = tuple(ShardSpec.model_validate(item) for item in payload["shards"])

    with LifecycleStateLock(tmp_path):
        with pytest.raises(LifecycleLockError):
            store.save(state)
        with pytest.raises(LifecycleLockError):
            registry.register_closed_wave(job, wave, shards)

    # The explicit caller-owned-lock path performs the publication without
    # trying to acquire the non-recursive interprocess lock again.
    second_wave = wave.model_copy(update={"wave_id": "second-wave"})
    with LifecycleStateLock(tmp_path):
        registry.register_closed_wave(
            job, second_wave, shards, _caller_holds_lifecycle_lock=True
        )


def test_apply_gc_reuses_one_reachability_build_for_many_candidates(tmp_path):
    policy = _write_policy(tmp_path, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    old = datetime.now(UTC) - timedelta(days=3)
    for number in range(8):
        ref = plane.write(f"orphan-{number}".encode())
        os.utime(
            artifact_payload_path(tmp_path, ref.sha256),
            (old.timestamp(), old.timestamp()),
        )

    original = gc_module.build_reachability_index
    with patch.object(gc_module, "build_reachability_index", wraps=original) as build:
        report = apply_gc(tmp_path, policy, mode="legacy")

    assert report.candidate_count == 8
    assert len(report.deleted) == 8
    assert build.call_count == 1


def test_apply_gc_builds_one_index_per_cross_mode_pending_intent(tmp_path):
    policy = _write_policy(tmp_path, legacy=0)
    plane = LocalFilesystemDataPlane(tmp_path)
    payload = b"pending-legacy-intent"
    ref = plane.write(payload)
    path = artifact_payload_path(tmp_path, ref.sha256)
    old = datetime.now(UTC) - timedelta(days=3)
    os.utime(path, (old.timestamp(), old.timestamp()))
    intent_store = DeletionIntentStore(tmp_path)
    intent_store.save(
        DeletionIntent(
            artifact_digest=ref.sha256,
            artifact_size_bytes=len(payload),
            logical_run_ids=(),
            producer_uid=None,
            consumer_uid=None,
            provenance=None,
            created_at=None,
            delivered_at=None,
            intended_at=old,
            mode="legacy",
            policy_identity="test",
        )
    )

    original = gc_module.build_reachability_index
    with patch.object(gc_module, "build_reachability_index", wraps=original) as build:
        apply_gc(tmp_path, policy, mode="normal")

    assert build.call_count == 2
    assert not path.exists()


def test_delivery_commit_idempotent_identical_replay(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    output = b'[{"id":1}]'
    _succeed_broker(service, "req-replay", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    first = service.handle_delivery_commit(
        _UID,
        {
            "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
            "request_id": "req-replay",
            "output_sha256": digest,
            "output_size_bytes": len(output),
        },
    )
    second = service.handle_delivery_commit(
        _UID,
        {
            "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
            "request_id": "req-replay",
            "output_sha256": digest,
            "output_size_bytes": len(output),
        },
    )
    assert first.status == "accepted"
    assert second.status == "accepted"
    first_delivered = DeliveryRecordStore(tmp_path).load("req-replay").delivered_at
    with patch(
        "portable_batch_execution.broker.service.utc_now",
        return_value=datetime.now(UTC) + timedelta(days=1),
    ):
        third = service.handle_delivery_commit(
            _UID,
            {
                "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
                "request_id": "req-replay",
                "output_sha256": digest,
                "output_size_bytes": len(output),
            },
        )
    assert third.status == "accepted"
    assert (
        DeliveryRecordStore(tmp_path).load("req-replay").delivered_at == first_delivered
    )


def test_delivery_commit_conflicting_replay_fails(tmp_path):
    config = _config(tmp_path)
    service = _local_service(tmp_path, config)
    output = b'[{"id":1}]'
    _succeed_broker(service, "req-conflict", output)
    digest = f"sha256:{sha256(output).hexdigest()}"
    accepted = service.handle_delivery_commit(
        _UID,
        {
            "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
            "request_id": "req-conflict",
            "output_sha256": digest,
            "output_size_bytes": len(output),
        },
    )
    assert accepted.status == "accepted"
    conflict = service.handle_delivery_commit(
        _UID,
        {
            "schema_version": "pbe.a1-unix-broker.delivery-commit.v1",
            "request_id": "req-conflict",
            "output_sha256": digest,
            "output_size_bytes": len(output) + 1,
        },
    )
    assert conflict.status == "failed"
    assert conflict.error_code == "delivery_commit_mismatch"


def test_delivery_save_acquires_lifecycle_lock(tmp_path):
    from portable_batch_execution.lifecycle.delivery import (
        DeliveryRecord,
        DeliveryRecordStore,
    )

    with patch.object(LifecycleStateLock, "acquire", autospec=True) as acquire:
        DeliveryRecordStore(tmp_path).save_new(
            DeliveryRecord(
                request_id="req-lock",
                logical_run_id="run-lock",
                artifact_digest="sha256:" + "d" * 64,
                artifact_size_bytes=1,
                producer_uid=None,
                consumer_uid=1,
                created_at=None,
                delivered_at=datetime.now(UTC),
            )
        )
    acquire.assert_called_once()


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
