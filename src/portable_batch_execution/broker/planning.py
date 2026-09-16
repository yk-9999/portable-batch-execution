"""Construct closed one-wave private runs for broker requests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from portable_batch_execution.contracts import (
    ExecutionPolicy,
    JobSpec,
    Provenance,
    RunManifest,
    ShardCorrectnessSpec,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.controller.a1_controller import job_spec_digest
from portable_batch_execution.controller.closed_wave_registry import ClosedWaveRegistry
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane
from portable_batch_execution.packs import MLPack, TabularPack


def canonical_operation_params(pack: str, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if pack == "tabular-batch":
        return TabularPack().validate_params(operation, params)
    if pack == "ml-batch":
        return MLPack().validate_params(operation, params)
    if pack == "media-batch":
        if params:
            raise ValueError("closed operation parameters")
        return {}
    raise ValueError("unsupported broker pack")


def broker_input_digest(input_bytes: bytes) -> str:
    return sha256(input_bytes).hexdigest()


def broker_execution_fingerprint(
    *,
    request_id: str,
    input_digest: str,
    pack: str,
    operation: str,
    operation_params: dict[str, Any],
    public_sha: str,
) -> str:
    material = json.dumps(
        {
            "request_id": request_id,
            "input_digest": input_digest,
            "pack": pack,
            "operation": operation,
            "operation_params": operation_params,
            "public_sha": public_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(material.encode("utf-8")).hexdigest()


def opaque_run_id(request_id: str) -> str:
    return f"run-{sha256(request_id.encode('utf-8')).hexdigest()[:16]}"


def opaque_wave_id(request_id: str) -> str:
    return f"wave-{sha256(request_id.encode('utf-8')).hexdigest()[:12]}"


def broker_job_id(request_id: str, execution_fingerprint: str) -> str:
    material = f"{request_id}\x1f{execution_fingerprint}".encode()
    return f"broker-{sha256(material).hexdigest()[:16]}"


def register_broker_private_run(
    *,
    state_root,
    request_id: str,
    pack: str,
    operation: str,
    operation_params: dict[str, Any],
    input_bytes: bytes,
    input_media_type: str,
    public_sha: str,
) -> tuple[JobSpec, WaveSpec, ShardSpec, RunManifest]:
    plane = LocalFilesystemDataPlane(state_root)
    registry = ClosedWaveRegistry(state_root / "controller")
    input_digest = broker_input_digest(input_bytes)
    validated_params = canonical_operation_params(pack, operation, operation_params)
    execution_fingerprint = broker_execution_fingerprint(
        request_id=request_id,
        input_digest=input_digest,
        pack=pack,
        operation=operation,
        operation_params=validated_params,
        public_sha=public_sha,
    )
    run_id = opaque_run_id(request_id)
    wave_id = opaque_wave_id(request_id)
    job_id = broker_job_id(request_id, execution_fingerprint)
    try:
        existing = registry.resolve_wave(run_id, wave_id)
    except KeyError:
        existing = None
    if existing is not None:
        job = JobSpec.model_validate(existing["job"])
        if job.job_id != job_id:
            raise ValueError("run already registered with a different job")
        wave = WaveSpec.model_validate(existing["wave"])
        shard = ShardSpec.model_validate(existing["shards"][0])
        manifest = plane.read_manifest(run_id)
        if manifest is None:
            raise ValueError("registered run missing manifest")
        return job, wave, shard, manifest
    input_ref = plane.write(input_bytes, input_media_type)
    now = datetime.now(UTC)
    job = JobSpec(
        job_id=job_id,
        logical_run_id=run_id,
        pack=pack,
        operation=operation,
        input_manifest_ref=input_ref,
        sharding=ShardCorrectnessSpec(mode="independent"),
        execution=ExecutionPolicy(
            max_parallel=1,
            max_attempts_per_shard=4,
            resume_enabled=True,
        ),
        security_profile="offline",
        provenance=Provenance(
            producer="a1-unix-broker",
            revision="v1",
            created_at=now,
        ),
        operation_params=validated_params,
    )
    shard = ShardSpec(
        logical_run_id=run_id,
        shard_id="shard-000000",
        ordinal=0,
        correctness=job.sharding,
        input_refs=(input_ref,),
        input_digest=input_digest,
        execution_fingerprint=execution_fingerprint,
    )
    wave = WaveSpec(
        logical_run_id=run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=(shard.shard_id,),
        max_parallel=1,
    )
    registry.register_closed_wave(job, wave, (shard,))
    manifest = RunManifest(
        logical_run_id=run_id,
        revision=0,
        job_spec_digest=job_spec_digest(job),
        status="planned",
        expected_shard_ids=(shard.shard_id,),
        created_at=now,
        updated_at=now,
        provenance=Provenance(
            producer="a1-unix-broker",
            revision="v1",
            created_at=now,
        ),
        waves=(wave,),
    )
    plane.write_next_manifest(manifest, -1)
    return job, wave, shard, manifest
