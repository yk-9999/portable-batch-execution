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
from portable_batch_execution.data_plane.base import RevisionConflictError
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane
from portable_batch_execution.packs import MLPack, TabularPack

_BINDING_CONFLICT = "request_binding_conflict"

_CLOSED_ML_BATCH_OPERATIONS = frozenset(
    {
        "ml.char_wb_tfidf_logistic_score",
        "ml.cosine_similarity_matrix",
    }
)


def canonical_operation_params(pack: str, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if pack == "tabular-batch":
        return TabularPack().validate_params(operation, params)
    if pack == "ml-batch":
        if operation in _CLOSED_ML_BATCH_OPERATIONS:
            if params:
                raise ValueError("closed operation parameters")
            return {}
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


def validate_registered_wave_binding(
    *,
    request_id: str,
    binding_input_digest: str,
    binding_pack: str,
    binding_operation: str,
    binding_operation_params: dict[str, Any],
    binding_execution_fingerprint: str,
    job: JobSpec,
    shard: ShardSpec,
) -> None:
    expected_job_id = broker_job_id(request_id, binding_execution_fingerprint)
    if job.job_id != expected_job_id:
        raise ValueError(_BINDING_CONFLICT)
    if job.pack != binding_pack or job.operation != binding_operation:
        raise ValueError(_BINDING_CONFLICT)
    if job.operation_params != binding_operation_params:
        raise ValueError(_BINDING_CONFLICT)
    if shard.input_digest != binding_input_digest:
        raise ValueError(_BINDING_CONFLICT)
    if shard.execution_fingerprint != binding_execution_fingerprint:
        raise ValueError(_BINDING_CONFLICT)


def _initial_manifest(job: JobSpec, wave: WaveSpec, shard: ShardSpec) -> RunManifest:
    created_at = job.provenance.created_at
    return RunManifest(
        logical_run_id=job.logical_run_id,
        revision=0,
        job_spec_digest=job_spec_digest(job),
        status="planned",
        expected_shard_ids=(shard.shard_id,),
        created_at=created_at,
        updated_at=created_at,
        provenance=job.provenance,
        waves=(wave,),
    )


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
        wave = WaveSpec.model_validate(existing["wave"])
        shard = ShardSpec.model_validate(existing["shards"][0])
        validate_registered_wave_binding(
            request_id=request_id,
            binding_input_digest=input_digest,
            binding_pack=pack,
            binding_operation=operation,
            binding_operation_params=validated_params,
            binding_execution_fingerprint=execution_fingerprint,
            job=job,
            shard=shard,
        )
        manifest = plane.read_manifest(run_id)
        if manifest is None:
            manifest = _initial_manifest(job, wave, shard)
            try:
                plane.write_next_manifest(manifest, -1)
            except RevisionConflictError:
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
    manifest = _initial_manifest(job, wave, shard)
    plane.write_next_manifest(manifest, -1)
    return job, wave, shard, manifest
