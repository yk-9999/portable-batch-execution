from datetime import UTC, datetime

import pytest

from portable_batch_execution.contracts import JobSpec, Provenance, ShardSpec, WaveSpec
from portable_batch_execution.controller.closed_wave_registry import ClosedWaveRegistry


def _job(run_id: str = "opaque-run") -> JobSpec:
    now = datetime.now(UTC)
    from portable_batch_execution.contracts import ArtifactRef, ExecutionPolicy

    ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + ("a" * 64),
    )
    return JobSpec(
        job_id="job",
        logical_run_id=run_id,
        pack="tabular-batch",
        operation="tabular.sort",
        input_manifest_ref=ref,
        sharding={},
        execution=ExecutionPolicy(max_parallel=1, max_attempts_per_shard=4),
        security_profile="offline",
        provenance=Provenance(producer="test", revision="1", created_at=now),
    )


def _wave(run_id: str = "opaque-run", wave_id: str = "opaque-wave") -> WaveSpec:
    return WaveSpec(
        logical_run_id=run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=("opaque-shard",),
        max_parallel=1,
    )


def _shard(run_id: str = "opaque-run") -> ShardSpec:
    from portable_batch_execution.contracts import ArtifactRef

    ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + ("a" * 64),
    )
    return ShardSpec(
        logical_run_id=run_id,
        shard_id="opaque-shard",
        ordinal=0,
        correctness={},
        input_refs=(ref,),
        input_digest="current",
        execution_fingerprint="fixed",
    )


def test_registry_resolves_exact_opaque_pair(tmp_path):
    registry = ClosedWaveRegistry(tmp_path)
    job, wave, shard = _job(), _wave(), _shard()
    registry.register_closed_wave(job, wave, (shard,))
    payload = registry.resolve_wave("opaque-run", "opaque-wave")
    assert payload["job"]["logical_run_id"] == "opaque-run"
    assert payload["wave"]["wave_id"] == "opaque-wave"
    assert payload["shards"][0]["shard_id"] == "opaque-shard"
    with pytest.raises(KeyError):
        registry.resolve_wave("opaque-run", "missing-wave")


def test_registry_rejects_path_like_identifiers(tmp_path):
    registry = ClosedWaveRegistry(tmp_path)
    with pytest.raises(ValueError, match="opaque identifier"):
        registry.resolve_wave("../run", "opaque-wave")
