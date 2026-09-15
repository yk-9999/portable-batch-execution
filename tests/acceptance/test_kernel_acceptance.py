from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    JobSpec,
    Provenance,
    RangeSpec,
    RunManifest,
    ShardAttemptRecord,
    ShardCorrectnessSpec,
    ShardSpec,
)
from portable_batch_execution.data_plane.base import RunStateStore
from portable_batch_execution.kernel import can_finalize, completeness, plan_waves

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def artifact(name: str = "input") -> ArtifactRef:
    return ArtifactRef(
        object_id=name,
        uri=f"file:///{name}",
        sha256="sha256:" + "a" * 64,
    )


def attempt(shard_id: str, status: str = "succeeded", attempt_id: str = "one"):
    return ShardAttemptRecord(
        logical_run_id="run-1",
        shard_id=shard_id,
        attempt_id=attempt_id,
        status=status,
        input_digest="input-digest",
        execution_fingerprint="execution-fingerprint",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        failure="expected failure" if status == "failed" else None,
    )


def shard(ordinal: int) -> ShardSpec:
    return ShardSpec(
        logical_run_id="run-1",
        shard_id=f"shard-{ordinal:03d}",
        ordinal=ordinal,
        primary_range=RangeSpec(kind="index", start=ordinal, end=ordinal + 1),
        correctness=ShardCorrectnessSpec(),
        input_refs=(artifact(),),
        input_digest="input-digest",
        execution_fingerprint="execution-fingerprint",
    )


def test_600_shards_plan_as_256_256_88():
    waves = plan_waves("run-1", [shard(i) for i in range(600)], max_parallel=256, max_per_wave=256)

    assert [len(wave.shard_ids) for wave in waves] == [256, 256, 88]
    assert [wave.ordinal for wave in waves] == [0, 1, 2]
    assert [wave.wave_id for wave in waves] == ["wave-0000", "wave-0001", "wave-0002"]
    assert waves[0].shard_ids[0] == "shard-000"
    assert waves[-1].shard_ids[-1] == "shard-599"


def test_resume_accepts_a_later_success_and_keeps_only_one_canonical_attempt():
    expected = {"a", "b"}
    attempts = [attempt("a", "failed", "first"), attempt("a", attempt_id="retry"), attempt("b")]

    canonical, missing, duplicate = completeness(expected, attempts)

    assert [record.attempt_id for record in canonical] == ["retry", "one"]
    assert missing == ()
    assert duplicate == ()
    assert can_finalize(expected, attempts)


@pytest.mark.parametrize(
    ("attempts", "reason"),
    [
        ([attempt("a"), attempt("a", attempt_id="second"), attempt("b")], "duplicate"),
        ([attempt("a")], "missing"),
    ],
)
def test_duplicate_or_missing_success_blocks_finalize(attempts, reason):
    expected = {"a", "b"}
    _, missing, duplicate = completeness(expected, attempts)

    assert not can_finalize(expected, attempts)
    if reason == "duplicate":
        assert duplicate == ("a",)
        assert missing == ()
    else:
        assert duplicate == ()
        assert missing == ("b",)


def test_run_state_contract_is_single_writer_compare_and_swap():
    method = RunStateStore.write_next_manifest
    assert tuple(inspect.signature(method).parameters) == (
        "self",
        "manifest",
        "expected_revision",
    )

    manifest = RunManifest(
        logical_run_id="run-1",
        revision=0,
        job_spec_digest="job-digest",
        status="planned",
        expected_shard_ids=("a",),
        created_at=NOW,
        updated_at=NOW,
        provenance=Provenance(producer="acceptance", revision="test", created_at=NOW),
    )
    with pytest.raises(ValidationError):
        manifest.revision = 1


def test_security_rejects_executable_or_secret_bearing_job_inputs():
    provenance = Provenance(producer="acceptance", revision="test", created_at=NOW)
    for uri in ("https://token@example.test/input", "https://example.test/input?secret=x"):
        with pytest.raises(ValidationError):
            ArtifactRef(object_id="bad", uri=uri, sha256="sha256:" + "b" * 64)

    for params in ({"command": "whoami"}, {"nested": {"sql": "select 1"}}):
        with pytest.raises(ValidationError):
            JobSpec(
                job_id="job-1",
                logical_run_id="run-1",
                pack="tabular-batch",
                operation="tabular.sort",
                input_manifest_ref=artifact(),
                sharding=ShardCorrectnessSpec(),
                execution=ExecutionPolicy(max_parallel=1, max_attempts_per_shard=1),
                security_profile="offline",
                provenance=provenance,
                operation_params=params,
            )

    with pytest.raises(ValidationError):
        RangeSpec(kind="index", start=float("nan"))
