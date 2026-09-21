from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
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


def test_600_shards_plan_as_256_256_88_with_github_capabilities():
    waves = plan_waves(
        "run-1",
        [shard(i) for i in range(600)],
        max_parallel=256,
        capabilities=GitHubActionsBackend("owner", "repo", "wave.yml").capabilities(),
    )

    assert [len(wave.shard_ids) for wave in waves] == [256, 256, 88]
    assert [wave.ordinal for wave in waves] == [0, 1, 2]
    assert [wave.wave_id for wave in waves] == ["wave-0000", "wave-0001", "wave-0002"]
    assert waves[0].shard_ids[0] == "shard-000"
    assert waves[-1].shard_ids[-1] == "shard-599"


def test_resume_accepts_a_later_success_and_keeps_only_one_canonical_attempt():
    expected = {"a", "b"}
    attempts = [
        attempt("a", "failed", "first"),
        attempt("a", attempt_id="retry"),
        attempt("b"),
    ]

    planned = [shard(0), shard(1)]
    planned[0] = planned[0].model_copy(update={"shard_id": "a"})
    planned[1] = planned[1].model_copy(update={"shard_id": "b"})
    canonical, missing, duplicate = completeness(expected, attempts, planned)

    assert [record.attempt_id for record in canonical] == ["retry", "one"]
    assert missing == ()
    assert duplicate == ()
    assert can_finalize(expected, attempts, planned)


@pytest.mark.parametrize(
    ("attempts", "reason"),
    [
        ([attempt("a"), attempt("a", attempt_id="second"), attempt("b")], "duplicate"),
        ([attempt("a")], "missing"),
    ],
)
def test_duplicate_or_missing_success_blocks_finalize(attempts, reason):
    expected = {"a", "b"}
    planned = [shard(0), shard(1)]
    planned[0] = planned[0].model_copy(update={"shard_id": "a"})
    planned[1] = planned[1].model_copy(update={"shard_id": "b"})
    _, missing, duplicate = completeness(expected, attempts, planned)

    assert not can_finalize(expected, attempts, planned)
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
    for uri in (
        "https://token@example.test/input",
        "https://example.test/input?secret=x",
    ):
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


import json
from datetime import UTC, datetime
from hashlib import sha256

from portable_batch_execution.worker.execute_wave import execute_private_wave


def test_private_mode_resolves_opaque_contracts_and_skips_completed_shards():
    rows = [{"group": "a", "value": 1}, {"group": "a", "value": 2}]
    input_ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + sha256(json.dumps(rows).encode()).hexdigest(),
    )
    now = datetime.now(UTC).isoformat()
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "tabular-batch",
        "operation": "tabular.rolling",
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {
            "column": "value",
            "window_size": 2,
            "output_column": "rolling",
        },
    }
    shard = {
        "logical_run_id": "opaque-run",
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [input_ref.model_dump(mode="json")],
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": ["opaque-shard"],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.resolved = []
            self.appended = []

        def resolve_wave(self, run_id, wave_id):
            self.resolved.append((run_id, wave_id))
            return {"job": job, "wave": wave, "shards": [shard]}

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            return json.dumps(rows).encode()

        def write(self, data, media_type):
            return ArtifactRef(
                object_id="output",
                uri="pbe://private/output",
                sha256="sha256:" + sha256(data).hexdigest(),
            )

        def append_attempt(self, record):
            self.appended.append(record)

    plane = Plane()
    assert len(execute_private_wave("opaque-run", "opaque-wave", plane=plane)) == 1
    assert plane.resolved == [("opaque-run", "opaque-wave")]
    assert execute_private_wave("opaque-run", "opaque-wave", plane=plane) == ()
