from datetime import UTC, datetime

import pytest

from portable_batch_execution.contracts import (
    Provenance,
    RunManifest,
    ShardAttemptRecord,
)
from portable_batch_execution.data_plane import (
    LocalFilesystemDataPlane,
    RevisionConflictError,
)


def manifest(revision: int) -> RunManifest:
    now = datetime.now(UTC)
    return RunManifest(
        logical_run_id="run",
        revision=revision,
        job_spec_digest="digest",
        status="planned",
        expected_shard_ids=(),
        created_at=now,
        updated_at=now,
        provenance=Provenance(producer="test", revision="r", created_at=now),
    )


def test_artifact_store_and_manifest_compare_and_swap(tmp_path):
    store = LocalFilesystemDataPlane(tmp_path)
    ref = store.write(b"payload", "text/plain")
    assert store.exists(ref) and store.read(ref) == b"payload" and store.verify(ref)
    first = manifest(0)
    assert store.write_next_manifest(first, -1) == first
    second = manifest(1)
    assert store.write_next_manifest(second, 0) == second
    assert store.read_manifest("run") == second
    assert len(list((tmp_path / "runs" / "run" / "manifests").glob("*.json"))) == 2
    with pytest.raises(RevisionConflictError):
        store.write_next_manifest(manifest(2), 0)


def _attempt_record(run_id: str, attempt_id: str) -> ShardAttemptRecord:
    now = datetime.now(UTC)
    return ShardAttemptRecord(
        logical_run_id=run_id,
        shard_id="shard",
        attempt_id=attempt_id,
        status="succeeded",
        input_digest="digest",
        execution_fingerprint="fingerprint",
        started_at=now,
        finished_at=now,
    )


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", r"..\escape", "run/evil"],
)
def test_run_id_must_be_safe_file_component(tmp_path, run_id):
    store = LocalFilesystemDataPlane(tmp_path)
    sentinel = tmp_path / "escaped-run.json"
    with pytest.raises(ValueError, match="opaque identifier"):
        store.write_next_manifest(manifest(0).model_copy(update={"logical_run_id": run_id}), -1)
    assert not sentinel.exists()
    assert list((tmp_path / "runs").glob("**/*")) == [] or run_id == ""


@pytest.mark.parametrize(
    "attempt_id",
    [".", "..", "../escaped", r"..\..\escaped", "evil/attempt"],
)
def test_attempt_id_must_be_safe_file_component(tmp_path, attempt_id):
    store = LocalFilesystemDataPlane(tmp_path)
    store.write_next_manifest(manifest(0), -1)
    sentinel = tmp_path / "escaped.json"
    with pytest.raises(ValueError, match="opaque identifier"):
        store.append_attempt(_attempt_record("run", attempt_id))
    assert not sentinel.exists()
    attempts_dir = tmp_path / "runs" / "run" / "attempts"
    if attempts_dir.exists():
        assert list(attempts_dir.glob("*.json")) == []
