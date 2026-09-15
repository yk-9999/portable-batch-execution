from datetime import UTC, datetime

import pytest

from portable_batch_execution.contracts import Provenance, RunManifest
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
