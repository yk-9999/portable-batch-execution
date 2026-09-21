import json
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.worker.runtime_profile import (
    RuntimeProfileError,
    resolve_private_profile,
    resolve_profile,
)


def _plane(*, pack="tabular-batch", operation="tabular.rolling",
           run_id="opaque-run", wave_id="opaque-wave"):
    payload = b"payload"
    input_ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    now = datetime.now(UTC).isoformat()
    job = {
        "job_id": "job",
        "logical_run_id": run_id,
        "pack": pack,
        "operation": operation,
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 1},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {},
    }
    shard = {
        "logical_run_id": run_id,
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [input_ref.model_dump(mode="json")],
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    wave = {
        "logical_run_id": run_id,
        "wave_id": wave_id,
        "ordinal": 0,
        "shard_ids": ["opaque-shard"],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.payload = {"job": job, "wave": wave, "shards": [shard]}
            self.resolved: list[tuple[str, str]] = []

        def resolve_wave(self, requested_run, requested_wave):
            self.resolved.append((requested_run, requested_wave))
            return json.loads(json.dumps(self.payload))

    return Plane()


def test_hf_direct_resolution_uses_control_metadata_only():
    plane = _plane(pack="tabular-batch", operation="tabular.join")
    assert not hasattr(plane, "read")
    resolution = resolve_private_profile(
        "opaque-run", "opaque-wave", plane=plane, mode="hf-direct"
    )
    assert resolution.mode == "hf-direct"
    assert resolution.profile == "tabular"
    assert resolution.run_id == "opaque-run"
    assert plane.resolved == [("opaque-run", "opaque-wave")]


def test_resolve_profile_dispatches_hf_direct_without_artifacts():
    plane = _plane()
    resolution = resolve_profile(
        "opaque-wave", mode="hf-direct", run_id="opaque-run", plane=plane
    )
    assert resolution.mode == "hf-direct"


def test_hf_direct_resolution_requires_run_id():
    with pytest.raises(RuntimeProfileError):
        resolve_profile("opaque-wave", mode="hf-direct")


def test_hf_direct_resolution_rejects_unavailable_operations():
    plane = _plane(operation="tabular.format_migration")
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile(
            "opaque-run", "opaque-wave", plane=plane, mode="hf-direct"
        )


def test_hf_direct_resolution_fails_closed_on_run_or_wave_mismatch():
    plane = _plane()
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile(
            "different-run", "opaque-wave", plane=plane, mode="hf-direct"
        )
    assert plane.resolved == [("different-run", "opaque-wave")]


def test_hf_direct_resolution_reads_no_artifact_reference():
    plane = _plane()
    before = plane.payload["shards"][0]["input_refs"][0]
    ArtifactRef.model_validate(before)
    resolve_private_profile("opaque-run", "opaque-wave", plane=plane, mode="hf-direct")
    assert plane.payload["shards"][0]["input_refs"][0] == before
