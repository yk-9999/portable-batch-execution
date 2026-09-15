from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    JobSpec,
    Provenance,
    ShardCorrectnessSpec,
    WaveSpec,
)


def artifact():
    return ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)


def test_contract_security_boundaries():
    provenance = Provenance(
        producer="test", revision="r", created_at=datetime.now(UTC)
    )
    with pytest.raises(ValidationError):
        ArtifactRef(object_id="x", uri="https://u:p@example/x", sha256="bad")
    with pytest.raises(ValidationError):
        JobSpec(
            job_id="j",
            logical_run_id="r",
            pack="tabular-batch",
            operation="tabular.sort",
            input_manifest_ref=artifact(),
            sharding=ShardCorrectnessSpec(),
            execution=ExecutionPolicy(max_parallel=1, max_attempts_per_shard=1),
            security_profile="offline",
            provenance=provenance,
            operation_params={"nested": {"sql": "x"}},
        )
    with pytest.raises(ValidationError):
        WaveSpec(
            logical_run_id="r",
            wave_id="w",
            ordinal=0,
            shard_ids=("a", "a"),
            max_parallel=1,
        )
