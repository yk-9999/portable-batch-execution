from datetime import UTC, datetime

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    RangeSpec,
    ShardCorrectnessSpec,
)
from portable_batch_execution.kernel import (
    expanded_range,
    plan_shards,
    plan_waves,
    retryable_shards,
)


def test_index_planner_and_backend_wave_limit():
    ref = ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)
    shards = plan_shards(
        "run",
        RangeSpec(kind="index", start=0, end=600),
        1,
        ShardCorrectnessSpec(lookback={"value": 2, "unit": "records"}),
        (ref,),
        "input",
        "fingerprint",
    )
    waves = plan_waves("run", shards, max_parallel=8)
    assert [len(wave.shard_ids) for wave in waves] == [256, 256, 88]
    assert shards[0].primary_range == RangeSpec(kind="index", start=0, end=1)
    assert expanded_range(shards[2]) == RangeSpec(kind="index", start=0, end=3)


def test_resume_stops_success_and_obeys_attempt_budget():
    ref = ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)
    shards = plan_shards(
        "run",
        RangeSpec(kind="index", start=0, end=2),
        1,
        ShardCorrectnessSpec(),
        (ref,),
        "input",
        "fingerprint",
    )
    now = datetime.now(UTC)
    attempt = {
        "logical_run_id": "run",
        "attempt_id": "a",
        "status": "succeeded",
        "input_digest": "input",
        "execution_fingerprint": "fingerprint",
        "started_at": now,
        "finished_at": now,
    }
    from portable_batch_execution.contracts import ShardAttemptRecord

    succeeded = ShardAttemptRecord(shard_id=shards[0].shard_id, **attempt)
    assert retryable_shards(
        shards, [succeeded], ExecutionPolicy(max_parallel=1, max_attempts_per_shard=1)
    ) == (shards[1],)
