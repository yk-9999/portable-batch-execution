from datetime import UTC, datetime

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    RangeSpec,
    ShardCorrectnessSpec,
)
from portable_batch_execution.kernel import (
    completeness,
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
    waves = plan_waves(
        "run",
        shards,
        max_parallel=8,
        capabilities=GitHubActionsBackend("o", "r", "w").capabilities(),
    )
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


def test_stale_success_is_missing_and_current_success_is_canonical():
    ref = ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)
    shard = plan_shards("run", RangeSpec(kind="index", start=0, end=1), 1, ShardCorrectnessSpec(), (ref,), "current", "fingerprint")[0]
    now = datetime.now(UTC)
    from portable_batch_execution.contracts import ShardAttemptRecord

    stale = ShardAttemptRecord(logical_run_id="run", shard_id=shard.shard_id, attempt_id="stale", status="succeeded", input_digest="stale", execution_fingerprint="fingerprint", started_at=now, finished_at=now)
    current = ShardAttemptRecord(logical_run_id="run", shard_id=shard.shard_id, attempt_id="current", status="succeeded", input_digest="current", execution_fingerprint="fingerprint", started_at=now, finished_at=now)

    canonical, missing, duplicate = completeness({shard.shard_id}, [stale], [shard])
    assert canonical == () and missing == (shard.shard_id,) and duplicate == ()
    canonical, missing, duplicate = completeness({shard.shard_id}, [stale, current], [shard])
    assert canonical == (current,) and missing == () and duplicate == ()


def test_stale_attempts_do_not_exhaust_current_retry_budget():
    ref = ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)
    shard = plan_shards("run", RangeSpec(kind="index", start=0, end=1), 1, ShardCorrectnessSpec(), (ref,), "current", "fingerprint")[0]
    now = datetime.now(UTC)
    from portable_batch_execution.contracts import ShardAttemptRecord

    stale = ShardAttemptRecord(logical_run_id="run", shard_id=shard.shard_id, attempt_id="stale", status="succeeded", input_digest="old", execution_fingerprint="old", started_at=now, finished_at=now)
    assert retryable_shards([shard], [stale], ExecutionPolicy(max_parallel=1, max_attempts_per_shard=1)) == (shard,)


def test_exhausted_shards_counts_initial_attempt_and_three_retries_only():
    from portable_batch_execution.contracts import ShardAttemptRecord
    from portable_batch_execution.kernel import exhausted_shards
    ref = ArtifactRef(object_id="x", uri="file:///x", sha256="sha256:" + "a" * 64)
    shard = plan_shards("run", RangeSpec(kind="index", start=0, end=1), 1, ShardCorrectnessSpec(), (ref,), "current", "fingerprint")[0]
    now = datetime.now(UTC)
    def attempt(attempt_id, status="failed", digest="current"):
        return ShardAttemptRecord(logical_run_id="run", shard_id=shard.shard_id, attempt_id=attempt_id, status=status, input_digest=digest, execution_fingerprint="fingerprint", started_at=now, finished_at=now, failure="failed" if status == "failed" else None)
    policy = ExecutionPolicy(max_parallel=1, max_attempts_per_shard=4)
    assert exhausted_shards([shard], [attempt(str(i)) for i in range(3)], policy) == ()
    assert exhausted_shards([shard], [attempt(str(i)) for i in range(4)], policy) == (shard,)
    assert exhausted_shards([shard], [attempt(str(i)) for i in range(3)] + [attempt("success", "succeeded")], policy) == ()
    assert exhausted_shards([shard], [attempt(str(i), digest="stale") for i in range(9)] + [attempt(str(i)) for i in range(3)], policy) == ()
