from __future__ import annotations

from collections import defaultdict

from portable_batch_execution.contracts import ShardAttemptRecord, ShardSpec, WaveSpec


def plan_waves(
    run_id: str,
    shards: list[ShardSpec],
    max_parallel: int,
    max_per_wave: int | None = None,
) -> list[WaveSpec]:
    limit = max_per_wave or len(shards)
    return [
        WaveSpec(
            logical_run_id=run_id,
            wave_id=f"wave-{i // limit:04d}",
            ordinal=i // limit,
            shard_ids=tuple(s.shard_id for s in shards[i : i + limit]),
            max_parallel=max_parallel,
        )
        for i in range(0, len(shards), limit)
    ]


def completeness(expected: set[str], attempts: list[ShardAttemptRecord]):
    by = defaultdict(list)
    for a in attempts:
        if a.status == "succeeded":
            by[a.shard_id].append(a)
    missing = tuple(sorted(x for x in expected if not by[x]))
    duplicate = tuple(sorted(x for x in expected if len(by[x]) > 1))
    canonical = tuple(by[x][0] for x in sorted(expected) if len(by[x]) == 1)
    return canonical, missing, duplicate


def can_finalize(expected: set[str], attempts: list[ShardAttemptRecord]) -> bool:
    _, missing, duplicate = completeness(expected, attempts)
    return not missing and not duplicate
