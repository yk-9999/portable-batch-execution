"""Deterministic planning and finalization primitives for v1 runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from typing import Protocol

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    RangeSpec,
    RunManifest,
    ShardAttemptRecord,
    ShardCorrectnessSpec,
    ShardSpec,
    WaveSpec,
)


class _Capabilities(Protocol):
    max_shards_per_wave: int


def expanded_range(shard: ShardSpec) -> RangeSpec | None:
    """Return the worker read window while preserving the shard primary range."""
    primary = shard.primary_range
    if primary is None:
        return None
    correctness = shard.correctness
    before = sum(
        extent.value
        for extent in (correctness.lookback, correctness.halo_before)
        if extent is not None
    )
    after = sum(
        extent.value
        for extent in (correctness.lookforward, correctness.halo_after)
        if extent is not None
    )
    if primary.kind == "index":
        return RangeSpec(
            kind="index",
            start=max(0, int(primary.start) - before),
            end=int(primary.end) + after,
        )
    if primary.kind == "time":
        start, end = primary.as_datetime_bounds()

        def duration(extent):
            if extent is None:
                return timedelta()
            return timedelta(**{extent.unit: extent.value})

        return RangeSpec(
            kind="time",
            start=(
                start
                - duration(correctness.lookback)
                - duration(correctness.halo_before)
            ).isoformat(),
            end=(
                end
                + duration(correctness.lookforward)
                + duration(correctness.halo_after)
            ).isoformat(),
        )
    return primary


def plan_shards(
    run_id: str,
    primary_range: RangeSpec,
    shard_size: float | timedelta,
    correctness: ShardCorrectnessSpec,
    input_refs: tuple[ArtifactRef, ...],
    input_digest: str,
    execution_fingerprint: str,
    *,
    partition_key: str | None = None,
) -> list[ShardSpec]:
    """Split a half-open index or time range into stable ordinal shard specs."""
    if primary_range.kind == "index":
        if (
            not isinstance(shard_size, int)
            or isinstance(shard_size, bool)
            or shard_size <= 0
        ):
            raise ValueError("index shard_size must be a positive integer")
        if primary_range.start is None or primary_range.end is None:
            raise ValueError("index ranges require start and end")
        start, end = int(primary_range.start), int(primary_range.end)
        ranges = [(i, min(i + shard_size, end)) for i in range(start, end, shard_size)]
    elif primary_range.kind == "time":
        step = (
            shard_size
            if isinstance(shard_size, timedelta)
            else timedelta(seconds=shard_size)
            if isinstance(shard_size, (int, float)) and not isinstance(shard_size, bool)
            else None
        )
        if (
            step is None
            or step <= timedelta(0)
            or primary_range.start is None
            or primary_range.end is None
        ):
            raise ValueError("time ranges require a positive size, start, and end")
        start, end = primary_range.as_datetime_bounds()
        ranges, cursor = [], start
        while cursor < end:
            nxt = min(cursor + step, end)
            ranges.append((cursor, nxt))
            cursor = nxt
    else:
        raise ValueError("key ranges require explicit shard boundaries")
    if correctness.mode == "partition_affinity" and not partition_key:
        raise ValueError("partition_key is required for partition_affinity")
    return [
        ShardSpec(
            logical_run_id=run_id,
            shard_id=f"shard-{ordinal:06d}",
            ordinal=ordinal,
            partition_key=partition_key,
            primary_range=RangeSpec(
                kind=primary_range.kind,
                start=start.isoformat() if primary_range.kind == "time" else start,
                end=end.isoformat() if primary_range.kind == "time" else end,
            ),
            correctness=correctness,
            input_refs=input_refs,
            input_digest=input_digest,
            execution_fingerprint=execution_fingerprint,
        )
        for ordinal, (start, end) in enumerate(ranges)
    ]


def plan_waves(
    run_id: str,
    shards: list[ShardSpec],
    max_parallel: int,
    max_per_wave: int | None = None,
    capabilities: _Capabilities | None = None,
) -> list[WaveSpec]:
    """Return stable waves capped by both caller and backend capacity."""
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    ordered = sorted(shards, key=lambda shard: (shard.ordinal, shard.shard_id))
    if len({shard.shard_id for shard in ordered}) != len(ordered):
        raise ValueError("shard IDs must be unique")
    if any(shard.logical_run_id != run_id for shard in ordered):
        raise ValueError("all shards must belong to run_id")
    if capabilities is None and max_per_wave is None:
        raise ValueError("backend capabilities or an explicit max_per_wave is required")
    limits = [
        limit
        for limit in (
            max_per_wave,
            capabilities.max_shards_per_wave if capabilities else None,
        )
        if limit is not None
    ]
    if any(limit <= 0 for limit in limits):
        raise ValueError("wave limits must be positive")
    limit = min(limits)
    return [
        WaveSpec(
            logical_run_id=run_id,
            wave_id=f"wave-{offset // limit:04d}",
            ordinal=offset // limit,
            shard_ids=tuple(
                shard.shard_id for shard in ordered[offset : offset + limit]
            ),
            max_parallel=max_parallel,
        )
        for offset in range(0, len(ordered), limit)
    ]


def completeness(
    expected: set[str],
    attempts: list[ShardAttemptRecord],
    shards: Iterable[ShardSpec],
) -> tuple[tuple[ShardAttemptRecord, ...], tuple[str, ...], tuple[str, ...]]:
    """Choose unambiguous successes; stale fingerprints never satisfy a shard."""
    shard_map = {shard.shard_id: shard for shard in shards}
    if set(shard_map) != expected:
        raise ValueError("planned shards must exactly match expected shard IDs")
    by_shard: dict[str, list[ShardAttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        shard = shard_map.get(attempt.shard_id)
        if (
            attempt.shard_id in expected
            and attempt.status == "succeeded"
            and attempt.input_digest == shard.input_digest
            and attempt.execution_fingerprint == shard.execution_fingerprint
        ):
            by_shard[attempt.shard_id].append(attempt)
    missing = tuple(sorted(shard_id for shard_id in expected if not by_shard[shard_id]))
    duplicate = tuple(
        sorted(shard_id for shard_id in expected if len(by_shard[shard_id]) > 1)
    )
    canonical = tuple(
        by_shard[shard_id][0]
        for shard_id in sorted(expected)
        if len(by_shard[shard_id]) == 1
    )
    return canonical, missing, duplicate


def retryable_shards(
    shards: Iterable[ShardSpec],
    attempts: Iterable[ShardAttemptRecord],
    policy: ExecutionPolicy,
) -> tuple[ShardSpec, ...]:
    """Return only shards that may legally receive another attempt."""
    by_shard: dict[str, list[ShardAttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        by_shard[attempt.shard_id].append(attempt)
    result = []
    for shard in sorted(shards, key=lambda item: (item.ordinal, item.shard_id)):
        records = [
            record
            for record in by_shard[shard.shard_id]
            if record.input_digest == shard.input_digest
            and record.execution_fingerprint == shard.execution_fingerprint
        ]
        success = any(
            record.status == "succeeded"
            and record.input_digest == shard.input_digest
            and record.execution_fingerprint == shard.execution_fingerprint
            for record in records
        )
        if (not success or not policy.resume_enabled) and len(
            records
        ) < policy.max_attempts_per_shard:
            result.append(shard)
    return tuple(result)


def can_finalize(
    expected: set[str],
    attempts: list[ShardAttemptRecord],
    shards: Iterable[ShardSpec],
) -> bool:
    _, missing, duplicate = completeness(expected, attempts, shards)
    return not missing and not duplicate


class RunController:
    """The sole manifest writer; workers append attempts but never finalize."""

    def __init__(self, state_store):
        self.state_store = state_store

    def refresh(
        self,
        manifest: RunManifest,
        expected_revision: int,
        planned_shards: Iterable[ShardSpec],
    ) -> RunManifest:
        attempts = self.state_store.read_attempts(manifest.logical_run_id)
        canonical, missing, duplicate = completeness(
            set(manifest.expected_shard_ids), list(attempts), planned_shards
        )
        ready = not missing and not duplicate
        next_manifest = manifest.model_copy(
            update={
                "revision": expected_revision + 1,
                "discovered_attempts": attempts,
                "canonical_attempts": canonical,
                "completed_shard_ids": tuple(record.shard_id for record in canonical),
                "missing_shard_ids": missing,
                "duplicate_shard_ids": duplicate,
                "finalization_status": "ready" if ready else "not_started",
            }
        )
        return self.state_store.write_next_manifest(next_manifest, expected_revision)

    def finalize(
        self,
        manifest: RunManifest,
        expected_revision: int,
        output_refs: tuple[ArtifactRef, ...],
    ) -> RunManifest:
        if (
            manifest.revision != expected_revision
            or manifest.finalization_status != "ready"
        ):
            raise ValueError("manifest is not ready for finalization")
        next_manifest = manifest.model_copy(
            update={
                "revision": expected_revision + 1,
                "status": "succeeded",
                "finalization_status": "finalized",
                "final_output_refs": output_refs,
            }
        )
        return self.state_store.write_next_manifest(next_manifest, expected_revision)


def exhausted_shards(
    shards: Iterable[ShardSpec],
    attempts: Iterable[ShardAttemptRecord],
    policy: ExecutionPolicy,
) -> tuple[ShardSpec, ...]:
    """Return current, unsuccessful shards that used their whole attempt budget."""
    by_shard: dict[str, list[ShardAttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        by_shard[attempt.shard_id].append(attempt)
    exhausted = []
    for shard in sorted(shards, key=lambda item: (item.ordinal, item.shard_id)):
        current = [
            record
            for record in by_shard[shard.shard_id]
            if record.input_digest == shard.input_digest
            and record.execution_fingerprint == shard.execution_fingerprint
        ]
        if len(current) >= policy.max_attempts_per_shard and not any(
            record.status == "succeeded" for record in current
        ):
            exhausted.append(shard)
    return tuple(exhausted)
