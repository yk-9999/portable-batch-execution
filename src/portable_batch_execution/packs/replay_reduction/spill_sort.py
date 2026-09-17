"""Bounded external sort and k-way merge for replay spill runs."""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from .canonicalize import _IDENTITY_SCAN_BATCH, StructuralCanonicalizeError

_SORT_RUN_ROWS = 65_536


def _sort_key_tuple(row: dict[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[key] for key in keys)


def external_sort_lazy_batches(
    batches: Iterator[pl.DataFrame],
    *,
    sort_keys: Sequence[str],
    spill_dir: Path,
    run_prefix: str,
) -> list[Path]:
    """Write sorted parquet runs from streamed batches; peak RAM is one batch plus one run."""
    spill_dir.mkdir(parents=True, exist_ok=True)
    run_paths: list[Path] = []
    run_index = 0
    for batch in batches:
        if batch.is_empty():
            continue
        missing = [key for key in sort_keys if key not in batch.columns]
        if missing:
            raise StructuralCanonicalizeError(f"sort keys missing columns: {missing}")
        sorted_batch = batch.sort(list(sort_keys))
        run_path = spill_dir / f"{run_prefix}_{run_index:05d}.parquet"
        sorted_batch.write_parquet(run_path)
        run_paths.append(run_path)
        run_index += 1
    return run_paths


def external_sort_lazy_frame(
    lazy: pl.LazyFrame,
    *,
    sort_keys: Sequence[str],
    spill_dir: Path,
    run_prefix: str,
    batch_size: int = _IDENTITY_SCAN_BATCH,
) -> list[Path]:
    row_count = int(lazy.select(pl.len()).collect().item())
    if row_count == 0:
        return []

    def _batch_iter() -> Iterator[pl.DataFrame]:
        offset = 0
        while offset < row_count:
            size = min(batch_size, row_count - offset)
            yield lazy.slice(offset, size).collect()
            offset += size

    return external_sort_lazy_batches(
        _batch_iter(),
        sort_keys=sort_keys,
        spill_dir=spill_dir,
        run_prefix=run_prefix,
    )


def iter_k_way_merge_dataframes(
    run_paths: Sequence[Path],
    *,
    sort_keys: Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Merge pre-sorted parquet runs without re-sorting."""
    if not run_paths:
        return
    if len(run_paths) == 1:
        frame = pl.read_parquet(str(run_paths[0]))
        for row in frame.iter_rows(named=True):
            yield dict(row)
        return

    iterators: list[Iterator[dict[str, Any]]] = []
    for path in run_paths:
        frame = pl.read_parquet(str(path))
        iterators.append(iter(dict(row) for row in frame.iter_rows(named=True)))

    heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for source_id, iterator in enumerate(iterators):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (_sort_key_tuple(row, sort_keys), source_id, row))

    while heap:
        _, source_id, row = heapq.heappop(heap)
        yield row
        iterator = iterators[source_id]
        try:
            next_row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (_sort_key_tuple(next_row, sort_keys), source_id, next_row),
        )
