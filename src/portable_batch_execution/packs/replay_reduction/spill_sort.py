"""Bounded external sort and k-way merge for replay spill runs."""

from __future__ import annotations

import heapq
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .canonicalize import _IDENTITY_SCAN_BATCH, StructuralCanonicalizeError

_SORT_RUN_ROWS = 65_536
_MERGE_FAN_IN = 16
_ROW_SCAN_BATCH = 65_536


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


@dataclass
class _ParquetRowStream:
    path: Path
    batch_size: int = _ROW_SCAN_BATCH
    _lazy: pl.LazyFrame = field(init=False, repr=False)
    _offset: int = field(init=False, default=0)
    _total: int = field(init=False, default=0)
    _buffer: Iterator[dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lazy = pl.scan_parquet(str(self.path))
        self._total = int(self._lazy.select(pl.len()).collect().item())
        self._buffer = iter(())
        self._refill_buffer()

    def _refill_buffer(self) -> None:
        if self._offset >= self._total:
            self._buffer = iter(())
            return
        size = min(self.batch_size, self._total - self._offset)
        frame = self._lazy.slice(self._offset, size).collect()
        self._offset += size
        self._buffer = (dict(row) for row in frame.iter_rows(named=True))

    def pop(self) -> dict[str, Any] | None:
        try:
            return next(self._buffer)
        except StopIteration:
            self._refill_buffer()
            try:
                return next(self._buffer)
            except StopIteration:
                return None


def _iter_k_way_merge_bounded(
    run_paths: Sequence[Path],
    *,
    sort_keys: Sequence[str],
) -> Iterator[dict[str, Any]]:
    if not run_paths:
        return
    streams = [_ParquetRowStream(path) for path in run_paths]
    heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for source_id, stream in enumerate(streams):
        row = stream.pop()
        if row is not None:
            heapq.heappush(heap, (_sort_key_tuple(row, sort_keys), source_id, row))
    while heap:
        _, source_id, row = heapq.heappop(heap)
        yield row
        next_row = streams[source_id].pop()
        if next_row is not None:
            heapq.heappush(
                heap,
                (_sort_key_tuple(next_row, sort_keys), source_id, next_row),
            )


def _merge_fan_in_group(
    run_paths: Sequence[Path],
    *,
    sort_keys: Sequence[str],
    spill_dir: Path,
    run_prefix: str,
) -> list[Path]:
    spill_dir.mkdir(parents=True, exist_ok=True)
    if len(run_paths) == 1:
        return [run_paths[0]]
    out_paths: list[Path] = []
    out_batch: list[dict[str, Any]] = []
    part_index = 0

    def flush_out() -> None:
        nonlocal part_index
        if not out_batch:
            return
        path = spill_dir / f"{run_prefix}_{part_index:05d}.parquet"
        pl.DataFrame(out_batch).write_parquet(path)
        out_paths.append(path)
        out_batch.clear()
        part_index += 1

    for row in _iter_k_way_merge_bounded(run_paths, sort_keys=sort_keys):
        out_batch.append(row)
        if len(out_batch) >= _SORT_RUN_ROWS:
            flush_out()
    flush_out()
    return out_paths


def _reduce_sorted_runs(
    run_paths: Sequence[Path],
    *,
    sort_keys: Sequence[str],
    spill_dir: Path,
    run_prefix: str,
) -> list[Path]:
    current = list(run_paths)
    if not current:
        return []
    level = 0
    while len(current) > _MERGE_FAN_IN:
        next_level: list[Path] = []
        level_dir = spill_dir / f"merge_level_{level:03d}"
        for group_index, start in enumerate(range(0, len(current), _MERGE_FAN_IN)):
            group = current[start : start + _MERGE_FAN_IN]
            merged = _merge_fan_in_group(
                group,
                sort_keys=sort_keys,
                spill_dir=level_dir / f"group_{group_index:05d}",
                run_prefix=f"{run_prefix}_l{level}",
            )
            next_level.extend(merged)
        current = next_level
        level += 1
    return current


def iter_k_way_merge_dataframes(
    run_paths: Sequence[Path],
    *,
    sort_keys: Sequence[str],
    spill_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Merge pre-sorted parquet runs with bounded fan-in and batched reads."""
    paths = list(run_paths)
    if not paths:
        return
    if len(paths) == 1:
        yield from _iter_k_way_merge_bounded(paths, sort_keys=sort_keys)
        return

    merge_root = spill_dir
    owns_merge_dir = False
    if merge_root is None:
        merge_root = Path(tempfile.mkdtemp(prefix="pbe-spill-merge-"))
        owns_merge_dir = True
    try:
        reduced = _reduce_sorted_runs(
            paths,
            sort_keys=sort_keys,
            spill_dir=merge_root,
            run_prefix="reduced",
        )
        yield from _iter_k_way_merge_bounded(reduced, sort_keys=sort_keys)
    finally:
        if owns_merge_dir:
            shutil.rmtree(merge_root, ignore_errors=True)
