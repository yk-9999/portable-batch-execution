"""Dense causal replay grid extraction in one bounded pass per contiguous partition."""

from __future__ import annotations

import heapq
import shutil
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .canonicalize import (
    _IDENTITY_SCAN_BATCH,
    StructuralCanonicalizeError,
    _require_columns,
    bucket_index,
    execute_structural_canonicalize,
)
from .event_window import _as_of_preferred, _row_is_sentinel, _validate_positive_row
from .models import CausalGridExtractRequest
from .spill_sort import (
    external_sort_lazy_batches,
    external_sort_lazy_frame,
    iter_k_way_merge_dataframes,
)

RESULT_SCHEMA_VERSION = "pbe.replay.causal-grid-extract-result.v1"
CARRY_SCHEMA_VERSION = "pbe.replay.causal-grid-carry.v3"
_BUCKET_SPILL_BATCH = 65_536
_COLLAPSED_RUN_BATCH = 65_536


def _validated(
    request: dict[str, Any] | CausalGridExtractRequest,
) -> CausalGridExtractRequest:
    if isinstance(request, CausalGridExtractRequest):
        return request
    return CausalGridExtractRequest.model_validate(request)


def _utc_date_key(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).date().isoformat()


def _segment_id(ms: int, missing_dates: frozenset[str]) -> int:
    day = _utc_date_key(ms)
    if day in missing_dates:
        raise StructuralCanonicalizeError("timestamp falls on hard missing date")
    segment = 0
    for missing in sorted(missing_dates):
        if day > missing:
            segment += 1
        else:
            break
    return segment


@dataclass
class _CausalSegmentFrontier:
    """Compact per-segment block frontier for frozen causal cutoff semantics."""

    segment_id: int
    monotone_ok: bool = True
    last_time: int | None = None
    last_block: int | None = None
    block_first_ms: dict[int, int] = field(default_factory=dict)

    def observe(self, *, exchange_time_ms: int, block_number: int) -> None:
        if (
            self.last_time is not None
            and (
                exchange_time_ms < self.last_time
                or (
                    self.last_block is not None
                    and block_number < self.last_block
                    and exchange_time_ms >= self.last_time
                )
            )
        ):
            self.monotone_ok = False
        prior = self.block_first_ms.get(block_number)
        if prior is None or exchange_time_ms < prior:
            self.block_first_ms[block_number] = exchange_time_ms
        self.last_time = exchange_time_ms
        self.last_block = block_number

    def causal_cutoff_block(self, decision_time_ms: int) -> int | None:
        if not self.monotone_ok:
            return None
        eligible = [
            block
            for block, first_ms in self.block_first_ms.items()
            if first_ms <= decision_time_ms
        ]
        if len(eligible) < 2:
            return None
        witness = max(eligible)
        lower = [block for block in eligible if block < witness]
        if not lower:
            return None
        return max(lower)

    def export_block_first_ms(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (self.segment_id, block, first_ms)
            for block, first_ms in sorted(self.block_first_ms.items())
        )

    def load_block_first_ms(self, entries: tuple[tuple[int, int, int], ...]) -> None:
        for segment_id, block_number, first_ms in entries:
            if segment_id != self.segment_id:
                continue
            prior = self.block_first_ms.get(block_number)
            if prior is None or first_ms < prior:
                self.block_first_ms[block_number] = first_ms


@dataclass
class _CausalIndex:
    missing_dates: frozenset[str]
    segments: dict[int, _CausalSegmentFrontier] = field(default_factory=dict)

    @classmethod
    def from_carry(
        cls,
        missing_dates: frozenset[str],
        carry_observations: tuple[tuple[int, int, int], ...],
        carry_block_first_ms: tuple[tuple[int, int, int], ...] = (),
    ) -> _CausalIndex:
        index = cls(missing_dates=missing_dates)
        if carry_block_first_ms:
            for segment_id, block_number, first_ms in carry_block_first_ms:
                state = index._state_for_segment(segment_id)
                state.load_block_first_ms(((segment_id, block_number, first_ms),))
        else:
            for segment_id, exchange_time_ms, block_number in carry_observations:
                index._state_for_segment(segment_id).observe(
                    exchange_time_ms=exchange_time_ms,
                    block_number=block_number,
                )
        return index

    def _state_for_time(self, exchange_ms: int) -> _CausalSegmentFrontier:
        segment = _segment_id(exchange_ms, self.missing_dates)
        return self._state_for_segment(segment)

    def _state_for_segment(self, segment_id: int) -> _CausalSegmentFrontier:
        if segment_id not in self.segments:
            self.segments[segment_id] = _CausalSegmentFrontier(segment_id=segment_id)
        return self.segments[segment_id]

    def observe(self, *, exchange_time_ms: int, block_number: int) -> None:
        self._state_for_time(exchange_time_ms).observe(
            exchange_time_ms=exchange_time_ms,
            block_number=block_number,
        )

    def cutoff(self, decision_time_ms: int) -> int | None:
        segment = _segment_id(decision_time_ms, self.missing_dates)
        state = self.segments.get(segment)
        if state is None:
            return None
        return state.causal_cutoff_block(decision_time_ms)

    def export_carry(self) -> tuple[tuple[int, int, int], ...]:
        exported: list[tuple[int, int, int]] = []
        for segment_id in sorted(self.segments):
            exported.extend(self.segments[segment_id].export_block_first_ms())
        return tuple(exported)


@dataclass(frozen=True)
class _TradeEvent:
    symbol: str
    exchange_time_ms: int
    block_number: int
    price: float
    notional_usd: float
    tie_break: tuple[Any, ...]


@dataclass
class _SymbolRolling:
    lookback_ms: int
    as_of_offsets_ms: tuple[int, ...]
    as_of_field: str
    trailing_specs: tuple[Any, ...]
    tie_break: tuple[str, ...]
    events: deque[_TradeEvent] = field(default_factory=deque)

    def _evict_before(self, watermark_ms: int) -> None:
        while self.events and self.events[0].exchange_time_ms < watermark_ms:
            self.events.popleft()

    def ingest(self, trade: _TradeEvent, *, watermark_ms: int) -> None:
        self._evict_before(watermark_ms)
        self.events.append(trade)

    def facts_at(self, *, decision_time_ms: int, cutoff: int | None) -> list[dict[str, Any]]:
        if cutoff is None:
            facts: list[dict[str, Any]] = []
            for offset_ms in self.as_of_offsets_ms:
                facts.append({"fact_id": f"as_of.{offset_ms}", "value": None})
            for spec in self.trailing_specs:
                facts.append({"fact_id": f"{spec.fact_id}.sum", "value": None})
                facts.append({"fact_id": f"{spec.fact_id}.count", "value": None})
            return facts

        as_of_best: dict[int, dict[str, Any] | None] = {
            int(offset): None for offset in self.as_of_offsets_ms
        }
        trailing_sums: dict[str, float] = {}
        trailing_counts: dict[str, int] = {}

        for row in self.events:
            if row.block_number > cutoff:
                continue
            timestamp_ms = row.exchange_time_ms
            if timestamp_ms > decision_time_ms:
                continue
            row_dict = {
                self.as_of_field: row.price,
                "price": row.price,
                "notional": row.notional_usd,
                "notional_usd": row.notional_usd,
                **{column: value for column, value in zip(self.tie_break, row.tie_break)},
            }
            for offset_ms in self.as_of_offsets_ms:
                target_ms = decision_time_ms - int(offset_ms)
                if timestamp_ms <= target_ms:
                    current = as_of_best[int(offset_ms)]
                    if _as_of_preferred(row_dict, current, self.tie_break):
                        as_of_best[int(offset_ms)] = row_dict
            for spec in self.trailing_specs:
                lower = decision_time_ms - int(spec.trailing_width_ms)
                if timestamp_ms > lower and timestamp_ms <= decision_time_ms:
                    key = spec.fact_id
                    measurement = row_dict.get(spec.measurement_field)
                    if measurement is None:
                        continue
                    trailing_sums[key] = trailing_sums.get(key, 0.0) + float(measurement)
                    trailing_counts[key] = trailing_counts.get(key, 0) + 1

        facts: list[dict[str, Any]] = []
        for offset_ms in self.as_of_offsets_ms:
            best = as_of_best[int(offset_ms)]
            value = None if best is None else best.get(self.as_of_field)
            facts.append({"fact_id": f"as_of.{offset_ms}", "value": value})
        for spec in self.trailing_specs:
            facts.append(
                {
                    "fact_id": f"{spec.fact_id}.sum",
                    "value": trailing_sums.get(spec.fact_id, 0.0),
                }
            )
            facts.append(
                {
                    "fact_id": f"{spec.fact_id}.count",
                    "value": int(trailing_counts.get(spec.fact_id, 0)),
                }
            )
        return facts


def _max_lookback_ms(model: CausalGridExtractRequest) -> int:
    offsets = max((int(value) for value in model.as_of_offsets_ms), default=0)
    trailing = max(
        (int(spec.trailing_width_ms) for spec in model.trailing_windows),
        default=0,
    )
    return max(offsets, trailing)


def _grid_timestamps(model: CausalGridExtractRequest) -> list[int]:
    emit = model.emit_grid
    part = model.partition
    start = max(emit.start_timestamp_ms, part.emit_start_ms)
    end = min(emit.end_timestamp_ms, part.emit_end_ms)
    if end < start:
        return []
    aligned = emit.start_timestamp_ms
    while aligned < start:
        aligned += emit.step_ms
    timestamps: list[int] = []
    cursor = aligned
    while cursor <= end:
        if cursor >= start:
            timestamps.append(cursor)
        cursor += emit.step_ms
    return timestamps


def _tie_break_tuple(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[column] for column in columns)


def _stream_positive_rows(path: Path, model: CausalGridExtractRequest) -> Iterator[dict[str, Any]]:
    trade_map = model.canonical_trade_mapping
    profile = trade_map.canonical_trade_profile
    sentinel = profile.sentinel
    identity_col = profile.identity_source_column
    normalized_col = profile.identity_normalized_column
    core_fields = profile.measurement_core_fields
    tie_break = model.tie_break_columns

    lazy = pl.scan_parquet(str(path))
    required = (
        trade_map.symbol_column,
        trade_map.block_column,
        trade_map.timestamp_column,
        identity_col,
        normalized_col,
        trade_map.price_field,
        trade_map.notional_field,
        *core_fields,
        *tie_break,
    )
    _require_columns(lazy.collect_schema(), required)
    if sentinel is not None:
        _require_columns(lazy.collect_schema(), tuple(sentinel.exact_match_fields))

    select_columns = list(dict.fromkeys(required))
    row_count = int(lazy.select(pl.len()).collect().item())
    offset = 0
    while offset < row_count:
        batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
        batch = (
            lazy.slice(offset, batch_size)
            .with_columns(
                pl.col(trade_map.block_column).cast(pl.Int64, strict=False).alias("_block_int"),
                pl.col(trade_map.timestamp_column)
                .cast(pl.Int64, strict=False)
                .alias("_timestamp_ms"),
                pl.col(identity_col).cast(pl.Int64, strict=False).alias("_identity_int"),
            )
            .select([*select_columns, "_block_int", "_timestamp_ms", "_identity_int"])
            .collect()
        )
        for row in batch.iter_rows(named=True):
            yield dict(row)
        offset += batch_size


def _row_dict_to_trade_event(row: dict[str, Any], model: CausalGridExtractRequest) -> _TradeEvent:
    tie = row.get("tie_break")
    if tie is None:
        tie = _tie_break_tuple(row, model.tie_break_columns)
    return _TradeEvent(
        symbol=str(row["symbol"]),
        exchange_time_ms=int(row["exchange_time_ms"]),
        block_number=int(row["block_number"]),
        price=float(row["price"]),
        notional_usd=float(row["notional_usd"]),
        tie_break=tuple(tie),
    )


def _row_preferred(
    candidate: dict[str, Any],
    current: dict[str, Any],
    tie_break: tuple[str, ...],
) -> bool:
    candidate_dict = {
        "price": candidate["price"],
        **{column: candidate[column] for column in tie_break},
    }
    current_dict = {
        "price": current["price"],
        **{column: current[column] for column in tie_break},
    }
    return _as_of_preferred(candidate_dict, current_dict, tie_break)


def _flush_bucket_spill_batch(batch: list[dict[str, Any]], *, part_path: Path) -> None:
    pl.DataFrame(batch).write_parquet(part_path)


def _collapse_bucket_identity_groups(
    bucket_dir: Path,
    model: CausalGridExtractRequest,
    *,
    core_fields: tuple[str, ...],
    spill_dir: Path,
    bucket_index_value: int,
) -> list[Path]:
    parts = sorted(bucket_dir.glob("part_*.parquet"))
    if not parts:
        return []
    lazy = pl.concat([pl.scan_parquet(str(path)) for path in parts], how="vertical_relaxed")
    sort_keys = (
        "identity",
        "exchange_time_ms",
        "block_number",
        *model.tie_break_columns,
    )
    sorted_runs = external_sort_lazy_frame(
        lazy,
        sort_keys=sort_keys,
        spill_dir=spill_dir / f"sort_b{bucket_index_value:05d}",
        run_prefix="identity",
    )
    collapsed_batch: list[dict[str, Any]] = []
    collapsed_runs: list[Path] = []
    run_index = 0
    current_identity: int | None = None
    current_core: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None

    def flush_collapsed() -> None:
        nonlocal run_index, best_row
        if best_row is None:
            return
        collapsed_batch.append(best_row)
        best_row = None
        if len(collapsed_batch) >= _COLLAPSED_RUN_BATCH:
            run_path = spill_dir / f"collapsed_b{bucket_index_value:05d}_{run_index:05d}.parquet"
            pl.DataFrame(collapsed_batch).write_parquet(run_path)
            collapsed_runs.append(run_path)
            collapsed_batch.clear()
            run_index += 1

    for row in iter_k_way_merge_dataframes(sorted_runs, sort_keys=sort_keys):
        identity = int(row["identity"])
        if current_identity is None or identity != current_identity:
            flush_collapsed()
            current_identity = identity
            current_core = {field: row[field] for field in core_fields}
            best_row = row
            continue
        if current_core is None:
            raise StructuralCanonicalizeError("measurement core fields disagree")
        for core_field in core_fields:
            if row[core_field] != current_core[core_field]:
                raise StructuralCanonicalizeError("measurement core fields disagree")
        if best_row is not None and _row_preferred(row, best_row, model.tie_break_columns):
            best_row = row
    flush_collapsed()
    if collapsed_batch:
        run_path = spill_dir / f"collapsed_b{bucket_index_value:05d}_{run_index:05d}.parquet"
        pl.DataFrame(collapsed_batch).write_parquet(run_path)
        collapsed_runs.append(run_path)
    return collapsed_runs


def _materialize_collapsed_trades(
    trade_paths: list[Path],
    model: CausalGridExtractRequest,
    *,
    spill_dir: Path,
    bucket_count: int,
) -> list[Path]:
    profile = model.canonical_trade_mapping.canonical_trade_profile
    sentinel = profile.sentinel
    identity_col = profile.identity_source_column
    normalized_col = profile.identity_normalized_column
    core_fields = tuple(profile.measurement_core_fields)
    trade_map = model.canonical_trade_mapping
    tie_break = model.tie_break_columns

    bucket_root = spill_dir / "trade_buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_batches: list[list[dict[str, Any]]] = [[] for _ in range(bucket_count)]
    bucket_part_counts = [0 for _ in range(bucket_count)]

    def flush_bucket_batch(bucket_value: int) -> None:
        batch = bucket_batches[bucket_value]
        if not batch:
            return
        bucket_dir = bucket_root / f"bucket_{bucket_value:05d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        part_path = bucket_dir / f"part_{bucket_part_counts[bucket_value]:05d}.parquet"
        _flush_bucket_spill_batch(batch, part_path=part_path)
        bucket_part_counts[bucket_value] += 1
        batch.clear()

    for path in trade_paths:
        for row in _stream_positive_rows(path, model):
            if _row_is_sentinel(row, sentinel):
                continue
            identity = _validate_positive_row(
                row,
                identity_col=identity_col,
                normalized_col=normalized_col,
                core_fields=core_fields,
            )
            bucket_value = bucket_index(identity, bucket_count)
            payload = {
                "identity": identity,
                "exchange_time_ms": int(row["_timestamp_ms"]),
                "block_number": int(row["_block_int"]),
                "symbol": str(row[trade_map.symbol_column]),
                "price": float(row[trade_map.price_field]),
                "notional_usd": float(row[trade_map.notional_field]),
                **{field: row[field] for field in core_fields},
                **{column: row[column] for column in tie_break},
            }
            bucket_batches[bucket_value].append(payload)
            if len(bucket_batches[bucket_value]) >= _BUCKET_SPILL_BATCH:
                flush_bucket_batch(bucket_value)

    for bucket_value in range(bucket_count):
        flush_bucket_batch(bucket_value)

    collapsed_unsorted: list[Path] = []
    for bucket_value in range(bucket_count):
        bucket_dir = bucket_root / f"bucket_{bucket_value:05d}"
        if not bucket_dir.exists():
            continue
        collapsed_unsorted.extend(
            _collapse_bucket_identity_groups(
                bucket_dir,
                model,
                core_fields=core_fields,
                spill_dir=spill_dir,
                bucket_index_value=bucket_value,
            )
        )
    if not collapsed_unsorted:
        return []
    if len(collapsed_unsorted) == 1:
        sort_source = pl.scan_parquet(str(collapsed_unsorted[0]))
    else:
        sort_source = pl.concat(
            [pl.scan_parquet(str(path)) for path in collapsed_unsorted],
            how="vertical_relaxed",
        )
    return external_sort_lazy_frame(
        sort_source,
        sort_keys=("exchange_time_ms", "block_number"),
        spill_dir=spill_dir / "collapsed_time_sort",
        run_prefix="collapsed",
    )


def _iter_sorted_collapsed_trades_for_model(
    sorted_runs: list[Path],
    model: CausalGridExtractRequest,
) -> Iterator[_TradeEvent]:
    sort_keys = ("exchange_time_ms", "block_number")
    for row in iter_k_way_merge_dataframes(sorted_runs, sort_keys=sort_keys):
        yield _row_dict_to_trade_event(row, model)


def _materialize_sorted_witness_runs(
    path_objs: list[Path],
    model: CausalGridExtractRequest,
    *,
    spill_dir: Path,
) -> list[Path]:
    mapping = model.causal_witness_mapping
    bindings = {binding.input_index: binding.role for binding in model.input_roles}
    witness_frames: list[pl.DataFrame] = []
    for input_index, path in enumerate(path_objs):
        if bindings.get(input_index) != "causal_witness":
            continue
        lazy = pl.scan_parquet(str(path))
        _require_columns(
            lazy.collect_schema(),
            (mapping.block_column, mapping.timestamp_column),
        )
        row_count = int(lazy.select(pl.len()).collect().item())
        offset = 0
        while offset < row_count:
            batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
            batch = (
                lazy.slice(offset, batch_size)
                .with_row_index("row_offset", offset=offset)
                .select(
                    pl.lit(input_index).alias("input_index"),
                    pl.col("row_offset").cast(pl.Int64),
                    pl.col(mapping.timestamp_column)
                    .cast(pl.Int64, strict=False)
                    .alias("exchange_time_ms"),
                    pl.col(mapping.block_column).cast(pl.Int64, strict=False).alias("block_number"),
                )
                .collect()
            )
            if int(batch.filter(pl.col("exchange_time_ms").is_null() | pl.col("block_number").is_null()).height):
                raise StructuralCanonicalizeError("witness block or timestamp missing")
            witness_frames.append(batch)
            offset += batch_size
    if not witness_frames:
        return []

    def _batch_iter() -> Iterator[pl.DataFrame]:
        yield from witness_frames

    return external_sort_lazy_batches(
        _batch_iter(),
        sort_keys=("exchange_time_ms", "block_number", "input_index", "row_offset"),
        spill_dir=spill_dir / "witness_sort",
        run_prefix="witness",
    )


def _iter_sorted_witness_events(
    sorted_runs: list[Path],
) -> Iterator[tuple[int, int, int, int]]:
    sort_keys = ("exchange_time_ms", "block_number", "input_index", "row_offset")
    for row in iter_k_way_merge_dataframes(sorted_runs, sort_keys=sort_keys):
        yield (
            int(row["exchange_time_ms"]),
            int(row["block_number"]),
            int(row["input_index"]),
            int(row["row_offset"]),
        )


def _chronological_stream(
    *,
    witness_runs: list[Path],
    collapsed_runs: list[Path],
    model: CausalGridExtractRequest,
) -> Iterator[tuple[Literal["witness", "trade"], int, int, _TradeEvent | None]]:
    witness_iter = _iter_sorted_witness_events(witness_runs)
    trade_iter = _iter_sorted_collapsed_trades_for_model(collapsed_runs, model)

    heap: list[
        tuple[int, int, int, int, int, Literal["witness", "trade"], _TradeEvent | None]
    ] = []
    try:
        time_ms, block, input_index, row_off = next(witness_iter)
        heapq.heappush(heap, (time_ms, block, input_index, row_off, 0, "witness", None))
    except StopIteration:
        pass
    try:
        first_trade = next(trade_iter)
        heapq.heappush(
            heap,
            (
                first_trade.exchange_time_ms,
                first_trade.block_number,
                0,
                0,
                1,
                "trade",
                first_trade,
            ),
        )
    except StopIteration:
        pass

    while heap:
        time_ms, block, input_index, row_off, _sid, kind, payload = heapq.heappop(heap)
        yield kind, block, time_ms, payload
        if kind == "witness":
            try:
                next_time, next_block, next_index, next_off = next(witness_iter)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (next_time, next_block, next_index, next_off, 0, "witness", None),
            )
        else:
            try:
                next_trade = next(trade_iter)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (
                    next_trade.exchange_time_ms,
                    next_trade.block_number,
                    0,
                    0,
                    1,
                    "trade",
                    next_trade,
                ),
            )


def _trade_carry_rows(
    rolling: dict[str, _SymbolRolling],
    *,
    carry_start_ms: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for state in rolling.values():
        for event in state.events:
            if event.exchange_time_ms >= carry_start_ms:
                rows.append(
                    {
                        "symbol": event.symbol,
                        "exchange_time_ms": event.exchange_time_ms,
                        "block_number": event.block_number,
                        "price": event.price,
                        "notional_usd": event.notional_usd,
                        "tie_break": event.tie_break,
                    }
                )
    return tuple(rows)


def execute_causal_grid_extract(
    paths: list[str | Path],
    request: dict[str, Any] | CausalGridExtractRequest,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one parquet input is required")
    model = _validated(request)
    if len(paths) > 64:
        raise ValueError("replay parquet input count exceeds limit")
    path_objs = [Path(path) for path in paths]
    trade_indices = sorted(
        binding.input_index
        for binding in model.input_roles
        if binding.role == "canonical_trade"
    )
    if not trade_indices:
        raise ValueError("at least one canonical_trade input is required")
    trade_paths = [path_objs[index] for index in trade_indices]
    profile = model.canonical_trade_mapping.canonical_trade_profile
    bucket_count = profile.structural_canonicalize_params().bucket_count
    validation_state = execute_structural_canonicalize(
        trade_paths,
        profile.structural_canonicalize_params(),
    )
    collapse_dir = tempfile.mkdtemp(prefix="pbe-grid-collapse-")
    try:
        collapsed_runs = _materialize_collapsed_trades(
            trade_paths,
            model,
            spill_dir=Path(collapse_dir),
            bucket_count=bucket_count,
        )
        witness_runs = _materialize_sorted_witness_runs(
            path_objs,
            model,
            spill_dir=Path(collapse_dir),
        )
        missing = frozenset(model.partition.hard_gap_missing_dates)
        carry_in = model.partition.incoming_carry
        causal = _CausalIndex.from_carry(
            missing,
            carry_in.causal_observations if carry_in is not None else (),
            carry_block_first_ms=carry_in.causal_block_first_ms if carry_in is not None else (),
        )
        lookback = _max_lookback_ms(model)
        scan_start = model.partition.emit_start_ms - lookback
        grid_times = _grid_timestamps(model)
        projected_rows = len(grid_times) * len(model.target_symbols)
        if projected_rows > model.max_output_rows:
            raise ValueError("replay grid output row count exceeds limit")

        rolling: dict[str, _SymbolRolling] = {
            symbol: _SymbolRolling(
                lookback_ms=lookback,
                as_of_offsets_ms=model.as_of_offsets_ms,
                as_of_field=model.as_of_measurement_field,
                trailing_specs=model.trailing_windows,
                tie_break=model.tie_break_columns,
            )
            for symbol in model.target_symbols
        }
        if carry_in is not None:
            for row in carry_in.trade_rows:
                symbol = str(row["symbol"])
                if symbol not in rolling:
                    continue
                event = _TradeEvent(
                    symbol=symbol,
                    exchange_time_ms=int(row["exchange_time_ms"]),
                    block_number=int(row["block_number"]),
                    price=float(row["price"]),
                    notional_usd=float(row["notional_usd"]),
                    tie_break=tuple(row["tie_break"]),
                )
                rolling[symbol].ingest(event, watermark_ms=scan_start)

        grid_index = 0
        rows: list[dict[str, Any]] = []
        trade_watermark = scan_start

        def _emit_grid_row(decision_ms: int) -> None:
            nonlocal rows
            if len(rows) >= model.max_output_rows:
                raise ValueError("replay grid output row count exceeds limit")
            cutoff = causal.cutoff(decision_ms)
            for symbol in model.target_symbols:
                if len(rows) >= model.max_output_rows:
                    raise ValueError("replay grid output row count exceeds limit")
                rows.append(
                    {
                        "symbol": symbol,
                        "grid_timestamp_ms": decision_ms,
                        "causal_cutoff_block": cutoff,
                        "facts": rolling[symbol].facts_at(
                            decision_time_ms=decision_ms,
                            cutoff=cutoff,
                        ),
                    }
                )

        for kind, block_number, exchange_time_ms, trade in _chronological_stream(
            witness_runs=witness_runs,
            collapsed_runs=collapsed_runs,
            model=model,
        ):
            if exchange_time_ms < scan_start:
                if kind == "witness" or trade is not None:
                    causal.observe(
                        exchange_time_ms=exchange_time_ms,
                        block_number=block_number,
                    )
                if trade is not None and trade.symbol in rolling:
                    rolling[trade.symbol].ingest(trade, watermark_ms=scan_start)
                continue
            while grid_index < len(grid_times) and grid_times[grid_index] < exchange_time_ms:
                _emit_grid_row(grid_times[grid_index])
                grid_index += 1
            if kind == "witness" or trade is not None:
                causal.observe(
                    exchange_time_ms=exchange_time_ms,
                    block_number=block_number,
                )
            if trade is not None and trade.symbol in rolling:
                trade_watermark = max(trade_watermark, exchange_time_ms - lookback)
                rolling[trade.symbol].ingest(trade, watermark_ms=trade_watermark)

        while grid_index < len(grid_times):
            _emit_grid_row(grid_times[grid_index])
            grid_index += 1

        outgoing_carry = {
            "schema_version": CARRY_SCHEMA_VERSION,
            "causal_observations": (),
            "causal_block_first_ms": causal.export_carry(),
            "trade_rows": _trade_carry_rows(
                rolling,
                carry_start_ms=model.partition.emit_end_ms - model.partition.overlap_ms,
            ),
        }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rows": rows,
            "outgoing_carry": outgoing_carry,
        }
    finally:
        shutil.rmtree(validation_state.spill_dir, ignore_errors=True)
        shutil.rmtree(collapse_dir, ignore_errors=True)
