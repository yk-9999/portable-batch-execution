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
    _UINT64_MASK,
    _UINT64_PACK,
    StructuralCanonicalizeError,
    _bucket_contains_identity,
    _bucket_file,
    _init_empty_spill,
    _require_columns,
    _sort_unique_bucket_file_in_place,
    bucket_index,
    execute_structural_canonicalize,
)
from .event_window import _as_of_preferred, _row_is_sentinel, _validate_positive_row
from .models import CausalGridExtractRequest

RESULT_SCHEMA_VERSION = "pbe.replay.causal-grid-extract-result.v1"
CARRY_SCHEMA_VERSION = "pbe.replay.causal-grid-carry.v2"
_COLLAPSE_BATCH_ROWS = 65_536
_MAX_CAUSAL_OBSERVATIONS_PER_SEGMENT = 65_536


class _EmittedIdentitySpill:
    """On-disk exact-ID membership for collapsed trade identities (bounded buckets)."""

    def __init__(self, *, bucket_count: int, spill_dir: Path) -> None:
        self.bucket_count = bucket_count
        self.spill_dir = spill_dir / "emitted_ids"
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        _init_empty_spill(bucket_count, self.spill_dir)

    def _contains(self, identity: int) -> bool:
        path = _bucket_file(self.spill_dir, bucket_index(identity, self.bucket_count))
        return _bucket_contains_identity(path, self.bucket_count, identity)

    def mark_emitted(self, identity: int) -> None:
        if self._contains(identity):
            raise StructuralCanonicalizeError(
                "positive identity recurs non-contiguously within one input"
            )
        index = bucket_index(identity, self.bucket_count)
        path = _bucket_file(self.spill_dir, index)
        with path.open("ab") as handle:
            handle.write(_UINT64_PACK.pack(identity & _UINT64_MASK))
        _sort_unique_bucket_file_in_place(path)


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
class _CausalSegmentCompact:
    segment_id: int
    monotone_ok: bool = True
    last_time: int | None = None
    last_block: int | None = None
    observations: deque[tuple[int, int]] = field(default_factory=deque)

    def observe(self, *, exchange_time_ms: int, block_number: int) -> None:
        if (
            self.last_time is not None
            and (
                exchange_time_ms < self.last_time
                or (
                    self.last_block is not None
                    and exchange_time_ms >= self.last_time
                    and block_number < self.last_block
                )
            )
        ):
            self.monotone_ok = False
        if len(self.observations) >= _MAX_CAUSAL_OBSERVATIONS_PER_SEGMENT:
            raise StructuralCanonicalizeError("causal observation carry exceeded safe bound")
        self.observations.append((exchange_time_ms, block_number))
        self.last_time = exchange_time_ms
        self.last_block = block_number

    def causal_cutoff_block(self, decision_time_ms: int) -> int | None:
        if not self.monotone_ok:
            return None
        eligible_blocks = {
            block
            for observed_time, block in self.observations
            if observed_time <= decision_time_ms
        }
        if len(eligible_blocks) < 2:
            return None
        witness = max(eligible_blocks)
        lower = [block for block in eligible_blocks if block < witness]
        if not lower:
            return None
        return max(lower)

    def export_carry(self, *, min_time_ms: int) -> tuple[tuple[int, int], ...]:
        return tuple((self.segment_id, tm, block) for tm, block in self.observations if tm >= min_time_ms)


@dataclass
class _CausalIndex:
    missing_dates: frozenset[str]
    segments: dict[int, _CausalSegmentCompact] = field(default_factory=dict)

    @classmethod
    def from_carry(
        cls,
        missing_dates: frozenset[str],
        carry_observations: tuple[tuple[int, int, int], ...],
    ) -> _CausalIndex:
        index = cls(missing_dates=missing_dates)
        for segment_id, exchange_time_ms, block_number in carry_observations:
            state = index._state_for_segment(segment_id)
            state.observe(
                exchange_time_ms=exchange_time_ms,
                block_number=block_number,
            )
        return index

    def _state_for_time(self, exchange_ms: int) -> _CausalSegmentCompact:
        segment = _segment_id(exchange_ms, self.missing_dates)
        return self._state_for_segment(segment)

    def _state_for_segment(self, segment_id: int) -> _CausalSegmentCompact:
        if segment_id not in self.segments:
            self.segments[segment_id] = _CausalSegmentCompact(segment_id=segment_id)
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

    def export_carry(self, *, min_time_ms: int) -> tuple[tuple[int, int, int], ...]:
        exported: list[tuple[int, int, int]] = []
        for segment_id, state in sorted(self.segments.items()):
            exported.extend(state.export_carry(min_time_ms=min_time_ms))
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


def _row_to_trade_event(row: dict[str, Any], model: CausalGridExtractRequest) -> _TradeEvent:
    trade_map = model.canonical_trade_mapping
    return _TradeEvent(
        symbol=str(row[trade_map.symbol_column]),
        exchange_time_ms=int(row["_timestamp_ms"]),
        block_number=int(row["_block_int"]),
        price=float(row[trade_map.price_field]),
        notional_usd=float(row[trade_map.notional_field]),
        tie_break=_tie_break_tuple(row, model.tie_break_columns),
    )


def _flush_collapsed_batch(
    batch: list[_TradeEvent],
    *,
    spill_dir: Path,
    part_index: int,
) -> Path:
    path = spill_dir / f"collapsed_{part_index:05d}.parquet"
    pl.DataFrame(
        {
            "exchange_time_ms": [item.exchange_time_ms for item in batch],
            "block_number": [item.block_number for item in batch],
            "symbol": [item.symbol for item in batch],
            "price": [item.price for item in batch],
            "notional_usd": [item.notional_usd for item in batch],
            "tie_break": [item.tie_break for item in batch],
        }
    ).write_parquet(path)
    return path


def _materialize_collapsed_trades(
    trade_paths: list[Path],
    model: CausalGridExtractRequest,
    *,
    spill_dir: Path,
    emitted: _EmittedIdentitySpill,
) -> list[Path]:
    profile = model.canonical_trade_mapping.canonical_trade_profile
    sentinel = profile.sentinel
    identity_col = profile.identity_source_column
    normalized_col = profile.identity_normalized_column
    core_fields = profile.measurement_core_fields

    pending_row: dict[str, Any] | None = None
    pending_identity: int | None = None
    pending_core: dict[str, Any] | None = None
    batch: list[_TradeEvent] = []
    part_paths: list[Path] = []

    def flush_pending() -> None:
        nonlocal pending_row, pending_identity, pending_core
        if pending_row is None or pending_identity is None:
            return
        emitted.mark_emitted(pending_identity)
        batch.append(_row_to_trade_event(pending_row, model))
        pending_row = None
        pending_identity = None
        pending_core = None
        if len(batch) >= _COLLAPSE_BATCH_ROWS:
            part_paths.append(
                _flush_collapsed_batch(batch, spill_dir=spill_dir, part_index=len(part_paths))
            )
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
            if pending_identity is None or identity != pending_identity:
                flush_pending()
                if emitted._contains(identity):
                    raise StructuralCanonicalizeError(
                        "positive identity recurs non-contiguously within one input"
                    )
                pending_identity = identity
                pending_core = {field_name: row[field_name] for field_name in core_fields}
            elif pending_core is None:
                raise StructuralCanonicalizeError("measurement core fields disagree")
            else:
                for field_name in core_fields:
                    if row[field_name] != pending_core[field_name]:
                        raise StructuralCanonicalizeError("measurement core fields disagree")
            pending_row = row
    flush_pending()
    if batch:
        part_paths.append(
            _flush_collapsed_batch(batch, spill_dir=spill_dir, part_index=len(part_paths))
        )
    return part_paths


def _iter_sorted_collapsed_trades(part_paths: list[Path]) -> Iterator[_TradeEvent]:
    if not part_paths:
        return
    if len(part_paths) == 1:
        lazy = pl.scan_parquet(str(part_paths[0])).sort(
            "exchange_time_ms", "block_number"
        )
    else:
        lazy = pl.concat(
            [pl.scan_parquet(str(path)) for path in part_paths],
            how="vertical_relaxed",
        ).sort("exchange_time_ms", "block_number")
    row_count = int(lazy.select(pl.len()).collect().item())
    offset = 0
    while offset < row_count:
        batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
        batch = lazy.slice(offset, batch_size).collect()
        for row in batch.iter_rows(named=True):
            yield _TradeEvent(
                symbol=str(row["symbol"]),
                exchange_time_ms=int(row["exchange_time_ms"]),
                block_number=int(row["block_number"]),
                price=float(row["price"]),
                notional_usd=float(row["notional_usd"]),
                tie_break=tuple(row["tie_break"]),
            )
        offset += batch_size


def _iter_witness_events(
    path: Path,
    *,
    input_index: int,
    mapping: Any,
) -> Iterator[tuple[int, int, int, int]]:
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
            .select(
                pl.col(mapping.block_column).cast(pl.Int64, strict=False),
                pl.col(mapping.timestamp_column).cast(pl.Int64, strict=False),
            )
            .collect()
        )
        for row_index, (block_raw, time_raw) in enumerate(batch.iter_rows()):
            if block_raw is None or time_raw is None:
                raise StructuralCanonicalizeError("witness block or timestamp missing")
            yield (
                int(time_raw),
                int(block_raw),
                input_index,
                offset + row_index,
            )
        offset += batch_size


def _chronological_stream(
    path_objs: list[Path],
    model: CausalGridExtractRequest,
    *,
    collapsed_parts: list[Path],
) -> Iterator[tuple[Literal["witness", "trade"], int, int, _TradeEvent | None]]:
    bindings = {binding.input_index: binding.role for binding in model.input_roles}
    witness_iters: list[Iterator[tuple[int, int, int, int]]] = []
    for input_index, path in enumerate(path_objs):
        role = bindings.get(input_index)
        if role == "causal_witness":
            witness_iters.append(
                _iter_witness_events(
                    path,
                    input_index=input_index,
                    mapping=model.causal_witness_mapping,
                )
            )
    trade_iter = _iter_sorted_collapsed_trades(collapsed_parts)

    heap: list[
        tuple[int, int, int, int, int, Literal["witness", "trade"], _TradeEvent | None]
    ] = []
    for source_id, witness in enumerate(witness_iters):
        try:
            time_ms, block, input_index, row_off = next(witness)
        except StopIteration:
            continue
        heapq.heappush(heap, (time_ms, block, input_index, row_off, source_id, "witness", None))
    try:
        first_trade = next(trade_iter)
    except StopIteration:
        first_trade = None
    trade_source_id = len(witness_iters)
    if first_trade is not None:
        heapq.heappush(
            heap,
            (
                first_trade.exchange_time_ms,
                first_trade.block_number,
                0,
                0,
                trade_source_id,
                "trade",
                first_trade,
            ),
        )

    while heap:
        time_ms, block, input_index, row_off, sid, kind, payload = heapq.heappop(heap)
        yield kind, block, time_ms, payload
        if kind == "witness":
            iterator = witness_iters[sid]
            try:
                next_time, next_block, next_index, next_off = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (next_time, next_block, next_index, next_off, sid, "witness", None),
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
                    trade_source_id,
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
    validation_state = execute_structural_canonicalize(
        trade_paths,
        profile.structural_canonicalize_params(),
    )
    collapse_dir = tempfile.mkdtemp(prefix="pbe-grid-collapse-")
    try:
        emitted = _EmittedIdentitySpill(
            bucket_count=profile.structural_canonicalize_params().bucket_count,
            spill_dir=Path(collapse_dir),
        )
        collapsed_parts = _materialize_collapsed_trades(
            trade_paths,
            model,
            spill_dir=Path(collapse_dir),
            emitted=emitted,
        )
        missing = frozenset(model.partition.hard_gap_missing_dates)
        carry_in = model.partition.incoming_carry
        causal = _CausalIndex.from_carry(
            missing,
            carry_in.causal_observations if carry_in is not None else (),
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
            path_objs,
            model,
            collapsed_parts=collapsed_parts,
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

        carry_start = model.partition.emit_end_ms - model.partition.overlap_ms
        outgoing_carry = {
            "schema_version": CARRY_SCHEMA_VERSION,
            "causal_observations": causal.export_carry(min_time_ms=carry_start),
            "trade_rows": _trade_carry_rows(rolling, carry_start_ms=carry_start),
        }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rows": rows,
            "outgoing_carry": outgoing_carry,
        }
    finally:
        shutil.rmtree(validation_state.spill_dir, ignore_errors=True)
        shutil.rmtree(collapse_dir, ignore_errors=True)
