"""Dense causal replay grid extraction in one bounded pass per contiguous partition."""

from __future__ import annotations

import heapq
import shutil
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
    execute_structural_canonicalize,
)
from .event_window import _as_of_preferred, _row_is_sentinel, _validate_positive_row
from .models import CausalGridExtractRequest

RESULT_SCHEMA_VERSION = "pbe.replay.causal-grid-extract-result.v1"
CARRY_SCHEMA_VERSION = "pbe.replay.causal-grid-carry.v1"


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
class _CausalBlockState:
    segment_id: int
    observations: list[tuple[int, int]] = field(default_factory=list)
    last_time: int | None = None
    last_block: int | None = None
    monotone_ok: bool = True

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

    def export_observations(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.observations)


@dataclass
class _CausalIndex:
    missing_dates: frozenset[str]
    segments: dict[int, _CausalBlockState] = field(default_factory=dict)

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

    def _state_for_time(self, exchange_ms: int) -> _CausalBlockState:
        segment = _segment_id(exchange_ms, self.missing_dates)
        return self._state_for_segment(segment)

    def _state_for_segment(self, segment_id: int) -> _CausalBlockState:
        if segment_id not in self.segments:
            self.segments[segment_id] = _CausalBlockState(segment_id=segment_id)
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
        for segment_id, state in sorted(self.segments.items()):
            for exchange_time_ms, block_number in state.observations:
                exported.append((segment_id, exchange_time_ms, block_number))
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
class _GridFactBuilder:
    tie_break: tuple[str, ...]
    as_of_offsets_ms: tuple[int, ...]
    as_of_field: str
    trailing_specs: tuple[Any, ...]

    def facts_for(
        self,
        *,
        symbol: str,
        decision_time_ms: int,
        cutoff: int | None,
        trades: list[_TradeEvent],
    ) -> list[dict[str, Any]]:
        if cutoff is None:
            facts: list[dict[str, Any]] = []
            for offset_ms in self.as_of_offsets_ms:
                facts.append({"fact_id": f"as_of.{offset_ms}", "value": None})
            for spec in self.trailing_specs:
                facts.append({"fact_id": f"{spec.fact_id}.sum", "value": None})
                facts.append({"fact_id": f"{spec.fact_id}.count", "value": None})
            return facts

        symbol_trades = [row for row in trades if row.symbol == symbol]
        as_of_best: dict[int, dict[str, Any] | None] = {
            int(offset): None for offset in self.as_of_offsets_ms
        }
        trailing_sums: dict[str, float] = {}
        trailing_counts: dict[str, int] = {}

        for row in symbol_trades:
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
                    candidate = row_dict
                    if _as_of_preferred(candidate, current, self.tie_break):
                        as_of_best[int(offset_ms)] = candidate
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
        for block_raw, time_raw in batch.iter_rows():
            if block_raw is None or time_raw is None:
                raise StructuralCanonicalizeError("witness block or timestamp missing")
            yield (
                int(time_raw),
                int(block_raw),
                input_index,
                offset,
            )
        offset += batch_size


def _iter_canonical_trade_events(
    path: Path,
    *,
    input_index: int,
    model: CausalGridExtractRequest,
) -> Iterator[tuple[int, int, int, _TradeEvent]]:
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
    previous_identity: int | None = None
    group_core: dict[str, Any] | None = None
    representative: dict[str, Any] | None = None
    row_sequence = 0
    collapsed_events: list[tuple[int, int, int, _TradeEvent]] = []

    def flush() -> None:
        nonlocal representative, row_sequence
        if representative is None:
            return
        row = representative
        event = _TradeEvent(
            symbol=str(row[trade_map.symbol_column]),
            exchange_time_ms=int(row["_timestamp_ms"]),
            block_number=int(row["_block_int"]),
            price=float(row[trade_map.price_field]),
            notional_usd=float(row[trade_map.notional_field]),
            tie_break=_tie_break_tuple(row, tie_break),
        )
        collapsed_events.append(
            (
                event.exchange_time_ms,
                event.block_number,
                row_sequence,
                event,
            )
        )
        row_sequence += 1
        representative = None

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
            row["_block_int"] = row["_block_int"]
            row["_timestamp_ms"] = row["_timestamp_ms"]
            if _row_is_sentinel(row, sentinel):
                continue
            identity = _validate_positive_row(
                row,
                identity_col=identity_col,
                normalized_col=normalized_col,
                core_fields=core_fields,
            )
            if previous_identity is None or identity != previous_identity:
                flush()
                previous_identity = identity
                group_core = {field_name: row[field_name] for field_name in core_fields}
            elif group_core is None:
                raise StructuralCanonicalizeError("measurement core fields disagree")
            else:
                for field_name in core_fields:
                    if row[field_name] != group_core[field_name]:
                        raise StructuralCanonicalizeError("measurement core fields disagree")
            representative = dict(row)
        offset += batch_size

    flush()
    collapsed_events.sort(key=lambda item: (item[0], item[1], item[2]))
    for exchange_time_ms, block_number, sequence, event in collapsed_events:
        yield (exchange_time_ms, block_number, input_index, sequence, event)


def _source_iterators(
    paths: list[Path],
    model: CausalGridExtractRequest,
) -> list[Iterator[tuple[int, int, int, int, Literal["witness", "trade"], _TradeEvent | None]]]:
    bindings = {binding.input_index: binding.role for binding in model.input_roles}
    iterators: list[
        Iterator[tuple[int, int, int, int, Literal["witness", "trade"], _TradeEvent | None]]
    ] = []
    for input_index, path in enumerate(paths):
        role = bindings.get(input_index)
        if role is None:
            raise ValueError(f"missing input role binding for input_index={input_index}")
        if role == "causal_witness":
            iterators.append(
                (
                    (time_ms, block, idx, row_off, "witness", None)
                    for time_ms, block, idx, row_off in _iter_witness_events(
                        path,
                        input_index=input_index,
                        mapping=model.causal_witness_mapping,
                    )
                )
            )
        elif role == "canonical_trade":
            iterators.append(
                (
                    (time_ms, block, idx, row_off, "trade", trade)
                    for time_ms, block, idx, row_off, trade in _iter_canonical_trade_events(
                        path,
                        input_index=input_index,
                        model=model,
                    )
                )
            )
        else:
            raise ValueError(f"unsupported input role: {role}")
    return iterators


def _merge_stream(
    paths: list[Path],
    model: CausalGridExtractRequest,
) -> Iterator[tuple[Literal["witness", "trade"], int, int, _TradeEvent | None]]:
    iterators = _source_iterators(paths, model)
    heap: list[
        tuple[int, int, int, int, int, Literal["witness", "trade"], _TradeEvent | None]
    ] = []
    for source_id, iterator in enumerate(iterators):
        try:
            time_ms, block, input_index, row_off, kind, trade = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (time_ms, block, input_index, row_off, source_id, kind, trade),
        )
    while heap:
        time_ms, block, input_index, row_off, source_id, kind, trade = heapq.heappop(heap)
        yield kind, block, time_ms, trade
        iterator = iterators[source_id]
        try:
            next_time, next_block, next_index, next_off, next_kind, next_trade = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (next_time, next_block, next_index, next_off, source_id, next_kind, next_trade),
        )


def _trade_carry_rows(
    trades: list[_TradeEvent],
    *,
    carry_start_ms: int,
) -> tuple[dict[str, Any], ...]:
    selected = [row for row in trades if row.exchange_time_ms >= carry_start_ms]
    return tuple(
        {
            "symbol": row.symbol,
            "exchange_time_ms": row.exchange_time_ms,
            "block_number": row.block_number,
            "price": row.price,
            "notional_usd": row.notional_usd,
            "tie_break": row.tie_break,
        }
        for row in selected
    )


def _load_trade_carry(
    carry_rows: tuple[dict[str, Any], ...],
    tie_break: tuple[str, ...],
) -> list[_TradeEvent]:
    loaded: list[_TradeEvent] = []
    for row in carry_rows:
        loaded.append(
            _TradeEvent(
                symbol=str(row["symbol"]),
                exchange_time_ms=int(row["exchange_time_ms"]),
                block_number=int(row["block_number"]),
                price=float(row["price"]),
                notional_usd=float(row["notional_usd"]),
                tie_break=tuple(row["tie_break"]),
            )
        )
    return loaded


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
    trade_indices = [
        binding.input_index
        for binding in model.input_roles
        if binding.role == "canonical_trade"
    ]
    if not trade_indices:
        raise ValueError("at least one canonical_trade input is required")
    trade_paths = [path_objs[index] for index in sorted(trade_indices)]
    profile = model.canonical_trade_mapping.canonical_trade_profile
    validation_state = execute_structural_canonicalize(
        trade_paths,
        profile.structural_canonicalize_params(),
    )
    try:
        missing = frozenset(model.partition.hard_gap_missing_dates)
        carry_in = model.partition.incoming_carry
        causal = _CausalIndex.from_carry(
            missing,
            carry_in.causal_observations if carry_in is not None else (),
        )
        trades = _load_trade_carry(
            carry_in.trade_rows if carry_in is not None else (),
            model.tie_break_columns,
        )
        lookback = _max_lookback_ms(model)
        scan_start = model.partition.emit_start_ms - lookback
        grid_times = _grid_timestamps(model)
        fact_builder = _GridFactBuilder(
            tie_break=model.tie_break_columns,
            as_of_offsets_ms=model.as_of_offsets_ms,
            as_of_field=model.as_of_measurement_field,
            trailing_specs=model.trailing_windows,
        )
        grid_index = 0
        rows: list[dict[str, Any]] = []

        for kind, block_number, exchange_time_ms, trade in _merge_stream(path_objs, model):
            if exchange_time_ms < scan_start:
                if kind == "witness" or trade is not None:
                    causal.observe(
                        exchange_time_ms=exchange_time_ms,
                        block_number=block_number,
                    )
                if trade is not None:
                    trades.append(trade)
                continue
            while grid_index < len(grid_times) and grid_times[grid_index] < exchange_time_ms:
                decision_ms = grid_times[grid_index]
                cutoff = causal.cutoff(decision_ms)
                for symbol in model.target_symbols:
                    rows.append(
                        {
                            "symbol": symbol,
                            "grid_timestamp_ms": decision_ms,
                            "causal_cutoff_block": cutoff,
                            "facts": fact_builder.facts_for(
                                symbol=symbol,
                                decision_time_ms=decision_ms,
                                cutoff=cutoff,
                                trades=trades,
                            ),
                        }
                    )
                grid_index += 1
            if kind == "witness" or trade is not None:
                causal.observe(
                    exchange_time_ms=exchange_time_ms,
                    block_number=block_number,
                )
            if trade is not None:
                trades.append(trade)

        while grid_index < len(grid_times):
            decision_ms = grid_times[grid_index]
            cutoff = causal.cutoff(decision_ms)
            for symbol in model.target_symbols:
                rows.append(
                    {
                        "symbol": symbol,
                        "grid_timestamp_ms": decision_ms,
                        "causal_cutoff_block": cutoff,
                        "facts": fact_builder.facts_for(
                            symbol=symbol,
                            decision_time_ms=decision_ms,
                            cutoff=cutoff,
                            trades=trades,
                        ),
                    }
                )
            grid_index += 1

        carry_start = model.partition.emit_end_ms - model.partition.overlap_ms
        outgoing_carry = {
            "schema_version": CARRY_SCHEMA_VERSION,
            "causal_observations": causal.export_carry(),
            "trade_rows": _trade_carry_rows(trades, carry_start_ms=carry_start),
        }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rows": rows,
            "outgoing_carry": outgoing_carry,
        }
    finally:
        shutil.rmtree(validation_state.spill_dir, ignore_errors=True)
