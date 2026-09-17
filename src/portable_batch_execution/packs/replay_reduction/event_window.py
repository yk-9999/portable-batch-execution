"""Closed event-window fact extraction using UTC timestamps and causal block guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .models import EventWindowExtractRequest

RESULT_SCHEMA_VERSION = "pbe.replay.event-window-extract-result.v2"


def _validated(request: dict[str, Any] | EventWindowExtractRequest) -> EventWindowExtractRequest:
    if isinstance(request, EventWindowExtractRequest):
        return request
    return EventWindowExtractRequest.model_validate(request)


def _scan_inputs(paths: list[str | Path]) -> pl.LazyFrame:
    if not paths:
        raise ValueError("at least one parquet input is required")
    frames = [pl.scan_parquet(str(path)) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed")


def _with_normalized_time(lazy: pl.LazyFrame, request: EventWindowExtractRequest) -> pl.LazyFrame:
    return lazy.with_columns(
        pl.col(request.block_column).cast(pl.Int64, strict=False).alias("_block_int"),
        pl.col(request.timestamp_column).cast(pl.Int64, strict=False).alias("_timestamp_ms"),
    )


def execute_event_window_extract(
    paths: list[str | Path],
    request: dict[str, Any] | EventWindowExtractRequest,
) -> dict[str, Any]:
    """Extract trailing, as-of, and future window facts using lazy parquet scans."""
    model = _validated(request)
    lazy = _with_normalized_time(_scan_inputs(paths), model)
    symbol_lazy = lazy.filter(pl.col(model.symbol_column) == model.symbol)
    causal = symbol_lazy.filter(pl.col("_block_int") <= model.causal_cutoff_block)
    decision_ms = int(model.decision_timestamp_ms)
    tie_break = list(model.tie_break_columns)

    facts: list[dict[str, Any]] = []

    for spec in model.trailing_windows:
        window = causal.filter(
            (pl.col("_timestamp_ms") > decision_ms - spec.trailing_width_ms)
            & (pl.col("_timestamp_ms") <= decision_ms)
        )
        summary = window.select(
            pl.col(spec.measurement_field).cast(pl.Float64, strict=False).sum().alias("sum"),
            pl.col(spec.measurement_field).is_not_null().sum().alias("count"),
        ).collect()
        row = summary.row(0, named=True)
        facts.append({"fact_id": f"{spec.fact_id}.sum", "value": row["sum"] or 0.0})
        facts.append({"fact_id": f"{spec.fact_id}.count", "value": int(row["count"] or 0)})

    for offset_ms in model.as_of_offsets_ms:
        target_ms = decision_ms - int(offset_ms)
        candidates = (
            causal.filter(pl.col("_timestamp_ms") <= target_ms)
            .sort(tie_break, descending=[True] * len(tie_break))
            .head(1)
        )
        chosen = candidates.select(model.as_of_measurement_field).collect()
        value = None if chosen.is_empty() else chosen.item(0, 0)
        facts.append({"fact_id": f"as_of.{offset_ms}", "value": value})

    for spec in model.future_windows:
        future = symbol_lazy.filter(
            (pl.col("_timestamp_ms") >= decision_ms + spec.start_offset_ms)
            & (pl.col("_timestamp_ms") <= decision_ms + spec.end_offset_ms)
            & (pl.col("_block_int") > spec.min_block_exclusive)
        ).sort(tie_break)
        chosen = future.select(spec.measurement_field).head(1).collect()
        value = None if chosen.is_empty() else chosen.item(0, 0)
        facts.append({"fact_id": spec.fact_id, "value": value})

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "facts": facts,
    }
