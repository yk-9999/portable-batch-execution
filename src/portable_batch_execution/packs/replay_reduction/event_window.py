"""Closed event-window fact extraction over parquet inputs without whole-history loads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .models import EventWindowExtractRequest

RESULT_SCHEMA_VERSION = "pbe.replay.event-window-extract-result.v1"


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


def execute_event_window_extract(
    paths: list[str | Path],
    request: dict[str, Any] | EventWindowExtractRequest,
) -> dict[str, Any]:
    """Extract trailing, as-of, and future window facts using lazy parquet scans."""
    model = _validated(request)
    symbol_column = model.symbol_column
    block_column = model.block_column
    lazy = _scan_inputs(paths).with_columns(
        pl.col(block_column).cast(pl.Int64, strict=False).alias("_block_int")
    )
    symbol_lazy = lazy.filter(pl.col(symbol_column) == model.symbol)
    causal = symbol_lazy.filter(pl.col("_block_int") <= model.causal_cutoff_block)

    facts: list[dict[str, Any]] = []

    for spec in model.trailing_windows:
        window = causal.filter(
            (pl.col("_block_int") > model.decision_block - spec.trailing_block_count)
            & (pl.col("_block_int") <= model.decision_block)
        )
        summary = window.select(
            pl.col(spec.measurement_field).cast(pl.Float64, strict=False).sum().alias("sum"),
            pl.col(spec.measurement_field).is_not_null().sum().alias("count"),
        ).collect()
        row = summary.row(0, named=True)
        facts.append({"fact_id": f"{spec.fact_id}.sum", "value": row["sum"] or 0.0})
        facts.append({"fact_id": f"{spec.fact_id}.count", "value": int(row["count"] or 0)})

    for offset in model.as_of_offsets:
        target_block = model.decision_block - offset
        candidates = causal.filter(pl.col("_block_int") == target_block).sort(
            list(model.tie_break_columns)
        )
        chosen = candidates.select(model.as_of_measurement_field).head(1).collect()
        value = None if chosen.is_empty() else chosen.item(0, 0)
        facts.append({"fact_id": f"as_of.{offset}", "value": value})

    for spec in model.future_windows:
        future = symbol_lazy.filter(
            (pl.col("_block_int") > model.causal_cutoff_block)
            & (pl.col("_block_int") >= model.decision_block + spec.start_offset_blocks)
            & (pl.col("_block_int") <= model.decision_block + spec.end_offset_blocks)
        ).sort(list(model.tie_break_columns))
        chosen = future.select(spec.measurement_field).head(1).collect()
        value = None if chosen.is_empty() else chosen.item(0, 0)
        facts.append({"fact_id": spec.fact_id, "value": value})

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "facts": facts,
    }
