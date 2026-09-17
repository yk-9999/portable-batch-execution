"""Closed event-window fact extraction over canonical replay records."""

from __future__ import annotations

from typing import Any

from .models import EventWindowExtractRequest

RESULT_SCHEMA_VERSION = "pbe.replay.event-window-extract-result.v1"


def _validated(request: dict[str, Any] | EventWindowExtractRequest) -> EventWindowExtractRequest:
    if isinstance(request, EventWindowExtractRequest):
        return request
    return EventWindowExtractRequest.model_validate(request)


def _tie_key(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _select_row(rows: list[dict[str, Any]], tie_break_columns: tuple[str, ...]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: _tie_key(row, tie_break_columns))


def execute_event_window_extract(
    records: list[dict[str, Any]],
    request: dict[str, Any] | EventWindowExtractRequest,
) -> dict[str, Any]:
    """Extract trailing, as-of, and future window facts for one decision event."""
    model = _validated(request)
    symbol_column = model.symbol_column
    block_column = model.block_column

    causal_rows = [
        row
        for row in records
        if row.get(symbol_column) == model.symbol and int(row[block_column]) <= model.causal_cutoff_block
    ]

    facts: list[dict[str, Any]] = []

    for spec in model.trailing_windows:
        window_rows = [
            row
            for row in causal_rows
            if model.decision_block - spec.trailing_block_count < int(row[block_column]) <= model.decision_block
        ]
        values = [row[spec.measurement_field] for row in window_rows if row.get(spec.measurement_field) is not None]
        numeric = [float(value) for value in values]
        facts.append({"fact_id": f"{spec.fact_id}.sum", "value": sum(numeric)})
        facts.append({"fact_id": f"{spec.fact_id}.count", "value": len(numeric)})

    for offset in model.as_of_offsets:
        target_block = model.decision_block - offset
        candidates = [row for row in causal_rows if int(row[block_column]) == target_block]
        chosen = _select_row(candidates, model.tie_break_columns)
        value = None if chosen is None else chosen.get(model.as_of_measurement_field)
        facts.append({"fact_id": f"as_of.{offset}", "value": value})

    for spec in model.future_windows:
        window_rows = [
            row
            for row in records
            if row.get(symbol_column) == model.symbol
            and model.decision_block + spec.start_offset_blocks
            <= int(row[block_column])
            <= model.decision_block + spec.end_offset_blocks
            and int(row[block_column]) > model.causal_cutoff_block
        ]
        chosen = _select_row(window_rows, model.tie_break_columns)
        value = None if chosen is None else chosen.get(spec.measurement_field)
        facts.append({"fact_id": spec.fact_id, "value": value})

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "facts": facts,
    }
