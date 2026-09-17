"""Structural canonicalization via compact ordered-range evidence (O(segments) state)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl

from .models import StructuralCanonicalizeParams

RESULT_SCHEMA_VERSION = "pbe.replay.structural-canonicalize-result.v3"


class StructuralCanonicalizeError(Exception):
    """Input rows violate structural canonicalization invariants."""


def _validated(params: dict[str, Any] | StructuralCanonicalizeParams) -> StructuralCanonicalizeParams:
    if isinstance(params, StructuralCanonicalizeParams):
        return params
    return StructuralCanonicalizeParams.model_validate(params)


def _require_columns(schema: pl.Schema, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in schema]
    if missing:
        raise StructuralCanonicalizeError(f"missing required columns: {', '.join(missing)}")


def _boundary_profile(row: dict[str, Any], core_fields: list[str]) -> dict[str, Any]:
    return {
        "identity": int(row["_identity_int"]),
        "core": {field: row[field] for field in core_fields},
    }


def _summarize_single_input(
    path: str | Path,
    source_index: int,
    model: StructuralCanonicalizeParams,
) -> tuple[int, dict[str, Any]]:
    identity_col = model.identity_source_column
    normalized_col = model.identity_normalized_column
    core_fields = list(model.measurement_core_fields)
    sentinel_value = model.sentinel.identity_equals if model.sentinel is not None else None

    lazy = pl.scan_parquet(str(path))
    _require_columns(lazy.collect_schema(), (identity_col, normalized_col, *core_fields))
    lazy = lazy.with_row_index("_row_index").with_columns(
        pl.col(identity_col).cast(pl.Int64, strict=False).alias("_identity_int"),
    )

    if sentinel_value is not None:
        witness_count = int(
            lazy.filter(pl.col("_identity_int") == sentinel_value)
            .select(pl.len())
            .collect()
            .item()
        )
    else:
        witness_count = 0

    nonpositive = pl.col("_identity_int") <= 0
    if sentinel_value is None:
        invalid_expr = pl.col("_identity_int").is_null() | nonpositive
    else:
        invalid_expr = pl.col("_identity_int").is_null() | (
            nonpositive & (pl.col("_identity_int") != sentinel_value)
        )
    if int(lazy.filter(invalid_expr).select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity is missing or not positive")

    economic = lazy.filter(pl.col("_identity_int") > 0).filter(
        pl.col(normalized_col).cast(pl.Utf8, strict=False)
        == pl.col("_identity_int").cast(pl.Utf8)
    )
    mismatch = lazy.filter(pl.col("_identity_int") > 0).filter(
        pl.col(normalized_col).cast(pl.Utf8, strict=False)
        != pl.col("_identity_int").cast(pl.Utf8)
    )
    if int(mismatch.select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity normalized column mismatch")

    positive_row_count = int(economic.select(pl.len()).collect().item())
    if positive_row_count == 0:
        return witness_count, {
            "source_input_index": source_index,
            "positive_row_count": 0,
            "distinct_identity_count": 0,
        }

    ordered = economic.sort("_row_index")
    decreasing = ordered.select(pl.col("_identity_int").diff().lt(0).any()).collect().item()
    if decreasing:
        raise StructuralCanonicalizeError("positive identities are not nondecreasing in source order")

    for field in core_fields:
        disagree = (
            economic.group_by("_identity_int")
            .agg(pl.col(field).n_unique().alias("_nunique"))
            .filter(pl.col("_nunique") > 1)
            .select(pl.len())
            .collect()
            .item()
        )
        if disagree:
            raise StructuralCanonicalizeError("measurement core fields disagree")

    stats = ordered.select(
        pl.col("_identity_int").min().alias("identity_min"),
        pl.col("_identity_int").max().alias("identity_max"),
        pl.col("_identity_int").n_unique().alias("distinct_identity_count"),
    ).collect()
    stat_row = stats.row(0, named=True)

    first_row = ordered.head(1).collect().row(0, named=True)
    last_row = ordered.tail(1).collect().row(0, named=True)

    segment = {
        "source_input_index": source_index,
        "positive_row_count": positive_row_count,
        "distinct_identity_count": int(stat_row["distinct_identity_count"]),
        "identity_min": int(stat_row["identity_min"]),
        "identity_max": int(stat_row["identity_max"]),
        "first_boundary": _boundary_profile(first_row, core_fields),
        "last_boundary": _boundary_profile(last_row, core_fields),
    }
    return witness_count, segment


def _segments_with_positive_data(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [segment for segment in segments if int(segment.get("positive_row_count", 0)) > 0]


def _link_adjacent_segments(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    dedup_shared_boundary_row: bool = False,
) -> dict[str, Any] | None:
    """Return coalesced segment when right continues left at a shared boundary identity."""
    left_max = int(left["identity_max"])
    right_min = int(right["identity_min"])

    if right_min > left_max:
        return None

    if right_min != left_max:
        raise StructuralCanonicalizeError("identity ranges overlap without a shared boundary")

    left_last = left["last_boundary"]
    right_first = right["first_boundary"]
    if left_last["identity"] != right_min or right_first["identity"] != right_min:
        raise StructuralCanonicalizeError("shared boundary identity mismatch")

    if left_last["core"] != right_first["core"]:
        raise StructuralCanonicalizeError("measurement core fields disagree at range boundary")

    row_count = int(left["positive_row_count"]) + int(right["positive_row_count"])
    if dedup_shared_boundary_row:
        row_count -= 1

    return {
        "source_input_index": int(left["source_input_index"]),
        "positive_row_count": row_count,
        "distinct_identity_count": int(left["distinct_identity_count"])
        + int(right["distinct_identity_count"])
        - 1,
        "identity_min": int(left["identity_min"]),
        "identity_max": int(right["identity_max"]),
        "first_boundary": deepcopy(left["first_boundary"]),
        "last_boundary": deepcopy(right["last_boundary"]),
    }


def _validate_and_compact_segment_chain(
    segments: list[dict[str, Any]],
    *,
    dedup_shared_boundary_at_link: int | None = None,
) -> list[dict[str, Any]]:
    """Validate cross-input ordering and compact shared-boundary segments."""
    if not segments:
        return []
    positives = _segments_with_positive_data(segments)
    if not positives:
        return list(segments)

    compact: list[dict[str, Any]] = []
    pending = deepcopy(positives[0])
    for link_index, segment in enumerate(positives[1:]):
        dedup = dedup_shared_boundary_at_link == link_index
        coalesced = _link_adjacent_segments(
            pending, segment, dedup_shared_boundary_row=dedup
        )
        if coalesced is not None:
            pending = coalesced
            continue
        if int(segment["identity_min"]) <= int(pending["identity_max"]):
            raise StructuralCanonicalizeError("identity ranges overlap without a shared boundary")
        compact.append(pending)
        pending = deepcopy(segment)
    compact.append(pending)

    empty_segments = [
        segment for segment in segments if int(segment.get("positive_row_count", 0)) == 0
    ]
    return empty_segments + compact


def _merge_segment_lists(
    left_segments: list[dict[str, Any]], right_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    left_positives = _segments_with_positive_data(left_segments)
    right_positives = _segments_with_positive_data(right_segments)
    dedup_link = len(left_positives) - 1 if left_positives and right_positives else None
    return _validate_and_compact_segment_chain(
        list(left_segments) + list(right_segments),
        dedup_shared_boundary_at_link=dedup_link,
    )


def execute_structural_canonicalize(
    paths: list[str | Path],
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> dict[str, Any]:
    """Prove structural order per bounded parquet input without O(distinct identities) state."""
    if not paths:
        raise StructuralCanonicalizeError("at least one parquet input is required")
    model = _validated(params)
    segments: list[dict[str, Any]] = []
    witness_row_count = 0

    for source_index, path in enumerate(paths):
        witness_count, segment = _summarize_single_input(path, source_index, model)
        witness_row_count += witness_count
        segments.append(segment)

    range_segments = _validate_and_compact_segment_chain(segments)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "range_segments": range_segments,
        "witness_row_count": witness_row_count,
    }


def merge_structural_canonicalize_states(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Merge two compact ordered-range states exactly across bounded waves."""
    if left.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("left canonicalization state schema mismatch")
    if right.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("right canonicalization state schema mismatch")

    merged_segments = _merge_segment_lists(
        list(left.get("range_segments", ())),
        list(right.get("range_segments", ())),
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "range_segments": merged_segments,
        "witness_row_count": int(left.get("witness_row_count", 0))
        + int(right.get("witness_row_count", 0)),
    }
