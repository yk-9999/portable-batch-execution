"""Nullable typed JSON scalar projections for paired-fill ingestion."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

from .json_scalar_projection import (
    _OUTPUT_COLUMN_PATTERN,
    _RESERVED_OUTPUT_COLUMNS,
    JSON_SCALAR_KEY_PATH_MAX_DEPTH,
    JSON_SCALAR_PROJECTION_MAX,
    JsonScalarProjectionError,
    _require_string_source_column,
)
from .models import NullableJsonScalarProjection


def validate_nullable_json_scalar_projection_bundle(
    projections: tuple[NullableJsonScalarProjection, ...],
) -> None:
    if len(projections) > JSON_SCALAR_PROJECTION_MAX:
        raise ValueError("nullable json scalar projection count exceeds bounded limit")
    outputs: set[str] = set()
    paths: set[tuple[str, tuple[str, ...]]] = set()
    projection_outputs: set[str] = set()
    for projection in projections:
        if len(projection.key_path) > JSON_SCALAR_KEY_PATH_MAX_DEPTH:
            raise ValueError(
                "nullable json scalar projection key_path exceeds bounded depth"
            )
        if projection.output_column in _RESERVED_OUTPUT_COLUMNS:
            raise ValueError(
                "nullable json scalar projection output column is reserved"
            )
        if not _OUTPUT_COLUMN_PATTERN.fullmatch(projection.output_column):
            raise ValueError("nullable json scalar projection output column is unsafe")
        if projection.output_column in outputs:
            raise ValueError("duplicate nullable json scalar projection output column")
        outputs.add(projection.output_column)
        projection_outputs.add(projection.output_column)
        path_key = (projection.source_column, projection.key_path)
        if path_key in paths:
            raise ValueError("duplicate nullable json scalar projection")
        paths.add(path_key)
        if projection.source_column in projection_outputs:
            raise ValueError("ambiguous nullable json scalar projection source column")


def validate_nullable_json_projections_against_schema(
    projections: tuple[NullableJsonScalarProjection, ...],
    schema: pl.Schema,
) -> None:
    if not projections:
        return
    validate_nullable_json_scalar_projection_bundle(projections)
    column_names = set(schema.names())
    for projection in projections:
        if projection.output_column in column_names:
            raise JsonScalarProjectionError(
                "nullable json scalar projection output column collides with input column"
            )
        _require_string_source_column(schema, projection.source_column)


def _extract_nullable_scalar(
    raw: Any,
    projection: NullableJsonScalarProjection,
) -> int | str | float | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    current: Any = parsed
    for key in projection.key_path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if projection.scalar_type == "integer":
        if isinstance(current, bool):
            return None
        if isinstance(current, int):
            return int(current)
        if (
            isinstance(current, float)
            and math.isfinite(current)
            and current == math.trunc(current)
        ):
            return int(current)
        return None
    if projection.scalar_type == "float":
        if isinstance(current, bool):
            return None
        if isinstance(current, (int, float)):
            number = float(current)
            return number if math.isfinite(number) else None
        return None
    if isinstance(current, str):
        return current
    return None


def project_nullable_json_scalar_batch(
    batch: pl.DataFrame,
    projections: tuple[NullableJsonScalarProjection, ...],
) -> pl.DataFrame:
    if not projections:
        return batch
    validate_nullable_json_projections_against_schema(projections, batch.schema)
    if batch.is_empty():
        dtypes = {
            "integer": pl.Int64,
            "float": pl.Float64,
            "string": pl.Utf8,
        }
        extra = {
            projection.output_column: pl.Series(
                projection.output_column,
                [],
                dtype=dtypes[projection.scalar_type],
            )
            for projection in projections
        }
        return batch.with_columns(**extra)
    new_columns: dict[str, list[Any]] = {
        projection.output_column: [] for projection in projections
    }
    unique_sources = list(
        dict.fromkeys(projection.source_column for projection in projections)
    )
    for row in batch.select(unique_sources).iter_rows():
        source_by_column = dict(zip(unique_sources, row, strict=True))
        for projection in projections:
            new_columns[projection.output_column].append(
                _extract_nullable_scalar(
                    source_by_column[projection.source_column], projection
                )
            )
    dtypes = {
        "integer": pl.Int64,
        "float": pl.Float64,
        "string": pl.Utf8,
    }
    series = [
        pl.Series(
            projection.output_column,
            new_columns[projection.output_column],
            dtype=dtypes[projection.scalar_type],
        )
        for projection in projections
    ]
    return batch.with_columns(series)


def _schema_with_nullable_projections(
    schema: pl.Schema, projections: tuple[NullableJsonScalarProjection, ...]
) -> pl.Schema:
    if not projections:
        return schema
    fields = dict(schema)
    dtypes = {
        "integer": pl.Int64,
        "float": pl.Float64,
        "string": pl.Utf8,
    }
    for projection in projections:
        fields[projection.output_column] = dtypes[projection.scalar_type]
    return pl.Schema(fields)


def apply_nullable_json_projections_to_lazy(
    lazy: pl.LazyFrame,
    projections: tuple[NullableJsonScalarProjection, ...],
) -> pl.LazyFrame:
    if not projections:
        return lazy
    input_schema = lazy.collect_schema()
    validate_nullable_json_projections_against_schema(projections, input_schema)
    output_schema = _schema_with_nullable_projections(input_schema, projections)

    def _map_batch(batch: pl.DataFrame) -> pl.DataFrame:
        return project_nullable_json_scalar_batch(batch, projections)

    return lazy.map_batches(_map_batch, schema=output_schema)
