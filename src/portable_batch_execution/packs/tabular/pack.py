"""Polars implementation of closed, portable tabular transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .models import (
    PARAM_MODELS,
    NormalizeParams,
    PitJoinParams,
    RollingParams,
    SortKey,
    StatisticsParams,
    WindowParams,
)


def rolling_halo(window_size: int) -> int:
    """Number of preceding rows a trailing rolling window needs from its neighbour."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    return window_size - 1


def rolling_halo_rows(params: RollingParams | dict[str, Any]) -> int:
    """Convenience helper for schedulers constructing shard lookback halos."""
    return rolling_halo(_validated(RollingParams, params).window_size)


def _validated(model, params):
    return params if isinstance(params, model) else model.model_validate(params)


def _sort(df: pl.DataFrame, keys: tuple[SortKey, ...]) -> pl.DataFrame:
    if not keys:
        return df
    return df.sort(
        [key.column for key in keys],
        descending=[key.descending for key in keys],
        nulls_last=[key.nulls_last for key in keys],
    )


def _cast_expr(column: str, dtype: str, strict: bool) -> pl.Expr:
    dtypes = {
        "string": pl.String, "integer": pl.Int64, "float": pl.Float64,
        "boolean": pl.Boolean, "date": pl.Date, "datetime": pl.Datetime,
    }
    return pl.col(column).cast(dtypes[dtype], strict=strict).alias(column)


def _read(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, list):
        return pl.DataFrame(value)
    if isinstance(value, dict):
        # Context values may be an ArtifactRef-like object only when URI is present.
        value = value.get("uri", value)
    if hasattr(value, "uri"):
        value = value.uri
    if not isinstance(value, (str, Path)):
        raise TypeError("tabular input must be a DataFrame, record list, or supported file")
    path = Path(str(value).removeprefix("file:///"))
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pl.read_ndjson(path) if suffix in {".jsonl", ".ndjson"} else pl.read_json(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError("supported tabular formats are CSV, JSON, JSONL, and Parquet")


def _write(df: pl.DataFrame, destination: str | Path, output_format: str) -> Path:
    path = Path(destination)
    if output_format == "csv": df.write_csv(path)
    elif output_format == "json": df.write_json(path)
    elif output_format == "jsonl": df.write_ndjson(path)
    elif output_format == "parquet": df.write_parquet(path)
    else: raise ValueError("unsupported output format")
    return path


class TabularPack:
    pack_id = "tabular-batch"
    supported_operations = tuple(PARAM_MODELS)

    def validate_params(self, operation: str, params: dict) -> dict:
        try:
            model = PARAM_MODELS[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported tabular operation: {operation}") from exc
        return model.model_validate(params).model_dump(mode="json")

    def run(self, operation: str, data: Any, params: dict | Any, *, right: Any = None, destination: str | Path | None = None):
        """Run one transformation.  ``right`` is only accepted by join operations."""
        if operation not in PARAM_MODELS:
            raise ValueError(f"unsupported tabular operation: {operation}")
        model = _validated(PARAM_MODELS[operation], params)
        df = _read(data)
        if operation == "tabular.normalize":
            result = self._normalize(df, model)
        elif operation == "tabular.cast":
            result = df.with_columns([_cast_expr(k, v, model.strict) for k, v in model.columns.items()])
        elif operation == "tabular.sort":
            result = _sort(df, model.by)
        elif operation == "tabular.dedup":
            keep = "none" if model.keep == "none" else model.keep
            result = df.unique(subset=model.subset, keep=keep, maintain_order=model.maintain_order)
        elif operation == "tabular.join":
            result = df.join(_read(right), on=list(model.on), how=model.how, suffix=model.suffix)
        elif operation == "tabular.pit_join":
            result = self._pit_join(df, _read(right), model)
        elif operation == "tabular.window":
            result = self._window(df, model)
        elif operation == "tabular.rolling":
            result = self._rolling(df, model)
        elif operation == "tabular.statistics":
            result = self._statistics(df, model)
        else:
            result = df
        if operation == "tabular.format_migration" and destination is None:
            raise ValueError("format_migration requires a destination")
        return _write(result, destination, model.output_format) if operation == "tabular.format_migration" else result

    def execute(self, job, shard, params, context):
        """DomainPack entry point; context supplies ``data``, optional ``right`` and destination."""
        operation = getattr(job, "operation", None) or context["operation"]
        return self.run(operation, context["data"], params, right=context.get("right"), destination=context.get("destination"))

    def finalize(self, job, canonical_attempts, context):
        return canonical_attempts

    @staticmethod
    def _normalize(df: pl.DataFrame, params: NormalizeParams) -> pl.DataFrame:
        columns = params.columns or tuple(df.columns)
        expressions = []
        for column in columns:
            expr = pl.col(column)
            # String normalization must not coerce numeric/date columns.
            if df.schema[column] == pl.String:
                if params.trim_strings: expr = expr.str.strip_chars()
                if params.lowercase: expr = expr.str.to_lowercase()
            if column in params.fill_nulls: expr = expr.fill_null(params.fill_nulls[column])
            expressions.append(expr.alias(column))
        return df.with_columns(expressions)

    @staticmethod
    def _pit_join(left: pl.DataFrame, right: pl.DataFrame, params: PitJoinParams) -> pl.DataFrame:
        # asof joins require both sides to be sorted; exact boundary matches are included.
        left = left.sort([*params.on, params.left_time])
        right = right.sort([*params.on, params.right_time])
        return left.join_asof(right, left_on=params.left_time, right_on=params.right_time,
                              by=list(params.on) or None, strategy=params.direction,
                              tolerance=params.tolerance, suffix=params.suffix,
                              check_sortedness=False)

    @staticmethod
    def _window(df: pl.DataFrame, params: WindowParams) -> pl.DataFrame:
        df = _sort(df, params.order_by)
        expr = {"row_number": pl.int_range(1, pl.len() + 1), "rank": pl.col(params.order_by[0].column).rank("min"), "dense_rank": pl.col(params.order_by[0].column).rank("dense")}[params.function]
        if params.partition_by: expr = expr.over(list(params.partition_by))
        return df.with_columns(expr.alias(params.output_column))

    @staticmethod
    def _rolling(df: pl.DataFrame, params: RollingParams) -> pl.DataFrame:
        df = _sort(df, params.order_by)
        expr = getattr(pl.col(params.column), f"rolling_{params.aggregation}")(params.window_size, min_samples=params.min_periods)
        if params.partition_by: expr = expr.over(list(params.partition_by))
        return df.with_columns(expr.alias(params.output_column))

    @staticmethod
    def _statistics(df: pl.DataFrame, params: StatisticsParams) -> pl.DataFrame:
        expressions = []
        for column in params.columns:
            for aggregation in params.aggregations:
                fn = getattr(pl.col(column), aggregation)
                expressions.append(fn().alias(f"{column}_{aggregation}"))
        return df.group_by(list(params.group_by)).agg(expressions) if params.group_by else df.select(expressions)
