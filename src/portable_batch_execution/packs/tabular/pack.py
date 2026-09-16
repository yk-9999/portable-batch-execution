"""Polars implementation of closed, portable tabular transformations."""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timedelta
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
    TextEventFeaturesParams,
    TrailingSparseWindowAggregateParams,
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


def _bounded_json_bytes(value: object, maximum: int) -> int:
    try:
        size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("input must be JSON-compatible") from exc
    if size > maximum:
        raise ValueError("input byte cap exceeded")
    return size


def _records(value: Any, required: frozenset[str], name: str, maximum: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} row cap exceeded")
    if not all(isinstance(row, dict) and frozenset(row) == required for row in value):
        raise ValueError(f"{name} rows must have exactly the required fields")
    return value


def _time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO 8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 string") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


def _opaque_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty opaque string")
    return value


def _halo_seconds(shard) -> int | None:
    if getattr(shard, "primary_range", None) is None:
        return None
    primary_range = shard.primary_range
    if primary_range.kind != "time":
        raise ValueError("trailing window shards require time primary ranges")
    halo = shard.correctness.halo_before
    if halo is None or halo.unit == "records":
        raise ValueError("trailing window shard is missing a time halo")
    return halo.value * {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[halo.unit]


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
        if operation == "tabular.text_event_features.v1":
            return self._text_event_features(data, model)
        if operation == "tabular.trailing_sparse_window_aggregate.v1":
            return self._trailing_sparse_window_aggregate(data, model)
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
        if operation == "tabular.trailing_sparse_window_aggregate.v1":
            return self._trailing_sparse_window_aggregate(
                context["data"],
                _validated(TrailingSparseWindowAggregateParams, params),
                shard=shard,
            )
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

    @staticmethod
    def _text_event_features(data: Any, params: TextEventFeaturesParams) -> pl.DataFrame:
        required = frozenset({"row_id", "partition_key", "segment_key", "event_time", "entity_id", "text"})
        rows = _records(data, required, "text event", params.max_input_rows)
        _bounded_json_bytes(rows, params.max_input_bytes)
        output: list[dict[str, Any]] = []
        unique_keys: set[tuple[str, str]] = set()
        for row in rows:
            row_id = _opaque_string(row["row_id"], "row_id")
            partition_key = _opaque_string(row["partition_key"], "partition_key")
            segment_key = _opaque_string(row["segment_key"], "segment_key")
            entity_id = _opaque_string(row["entity_id"], "entity_id")
            event_time = _time(row["event_time"], "event_time").isoformat()
            if not isinstance(row["text"], str):
                raise TypeError("text must be a string")
            normalized = unicodedata.normalize(params.unicode_normalization, row["text"])
            normalized_non_whitespace_length = sum(
                not character.isspace() for character in normalized
            )
            text = normalized
            if params.lowercase:
                text = text.lower()
            if params.whitespace_mode == "collapse":
                text = " ".join(text.split())
            elif params.whitespace_mode == "remove":
                text = "".join(character for character in text if not character.isspace())
            base = {"row_id": row_id, "partition_key": partition_key, "segment_key": segment_key,
                    "event_time": event_time, "entity_id": entity_id}
            contributions = [
                ("message_presence", "present", 1),
                ("normalized_non_whitespace_length", "value", normalized_non_whitespace_length),
            ]
            for ngram_size in sorted(params.ngram_sizes):
                occurrences: dict[str, int] = {}
                for index in range(max(0, len(text) - ngram_size + 1)):
                    gram = text[index:index + ngram_size]
                    occurrences[gram] = occurrences.get(gram, 0) + 1
                contributions.extend(("character_ngram", gram, count) for gram, count in occurrences.items())
            for feature_kind, feature_key, weight in contributions:
                unique_keys.add((feature_kind, feature_key))
                if len(unique_keys) > params.max_unique_keys:
                    raise ValueError("unique key cap exceeded")
                output.append({**base, "feature_kind": feature_kind, "feature_key": feature_key, "weight": weight})
                if len(output) > params.max_output_rows:
                    raise ValueError("output row cap exceeded")
        return pl.DataFrame(sorted(output, key=lambda item: (
            item["partition_key"], item["segment_key"], item["event_time"], item["row_id"],
            item["entity_id"], item["feature_kind"], item["feature_key"])), schema={
                "row_id": pl.String, "partition_key": pl.String, "segment_key": pl.String,
                "event_time": pl.String, "entity_id": pl.String, "feature_kind": pl.String,
                "feature_key": pl.String, "weight": pl.Int64,
            })

    @staticmethod
    def _trailing_sparse_window_aggregate(
        data: Any, params: TrailingSparseWindowAggregateParams, *, shard=None
    ) -> pl.DataFrame:
        if not isinstance(data, dict) or frozenset(data) != frozenset({"features", "requests"}):
            raise ValueError("trailing window input must contain only features and requests")
        feature_fields = frozenset({"row_id", "partition_key", "segment_key", "event_time", "entity_id", "feature_kind", "feature_key", "weight"})
        request_fields = frozenset({"request_id", "partition_key", "segment_key", "endpoint_time", "window_seconds", "window_valid"})
        features = _records(data["features"], feature_fields, "feature", params.max_input_rows)
        requests = _records(data["requests"], request_fields, "request", params.max_input_rows)
        _bounded_json_bytes(data, params.max_input_bytes)
        parsed_requests: list[tuple[dict[str, Any], datetime, int]] = []
        maximum_window = 0
        for request in requests:
            _opaque_string(request["request_id"], "request_id")
            _opaque_string(request["partition_key"], "partition_key")
            _opaque_string(request["segment_key"], "segment_key")
            endpoint = _time(request["endpoint_time"], "endpoint_time")
            seconds = request["window_seconds"]
            if isinstance(seconds, bool) or not isinstance(seconds, int) or not 0 < seconds <= params.max_window_seconds:
                raise ValueError("window_seconds is outside the permitted range")
            if request["window_valid"] is not True:
                raise ValueError("window request is not valid")
            maximum_window = max(maximum_window, seconds)
            parsed_requests.append((request, endpoint, seconds))
        available_halo = _halo_seconds(shard) if shard is not None else None
        if available_halo is not None and available_halo < maximum_window:
            raise ValueError("trailing window shard halo is insufficient")
        parsed_features: list[tuple[dict[str, Any], datetime]] = []
        unique_keys: set[tuple[str, str]] = set()
        for feature in features:
            for name in ("row_id", "partition_key", "segment_key", "entity_id", "feature_kind", "feature_key"):
                _opaque_string(feature[name], name)
            event_time = _time(feature["event_time"], "event_time")
            weight = feature["weight"]
            if isinstance(weight, bool) or not isinstance(weight, int):
                raise TypeError("weight must be an integer")
            unique_keys.add((feature["feature_kind"], feature["feature_key"]))
            if len(unique_keys) > params.max_unique_keys:
                raise ValueError("unique key cap exceeded")
            parsed_features.append((feature, event_time))
        output: list[dict[str, Any]] = []
        for request, endpoint, seconds in parsed_requests:
            start = endpoint - timedelta(seconds=seconds)
            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for feature, event_time in parsed_features:
                if (feature["partition_key"] == request["partition_key"] and feature["segment_key"] == request["segment_key"]
                        and start <= event_time < endpoint):
                    groups.setdefault((feature["feature_kind"], feature["feature_key"]), []).append(feature)
            for (feature_kind, feature_key), matches in groups.items():
                output.append({"request_id": request["request_id"], "partition_key": request["partition_key"],
                               "segment_key": request["segment_key"], "endpoint_time": endpoint.isoformat(),
                               "window_seconds": seconds, "feature_kind": feature_kind, "feature_key": feature_key,
                               "sum_weight": sum(item["weight"] for item in matches),
                               "distinct_row_count": len({item["row_id"] for item in matches}),
                               "distinct_entity_count": len({item["entity_id"] for item in matches})})
                if len(output) > params.max_output_rows:
                    raise ValueError("output row cap exceeded")
        return pl.DataFrame(sorted(output, key=lambda item: (
            item["partition_key"], item["segment_key"], item["endpoint_time"], item["request_id"],
            item["window_seconds"], item["feature_kind"], item["feature_key"])), schema={
                "request_id": pl.String, "partition_key": pl.String, "segment_key": pl.String,
                "endpoint_time": pl.String, "window_seconds": pl.Int64, "feature_kind": pl.String,
                "feature_key": pl.String, "sum_weight": pl.Int64, "distinct_row_count": pl.Int64,
                "distinct_entity_count": pl.Int64,
            })
