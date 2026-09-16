"""Closed parameter models for the tabular pack.

The models deliberately describe transformations rather than accepting expression
strings.  This keeps a tabular job portable and prevents a job specification from
becoming an arbitrary SQL/program execution surface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TabularParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SortKey(TabularParams):
    column: str = Field(min_length=1)
    descending: bool = False
    nulls_last: bool = False


class NormalizeParams(TabularParams):
    columns: tuple[str, ...] | None = None
    trim_strings: bool = True
    lowercase: bool = False
    fill_nulls: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CastParams(TabularParams):
    columns: dict[str, Literal["string", "integer", "float", "boolean", "date", "datetime"]]
    strict: bool = True


class SortParams(TabularParams):
    by: tuple[SortKey, ...] = Field(min_length=1)


class DedupParams(TabularParams):
    subset: tuple[str, ...] | None = None
    keep: Literal["first", "last", "none"] = "first"
    maintain_order: bool = True


class JoinParams(TabularParams):
    on: tuple[str, ...] = Field(min_length=1)
    how: Literal["inner", "left", "right", "full", "semi", "anti"] = "inner"
    suffix: str = "_right"


class PitJoinParams(TabularParams):
    on: tuple[str, ...] = ()
    left_time: str = Field(min_length=1)
    right_time: str = Field(min_length=1)
    direction: Literal["backward", "forward"] = "backward"
    tolerance: str | int | float | None = None
    suffix: str = "_right"


class WindowParams(TabularParams):
    partition_by: tuple[str, ...] = ()
    order_by: tuple[SortKey, ...] = Field(min_length=1)
    function: Literal["row_number", "rank", "dense_rank"] = "row_number"
    output_column: str = Field(default="window_value", min_length=1)


class RollingParams(TabularParams):
    column: str = Field(min_length=1)
    window_size: int = Field(gt=0)
    aggregation: Literal["mean", "sum", "min", "max", "std", "count"] = "mean"
    output_column: str = Field(default="rolling", min_length=1)
    partition_by: tuple[str, ...] = ()
    order_by: tuple[SortKey, ...] = ()
    min_periods: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def periods_fit_window(self):
        if self.min_periods > self.window_size:
            raise ValueError("min_periods cannot exceed window_size")
        return self


class StatisticsParams(TabularParams):
    columns: tuple[str, ...] = Field(min_length=1)
    aggregations: tuple[Literal["count", "null_count", "mean", "sum", "min", "max", "std", "median"], ...] = ("count", "mean")
    group_by: tuple[str, ...] = ()


class FormatMigrationParams(TabularParams):
    output_format: Literal["csv", "json", "jsonl", "parquet"]


class TextEventFeaturesParams(TabularParams):
    """Bounded, deterministic parameters for character feature expansion."""

    unicode_normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    lowercase: bool = False
    whitespace_mode: Literal["preserve", "collapse", "remove"] = "collapse"
    ngram_sizes: tuple[Literal[1, 2, 3, 4, 5], ...] = Field(min_length=1)
    max_input_rows: int = Field(default=100_000, ge=1, le=1_000_000)
    max_input_bytes: int = Field(default=64 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_unique_keys: int = Field(default=100_000, ge=1, le=1_000_000)
    max_output_rows: int = Field(default=1_000_000, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def unique_ngram_sizes(self):
        if len(set(self.ngram_sizes)) != len(self.ngram_sizes):
            raise ValueError("ngram_sizes must be unique")
        return self


class TrailingSparseWindowAggregateParams(TabularParams):
    """Bounded parameters for exact trailing contribution aggregation."""

    max_window_seconds: int = Field(default=604_800, ge=1, le=31_536_000)
    max_input_rows: int = Field(default=500_000, ge=1, le=2_000_000)
    max_input_bytes: int = Field(default=128 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_unique_keys: int = Field(default=250_000, ge=1, le=2_000_000)
    max_output_rows: int = Field(default=1_000_000, ge=1, le=10_000_000)


OperationParams = (
    NormalizeParams | CastParams | SortParams | DedupParams | JoinParams |
    PitJoinParams | WindowParams | RollingParams | StatisticsParams |
    FormatMigrationParams | TextEventFeaturesParams |
    TrailingSparseWindowAggregateParams
)

PARAM_MODELS = {
    "tabular.normalize": NormalizeParams,
    "tabular.cast": CastParams,
    "tabular.sort": SortParams,
    "tabular.dedup": DedupParams,
    "tabular.join": JoinParams,
    "tabular.pit_join": PitJoinParams,
    "tabular.window": WindowParams,
    "tabular.rolling": RollingParams,
    "tabular.statistics": StatisticsParams,
    "tabular.format_migration": FormatMigrationParams,
    "tabular.text_event_features.v1": TextEventFeaturesParams,
    "tabular.trailing_sparse_window_aggregate.v1": TrailingSparseWindowAggregateParams,
}
