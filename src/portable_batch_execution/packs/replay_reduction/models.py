"""Typed closed parameters for generic replay reduction primitives."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from portable_batch_execution.contracts.models import Frozen


class SentinelPredicate(Frozen):
    identity_equals: int


class StructuralCanonicalizeParams(Frozen):
    schema_version: Literal["pbe.replay.structural-canonicalize.v1"]
    identity_source_column: str
    identity_normalized_column: str
    measurement_core_fields: tuple[str, ...] = Field(min_length=1)
    sentinel: SentinelPredicate | None = None


class TrailingWindowSpec(Frozen):
    fact_id: str
    measurement_field: str
    trailing_width_ms: int = Field(ge=1)


class FutureWindowSpec(Frozen):
    fact_id: str
    measurement_field: str
    start_offset_ms: int = Field(ge=0)
    end_offset_ms: int = Field(ge=0)
    min_block_exclusive: int

    @model_validator(mode="after")
    def _end_not_before_start(self) -> FutureWindowSpec:
        if self.end_offset_ms < self.start_offset_ms:
            raise ValueError("end_offset_ms must be >= start_offset_ms")
        return self


class EventWindowExtractJobParams(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract-job.v1"]


class EventWindowExtractRequest(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract.v2"]
    request_id: str
    symbol: str
    symbol_column: str
    block_column: str
    timestamp_column: str
    decision_timestamp_ms: int
    causal_cutoff_block: int
    as_of_measurement_field: str
    as_of_offsets_ms: tuple[int, ...] = (0,)
    trailing_windows: tuple[TrailingWindowSpec, ...] = ()
    future_windows: tuple[FutureWindowSpec, ...] = ()
    tie_break_columns: tuple[str, ...] = ("timestamp_ms",)


class StructuralCanonicalizeMergeParams(Frozen):
    schema_version: Literal["pbe.replay.structural-canonicalize-merge.v1"]


PARAM_MODELS = {
    "replay.structural_canonicalize": StructuralCanonicalizeParams,
    "replay.event_window_extract": EventWindowExtractJobParams,
    "replay.structural_canonicalize_merge": StructuralCanonicalizeMergeParams,
}
