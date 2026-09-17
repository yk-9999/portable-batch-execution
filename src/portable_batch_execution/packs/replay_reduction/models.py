"""Typed closed parameters for generic replay reduction primitives."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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
    trailing_block_count: int = Field(ge=1)


class FutureWindowSpec(Frozen):
    fact_id: str
    measurement_field: str
    start_offset_blocks: int = Field(ge=1)
    end_offset_blocks: int = Field(ge=1)


class EventWindowExtractJobParams(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract-job.v1"]


class EventWindowExtractRequest(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract.v1"]
    request_id: str
    symbol: str
    symbol_column: str
    block_column: str
    causal_cutoff_block: int
    decision_block: int
    as_of_measurement_field: str
    as_of_offsets: tuple[int, ...] = (0,)
    trailing_windows: tuple[TrailingWindowSpec, ...] = ()
    future_windows: tuple[FutureWindowSpec, ...] = ()
    tie_break_columns: tuple[str, ...] = ("block",)


PARAM_MODELS = {
    "replay.structural_canonicalize": StructuralCanonicalizeParams,
    "replay.event_window_extract": EventWindowExtractJobParams,
}
