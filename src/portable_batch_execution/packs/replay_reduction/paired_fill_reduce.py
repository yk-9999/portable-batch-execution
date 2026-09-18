"""Generic bounded paired-fill reduction with deterministic source order."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

import polars as pl

from .canonicalize import (
    _IDENTITY_SCAN_BATCH,
    StructuralCanonicalizeError,
    _require_columns,
)
from .event_window import _row_is_sentinel, _validate_positive_row
from .models import (
    PairedFillReduceRequest,
)
from .nullable_json_projection import apply_nullable_json_projections_to_lazy
from .row_invariants import invariant_columns, validate_row_invariants

RESULT_SCHEMA_VERSION = "pbe.replay.paired-fill-reduce-result.v1"
CARRY_SCHEMA_VERSION = "pbe.replay.paired-fill-reduce-carry.v1"
_SOURCE_INPUT_INDEX = "_source_input_index"
_SOURCE_ROW_OFFSET = "_source_row_offset"
_MAX_INPUT_FILES = 64


def _validated(
    request: dict[str, Any] | PairedFillReduceRequest,
) -> PairedFillReduceRequest:
    if isinstance(request, PairedFillReduceRequest):
        return request
    return PairedFillReduceRequest.model_validate(request)


def _mechanical_quantities(
    start_position: float, signed_execution: float
) -> tuple[float, float, float]:
    pre = float(start_position)
    qty = float(signed_execution)
    if pre == 0.0 or qty == 0.0:
        closing = 0.0
    elif (pre > 0.0 and qty < 0.0) or (pre < 0.0 and qty > 0.0):
        closing = min(abs(pre), abs(qty))
    else:
        closing = 0.0
    opening = abs(qty) - closing
    post = pre + qty
    return opening, closing, post


def _role_name(
    row: dict[str, Any],
    *,
    pair_role_column: str,
    aggressor_value: Any,
    passive_value: Any,
) -> Literal["aggressor", "passive"] | None:
    actual = row.get(pair_role_column)
    if actual == aggressor_value:
        return "aggressor"
    if actual == passive_value:
        return "passive"
    return None


def _participant_record(
    row: dict[str, Any],
    *,
    role: Literal["aggressor", "passive"],
    start_position_column: str,
    signed_execution_column: str,
) -> dict[str, Any]:
    start_position = float(row[start_position_column])
    signed_execution = float(row[signed_execution_column])
    opening, closing, post = _mechanical_quantities(start_position, signed_execution)
    return {
        "role": role,
        "start_position": start_position,
        "signed_execution": signed_execution,
        "opening_quantity": opening,
        "closing_quantity": closing,
        "post_position": post,
        "source_input_index": int(row[_SOURCE_INPUT_INDEX]),
        "source_row_offset": int(row[_SOURCE_ROW_OFFSET]),
    }


def _measurement_core(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: row[field_name] for field_name in fields}


def _cores_match(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    if not rows:
        return True
    reference = _measurement_core(rows[0], fields)
    for row in rows[1:]:
        for field_name in fields:
            if row.get(field_name) != reference[field_name]:
                return False
    return True


def _ledger_row(
    rows: list[dict[str, Any]],
    *,
    classification: Literal["complete_pair", "singleton"],
    identity: int,
    pair_mapping,
) -> dict[str, Any]:
    if not _cores_match(rows, pair_mapping.measurement_core_fields):
        raise StructuralCanonicalizeError("measurement core fields disagree")
    participants: list[dict[str, Any]] = []
    for row in rows:
        role = _role_name(
            row,
            pair_role_column=pair_mapping.pair_role_column,
            aggressor_value=pair_mapping.aggressor_role_value,
            passive_value=pair_mapping.passive_role_value,
        )
        if role is None:
            raise StructuralCanonicalizeError("pair role invalid")
        participants.append(
            _participant_record(
                row,
                role=role,
                start_position_column=pair_mapping.start_position_column,
                signed_execution_column=pair_mapping.signed_execution_column,
            )
        )
    if classification == "complete_pair":
        roles = {item["role"] for item in participants}
        if roles != {"aggressor", "passive"}:
            raise StructuralCanonicalizeError("pair role invalid")
    return {
        "ledger_identity": identity,
        "classification": classification,
        "measurement_core": _measurement_core(
            rows[0], pair_mapping.measurement_core_fields
        ),
        "source_input_index": int(rows[0][_SOURCE_INPUT_INDEX]),
        "source_row_offset": int(rows[0][_SOURCE_ROW_OFFSET]),
        "participants": participants,
    }


@dataclass
class _Reducer:
    model: PairedFillReduceRequest
    closed_identities: set[int] = field(default_factory=set)
    pending: list[dict[str, Any]] = field(default_factory=list)
    pending_identity: int | None = None
    boundary_carry: dict[str, Any] | None = None
    boundary_carry_identity: int | None = None
    ledger_rows: list[dict[str, Any]] = field(default_factory=list)
    administrative_row_count: int = 0

    def __post_init__(self) -> None:
        carry = self.model.partition.incoming_carry
        if carry is not None and carry.pending_row is not None:
            if carry.pending_identity is None:
                raise StructuralCanonicalizeError("carry identity missing")
            self.boundary_carry = dict(carry.pending_row)
            self.boundary_carry_identity = int(carry.pending_identity)

    def _check_output_bounds(self) -> None:
        if len(self.ledger_rows) > self.model.max_output_rows:
            raise ValueError("paired fill output row count exceeds limit")

    def _append_ledger(self, row: dict[str, Any]) -> None:
        self.ledger_rows.append(row)
        self._check_output_bounds()

    def _clear_pending(self) -> None:
        self.pending = []
        self.pending_identity = None

    def _finalize_singleton(self, identity: int) -> None:
        if len(self.pending) != 1:
            raise StructuralCanonicalizeError("singleton group size invalid")
        self._append_ledger(
            _ledger_row(
                self.pending,
                classification="singleton",
                identity=identity,
                pair_mapping=self.model.pair_mapping,
            )
        )
        self.closed_identities.add(identity)
        self._clear_pending()

    def _finalize_pair(self, identity: int) -> None:
        if len(self.pending) != 2:
            raise StructuralCanonicalizeError("pair group size invalid")
        self._append_ledger(
            _ledger_row(
                self.pending,
                classification="complete_pair",
                identity=identity,
                pair_mapping=self.model.pair_mapping,
            )
        )
        self.closed_identities.add(identity)
        self._clear_pending()

    def _ingest_economic_row(self, row: dict[str, Any], identity: int) -> None:
        if identity in self.closed_identities:
            raise StructuralCanonicalizeError("identity is not contiguous")

        if self.boundary_carry is not None:
            carry = self.boundary_carry
            carry_identity = self.boundary_carry_identity
            self.boundary_carry = None
            self.boundary_carry_identity = None
            if carry_identity is None:
                raise StructuralCanonicalizeError("carry identity missing")
            if identity == carry_identity:
                self.pending = [carry, row]
                self.pending_identity = identity
                self._finalize_pair(identity)
                return
            self.pending = [carry]
            self.pending_identity = carry_identity
            self._finalize_singleton(carry_identity)

        if not self.pending:
            self.pending = [row]
            self.pending_identity = identity
            return

        if identity != self.pending_identity:
            self._finalize_singleton(int(self.pending_identity))
            self.pending = [row]
            self.pending_identity = identity
            return

        self.pending.append(row)
        if len(self.pending) > 2:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )
        if len(self.pending) == 2:
            self._finalize_pair(identity)

    def ingest(self, row: dict[str, Any]) -> None:
        admin = self.model.administrative_row_handling
        if admin is not None and _row_is_sentinel(row, admin.predicate):
            self.administrative_row_count += 1
            return

        identity_col = self.model.identity_mapping.identity_source_column
        normalized_col = self.model.identity_mapping.identity_normalized_column
        identity = _validate_positive_row(
            row,
            identity_col=identity_col,
            normalized_col=normalized_col,
            core_fields=self.model.pair_mapping.measurement_core_fields,
        )
        validate_row_invariants(row, self.model.row_invariants)
        self._ingest_economic_row(row, identity)

    def end_input_boundary(self) -> None:
        if len(self.pending) == 2:
            if self.pending_identity is None:
                raise StructuralCanonicalizeError("pending identity missing")
            self._finalize_pair(int(self.pending_identity))
        elif len(self.pending) == 1:
            if self.boundary_carry is not None:
                raise StructuralCanonicalizeError("carry overflow")
            if self.pending_identity is None:
                raise StructuralCanonicalizeError("pending identity missing")
            self.boundary_carry = dict(self.pending[0])
            self.boundary_carry_identity = int(self.pending_identity)
            self._clear_pending()
        elif len(self.pending) > 2:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )

    def finish(self) -> dict[str, Any] | None:
        if self.boundary_carry is not None:
            if not self.model.partition.terminal:
                return {
                    "schema_version": CARRY_SCHEMA_VERSION,
                    "pending_row": self.boundary_carry,
                    "pending_identity": self.boundary_carry_identity,
                }
            if self.boundary_carry_identity is None:
                raise StructuralCanonicalizeError("carry identity missing")
            self.pending = [self.boundary_carry]
            self.pending_identity = int(self.boundary_carry_identity)
            self.boundary_carry = None
            self.boundary_carry_identity = None

        if len(self.pending) == 1:
            if self.pending_identity is None:
                raise StructuralCanonicalizeError("pending identity missing")
            self._finalize_singleton(int(self.pending_identity))
        elif len(self.pending) == 2:
            if self.pending_identity is None:
                raise StructuralCanonicalizeError("pending identity missing")
            self._finalize_pair(int(self.pending_identity))
        elif len(self.pending) > 2:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )

        if self.boundary_carry is not None:
            raise StructuralCanonicalizeError("carry overflow")
        return None


def _required_columns(model: PairedFillReduceRequest) -> tuple[str, ...]:
    pair = model.pair_mapping
    identity = model.identity_mapping
    admin = model.administrative_row_handling
    admin_fields = ()
    if admin is not None:
        admin_fields = tuple(admin.predicate.exact_match_fields)
    return tuple(
        dict.fromkeys(
            [
                identity.identity_source_column,
                identity.identity_normalized_column,
                pair.pair_role_column,
                pair.start_position_column,
                pair.signed_execution_column,
                *pair.measurement_core_fields,
                *invariant_columns(model.row_invariants),
                *admin_fields,
                *(
                    projection.source_column
                    for projection in model.nullable_json_scalar_projections
                ),
            ]
        )
    )


def _stream_rows(
    paths: list[str | Any],
    model: PairedFillReduceRequest,
) -> tuple[_Reducer, dict[str, Any] | None]:
    reducer = _Reducer(model)
    required = _required_columns(model)
    identity_col = model.identity_mapping.identity_source_column

    for input_index, path in enumerate(paths):
        lazy = pl.scan_parquet(str(path)).with_row_index(_SOURCE_ROW_OFFSET)
        lazy = apply_nullable_json_projections_to_lazy(
            lazy, model.nullable_json_scalar_projections
        )
        _require_columns(lazy.collect_schema(), required)
        row_count = int(lazy.select(pl.len()).collect().item())
        offset = 0
        while offset < row_count:
            batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
            batch = (
                lazy.slice(offset, batch_size)
                .with_columns(
                    pl.lit(input_index).cast(pl.Int64).alias(_SOURCE_INPUT_INDEX),
                    pl.col(_SOURCE_ROW_OFFSET).cast(pl.Int64),
                    pl.col(identity_col)
                    .cast(pl.Int64, strict=False)
                    .alias("_identity_int"),
                )
                .collect()
            )
            select_columns = list(
                dict.fromkeys(
                    [
                        *required,
                        _SOURCE_INPUT_INDEX,
                        _SOURCE_ROW_OFFSET,
                        "_identity_int",
                    ]
                )
            )
            for row in batch.select(select_columns).iter_rows(named=True):
                reducer.ingest(dict(row))
            offset += batch_size
        if input_index < len(paths) - 1:
            reducer.end_input_boundary()
    if not model.partition.terminal:
        reducer.end_input_boundary()

    outgoing_carry = reducer.finish()
    return reducer, outgoing_carry


def _content_identity(
    ledger_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    exceptions: list[dict[str, Any]],
) -> str:
    payload = {
        "ledger_rows": ledger_rows,
        "summary": summary,
        "exceptions": exceptions,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def execute_paired_fill_reduce(
    paths: list[str | Any],
    request: dict[str, Any] | PairedFillReduceRequest,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one parquet input is required")
    if len(paths) > _MAX_INPUT_FILES:
        raise ValueError("replay parquet input count exceeds limit")
    model = _validated(request)
    reducer, outgoing_carry = _stream_rows(paths, model)
    exceptions: list[dict[str, Any]] = []
    summary = {
        "request_id": model.request_id,
        "ledger_row_count": len(reducer.ledger_rows),
        "administrative_row_count": reducer.administrative_row_count,
        "exception_row_count": len(exceptions),
    }
    content_identity = _content_identity(reducer.ledger_rows, summary, exceptions)
    summary["content_identity"] = content_identity

    ledger_buffer = io.BytesIO()
    if reducer.ledger_rows:
        pl.DataFrame(reducer.ledger_rows).write_parquet(ledger_buffer)
    ledger_bytes = ledger_buffer.getvalue()
    ledger_parquet_identity = (
        f"sha256:{sha256(ledger_bytes).hexdigest()}" if ledger_bytes else None
    )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "summary": summary,
        "exceptions": exceptions,
        "ledger_rows": reducer.ledger_rows,
        "ledger_parquet_bytes": ledger_bytes,
        "ledger_parquet_identity": ledger_parquet_identity,
        "outgoing_carry": outgoing_carry
        or {
            "schema_version": CARRY_SCHEMA_VERSION,
            "pending_row": None,
            "pending_identity": None,
        },
    }
