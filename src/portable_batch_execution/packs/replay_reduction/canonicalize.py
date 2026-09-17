"""Exact structural canonicalization via a partitioned exact-ID state.

Positive identity groups are formed from consecutive equal positive identities in
source order; identity values are never required to be numerically ordered and are
never compared through hashes. Exact identity membership is carried by closed
binary bucket artifacts (deterministic sorted unique uint64 sets), while the
mergeable summary JSON stays O(bucket_count) rather than O(identity_count).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .models import (
    BUCKET_COUNT_MAX,
    BUCKET_COUNT_MIN,
    SentinelPredicate,
    SentinelScalar,
    StructuralCanonicalizeParams,
)

STATE_SCHEMA_VERSION = "pbe.replay.structural-canonicalize-state.v4"
BUCKET_FORMAT = "pbe.replay.exact-id-bucket.v1"
BUCKET_FORMAT_VERSION = 1
BUCKET_MEDIA_TYPE = "application/vnd.pbe.exact-id-bucket.v1"
BUCKET_MAGIC = b"PBEBKT01"
_BUCKET_HEADER = struct.Struct("<8sHIIQ")
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
_SPLITMIX_MUL_A = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL_B = 0x94D049BB133111EB


class StructuralCanonicalizeError(Exception):
    """Input rows violate structural canonicalization invariants."""


@dataclass(frozen=True)
class BucketSet:
    """One deterministic sorted unique uint64 identity bucket."""

    bucket_index: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalizeState:
    """A complete mergeable canonicalization state (summary plus bucket sets)."""

    bucket_count: int
    bucket_sets: tuple[BucketSet, ...]
    positive_row_count: int
    positive_group_count: int
    witness_row_count: int
    first_boundary: dict[str, Any] | None
    last_boundary: dict[str, Any] | None


def splitmix64(value: int) -> int:
    """Bijective uint64 mixing permutation (SplitMix64 finalizer over a golden-ratio add).

    The permutation is a bijection on the full uint64 domain, so distinct
    identities stay distinct; bucket assignment may collide by bucket bits, but
    identity equality is never decided by the mixed value.
    """
    z = (value + _SPLITMIX_GOLDEN) & _UINT64_MASK
    z = ((z ^ (z >> 30)) * _SPLITMIX_MUL_A) & _UINT64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_MUL_B) & _UINT64_MASK
    return (z ^ (z >> 31)) & _UINT64_MASK


def bucket_index(value: int, bucket_count: int) -> int:
    """Assign a uint64 identity to one of bucket_count buckets via mixed low bits."""
    return splitmix64(value) & (bucket_count - 1)


def _validated(
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> StructuralCanonicalizeParams:
    if isinstance(params, StructuralCanonicalizeParams):
        return params
    return StructuralCanonicalizeParams.model_validate(params)


def _require_columns(schema: pl.Schema, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in schema]
    if missing:
        raise StructuralCanonicalizeError(
            f"missing required columns: {', '.join(missing)}"
        )


def _exact_match_expr(field: str, expected: SentinelScalar) -> pl.Expr:
    if expected is None:
        return pl.col(field).is_null()
    return (pl.col(field) == expected).fill_null(False)


def _sentinel_witness_expr(sentinel: SentinelPredicate) -> pl.Expr:
    expr = (pl.col("_identity_int") == sentinel.identity_equals).fill_null(False)
    for field, expected in sentinel.exact_match_fields.items():
        expr = expr & _exact_match_expr(field, expected)
    return expr


def _boundary_profile(row: dict[str, Any], core_fields: list[str]) -> dict[str, Any]:
    return {
        "identity": int(row["_identity_int"]),
        "core": {field: row[field] for field in core_fields},
    }


def _uint64_literal(value: int) -> pl.Expr:
    return pl.lit(value, dtype=pl.UInt64)


def _logical_right_shift(expr: pl.Expr, bits: int) -> pl.Expr:
    return expr // _uint64_literal(1 << bits)


def _bucket_expr(bucket_count: int) -> pl.Expr:
    """Vectorized form of bucket_index over the _identity_u64 column."""
    expr = pl.col("_identity_u64") + _uint64_literal(_SPLITMIX_GOLDEN)
    expr = (expr ^ _logical_right_shift(expr, 30)) * _uint64_literal(_SPLITMIX_MUL_A)
    expr = (expr ^ _logical_right_shift(expr, 27)) * _uint64_literal(_SPLITMIX_MUL_B)
    expr = expr ^ _logical_right_shift(expr, 31)
    return (expr & _uint64_literal(bucket_count - 1)).cast(pl.UInt32)


def _bucket_values(ordered: pl.LazyFrame, bucket_count: int) -> list[list[int]]:
    rows = (
        ordered.select(pl.col("_identity_int").cast(pl.UInt64).alias("_identity_u64"))
        .unique()
        .with_columns(_bucket_expr(bucket_count).alias("_bucket"))
        .group_by("_bucket")
        .agg(pl.col("_identity_u64").sort().alias("_ids"))
        .collect()
    )
    per_bucket: list[list[int]] = [[] for _ in range(bucket_count)]
    for item in rows.iter_rows(named=True):
        per_bucket[int(item["_bucket"])] = [int(value) for value in item["_ids"]]
    return per_bucket


def _empty_state(bucket_count: int, witness_row_count: int) -> CanonicalizeState:
    return CanonicalizeState(
        bucket_count=bucket_count,
        bucket_sets=tuple(BucketSet(index, ()) for index in range(bucket_count)),
        positive_row_count=0,
        positive_group_count=0,
        witness_row_count=witness_row_count,
        first_boundary=None,
        last_boundary=None,
    )


def _single_input_state(
    path: str | Path,
    model: StructuralCanonicalizeParams,
) -> CanonicalizeState:
    identity_col = model.identity_source_column
    normalized_col = model.identity_normalized_column
    core_fields = list(model.measurement_core_fields)
    sentinel = model.sentinel
    bucket_count = model.bucket_count

    lazy = pl.scan_parquet(str(path))
    required = (identity_col, normalized_col, *core_fields)
    if sentinel is not None:
        required = required + tuple(sentinel.exact_match_fields)
    _require_columns(lazy.collect_schema(), required)
    lazy = lazy.with_row_index("_row_index").with_columns(
        pl.col(identity_col).cast(pl.Int64, strict=False).alias("_identity_int"),
    )

    if sentinel is not None:
        witness_expr = _sentinel_witness_expr(sentinel)
        witness_count = int(lazy.filter(witness_expr).select(pl.len()).collect().item())
    else:
        witness_expr = None
        witness_count = 0

    nonpositive = pl.col("_identity_int") <= 0
    if witness_expr is None:
        invalid_expr = pl.col("_identity_int").is_null() | nonpositive
    else:
        invalid_expr = pl.col("_identity_int").is_null() | (
            nonpositive & witness_expr.not_()
        )
    if int(lazy.filter(invalid_expr).select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity is missing or not positive")

    positive = lazy.filter(pl.col("_identity_int") > 0)
    mismatch = positive.filter(
        pl.col(normalized_col).cast(pl.Utf8, strict=False)
        != pl.col("_identity_int").cast(pl.Utf8)
    )
    if int(mismatch.select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity normalized column mismatch")

    positive_row_count = int(positive.select(pl.len()).collect().item())
    if positive_row_count == 0:
        return _empty_state(bucket_count, witness_count)

    ordered = positive.sort("_row_index")
    grouped = ordered.with_columns(
        (pl.col("_identity_int") != pl.col("_identity_int").shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("_group")
    )

    group_count = int(grouped.select(pl.col("_group").n_unique()).collect().item())
    distinct_count = int(ordered.select(pl.col("_identity_int").n_unique()).collect().item())
    if group_count != distinct_count:
        raise StructuralCanonicalizeError(
            "positive identity recurs non-contiguously within one input"
        )

    nunique = grouped.group_by("_group").agg(
        [pl.col(field).n_unique().alias(field) for field in core_fields]
    )
    disagree = nunique.filter(
        pl.any_horizontal([pl.col(field) > 1 for field in core_fields])
    )
    if int(disagree.select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("measurement core fields disagree")

    first_row = ordered.head(1).collect().row(0, named=True)
    last_row = ordered.tail(1).collect().row(0, named=True)
    per_bucket = _bucket_values(ordered, bucket_count)

    return CanonicalizeState(
        bucket_count=bucket_count,
        bucket_sets=tuple(
            BucketSet(index, tuple(values)) for index, values in enumerate(per_bucket)
        ),
        positive_row_count=positive_row_count,
        positive_group_count=group_count,
        witness_row_count=witness_count,
        first_boundary=_boundary_profile(first_row, core_fields),
        last_boundary=_boundary_profile(last_row, core_fields),
    )


def _merge_sorted_unique(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    merged: list[int] = []
    shared: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if a == b:
            merged.append(a)
            shared.append(a)
            i += 1
            j += 1
        elif a < b:
            merged.append(a)
            i += 1
        else:
            merged.append(b)
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return tuple(merged), tuple(shared)


def _validate_state(state: CanonicalizeState) -> None:
    if (
        not isinstance(state.bucket_count, int)
        or isinstance(state.bucket_count, bool)
        or state.bucket_count < BUCKET_COUNT_MIN
        or state.bucket_count > BUCKET_COUNT_MAX
        or state.bucket_count & (state.bucket_count - 1)
    ):
        raise StructuralCanonicalizeError("invalid bucket_count")
    if len(state.bucket_sets) != state.bucket_count:
        raise StructuralCanonicalizeError("bucket set cardinality mismatch")
    total_identities = 0
    for index, bucket in enumerate(state.bucket_sets):
        if bucket.bucket_index != index:
            raise StructuralCanonicalizeError("bucket order mismatch")
        previous = 0
        for value in bucket.values:
            if not isinstance(value, int) or value <= previous:
                raise StructuralCanonicalizeError("bucket values must be sorted unique")
            if bucket_index(value, state.bucket_count) != index:
                raise StructuralCanonicalizeError("bucket value is not in its bucket")
            previous = value
        total_identities += len(bucket.values)
    if total_identities != state.positive_group_count:
        raise StructuralCanonicalizeError("bucket identity total does not match group count")
    if state.positive_row_count < state.positive_group_count or state.witness_row_count < 0:
        raise StructuralCanonicalizeError("invalid positive or witness counts")
    if state.positive_group_count == 0:
        if state.positive_row_count != 0:
            raise StructuralCanonicalizeError("empty state cannot carry positive rows")
        if state.first_boundary is not None or state.last_boundary is not None:
            raise StructuralCanonicalizeError("empty state cannot carry boundaries")
        return
    for name, boundary in (
        ("first", state.first_boundary),
        ("last", state.last_boundary),
    ):
        if not isinstance(boundary, dict) or "identity" not in boundary or "core" not in boundary:
            raise StructuralCanonicalizeError(f"{name} boundary is missing")
        identity = boundary["identity"]
        if not isinstance(identity, int) or isinstance(identity, bool) or identity <= 0:
            raise StructuralCanonicalizeError(f"{name} boundary identity is not positive")
        if identity not in state.bucket_sets[bucket_index(identity, state.bucket_count)].values:
            raise StructuralCanonicalizeError(f"{name} boundary identity is not in its bucket")


def _merge_states(
    left: CanonicalizeState,
    right: CanonicalizeState,
    *,
    subtract_boundary_row: bool,
) -> CanonicalizeState:
    _validate_state(left)
    _validate_state(right)
    if left.bucket_count != right.bucket_count:
        raise StructuralCanonicalizeError(
            "bucket_count mismatch between canonicalization states"
        )

    left_positive = left.positive_group_count > 0
    right_positive = right.positive_group_count > 0
    shared_boundary = (
        left_positive
        and right_positive
        and left.last_boundary["identity"] == right.first_boundary["identity"]
    )
    if shared_boundary and left.last_boundary["core"] != right.first_boundary["core"]:
        raise StructuralCanonicalizeError(
            "measurement core fields disagree at boundary identity"
        )
    allowed_identity = left.last_boundary["identity"] if shared_boundary else None

    merged_sets: list[BucketSet] = []
    for index in range(left.bucket_count):
        merged, shared = _merge_sorted_unique(
            left.bucket_sets[index].values, right.bucket_sets[index].values
        )
        for value in shared:
            if allowed_identity is None or value != allowed_identity:
                raise StructuralCanonicalizeError(
                    "non-contiguous recurrence of a positive identity"
                )
        merged_sets.append(BucketSet(index, merged))

    if not left_positive:
        first_boundary = right.first_boundary
        last_boundary = right.last_boundary
        group_count = right.positive_group_count
        row_count = right.positive_row_count
    elif not right_positive:
        first_boundary = left.first_boundary
        last_boundary = left.last_boundary
        group_count = left.positive_group_count
        row_count = left.positive_row_count
    else:
        first_boundary = left.first_boundary
        last_boundary = right.last_boundary
        if shared_boundary:
            group_count = left.positive_group_count + right.positive_group_count - 1
            row_count = left.positive_row_count + right.positive_row_count - (
                1 if subtract_boundary_row else 0
            )
        else:
            group_count = left.positive_group_count + right.positive_group_count
            row_count = left.positive_row_count + right.positive_row_count

    merged_state = CanonicalizeState(
        bucket_count=left.bucket_count,
        bucket_sets=tuple(merged_sets),
        positive_row_count=row_count,
        positive_group_count=group_count,
        witness_row_count=left.witness_row_count + right.witness_row_count,
        first_boundary=first_boundary,
        last_boundary=last_boundary,
    )
    _validate_state(merged_state)
    return merged_state


def execute_structural_canonicalize(
    paths: list[str | Path],
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> CanonicalizeState:
    """Build one exact partitioned state from ordered parquet inputs."""
    if not paths:
        raise StructuralCanonicalizeError("at least one parquet input is required")
    model = _validated(params)
    states = [_single_input_state(path, model) for path in paths]
    state = states[0]
    for part in states[1:]:
        state = _merge_states(state, part, subtract_boundary_row=False)
    return state


def merge_structural_canonicalize_states(
    left: CanonicalizeState,
    right: CanonicalizeState,
) -> CanonicalizeState:
    """Merge two complete states exactly across bounded waves."""
    return _merge_states(left, right, subtract_boundary_row=True)


def encode_bucket(bucket_count: int, bucket: BucketSet) -> bytes:
    """Encode one bucket as the closed deterministic binary sorted unique set."""
    header = _BUCKET_HEADER.pack(
        BUCKET_MAGIC,
        BUCKET_FORMAT_VERSION,
        bucket_count,
        bucket.bucket_index,
        len(bucket.values),
    )
    if not bucket.values:
        return header
    return header + struct.pack(f"<{len(bucket.values)}Q", *bucket.values)


def encode_state_buckets(state: CanonicalizeState) -> tuple[bytes, ...]:
    """Encode every bucket artifact in deterministic bucket order."""
    _validate_state(state)
    return tuple(
        encode_bucket(state.bucket_count, bucket) for bucket in state.bucket_sets
    )


def decode_bucket(payload: bytes, bucket_count: int, expected_index: int) -> tuple[int, ...]:
    """Decode and validate one deterministic bucket artifact."""
    if len(payload) < _BUCKET_HEADER.size:
        raise StructuralCanonicalizeError("bucket artifact is truncated")
    magic, version, encoded_count, encoded_index, value_count = _BUCKET_HEADER.unpack_from(
        payload, 0
    )
    if magic != BUCKET_MAGIC:
        raise StructuralCanonicalizeError("bucket artifact magic mismatch")
    if version != BUCKET_FORMAT_VERSION:
        raise StructuralCanonicalizeError("bucket artifact version mismatch")
    if encoded_count != bucket_count:
        raise StructuralCanonicalizeError("bucket artifact bucket_count mismatch")
    if encoded_index != expected_index:
        raise StructuralCanonicalizeError("bucket artifact order mismatch")
    if len(payload) != _BUCKET_HEADER.size + value_count * 8:
        raise StructuralCanonicalizeError("bucket artifact length mismatch")
    values = (
        struct.unpack_from(f"<{value_count}Q", payload, _BUCKET_HEADER.size)
        if value_count
        else ()
    )
    previous = 0
    for value in values:
        if value <= previous:
            raise StructuralCanonicalizeError("bucket artifact values are not sorted unique")
        previous = value
    return values


def state_summary(state: CanonicalizeState) -> dict[str, Any]:
    """Return the O(bucket_count) summary JSON of a state."""
    _validate_state(state)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "bucket_count": state.bucket_count,
        "bucket_format": BUCKET_FORMAT,
        "bucket_format_version": BUCKET_FORMAT_VERSION,
        "positive_row_count": state.positive_row_count,
        "positive_group_count": state.positive_group_count,
        "witness_row_count": state.witness_row_count,
        "first_boundary": state.first_boundary,
        "last_boundary": state.last_boundary,
        "bucket_counts": [len(bucket.values) for bucket in state.bucket_sets],
    }


def attach_bucket_refs(
    summary: dict[str, Any], bucket_refs: tuple[Any, ...]
) -> dict[str, Any]:
    """Return the summary carrying bucket artifact references in bucket order."""
    if len(bucket_refs) != int(summary["bucket_count"]):
        raise StructuralCanonicalizeError("bucket reference count mismatch")
    enriched = dict(summary)
    enriched["bucket_refs"] = [
        dict(ref) if isinstance(ref, dict) else ref.model_dump(mode="json")
        for ref in bucket_refs
    ]
    return enriched


def decode_state(
    summary: dict[str, Any], bucket_payloads: tuple[bytes, ...]
) -> CanonicalizeState:
    """Rebuild and validate a state from a summary and its bucket artifacts."""
    if not isinstance(summary, dict):
        raise StructuralCanonicalizeError("canonicalization summary must be an object")
    if summary.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("canonicalization state schema mismatch")
    if summary.get("bucket_format") != BUCKET_FORMAT:
        raise StructuralCanonicalizeError("canonicalization bucket format mismatch")
    if summary.get("bucket_format_version") != BUCKET_FORMAT_VERSION:
        raise StructuralCanonicalizeError("canonicalization bucket format version mismatch")
    bucket_count = summary.get("bucket_count")
    if (
        not isinstance(bucket_count, int)
        or isinstance(bucket_count, bool)
        or bucket_count < BUCKET_COUNT_MIN
        or bucket_count > BUCKET_COUNT_MAX
        or bucket_count & (bucket_count - 1)
    ):
        raise StructuralCanonicalizeError("canonicalization bucket_count is invalid")
    bucket_counts = summary.get("bucket_counts")
    if not isinstance(bucket_counts, list) or len(bucket_counts) != bucket_count:
        raise StructuralCanonicalizeError("canonicalization bucket_counts are invalid")
    if not all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in bucket_counts
    ):
        raise StructuralCanonicalizeError("canonicalization bucket_counts are invalid")
    if len(bucket_payloads) != bucket_count:
        raise StructuralCanonicalizeError("canonicalization bucket artifacts are incomplete")

    bucket_sets: list[BucketSet] = []
    for index, payload in enumerate(bucket_payloads):
        values = decode_bucket(payload, bucket_count, index)
        if len(values) != bucket_counts[index]:
            raise StructuralCanonicalizeError("canonicalization bucket count mismatch")
        bucket_sets.append(BucketSet(index, values))

    for name in ("positive_row_count", "positive_group_count", "witness_row_count"):
        value = summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StructuralCanonicalizeError(f"canonicalization {name} is invalid")

    state = CanonicalizeState(
        bucket_count=bucket_count,
        bucket_sets=tuple(bucket_sets),
        positive_row_count=int(summary["positive_row_count"]),
        positive_group_count=int(summary["positive_group_count"]),
        witness_row_count=int(summary["witness_row_count"]),
        first_boundary=summary.get("first_boundary"),
        last_boundary=summary.get("last_boundary"),
    )
    _validate_state(state)
    return state
