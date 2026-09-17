"""Structural canonicalization with bounded per-input aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .models import StructuralCanonicalizeParams

RESULT_SCHEMA_VERSION = "pbe.replay.structural-canonicalize-result.v2"


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


def _summarize_single_input(
    path: str | Path,
    source_index: int,
    model: StructuralCanonicalizeParams,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    identity_col = model.identity_source_column
    normalized_col = model.identity_normalized_column
    core_fields = list(model.measurement_core_fields)
    sentinel_value = model.sentinel.identity_equals if model.sentinel is not None else None

    lazy = pl.scan_parquet(str(path))
    _require_columns(lazy.collect_schema(), (identity_col, normalized_col, *core_fields))
    lazy = lazy.with_row_index("_row_index").with_columns(
        pl.lit(source_index).alias("_source_input_index"),
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
    invalid = lazy.filter(invalid_expr)
    if int(invalid.select(pl.len()).collect().item()) > 0:
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

    boundary = (
        economic.group_by("_source_input_index")
        .agg(
            pl.col("_identity_int").min().alias("identity_min"),
            pl.col("_identity_int").max().alias("identity_max"),
            pl.col("_identity_int").n_unique().alias("identity_count"),
        )
        .sort("_source_input_index")
        .collect()
    )
    boundary_evidence = [
        {
            "source_input_index": int(row["_source_input_index"]),
            "identity_min": int(row["identity_min"]),
            "identity_max": int(row["identity_max"]),
            "identity_count": int(row["identity_count"]),
        }
        for row in boundary.iter_rows(named=True)
    ]

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("_source_input_index").sort_by("_row_index").first().alias(
            "_chosen_source_input_index"
        ),
        pl.col("_row_index").min().alias("_chosen_row_index"),
    ]
    for field in core_fields:
        agg_exprs.append(pl.col(field).n_unique().alias(f"_core_nunique_{field}"))
        agg_exprs.append(
            pl.col(field).sort_by("_row_index").first().alias(f"_core_{field}")
        )

    grouped = economic.group_by("_identity_int").agg(agg_exprs).sort("_identity_int").collect()
    profiles: list[dict[str, Any]] = []
    for row in grouped.iter_rows(named=True):
        core = {field: row[f"_core_{field}"] for field in core_fields}
        for field in core_fields:
            if int(row[f"_core_nunique_{field}"]) != 1:
                raise StructuralCanonicalizeError("measurement core fields disagree")
        profiles.append(
            {
                "identity": int(row["_identity_int"]),
                "core": core,
                "source_input_indices": [int(row["_chosen_source_input_index"])],
                "row_count": int(row["row_count"]),
            }
        )
    return witness_count, profiles, boundary_evidence


def _merge_profiles(
    left: dict[int, dict[str, Any]], profile: dict[str, Any]
) -> None:
    identity = int(profile["identity"])
    if identity not in left:
        left[identity] = {
            "identity": identity,
            "core": dict(profile["core"]),
            "source_input_indices": list(profile["source_input_indices"]),
            "row_count": int(profile["row_count"]),
        }
        return
    existing = left[identity]
    if existing["core"] != profile["core"]:
        raise StructuralCanonicalizeError("measurement core fields disagree")
    existing["row_count"] = int(existing["row_count"]) + int(profile["row_count"])
    merged_indices = sorted(
        set(existing["source_input_indices"]) | set(profile["source_input_indices"])
    )
    existing["source_input_indices"] = merged_indices


def execute_structural_canonicalize(
    paths: list[str | Path],
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> dict[str, Any]:
    """Aggregate identities per bounded parquet input without whole-corpus materialization."""
    if not paths:
        raise StructuralCanonicalizeError("at least one parquet input is required")
    model = _validated(params)
    profiles_by_identity: dict[int, dict[str, Any]] = {}
    boundary_evidence: list[dict[str, Any]] = []
    witness_row_count = 0

    for source_index, path in enumerate(paths):
        witness_count, profiles, boundary = _summarize_single_input(path, source_index, model)
        witness_row_count += witness_count
        boundary_evidence.extend(boundary)
        for profile in profiles:
            _merge_profiles(profiles_by_identity, profile)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity_profiles": [
            profiles_by_identity[identity]
            for identity in sorted(profiles_by_identity)
        ],
        "witness_row_count": witness_row_count,
        "boundary_evidence": boundary_evidence,
    }


def merge_structural_canonicalize_states(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Merge two compact canonicalization states exactly across bounded waves."""
    if left.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("left canonicalization state schema mismatch")
    if right.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("right canonicalization state schema mismatch")

    profiles_by_identity: dict[int, dict[str, Any]] = {}
    for profile in left.get("identity_profiles", ()):
        profiles_by_identity[int(profile["identity"])] = {
            "identity": int(profile["identity"]),
            "core": dict(profile["core"]),
            "source_input_indices": list(profile["source_input_indices"]),
            "row_count": int(profile["row_count"]),
        }
    for profile in right.get("identity_profiles", ()):
        _merge_profiles(profiles_by_identity, profile)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity_profiles": [
            profiles_by_identity[identity]
            for identity in sorted(profiles_by_identity)
        ],
        "witness_row_count": int(left.get("witness_row_count", 0))
        + int(right.get("witness_row_count", 0)),
        "boundary_evidence": list(left.get("boundary_evidence", ()))
        + list(right.get("boundary_evidence", ())),
    }
