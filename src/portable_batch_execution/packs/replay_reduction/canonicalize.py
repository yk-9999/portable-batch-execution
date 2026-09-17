"""Structural canonicalization over ordered parquet shard inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .models import StructuralCanonicalizeParams

RESULT_SCHEMA_VERSION = "pbe.replay.structural-canonicalize-result.v1"


class StructuralCanonicalizeError(Exception):
    """Input rows violate structural canonicalization invariants."""


def _validated(params: dict[str, Any] | StructuralCanonicalizeParams) -> StructuralCanonicalizeParams:
    if isinstance(params, StructuralCanonicalizeParams):
        return params
    return StructuralCanonicalizeParams.model_validate(params)


def _source_indices_contiguous(indices: list[int]) -> bool:
    if not indices:
        return True
    ordered = sorted(set(indices))
    return ordered == list(range(ordered[0], ordered[0] + len(ordered)))


def execute_structural_canonicalize(
    paths: list[str | Path],
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> dict[str, Any]:
    """Collapse duplicate identities across ordered parquet inputs into canonical records."""
    if not paths:
        raise StructuralCanonicalizeError("at least one parquet input is required")
    model = _validated(params)
    identity_col = model.identity_source_column
    normalized_col = model.identity_normalized_column
    core_fields = list(model.measurement_core_fields)
    sentinel_value = model.sentinel.identity_equals if model.sentinel is not None else None

    frames: list[pl.DataFrame] = []
    for source_index, path in enumerate(paths):
        frame = pl.read_parquet(str(path)).with_columns(
            pl.lit(source_index).alias("_source_input_index"),
            pl.int_range(0, pl.len()).alias("_row_index"),
        )
        missing = [column for column in (identity_col, normalized_col, *core_fields) if column not in frame.columns]
        if missing:
            raise StructuralCanonicalizeError(f"missing required columns: {', '.join(missing)}")
        frames.append(frame)

    combined = pl.concat(frames, how="diagonal_relaxed")
    rows = combined.sort(["_source_input_index", "_row_index"]).to_dicts()

    witness_facts: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    boundary_by_source: dict[int, set[int]] = {}

    for row in rows:
        source_index = int(row["_source_input_index"])
        raw_identity = row[identity_col]
        if raw_identity is None:
            if sentinel_value is None:
                raise StructuralCanonicalizeError("identity is missing")
            raise StructuralCanonicalizeError("identity is missing")
        try:
            identity = int(raw_identity)
        except (TypeError, ValueError):
            raise StructuralCanonicalizeError("identity is not an integer") from None

        if sentinel_value is not None and identity == sentinel_value:
            witness_facts.append(_strip_internal_columns(row))
            continue

        if identity <= 0:
            raise StructuralCanonicalizeError("identity is not positive")

        normalized = row[normalized_col]
        if normalized is None or str(normalized) != str(identity):
            raise StructuralCanonicalizeError("identity normalized column mismatch")

        positive_rows.append(row)
        boundary_by_source.setdefault(source_index, set()).add(identity)

    boundary_evidence = [
        {
            "source_input_index": source_index,
            "identities": sorted(boundary_by_source[source_index]),
        }
        for source_index in sorted(boundary_by_source)
    ]

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in positive_rows:
        identity = int(row[identity_col])
        grouped.setdefault(identity, []).append(row)

    canonical_records: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        group_rows = grouped[identity]
        source_indices = sorted({int(item["_source_input_index"]) for item in group_rows})
        if not _source_indices_contiguous(source_indices):
            raise StructuralCanonicalizeError("identity spans non-contiguous source inputs")

        for field in core_fields:
            values = {item[field] for item in group_rows}
            if len(values) != 1:
                raise StructuralCanonicalizeError("measurement core fields disagree")

        chosen = min(group_rows, key=lambda item: (int(item["_source_input_index"]), int(item["_row_index"])))
        canonical_records.append(_strip_internal_columns(chosen))

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "canonical_records": canonical_records,
        "witness_facts": witness_facts,
        "boundary_evidence": boundary_evidence,
    }


def _strip_internal_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }
