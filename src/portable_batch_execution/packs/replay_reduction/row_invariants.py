"""Generic per-row invariant validation for canonical trade replay ingestion."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any, Literal

from .canonicalize import StructuralCanonicalizeError
from .models import (
    ExactTextRowInvariant,
    NumericRowInvariant,
    PositiveFiniteRowInvariant,
    ProductWithToleranceRowInvariant,
    RowInvariant,
    TimestampMsEquivalenceRowInvariant,
)

ROW_INVARIANT_MAX = 64
_ISO_EXPLICIT_OFFSET = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def validate_row_invariant_bundle(invariants: tuple[RowInvariant, ...]) -> None:
    if len(invariants) > ROW_INVARIANT_MAX:
        raise ValueError("row invariant count exceeds bounded limit")


def invariant_columns(invariants: tuple[RowInvariant, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for invariant in invariants:
        if isinstance(invariant, ExactTextRowInvariant):
            columns.append(invariant.column)
        elif isinstance(
            invariant, (NumericRowInvariant, TimestampMsEquivalenceRowInvariant)
        ):
            columns.extend((invariant.left_column, invariant.right_column))
        elif isinstance(invariant, PositiveFiniteRowInvariant):
            columns.append(invariant.column)
        elif isinstance(invariant, ProductWithToleranceRowInvariant):
            columns.extend(invariant.factor_columns)
    return tuple(dict.fromkeys(columns))


def _timestamp_to_epoch_ms(value: Any, mode: Literal["integer_ms", "iso8601"]) -> int:
    if value is None:
        raise StructuralCanonicalizeError("row invariant timestamp missing")
    if mode == "integer_ms":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise StructuralCanonicalizeError("row invariant timestamp invalid") from exc
    if not isinstance(value, str):
        raise StructuralCanonicalizeError("row invariant timestamp invalid")
    if not _ISO_EXPLICIT_OFFSET.search(value):
        raise StructuralCanonicalizeError("row invariant timestamp invalid")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise StructuralCanonicalizeError("row invariant timestamp invalid") from exc
    if parsed.tzinfo is None:
        raise StructuralCanonicalizeError("row invariant timestamp invalid")
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def _numeric_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        left_num = float(left)
        right_num = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return False
    return left_num == right_num


def _positive_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def validate_row_invariants(row: dict[str, Any], invariants: tuple[RowInvariant, ...]) -> None:
    if not invariants:
        return
    validate_row_invariant_bundle(invariants)
    for invariant in invariants:
        if isinstance(invariant, ExactTextRowInvariant):
            actual = row.get(invariant.column)
            if actual is None or str(actual) != invariant.expected:
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, NumericRowInvariant):
            if not _numeric_equal(row.get(invariant.left_column), row.get(invariant.right_column)):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, TimestampMsEquivalenceRowInvariant):
            left_ms = _timestamp_to_epoch_ms(row.get(invariant.left_column), invariant.left_mode)
            right_ms = _timestamp_to_epoch_ms(
                row.get(invariant.right_column), invariant.right_mode
            )
            if left_ms != right_ms:
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, PositiveFiniteRowInvariant):
            if not _positive_finite(row.get(invariant.column)):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, ProductWithToleranceRowInvariant):
            product = 1.0
            for column in invariant.factor_columns:
                raw = row.get(column)
                if raw is None:
                    raise StructuralCanonicalizeError("row invariant violated")
                try:
                    factor = float(raw)
                except (TypeError, ValueError):
                    raise StructuralCanonicalizeError("row invariant violated")
                if not math.isfinite(factor):
                    raise StructuralCanonicalizeError("row invariant violated")
                product *= factor
            expected = float(invariant.expected)
            tolerance = invariant.absolute_tolerance + (
                invariant.relative_tolerance * abs(expected)
            )
            if abs(product - expected) > tolerance:
                raise StructuralCanonicalizeError("row invariant violated")
        else:
            raise StructuralCanonicalizeError("row invariant violated")
