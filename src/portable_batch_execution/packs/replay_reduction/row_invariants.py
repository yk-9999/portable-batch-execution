"""Generic per-row invariant validation for canonical trade replay ingestion."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any, Literal

from .canonicalize import StructuralCanonicalizeError
from .models import (
    BooleanRowInvariant,
    ExactTextRowInvariant,
    FiniteNumericRowInvariant,
    NonEmptyTextRowInvariant,
    NumericRowInvariant,
    PositiveFiniteRowInvariant,
    ProductColumnWithToleranceRowInvariant,
    ProductWithToleranceRowInvariant,
    RowInvariant,
    TextColumnEquivalenceRowInvariant,
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
            invariant,
            (
                NumericRowInvariant,
                TimestampMsEquivalenceRowInvariant,
                TextColumnEquivalenceRowInvariant,
            ),
        ):
            columns.extend((invariant.left_column, invariant.right_column))
        elif isinstance(
            invariant,
            (
                PositiveFiniteRowInvariant,
                NonEmptyTextRowInvariant,
                BooleanRowInvariant,
                FiniteNumericRowInvariant,
            ),
        ):
            columns.append(invariant.column)
        elif isinstance(invariant, ProductWithToleranceRowInvariant):
            columns.extend(invariant.factor_columns)
        elif isinstance(invariant, ProductColumnWithToleranceRowInvariant):
            columns.extend((*invariant.factor_columns, invariant.expected_column))
    return tuple(dict.fromkeys(columns))


def _timestamp_to_epoch_ms(value: Any, mode: Literal["integer_ms", "iso8601"]) -> int:
    if value is None:
        raise StructuralCanonicalizeError("row invariant timestamp missing")
    if mode == "integer_ms":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise StructuralCanonicalizeError(
                "row invariant timestamp invalid"
            ) from exc
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


def _finite_numeric(value: Any) -> float:
    if value is None:
        raise StructuralCanonicalizeError("row invariant violated")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise StructuralCanonicalizeError("row invariant violated")
    if not math.isfinite(number):
        raise StructuralCanonicalizeError("row invariant violated")
    return number


def _product_of_columns(row: dict[str, Any], factor_columns: tuple[str, ...]) -> float:
    product = 1.0
    for column in factor_columns:
        product *= _finite_numeric(row.get(column))
    return product


def _within_product_tolerance(
    product: float,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    tolerance = absolute_tolerance + (relative_tolerance * abs(expected))
    return abs(product - expected) <= tolerance


def validate_row_invariants(
    row: dict[str, Any], invariants: tuple[RowInvariant, ...]
) -> None:
    if not invariants:
        return
    validate_row_invariant_bundle(invariants)
    for invariant in invariants:
        if isinstance(invariant, ExactTextRowInvariant):
            actual = row.get(invariant.column)
            if actual is None or str(actual) != invariant.expected:
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, NumericRowInvariant):
            if not _numeric_equal(
                row.get(invariant.left_column), row.get(invariant.right_column)
            ):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, TimestampMsEquivalenceRowInvariant):
            left_ms = _timestamp_to_epoch_ms(
                row.get(invariant.left_column), invariant.left_mode
            )
            right_ms = _timestamp_to_epoch_ms(
                row.get(invariant.right_column), invariant.right_mode
            )
            if left_ms != right_ms:
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, PositiveFiniteRowInvariant):
            if not _positive_finite(row.get(invariant.column)):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, ProductWithToleranceRowInvariant):
            product = _product_of_columns(row, invariant.factor_columns)
            expected = float(invariant.expected)
            if not _within_product_tolerance(
                product,
                expected,
                absolute_tolerance=invariant.absolute_tolerance,
                relative_tolerance=invariant.relative_tolerance,
            ):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, ProductColumnWithToleranceRowInvariant):
            product = _product_of_columns(row, invariant.factor_columns)
            expected = _finite_numeric(row.get(invariant.expected_column))
            if not _within_product_tolerance(
                product,
                expected,
                absolute_tolerance=invariant.absolute_tolerance,
                relative_tolerance=invariant.relative_tolerance,
            ):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, TextColumnEquivalenceRowInvariant):
            left = row.get(invariant.left_column)
            right = row.get(invariant.right_column)
            if not isinstance(left, str) or not isinstance(right, str):
                raise StructuralCanonicalizeError("row invariant violated")
            if left != right:
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, NonEmptyTextRowInvariant):
            value = row.get(invariant.column)
            if not isinstance(value, str) or value == "":
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, BooleanRowInvariant):
            value = row.get(invariant.column)
            if not isinstance(value, bool):
                raise StructuralCanonicalizeError("row invariant violated")
        elif isinstance(invariant, FiniteNumericRowInvariant):
            _finite_numeric(row.get(invariant.column))
        else:
            raise StructuralCanonicalizeError("row invariant violated")
