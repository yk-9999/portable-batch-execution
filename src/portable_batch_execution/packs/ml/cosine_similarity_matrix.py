"""Deterministic cosine similarity matrix for closed private Data Plane vector input."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_ROW_KEYS = frozenset({"row_id", "vector"})
_INPUT_KEYS = frozenset({"left", "right"})


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _parse_side(name: str, side: Any) -> tuple[list[str], np.ndarray]:
    if not isinstance(side, list):
        raise TypeError(f"{name} must be an array")
    if not side:
        raise ValueError(f"{name} must be non-empty")
    row_ids: list[str] = []
    seen_ids: set[str] = set()
    vectors: list[list[float]] = []
    dimension: int | None = None
    for row in side:
        if not isinstance(row, dict):
            raise TypeError(f"{name} entries must be objects")
        if frozenset(row.keys()) != _ROW_KEYS:
            raise ValueError(f"{name} entry has unexpected fields")
        row_id = row.get("row_id")
        vector = row.get("vector")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string")
        if row_id in seen_ids:
            raise ValueError("duplicate row_id")
        seen_ids.add(row_id)
        if not isinstance(vector, list) or not vector:
            raise TypeError("vector must be a non-empty array")
        parsed_vector: list[float] = []
        for value in vector:
            if not _is_finite_number(value):
                raise ValueError("vector entries must be finite numbers")
            parsed_vector.append(float(value))
        if dimension is None:
            dimension = len(parsed_vector)
        elif len(parsed_vector) != dimension:
            raise ValueError("inconsistent vector dimensions")
        row_ids.append(row_id)
        vectors.append(parsed_vector)
    return row_ids, np.asarray(vectors, dtype=float)


def validate_cosine_similarity_matrix_input(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("input must be a JSON object")
    if frozenset(payload.keys()) != _INPUT_KEYS:
        raise ValueError("input has unexpected fields")
    left_ids, left = _parse_side("left", payload["left"])
    right_ids, right = _parse_side("right", payload["right"])
    if left.shape[1] != right.shape[1]:
        raise ValueError("left and right vector dimensions must match")
    return {"left_ids": left_ids, "right_ids": right_ids, "left": left, "right": right}


def execute_cosine_similarity_matrix(payload: Any) -> dict[str, Any]:
    validated = validate_cosine_similarity_matrix_input(payload)
    matrix = cosine_similarity(validated["left"], validated["right"])
    scores = [
        [float(value) for value in row]
        for row in matrix.tolist()
    ]
    for row in scores:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("similarity score is not finite")
    return {
        "left_ids": validated["left_ids"],
        "right_ids": validated["right_ids"],
        "scores": scores,
    }
