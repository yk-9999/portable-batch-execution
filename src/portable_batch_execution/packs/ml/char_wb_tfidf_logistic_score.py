"""Deterministic char_wb TF-IDF + binary logistic scoring for private Data Plane input."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

INPUT_SCHEMA_VERSION = "pbe.ml.char-wb-tfidf-logistic-score.v1"
RESULT_SCHEMA_VERSION = "pbe.ml.char-wb-tfidf-logistic-score-result.v1"

_NGRAM_MIN = 2
_NGRAM_MAX = 4
_WHITESPACE_RUNS = re.compile(r"\s{2,}")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _preprocess_text(text: str) -> str:
    return _WHITESPACE_RUNS.sub(" ", text.lower())


def _char_wb_ngrams(text_document: str) -> list[str]:
    """Sklearn-compatible char_wb n-grams on already lowercased, collapsed text."""
    ngrams: list[str] = []
    ngrams_append = ngrams.append
    for word in text_document.split():
        padded = f" {word} "
        word_len = len(padded)
        for n in range(_NGRAM_MIN, _NGRAM_MAX + 1):
            offset = 0
            ngrams_append(padded[offset : offset + n])
            while offset + n < word_len:
                offset += 1
                ngrams_append(padded[offset : offset + n])
            if offset == 0:
                break
    return ngrams


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _validate_model(model: Any) -> tuple[tuple[tuple[str, float, float], ...], float]:
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    features = model.get("features")
    intercept = model.get("intercept")
    if not isinstance(features, list):
        raise TypeError("model.features must be an array")
    if not _is_finite_number(intercept):
        raise ValueError("model.intercept must be a finite number")
    parsed: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for item in features:
        if not isinstance(item, dict):
            raise TypeError("model.features entries must be objects")
        token = item.get("feature")
        idf = item.get("idf")
        coefficient = item.get("coefficient")
        if not isinstance(token, str):
            raise TypeError("feature token must be a string")
        if token in seen:
            raise ValueError("duplicate feature token")
        seen.add(token)
        if not _is_finite_number(idf) or not _is_finite_number(coefficient):
            raise ValueError("feature weights must be finite numbers")
        parsed.append((token, float(idf), float(coefficient)))
    return tuple(parsed), float(intercept)


def validate_char_wb_tfidf_logistic_score_input(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("input must be a JSON object")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input schema_version")
    features, intercept = _validate_model(payload.get("model"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("rows must be an array")
    parsed_rows: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("rows entries must be objects")
        row_id = row.get("row_id")
        text = row.get("text")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string")
        if row_id in seen_ids:
            raise ValueError("duplicate row_id")
        seen_ids.add(row_id)
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        parsed_rows.append((row_id, text))
    return {
        "features": features,
        "intercept": intercept,
        "rows": parsed_rows,
    }


def _score_text(
    text: str,
    *,
    features: tuple[tuple[str, float, float], ...],
    intercept: float,
) -> float:
    vocabulary = {feature: (idf, coefficient) for feature, idf, coefficient in features}
    counts = Counter(
        token
        for token in _char_wb_ngrams(_preprocess_text(text))
        if token in vocabulary
    )
    weighted = []
    for feature, idf, coefficient in features:
        count = counts.get(feature, 0)
        if count:
            weighted.append((1.0 + math.log(count)) * idf)
        else:
            weighted.append(0.0)
    norm = math.sqrt(sum(value * value for value in weighted))
    if norm:
        weighted = [value / norm for value in weighted]
    logit = intercept
    for index, (_, _, coefficient) in enumerate(features):
        logit += weighted[index] * coefficient
    score = _sigmoid(logit)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("score is not a finite probability")
    return score


def execute_char_wb_tfidf_logistic_score(payload: Any) -> dict[str, Any]:
    validated = validate_char_wb_tfidf_logistic_score_input(payload)
    result_rows = []
    for row_id, text in validated["rows"]:
        score = _score_text(
            text,
            features=validated["features"],
            intercept=validated["intercept"],
        )
        result_rows.append({"row_id": row_id, "score": score})
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "rows": result_rows,
    }
