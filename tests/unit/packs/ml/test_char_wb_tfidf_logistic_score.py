import math
import re

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from portable_batch_execution.packs.ml.char_wb_tfidf_logistic_score import (
    INPUT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    execute_char_wb_tfidf_logistic_score,
    validate_char_wb_tfidf_logistic_score_input,
)


def _preprocess(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text.lower())


def _reference_score(
    text: str,
    features: list[dict[str, float | str]],
    intercept: float,
) -> float:
    vocabulary = {item["feature"]: index for index, item in enumerate(features)}
    idf = np.array([float(item["idf"]) for item in features], dtype=float)
    coefficients = np.array(
        [float(item["coefficient"]) for item in features], dtype=float
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
        preprocessor=_preprocess,
        sublinear_tf=True,
        norm=None,
        use_idf=False,
        vocabulary=vocabulary,
    )
    vectorizer.fit([text])
    tf = vectorizer.transform([text]).toarray()[0]
    weighted = tf * idf
    norm = np.linalg.norm(weighted)
    if norm:
        weighted = weighted / norm
    logit = intercept + float(np.dot(weighted, coefficients))
    return 1.0 / (1.0 + math.exp(-logit))


def _payload(*, rows, features, intercept=0.0):
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "model": {"features": features, "intercept": intercept},
        "rows": rows,
    }


def test_scores_match_sklearn_reference():
    features = [
        {"feature": "ab", "idf": 1.1, "coefficient": 0.4},
        {"feature": "bc", "idf": 0.9, "coefficient": -0.2},
        {"feature": " xy", "idf": 1.3, "coefficient": 0.7},
        {"feature": "yz", "idf": 1.0, "coefficient": 0.1},
    ]
    rows = [
        {"row_id": "r1", "text": "ab bc"},
        {"row_id": "r2", "text": "xy  yz"},
    ]
    payload = _payload(rows=rows, features=features, intercept=-0.05)
    result = execute_char_wb_tfidf_logistic_score(payload)
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert [item["row_id"] for item in result["rows"]] == ["r1", "r2"]
    for item, source in zip(result["rows"], rows, strict=True):
        expected = _reference_score(source["text"], features, -0.05)
        assert math.isclose(item["score"], expected, rel_tol=0, abs_tol=1e-12)
        assert 0.0 <= item["score"] <= 1.0


def test_empty_rows_returns_empty_result():
    payload = _payload(
        rows=[], features=[{"feature": "ab", "idf": 1.0, "coefficient": 0.0}]
    )
    result = execute_char_wb_tfidf_logistic_score(payload)
    assert result == {
        "schema_version": RESULT_SCHEMA_VERSION,
        "rows": [],
    }


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload.update({"schema_version": "other"}),
        lambda payload: payload.pop("model"),
        lambda payload: payload["model"]["features"].append(
            {"feature": "ab", "idf": 1.0, "coefficient": 0.0}
        ),
        lambda payload: payload["model"]["features"][0].update({"idf": float("nan")}),
        lambda payload: payload["rows"].append({"row_id": "dup", "text": "one"}),
        lambda payload: payload["rows"].append({"row_id": "dup", "text": "two"}),
        lambda payload: payload["rows"].append({"row_id": "", "text": "x"}),
        lambda payload: payload["rows"].append({"row_id": "x", "text": 1}),
    ),
)
def test_rejects_malformed_contract(mutator):
    payload = _payload(
        rows=[{"row_id": "dup", "text": "alpha"}],
        features=[{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
    )
    mutator(payload)
    with pytest.raises((TypeError, ValueError)):
        validate_char_wb_tfidf_logistic_score_input(payload)
