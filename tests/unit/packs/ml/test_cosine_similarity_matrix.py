import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from portable_batch_execution.packs.ml.cosine_similarity_matrix import (
    execute_cosine_similarity_matrix,
    validate_cosine_similarity_matrix_input,
)


def _payload(left, right):
    return {"left": left, "right": right}


def _row(row_id: str, vector: list[float]) -> dict:
    return {"row_id": row_id, "vector": vector}


def test_scores_match_sklearn_for_nonzero_norm_pairs():
    left = [
        _row("l1", [1.0, 0.0, 2.0]),
        _row("l2", [0.0, 0.0, 0.0]),
        _row("l3", [-1.5, 2.0, 0.5]),
    ]
    right = [
        _row("r1", [0.5, 1.0, -1.0]),
        _row("r2", [3.0, 0.0, 1.0]),
    ]
    payload = _payload(left, right)
    left_matrix = np.asarray([item["vector"] for item in left], dtype=float)
    right_matrix = np.asarray([item["vector"] for item in right], dtype=float)
    sklearn_matrix = cosine_similarity(left_matrix, right_matrix)

    result = execute_cosine_similarity_matrix(payload)
    assert result["left_ids"] == ["l1", "l2", "l3"]
    assert result["right_ids"] == ["r1", "r2"]
    assert len(result["scores"]) == 3
    assert all(len(row) == 2 for row in result["scores"])
    assert result["scores"][0][0] == pytest.approx(float(sklearn_matrix[0, 0]))
    assert result["scores"][0][1] == pytest.approx(float(sklearn_matrix[0, 1]))
    assert result["scores"][1] == [None, None]
    assert result["scores"][2][0] == pytest.approx(float(sklearn_matrix[2, 0]))
    assert result["scores"][2][1] == pytest.approx(float(sklearn_matrix[2, 1]))


def test_zero_left_vector_yields_null_row_cells():
    left = [
        _row("nz", [1.0, 0.0]),
        _row("zero", [0.0, 0.0]),
    ]
    right = [_row("r1", [1.0, 0.0]), _row("r2", [0.0, 1.0])]
    result = execute_cosine_similarity_matrix(_payload(left, right))
    assert result["scores"][0] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert result["scores"][1] == [None, None]


def test_zero_right_vector_yields_null_column_cells():
    left = [_row("l1", [1.0, 0.0]), _row("l2", [0.0, 1.0])]
    right = [
        _row("nz", [1.0, 0.0]),
        _row("zero", [0.0, 0.0]),
        _row("nz2", [0.0, 1.0]),
    ]
    result = execute_cosine_similarity_matrix(_payload(left, right))
    assert result["scores"][0][0] == pytest.approx(1.0)
    assert result["scores"][0][1] is None
    assert result["scores"][0][2] == pytest.approx(0.0)
    assert result["scores"][1][0] == pytest.approx(0.0)
    assert result["scores"][1][1] is None
    assert result["scores"][1][2] == pytest.approx(1.0)


def test_orthogonal_nonzero_vectors_remain_numeric_zero():
    left = [_row("a", [1.0, 0.0])]
    right = [_row("b", [0.0, 1.0])]
    result = execute_cosine_similarity_matrix(_payload(left, right))
    assert result["scores"] == [[0.0]]


def test_order_preservation_and_rectangular_output():
    left = [_row("a", [1.0, 0.0]), _row("b", [0.0, 1.0])]
    right = [_row("x", [1.0, 1.0]), _row("y", [1.0, -1.0]), _row("z", [0.0, 1.0])]
    result = execute_cosine_similarity_matrix(_payload(left, right))
    assert result["left_ids"] == ["a", "b"]
    assert result["right_ids"] == ["x", "y", "z"]
    assert len(result["scores"]) == 2
    assert [len(row) for row in result["scores"]] == [3, 3]


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload.pop("right"),
        lambda payload: payload.update({"left": []}),
        lambda payload: payload.update({"right": []}),
        lambda payload: payload["left"].append({"row_id": "a", "vector": [0.0]}),
        lambda payload: payload["left"].append({"row_id": "x", "vector": [1.0, 2.0]}),
        lambda payload: payload["right"].append({"row_id": "y", "vector": [1.0, 2.0]}),
        lambda payload: payload["left"].append({"row_id": "z", "vector": [True]}),
        lambda payload: payload["left"].append(
            {"row_id": "z", "vector": [float("nan")]}
        ),
        lambda payload: payload["left"].append(
            {"row_id": "z", "vector": [float("inf")]}
        ),
        lambda payload: payload["left"].append({"row_id": "z", "vector": ["1"]}),
        lambda payload: payload["left"].append({"row_id": "z", "vector": []}),
        lambda payload: payload["left"].append({"row_id": "", "vector": [1.0]}),
        lambda payload: payload["left"].append({"row_id": "z", "vector": [1.0], "extra": 1}),
        lambda payload: payload.update({"callback": "run"}),
        lambda payload: payload["left"].append(1),
    ),
)
def test_rejects_malformed_contract(mutator):
    payload = _payload(
        [{"row_id": "a", "vector": [1.0]}],
        [{"row_id": "b", "vector": [1.0]}],
    )
    mutator(payload)
    with pytest.raises((TypeError, ValueError)):
        validate_cosine_similarity_matrix_input(payload)
