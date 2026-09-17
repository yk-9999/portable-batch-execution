import io

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
    execute_structural_canonicalize,
)

_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {"identity_equals": -1},
}


def _write_parquet(rows: list[dict]) -> io.BytesIO:
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    buffer.seek(0)
    return buffer


def _paths(tmp_path, *row_groups):
    paths = []
    for index, rows in enumerate(row_groups):
        path = tmp_path / f"part-{index}.parquet"
        pl.DataFrame(rows).write_parquet(path)
        paths.append(path)
    return paths


def test_structural_canonicalize_collapses_identity_across_inputs(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {"identity": 1, "identity_norm": "1", "price": 10.0, "symbol": "AAA"},
            {"identity": 2, "identity_norm": "2", "price": 20.0, "symbol": "AAA"},
        ],
        [
            {"identity": 1, "identity_norm": "1", "price": 10.0, "symbol": "AAA", "note": "later"},
        ],
    )
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result["schema_version"] == "pbe.replay.structural-canonicalize-result.v1"
    assert len(result["canonical_records"]) == 2
    first = next(item for item in result["canonical_records"] if item["identity"] == 1)
    assert first.get("note") is None
    assert result["boundary_evidence"] == [
        {"source_input_index": 0, "identities": [1, 2]},
        {"source_input_index": 1, "identities": [1]},
    ]


def test_structural_canonicalize_collects_witness_facts(tmp_path):
    paths = _paths(
        tmp_path,
        [{"identity": -1, "identity_norm": "x", "price": 0.0}],
    )
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result["witness_facts"] == [{"identity": -1, "identity_norm": "x", "price": 0.0}]
    assert result["canonical_records"] == []


@pytest.mark.parametrize(
    "rows",
    [
        [{"identity": None, "identity_norm": "1", "price": 1.0}],
        [{"identity": 0, "identity_norm": "0", "price": 1.0}],
        [{"identity": -2, "identity_norm": "-2", "price": 1.0}],
    ],
)
def test_structural_canonicalize_rejects_missing_or_nonpositive(tmp_path, rows):
    paths = _paths(tmp_path, rows)
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _PARAMS)


def test_structural_canonicalize_rejects_normalized_mismatch(tmp_path):
    paths = _paths(tmp_path, [{"identity": 1, "identity_norm": "2", "price": 1.0}])
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _PARAMS)


def test_structural_canonicalize_rejects_core_field_disagreement(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {"identity": 1, "identity_norm": "1", "price": 1.0},
            {"identity": 1, "identity_norm": "1", "price": 2.0},
        ],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _PARAMS)


def test_structural_canonicalize_rejects_non_contiguous_source_indices(tmp_path):
    paths = _paths(
        tmp_path,
        [{"identity": 1, "identity_norm": "1", "price": 1.0}],
        [{"identity": 2, "identity_norm": "2", "price": 2.0}],
        [{"identity": 1, "identity_norm": "1", "price": 1.0}],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _PARAMS)


def test_structural_canonicalize_prefers_earliest_row(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {"identity": 1, "identity_norm": "1", "price": 5.0, "tag": "first"},
            {"identity": 1, "identity_norm": "1", "price": 5.0, "tag": "second"},
        ],
    )
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result["canonical_records"][0]["tag"] == "first"
