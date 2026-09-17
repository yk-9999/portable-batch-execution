import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
    execute_structural_canonicalize,
    merge_structural_canonicalize_states,
)

_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {"identity_equals": -1},
}


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
            {"identity": 1, "identity_norm": "1", "price": 10.0},
            {"identity": 2, "identity_norm": "2", "price": 20.0},
        ],
        [{"identity": 1, "identity_norm": "1", "price": 10.0}],
    )
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result["schema_version"] == "pbe.replay.structural-canonicalize-result.v2"
    profile = next(item for item in result["identity_profiles"] if item["identity"] == 1)
    assert profile["core"] == {"price": 10.0}
    assert profile["source_input_indices"] == [0, 1]
    assert profile["row_count"] == 2


def test_structural_canonicalize_counts_sentinel_witness_rows(tmp_path):
    paths = _paths(tmp_path, [{"identity": -1, "identity_norm": "x", "price": 0.0}])
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result["witness_row_count"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [{"identity": None, "identity_norm": "1", "price": 1.0}],
        [{"identity": 0, "identity_norm": "0", "price": 1.0}],
        [{"identity": -2, "identity_norm": "-2", "price": 1.0}],
    ],
)
def test_structural_canonicalize_rejects_missing_or_nonpositive(tmp_path, rows):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(_paths(tmp_path, rows), _PARAMS)


def test_structural_canonicalize_rejects_core_field_disagreement(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [
                    {"identity": 1, "identity_norm": "1", "price": 1.0},
                    {"identity": 1, "identity_norm": "1", "price": 2.0},
                ],
            ),
            _PARAMS,
        )


def test_structural_canonicalize_allows_non_contiguous_source_indices(tmp_path):
    result = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [{"identity": 1, "identity_norm": "1", "price": 1.0}],
            [{"identity": 2, "identity_norm": "2", "price": 2.0}],
            [{"identity": 1, "identity_norm": "1", "price": 1.0}],
        ),
        _PARAMS,
    )
    profile = next(item for item in result["identity_profiles"] if item["identity"] == 1)
    assert profile["source_input_indices"] == [0, 2]


def test_merge_detects_cross_wave_core_conflict(tmp_path):
    wave_a = execute_structural_canonicalize(
        _paths(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 1.0}]),
        _PARAMS,
    )
    wave_b = execute_structural_canonicalize(
        _paths(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 2.0}]),
        _PARAMS,
    )
    with pytest.raises(StructuralCanonicalizeError):
        merge_structural_canonicalize_states(wave_a, wave_b)


def test_merge_finalizes_same_identity_across_waves(tmp_path):
    wave_a = execute_structural_canonicalize(
        _paths(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 5.0}]),
        _PARAMS,
    )
    wave_b = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [{"identity": 1, "identity_norm": "1", "price": 5.0}],
            [{"identity": 2, "identity_norm": "2", "price": 6.0}],
        ),
        _PARAMS,
    )
    merged = merge_structural_canonicalize_states(wave_a, wave_b)
    profiles = {item["identity"]: item for item in merged["identity_profiles"]}
    assert profiles[1]["row_count"] == 2
    assert profiles[2]["core"] == {"price": 6.0}
