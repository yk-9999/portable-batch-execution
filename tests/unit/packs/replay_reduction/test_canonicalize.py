import json

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    RESULT_SCHEMA_VERSION,
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


def _positive_segments(result):
    return [
        segment
        for segment in result["range_segments"]
        if int(segment.get("positive_row_count", 0)) > 0
    ]


def test_many_distinct_identities_keep_constant_segment_cardinality(tmp_path):
    rows = [
        {"identity": i, "identity_norm": str(i), "price": float(i)} for i in range(1, 5001)
    ]
    result = execute_structural_canonicalize(_paths(tmp_path, rows), _PARAMS)
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert len(_positive_segments(result)) == 1
    assert len(json.dumps(result)) < 5000


def test_ordered_disjoint_shards_pass(tmp_path):
    result = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [
                {"identity": 1, "identity_norm": "1", "price": 1.0},
                {"identity": 2, "identity_norm": "2", "price": 2.0},
            ],
            [{"identity": 3, "identity_norm": "3", "price": 3.0}],
        ),
        _PARAMS,
    )
    assert len(_positive_segments(result)) == 2


def test_shared_boundary_split_across_adjacent_shards_passes(tmp_path):
    result = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [{"identity": 1, "identity_norm": "1", "price": 10.0}],
            [
                {"identity": 1, "identity_norm": "1", "price": 10.0},
                {"identity": 2, "identity_norm": "2", "price": 20.0},
            ],
        ),
        _PARAMS,
    )
    segments = _positive_segments(result)
    assert len(segments) == 1
    assert segments[0]["positive_row_count"] == 3
    assert segments[0]["identity_max"] == 2


def test_shared_boundary_core_conflict_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 1, "identity_norm": "1", "price": 2.0}],
            ),
            _PARAMS,
        )


def test_overlap_without_shared_boundary_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [
                    {"identity": 1, "identity_norm": "1", "price": 1.0},
                    {"identity": 5, "identity_norm": "5", "price": 5.0},
                ],
                [
                    {"identity": 3, "identity_norm": "3", "price": 3.0},
                    {"identity": 4, "identity_norm": "4", "price": 4.0},
                ],
            ),
            _PARAMS,
        )


def test_non_contiguous_recurrence_across_shards_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 2, "identity_norm": "2", "price": 2.0}],
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
            ),
            _PARAMS,
        )


def test_unordered_identities_within_shard_fail_closed(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(tmp_path, [{"identity": 2, "identity_norm": "2", "price": 2.0}, {"identity": 1, "identity_norm": "1", "price": 1.0}]),
            _PARAMS,
        )


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


def test_structural_canonicalize_rejects_core_field_disagreement_within_shard(tmp_path):
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


def test_merge_coalesces_shared_boundary_across_waves(tmp_path):
    wave_a = execute_structural_canonicalize(
        _paths(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 5.0}]),
        _PARAMS,
    )
    wave_b = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [{"identity": 1, "identity_norm": "1", "price": 5.0}],
            [
                {"identity": 1, "identity_norm": "1", "price": 5.0},
                {"identity": 2, "identity_norm": "2", "price": 6.0},
            ],
        ),
        _PARAMS,
    )
    merged = merge_structural_canonicalize_states(wave_a, wave_b)
    segments = _positive_segments(merged)
    assert len(segments) == 1
    assert segments[0]["positive_row_count"] == 3
    assert segments[0]["identity_max"] == 2
