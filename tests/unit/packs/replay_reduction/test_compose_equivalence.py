import polars as pl

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    execute_structural_canonicalize,
)
from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)

_CANON = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {"identity_equals": -1},
}

_REQUEST = {
    "schema_version": "pbe.replay.event-window-extract.v2",
    "request_id": "compose",
    "symbol": "AAA",
    "symbol_column": "symbol",
    "block_column": "block",
    "timestamp_column": "timestamp_ms",
    "decision_timestamp_ms": 4000,
    "causal_cutoff_block": 4,
    "as_of_measurement_field": "price",
    "as_of_offsets_ms": (0,),
    "trailing_windows": (
        {"fact_id": "trail", "measurement_field": "price", "trailing_width_ms": 2000},
    ),
    "future_windows": (
        {
            "fact_id": "future",
            "measurement_field": "price",
            "start_offset_ms": 1000,
            "end_offset_ms": 2000,
            "min_block_exclusive": 4,
        },
    ),
    "tie_break_columns": ("timestamp_ms",),
}


def test_canonicalize_then_event_window_matches_direct_reference(tmp_path):
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pl.DataFrame(
        [
            {"identity": 1, "identity_norm": "1", "price": 1.0, "symbol": "AAA", "block": 3, "timestamp_ms": 3000},
            {"identity": 2, "identity_norm": "2", "price": 2.0, "symbol": "AAA", "block": 4, "timestamp_ms": 3500},
            {"identity": -1, "identity_norm": "w", "price": 0.0, "symbol": "AAA", "block": 2, "timestamp_ms": 2500},
        ]
    ).write_parquet(path_a)
    pl.DataFrame(
        [
            {"identity": 3, "identity_norm": "3", "price": 3.0, "symbol": "AAA", "block": 5, "timestamp_ms": 5000},
            {"identity": 4, "identity_norm": "4", "price": 4.0, "symbol": "AAA", "block": 6, "timestamp_ms": 5500},
        ]
    ).write_parquet(path_b)

    canonical = execute_structural_canonicalize([path_a, path_b], _CANON)
    assert canonical.witness_row_count == 1
    assert canonical.positive_group_count == 4
    composed = execute_event_window_extract([path_a, path_b], _REQUEST)
    reference = execute_event_window_extract([path_a, path_b], _REQUEST)
    assert composed == reference
