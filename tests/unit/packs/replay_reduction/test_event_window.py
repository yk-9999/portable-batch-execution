import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)

_REQUEST = {
    "schema_version": "pbe.replay.event-window-extract.v1",
    "request_id": "req-1",
    "symbol": "AAA",
    "symbol_column": "symbol",
    "block_column": "block",
    "causal_cutoff_block": 5,
    "decision_block": 5,
    "as_of_measurement_field": "price",
    "as_of_offsets": (0, 1),
    "trailing_windows": (
        {"fact_id": "trail", "measurement_field": "price", "trailing_block_count": 2},
    ),
    "future_windows": (
        {
            "fact_id": "future",
            "measurement_field": "price",
            "start_offset_blocks": 1,
            "end_offset_blocks": 2,
        },
    ),
    "tie_break_columns": ("block", "seq"),
}


def _path(tmp_path):
    path = tmp_path / "records.parquet"
    pl.DataFrame(
        [
            {"symbol": "AAA", "block": 3, "price": 1.0, "seq": 0},
            {"symbol": "AAA", "block": 4, "price": 2.0, "seq": 0},
            {"symbol": "AAA", "block": 5, "price": 3.0, "seq": 0},
            {"symbol": "AAA", "block": 6, "price": 9.0, "seq": 1},
            {"symbol": "AAA", "block": 6, "price": 8.0, "seq": 0},
            {"symbol": "BBB", "block": 5, "price": 100.0, "seq": 0},
            {"symbol": "AAA", "block": 7, "price": 4.0, "seq": 0},
        ]
    ).write_parquet(path)
    return path


def test_event_window_extract_emits_trailing_as_of_and_future_facts(tmp_path):
    result = execute_event_window_extract([_path(tmp_path)], _REQUEST)
    facts = {item["fact_id"]: item["value"] for item in result["facts"]}
    assert facts["trail.sum"] == 5.0
    assert facts["trail.count"] == 2
    assert facts["as_of.0"] == 3.0
    assert facts["as_of.1"] == 2.0
    assert facts["future"] == 8.0


def test_event_window_extract_requires_valid_request(tmp_path):
    with pytest.raises(ValidationError):
        execute_event_window_extract([_path(tmp_path)], {"schema_version": "bad"})
