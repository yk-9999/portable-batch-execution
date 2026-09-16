from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from portable_batch_execution.contracts import Extent, RangeSpec, ShardCorrectnessSpec
from portable_batch_execution.packs.tabular import TabularPack


def _text_row(**overrides):
    row = {
        "row_id": "row-1", "partition_key": "p", "segment_key": "s",
        "event_time": "2026-01-01T00:00:00+00:00", "entity_id": "entity-1", "text": "",
    }
    row.update(overrides)
    return row


def _aggregate_input():
    return {
        "features": [
            {"row_id": "at-start", "partition_key": "p", "segment_key": "s", "event_time": "2026-01-01T00:00:00+00:00", "entity_id": "e1", "feature_kind": "k", "feature_key": "x", "weight": 2},
            {"row_id": "inside", "partition_key": "p", "segment_key": "s", "event_time": "2026-01-01T00:00:05+00:00", "entity_id": "e1", "feature_kind": "k", "feature_key": "x", "weight": 3},
            {"row_id": "inside", "partition_key": "p", "segment_key": "s", "event_time": "2026-01-01T00:00:06+00:00", "entity_id": "e1", "feature_kind": "k", "feature_key": "x", "weight": 4},
            {"row_id": "at-end", "partition_key": "p", "segment_key": "s", "event_time": "2026-01-01T00:00:10+00:00", "entity_id": "e2", "feature_kind": "k", "feature_key": "x", "weight": 7},
            {"row_id": "other-partition", "partition_key": "other", "segment_key": "s", "event_time": "2026-01-01T00:00:05+00:00", "entity_id": "e3", "feature_kind": "k", "feature_key": "x", "weight": 99},
            {"row_id": "other-segment", "partition_key": "p", "segment_key": "other", "event_time": "2026-01-01T00:00:05+00:00", "entity_id": "e4", "feature_kind": "k", "feature_key": "x", "weight": 99},
        ],
        "requests": [
            {"request_id": "large", "partition_key": "p", "segment_key": "s", "endpoint_time": "2026-01-01T00:00:10+00:00", "window_seconds": 10, "window_valid": True},
            {"request_id": "small", "partition_key": "p", "segment_key": "s", "endpoint_time": "2026-01-01T00:00:10+00:00", "window_seconds": 5, "window_valid": True},
        ],
    }


def test_text_event_features_normalizes_unicode_and_counts_overlapping_ngrams():
    result = TabularPack().run("tabular.text_event_features.v1", [
        _text_row(row_id="z", text="  ÅÅ  "),
        _text_row(row_id="a", text="ＡＡＡ"),
    ], {"unicode_normalization": "NFKC", "lowercase": True, "ngram_sizes": [2]})
    rows = result.to_dicts()
    assert rows == sorted(rows, key=lambda row: (row["partition_key"], row["segment_key"], row["event_time"], row["row_id"], row["entity_id"], row["feature_kind"], row["feature_key"]))
    assert next(row for row in rows if row["row_id"] == "a" and row["feature_key"] == "aa")["weight"] == 2
    assert next(row for row in rows if row["row_id"] == "z" and row["feature_kind"] == "normalized_non_whitespace_length")["weight"] == 2


def test_text_event_features_empty_whitespace_caps_and_invalid_parameters():
    pack = TabularPack()
    empty = pack.run("tabular.text_event_features.v1", [_text_row(text=" \t\n ")], {"ngram_sizes": [1]})
    assert empty.filter(empty["feature_kind"] == "message_presence")["weight"].item() == 0
    assert empty.filter(empty["feature_kind"] == "normalized_non_whitespace_length")["weight"].item() == 0
    with pytest.raises(ValueError, match="row cap"):
        pack.run("tabular.text_event_features.v1", [_text_row(), _text_row(row_id="two")], {"ngram_sizes": [1], "max_input_rows": 1})
    with pytest.raises(ValueError, match="output row cap"):
        pack.run("tabular.text_event_features.v1", [_text_row(text="abcd")], {"ngram_sizes": [1], "max_output_rows": 2})
    with pytest.raises(ValueError, match="byte cap"):
        pack.run("tabular.text_event_features.v1", [_text_row(text="é")], {"ngram_sizes": [1], "max_input_bytes": 1})
    with pytest.raises(ValueError, match="unique key cap"):
        pack.run("tabular.text_event_features.v1", [_text_row(text="ab")], {"ngram_sizes": [1], "max_unique_keys": 2})
    with pytest.raises(ValidationError):
        pack.validate_params("tabular.text_event_features.v1", {"ngram_sizes": [6]})
    with pytest.raises(ValidationError):
        pack.validate_params("tabular.text_event_features.v1", {"ngram_sizes": [1], "script": "x"})


def test_trailing_sparse_window_aggregate_is_exact_and_isolated():
    result = TabularPack().run("tabular.trailing_sparse_window_aggregate.v1", _aggregate_input(), {})
    rows = result.to_dicts()
    assert rows == sorted(rows, key=lambda row: (row["partition_key"], row["segment_key"], row["endpoint_time"], row["request_id"], row["window_seconds"], row["feature_kind"], row["feature_key"]))
    large = next(row for row in rows if row["request_id"] == "large")
    small = next(row for row in rows if row["request_id"] == "small")
    assert large["sum_weight"] == 9  # includes start, excludes endpoint and isolated rows
    assert large["distinct_row_count"] == 2
    assert large["distinct_entity_count"] == 1
    assert small["sum_weight"] == 7


def test_trailing_sparse_window_rejects_invalid_requests_halo_and_caps():
    pack = TabularPack()
    invalid = _aggregate_input()
    invalid["requests"][0]["window_valid"] = False
    with pytest.raises(ValueError, match="not valid"):
        pack.run("tabular.trailing_sparse_window_aggregate.v1", invalid, {})
    shard = SimpleNamespace(
        primary_range=RangeSpec(kind="time", start="2026-01-01T00:00:00+00:00", end="2026-01-01T00:01:00+00:00"),
        correctness=ShardCorrectnessSpec(halo_before=Extent(value=9, unit="seconds")),
    )
    job = SimpleNamespace(operation="tabular.trailing_sparse_window_aggregate.v1")
    with pytest.raises(ValueError, match="halo"):
        pack.execute(job, shard, {}, {"data": _aggregate_input()})
    with pytest.raises(ValueError, match="output row cap"):
        pack.run("tabular.trailing_sparse_window_aggregate.v1", _aggregate_input(), {"max_output_rows": 1})
    with pytest.raises(ValueError, match="feature row cap"):
        pack.run("tabular.trailing_sparse_window_aggregate.v1", _aggregate_input(), {"max_input_rows": 1})
    with pytest.raises(ValueError, match="byte cap"):
        pack.run("tabular.trailing_sparse_window_aggregate.v1", _aggregate_input(), {"max_input_bytes": 1})
    varied = _aggregate_input()
    varied["features"][1]["feature_key"] = "different"
    with pytest.raises(ValueError, match="unique key cap"):
        pack.run("tabular.trailing_sparse_window_aggregate.v1", varied, {"max_unique_keys": 1})
