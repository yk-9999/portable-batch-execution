import json

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction import canonicalize, event_window

_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
}


def test_canonicalize_rejects_whole_corpus_read_and_concat(tmp_path, monkeypatch):
    path = tmp_path / "part.parquet"
    pl.DataFrame([{"identity": 1, "identity_norm": "1", "price": 1.0}]).write_parquet(path)

    def forbid_read(*args, **kwargs):
        raise AssertionError("read_parquet must not be used")

    def forbid_concat(*args, **kwargs):
        raise AssertionError("concat must not be used for whole-corpus materialization")

    monkeypatch.setattr(pl, "read_parquet", forbid_read)
    monkeypatch.setattr(pl, "concat", forbid_concat)

    result = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    assert result.positive_group_count == 1
    assert len(json.dumps(canonicalize.state_summary(result))) < 4000


def test_event_window_rejects_whole_history_to_dicts(tmp_path, monkeypatch):
    path = tmp_path / "records.parquet"
    pl.DataFrame(
        [{"symbol": "AAA", "block": 1, "timestamp_ms": 1, "price": 1.0, "seq": 0}]
    ).write_parquet(path)

    class _Frame:
        def to_dicts(self):
            raise AssertionError("to_dicts must not be used")

    def forbid_read(*args, **kwargs):
        return _Frame()

    monkeypatch.setattr(pl, "read_parquet", forbid_read)

    request = {
        "schema_version": "pbe.replay.event-window-extract.v2",
        "request_id": "req",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": 1000,
        "causal_cutoff_block": 1,
        "as_of_measurement_field": "price",
    }
    result = event_window.execute_event_window_extract([path], request)
    assert result["facts"]


def test_canonicalize_does_not_materialize_all_bucket_lists(tmp_path):
    assert not hasattr(canonicalize, "_bucket_values")
    rows = [
        {"identity": i, "identity_norm": str(i), "price": float(i)} for i in range(1, 5001)
    ]
    path = tmp_path / "part.parquet"
    pl.DataFrame(rows).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    assert state.positive_group_count == 5000
    assert len(json.dumps(canonicalize.state_summary(state))) < 4000


def test_canonicalize_materialization_guard_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "_MAX_IDENTITY_MATERIALIZATION", 32)
    rows = [{"identity": i, "identity_norm": str(i), "price": 1.0} for i in range(1, 200)]
    path = tmp_path / "dense.parquet"
    pl.DataFrame(rows).write_parquet(path)
    with pytest.raises(canonicalize.StructuralCanonicalizeError):
        canonicalize.execute_structural_canonicalize([path], _PARAMS)
