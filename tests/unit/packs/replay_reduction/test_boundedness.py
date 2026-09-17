import polars as pl

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
    assert result["range_segments"][0]["identity_min"] == 1


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
