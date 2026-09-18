import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
)
from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)
from portable_batch_execution.packs.replay_reduction.row_invariants import (
    validate_row_invariants,
)

_PROFILE = {
    "schema_version": "pbe.replay.canonical-trade-profile.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
}


def _row(**fields):
    base = {
        "identity": 1,
        "identity_norm": "1",
        "symbol": "AAA",
        "block": 5,
        "timestamp_ms": 3000,
        "price": 1.0,
        "seq": 0,
    }
    base.update(fields)
    return base


def test_row_invariant_kinds_pass_and_fail():
    row = {
        "tag": "ok",
        "left_num": 2,
        "right_num": 2.0,
        "ts_ms": 5000,
        "ts_iso": "1970-01-01T00:00:05Z",
        "qty": 3.0,
        "unit_price": 4.0,
    }
    invariants = (
        {
            "schema_version": "pbe.replay.row-invariant.exact-text.v1",
            "column": "tag",
            "expected": "ok",
        },
        {
            "schema_version": "pbe.replay.row-invariant.numeric.v1",
            "left_column": "left_num",
            "right_column": "right_num",
        },
        {
            "schema_version": "pbe.replay.row-invariant.timestamp-ms-equivalence.v1",
            "left_column": "ts_ms",
            "right_column": "ts_iso",
            "left_mode": "integer_ms",
            "right_mode": "iso8601",
        },
        {
            "schema_version": "pbe.replay.row-invariant.positive-finite.v1",
            "column": "qty",
        },
        {
            "schema_version": "pbe.replay.row-invariant.product-with-tolerance.v1",
            "factor_columns": ["qty", "unit_price"],
            "expected": 12.0,
            "absolute_tolerance": 1e-9,
        },
    )
    from portable_batch_execution.packs.replay_reduction.models import (
        CanonicalTradeProfile,
    )

    profile = CanonicalTradeProfile.model_validate({**_PROFILE, "row_invariants": invariants})
    validate_row_invariants(row, profile.row_invariants)

    with pytest.raises(StructuralCanonicalizeError):
        validate_row_invariants({**row, "tag": "bad"}, profile.row_invariants)
    with pytest.raises(StructuralCanonicalizeError):
        validate_row_invariants({**row, "qty": 0.0}, profile.row_invariants)
    with pytest.raises(StructuralCanonicalizeError):
        validate_row_invariants({**row, "ts_iso": "1970-01-01T00:00:05"}, profile.row_invariants)


def test_event_window_fails_closed_before_facts_on_row_invariant(tmp_path):
    path = tmp_path / "rows.parquet"
    pl.DataFrame([_row(price=1.0)]).write_parquet(path)
    request = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "req",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": 5000,
        "causal_cutoff_block": 5,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": {
            **_PROFILE,
            "row_invariants": [
                {
                    "schema_version": "pbe.replay.row-invariant.exact-text.v1",
                    "column": "symbol",
                    "expected": "ZZZ",
                }
            ],
        },
        "as_of_offsets_ms": (0,),
        "tie_break_columns": ("timestamp_ms", "seq"),
    }
    with pytest.raises(StructuralCanonicalizeError):
        execute_event_window_extract([path], request)


def test_equal_block_timestamp_source_order_changes_as_of_price(tmp_path):
    path = tmp_path / "rows.parquet"
    pl.DataFrame(
        [
            _row(identity=1, block=100, timestamp_ms=5000, price=1.5964, seq=0),
            _row(identity=2, identity_norm="2", block=100, timestamp_ms=5000, price=1.5966, seq=0),
        ]
    ).write_parquet(path)
    base = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "req",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": 5000,
        "causal_cutoff_block": 100,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": _PROFILE,
        "as_of_offsets_ms": (0,),
        "trailing_windows": (),
        "future_windows": (),
        "tie_break_columns": ("timestamp_ms", "seq"),
    }
    off = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], base)["facts"]
    }
    on = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract(
            [path], {**base, "source_order_tie_break": True}
        )["facts"]
    }
    assert off["as_of.0"] == 1.5964
    assert on["as_of.0"] == 1.5966
