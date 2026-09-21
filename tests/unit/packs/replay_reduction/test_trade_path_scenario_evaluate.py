import pytest

from portable_batch_execution.contracts import JobSpec
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate import (
    REQUEST_SCHEMA_VERSION,
    TradePathScenarioEvaluateError,
    execute_trade_path_scenario_evaluate,
)


def _closed_record(**overrides):
    base = {
        "record_id": "r1",
        "model": "M1",
        "window": "6m",
        "group_label": "G",
        "month": "2010-01",
        "status": "closed",
        "cancellation_reason": None,
        "reference_notional": 1.0,
        "price_pnl": 10.0,
        "dividend_pnl": 1.0,
        "gross_pnl": 11.0,
        "entry_notional": 100.0,
        "exit_notional": 100.0,
        "traded_notional": 200.0,
        "short_notional": 50.0,
        "holding_days": 10.0,
        "holding_sessions": 10,
        "mae": -2.0,
        "mfe": 3.0,
        "adv_ratio_20d": 0.01,
        "adv_ratio_60d": 0.02,
    }
    base.update(overrides)
    return base


def _request(**scenario):
    scen = {
        "commission_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "borrow_bps_per_year": 100.0,
        "short_available": True,
    }
    scen.update(scenario)
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "batch_id": "batch-1",
        "reference_notional": 1.0,
        "scenario": scen,
        "records": [_closed_record()],
    }


def test_hand_computed_arithmetic():
    payload = _request()
    result = execute_trade_path_scenario_evaluate(payload)
    row = result["records"][0]
    entry_cost = 100.0 * (10.0 + 5.0) / 1e4
    exit_cost = 100.0 * (10.0 + 5.0) / 1e4
    borrow_cost = 50.0 * (100.0 / 1e4) * (10.0 / 365.0)
    assert row["entry_cost"] == pytest.approx(entry_cost)
    assert row["exit_cost"] == pytest.approx(exit_cost)
    assert row["borrow_cost"] == pytest.approx(borrow_cost)
    assert row["net_pnl"] == pytest.approx(11.0 - entry_cost - exit_cost - borrow_cost)


def test_short_unavailable_cancels_short_records():
    payload = _request(short_available=False)
    result = execute_trade_path_scenario_evaluate(payload)
    row = result["records"][0]
    assert row["status"] == "cancelled"
    assert row["cancellation_reason"] == "short_unavailable"
    assert result["event_summary"]["n"] == 0
    assert result["event_summary"]["cancelled"]["short_unavailable"] == 1


def test_aggregate_stats():
    payload = _request()
    payload["records"] = [
        _closed_record(record_id="a", gross_pnl=10.0, model="A", window="6m"),
        _closed_record(record_id="b", gross_pnl=-4.0, model="B", window="24m"),
        {
            **_closed_record(record_id="c"),
            "status": "censored",
            "gross_pnl": 0.0,
            "entry_notional": 0.0,
            "exit_notional": 0.0,
            "traded_notional": 0.0,
            "short_notional": 0.0,
        },
    ]
    summary = execute_trade_path_scenario_evaluate(payload)["event_summary"]
    assert summary["n"] == 2
    assert summary["censored_dev_unresolved"] == 1
    assert summary["gross_pnl_sum"] == pytest.approx(6.0)
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert "A/6m" in summary["per_window_model"]


def test_job_spec_rejects_executable_operation_params():
    with pytest.raises(ValueError, match="closed operation parameters"):
        JobSpec.model_validate(
            {
                "job_id": "job",
                "logical_run_id": "run",
                "pack": "replay-batch",
                "operation": "replay.trade_path_scenario_evaluate",
                "input_manifest_ref": {
                    "object_id": "in",
                    "uri": "pbe://private/in",
                    "sha256": "sha256:" + ("a" * 64),
                    "size_bytes": 1,
                    "media_type": "application/json",
                },
                "sharding": {"mode": "independent"},
                "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
                "security_profile": "offline",
                "provenance": {
                    "producer": "t",
                    "revision": "1",
                    "created_at": "2020-01-01T00:00:00Z",
                },
                "operation_params": {
                    "schema_version": "pbe.replay.trade-path-scenario-evaluate-job.v1",
                    "executable": "rm -rf /",
                },
            }
        )


def test_forbidden_url_in_record():
    payload = _request()
    payload["records"][0]["model"] = "http://evil"
    with pytest.raises(TradePathScenarioEvaluateError):
        execute_trade_path_scenario_evaluate(payload)


def test_deterministic_content_identity():
    payload = _request()
    first = execute_trade_path_scenario_evaluate(payload)
    second = execute_trade_path_scenario_evaluate(payload)
    assert first["summary"]["content_identity"] == second["summary"]["content_identity"]
