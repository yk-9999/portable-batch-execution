from __future__ import annotations

import json

from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate import (
    execute_trade_path_scenario_evaluate,
)
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SCENARIO_NAMES,
    _FIXED_SCENARIO_PARAMS,
    execute_trade_path_scenario_evaluate_fixed_set,
)


def test_all_five_scenarios_match_single_path():
    payload = {
        "schema_version": "pbe.replay.trade-path-scenario-evaluate.v1",
        "batch_id": "batch-00000",
        "reference_notional": 1.0,
        "scenario": {
            "commission_bps_per_side": 0.0,
            "slippage_bps_per_side": 0.0,
            "borrow_bps_per_year": 0.0,
            "short_available": True,
        },
        "records": [
            {
                "record_id": "r1",
                "model": "DIST",
                "window": "6m",
                "group_label": "g",
                "month": "2014-06",
                "status": "closed",
                "reference_notional": 1.0,
                "price_pnl": 2.0,
                "dividend_pnl": 0.0,
                "gross_pnl": 2.0,
                "entry_notional": 100.0,
                "exit_notional": 100.0,
                "traded_notional": 200.0,
                "short_notional": 0.0,
                "holding_days": 5.0,
                "holding_sessions": 5,
            }
        ],
    }
    combined = execute_trade_path_scenario_evaluate_fixed_set(payload)
    body = json.loads(combined["result_json_bytes"])
    for name in FIXED_SCENARIO_NAMES:
        injected = dict(payload)
        injected["scenario"] = _FIXED_SCENARIO_PARAMS[name].model_dump(mode="json")
        single = execute_trade_path_scenario_evaluate(injected)
        combined_summary = body["scenarios"][name]["event_summary"]
        for key in ("n", "gross_pnl_sum", "net_pnl_mean", "hit_rate"):
            assert combined_summary[key] == single["event_summary"][key]
