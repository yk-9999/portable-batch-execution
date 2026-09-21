"""Evaluate the frozen five cost scenarios in one pass over one normalized batch."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from portable_batch_execution.contracts.models import Frozen

from .trade_path_scenario_evaluate import (
    REQUEST_SCHEMA_VERSION,
    TRADE_PATH_SCENARIO_MAX_INPUT_BYTES,
    TRADE_PATH_SCENARIO_MAX_RECORDS,
    TradePathScenarioEvaluateError,
    TradePathScenarioEvaluateRequest,
    TradePathScenarioParams,
    execute_trade_path_scenario_evaluate,
)

FIXED_SET_JOB_SCHEMA_VERSION = (
    "pbe.replay.trade-path-scenario-evaluate-fixed-set-job.v1"
)
FIXED_SET_RESULT_SCHEMA_VERSION = (
    "pbe.replay.trade-path-scenario-evaluate-fixed-set-result.v1"
)
FIXED_SET_OPERATION = "replay.trade_path_scenario_evaluate_fixed_set"

# Frozen scenario definitions (must match research/config/pair_trading_v1/v1_config.json).
FIXED_SCENARIO_NAMES: tuple[str, ...] = (
    "ZERO_BASELINE",
    "HYPOTHESIS_LOW",
    "HYPOTHESIS_MID",
    "HYPOTHESIS_HIGH",
    "SHORT_UNAVAILABLE",
)

_FIXED_SCENARIO_PARAMS: dict[str, TradePathScenarioParams] = {
    "ZERO_BASELINE": TradePathScenarioParams(
        commission_bps_per_side=0.0,
        slippage_bps_per_side=0.0,
        borrow_bps_per_year=0.0,
        short_available=True,
    ),
    "HYPOTHESIS_LOW": TradePathScenarioParams(
        commission_bps_per_side=5.0,
        slippage_bps_per_side=5.0,
        borrow_bps_per_year=50.0,
        short_available=True,
    ),
    "HYPOTHESIS_MID": TradePathScenarioParams(
        commission_bps_per_side=10.0,
        slippage_bps_per_side=10.0,
        borrow_bps_per_year=100.0,
        short_available=True,
    ),
    "HYPOTHESIS_HIGH": TradePathScenarioParams(
        commission_bps_per_side=20.0,
        slippage_bps_per_side=20.0,
        borrow_bps_per_year=300.0,
        short_available=True,
    ),
    "SHORT_UNAVAILABLE": TradePathScenarioParams(
        commission_bps_per_side=10.0,
        slippage_bps_per_side=10.0,
        borrow_bps_per_year=0.0,
        short_available=False,
    ),
}


class TradePathScenarioEvaluateFixedSetJobParams(Frozen):
    schema_version: Literal["pbe.replay.trade-path-scenario-evaluate-fixed-set-job.v1"]
    transport_profile: Literal["hf_bucket_direct"] = "hf_bucket_direct"
    bucket_prefix: str | None = None


_SCENARIO_DEPENDENT_FIELDS = (
    "gross_pnl",
    "net_pnl",
    "entry_cost",
    "exit_cost",
    "borrow_cost",
    "cost",
    "status",
    "cancellation_reason",
)


def _compact_record_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in records:
        delta = {key: row[key] for key in _SCENARIO_DEPENDENT_FIELDS if key in row}
        delta["record_id"] = row["record_id"]
        compact.append(delta)
    compact.sort(key=lambda item: str(item["record_id"]))
    return compact


def execute_trade_path_scenario_evaluate_fixed_set(
    payload: dict[str, Any] | TradePathScenarioEvaluateRequest,
    *,
    encoded_size: int | None = None,
) -> dict[str, Any]:
    if isinstance(payload, TradePathScenarioEvaluateRequest):
        base_request = payload
    else:
        base_request = TradePathScenarioEvaluateRequest.model_validate(payload)
    if encoded_size is not None and encoded_size > TRADE_PATH_SCENARIO_MAX_INPUT_BYTES:
        raise TradePathScenarioEvaluateError("input byte limit exceeded")
    if len(base_request.records) > TRADE_PATH_SCENARIO_MAX_RECORDS:
        raise TradePathScenarioEvaluateError("record limit exceeded")
    if base_request.schema_version != REQUEST_SCHEMA_VERSION:
        raise TradePathScenarioEvaluateError("batch payload schema mismatch")

    scenario_payloads: dict[str, Any] = {}
    for name in FIXED_SCENARIO_NAMES:
        scen = _FIXED_SCENARIO_PARAMS[name]
        injected = base_request.model_copy(update={"scenario": scen})
        single = execute_trade_path_scenario_evaluate(
            injected.model_dump(mode="json"),
            encoded_size=encoded_size,
        )
        scenario_payloads[name] = {
            "scenario_name": name,
            "scenario": scen.model_dump(mode="json"),
            "event_summary": single["event_summary"],
            "content_identity": single["summary"]["content_identity"],
            "record_deltas": _compact_record_deltas(single["records"]),
        }

    body_obj = {
        "schema_version": FIXED_SET_RESULT_SCHEMA_VERSION,
        "batch_id": base_request.batch_id,
        "reference_notional": base_request.reference_notional,
        "base_content_identity": sha256(
            json.dumps(
                [record.model_dump(mode="json") for record in base_request.records],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "scenarios": scenario_payloads,
    }
    body = json.dumps(body_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity = f"sha256:{sha256(body).hexdigest()}"
    return {
        "schema_version": FIXED_SET_RESULT_SCHEMA_VERSION,
        "batch_id": base_request.batch_id,
        "summary": {
            "record_count": len(base_request.records),
            "scenario_count": len(FIXED_SCENARIO_NAMES),
            "content_identity": identity,
        },
        "result_json_bytes": body,
    }
