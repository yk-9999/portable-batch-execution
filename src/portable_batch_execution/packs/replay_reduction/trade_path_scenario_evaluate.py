"""Generic bounded trade-path scenario evaluation over normalized records."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from portable_batch_execution.contracts.models import Frozen

from .canonicalize import StructuralCanonicalizeError

RESULT_SCHEMA_VERSION = "pbe.replay.trade-path-scenario-evaluate-result.v1"
REQUEST_SCHEMA_VERSION = "pbe.replay.trade-path-scenario-evaluate.v1"
JOB_SCHEMA_VERSION = "pbe.replay.trade-path-scenario-evaluate-job.v1"

TRADE_PATH_SCENARIO_MAX_INPUT_BYTES = 134_217_728
TRADE_PATH_SCENARIO_MAX_RECORDS = 250_000

_FORBIDDEN_PARAM_KEYS = frozenset(
    {
        "shell",
        "command",
        "cmd",
        "python",
        "python_code",
        "script",
        "sql",
        "import_path",
        "entrypoint",
        "executable",
        "file_path",
        "path",
        "url",
        "uri",
        "callback",
    }
)
_SUSPICIOUS_STRING = re.compile(r"(://)|(\\)|(\.py\b)|(^/)", re.IGNORECASE)


class TradePathScenarioEvaluateError(StructuralCanonicalizeError):
    """Fail-closed trade-path scenario evaluation error."""


class TradePathScenarioParams(Frozen):
    commission_bps_per_side: float = Field(ge=0.0)
    slippage_bps_per_side: float = Field(ge=0.0)
    borrow_bps_per_year: float = Field(ge=0.0)
    short_available: bool = True


class NormalizedTradePathRecord(Frozen):
    record_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    window: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    month: str = Field(min_length=1)
    status: Literal["closed", "censored", "cancelled"]
    cancellation_reason: str | None = None
    reference_notional: float = Field(gt=0.0)
    price_pnl: float = 0.0
    dividend_pnl: float = 0.0
    gross_pnl: float = 0.0
    entry_notional: float = Field(ge=0.0)
    exit_notional: float = Field(ge=0.0)
    traded_notional: float = Field(ge=0.0)
    short_notional: float = Field(ge=0.0)
    holding_days: float = Field(ge=0.0)
    holding_sessions: int = Field(ge=0)
    mae: float = 0.0
    mfe: float = 0.0
    adv_ratio_20d: float | None = None
    adv_ratio_60d: float | None = None
    residual_market_beta: float | None = None
    residual_sector_beta: float | None = None

    @field_validator(
        "record_id",
        "model",
        "window",
        "group_label",
        "month",
        "cancellation_reason",
        mode="before",
    )
    @classmethod
    def _reject_suspicious_text(cls, value: object) -> object:
        if value is None:
            return value
        text = str(value)
        if _SUSPICIOUS_STRING.search(text):
            raise ValueError("forbidden record text")
        return text


class TradePathScenarioEvaluateRequest(Frozen):
    schema_version: Literal["pbe.replay.trade-path-scenario-evaluate.v1"]
    batch_id: str = Field(min_length=1)
    reference_notional: float = Field(gt=0.0)
    scenario: TradePathScenarioParams
    records: tuple[NormalizedTradePathRecord, ...] = Field(min_length=0)
    max_input_bytes: int = Field(
        default=TRADE_PATH_SCENARIO_MAX_INPUT_BYTES,
        ge=1,
        le=TRADE_PATH_SCENARIO_MAX_INPUT_BYTES,
    )
    max_records: int = Field(
        default=TRADE_PATH_SCENARIO_MAX_RECORDS,
        ge=1,
        le=TRADE_PATH_SCENARIO_MAX_RECORDS,
    )

    @model_validator(mode="after")
    def _bounds(self):
        if len(self.records) > self.max_records:
            raise ValueError("record limit exceeded")
        return self


def _finite(value: float, *, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TradePathScenarioEvaluateError(f"{label} invalid") from None
    if not math.isfinite(out):
        raise TradePathScenarioEvaluateError(f"{label} non-finite")
    return out


def _walk_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_PARAM_KEYS & set(value):
            raise TradePathScenarioEvaluateError("forbidden payload key")
        for item in value.values():
            _walk_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden(item)
    elif isinstance(value, str) and _SUSPICIOUS_STRING.search(value):
        raise TradePathScenarioEvaluateError("forbidden payload text")


def _validated_request(
    payload: dict[str, Any] | TradePathScenarioEvaluateRequest,
    *,
    encoded_size: int | None = None,
) -> TradePathScenarioEvaluateRequest:
    if isinstance(payload, TradePathScenarioEvaluateRequest):
        model = payload
    else:
        _walk_forbidden(payload)
        model = TradePathScenarioEvaluateRequest.model_validate(payload)
    if encoded_size is not None and encoded_size > model.max_input_bytes:
        raise TradePathScenarioEvaluateError("input byte limit exceeded")
    return model


def _breakeven_round_trip_bps(gross_pnl: float, traded_notional: float) -> float:
    if traded_notional <= 0:
        return float("nan")
    return 1e4 * gross_pnl / traded_notional


def _breakeven_borrow_bps_per_year(
    price_pnl: float,
    dividend_pnl: float,
    short_notional: float,
    holding_days: float,
) -> float:
    if short_notional <= 0 or holding_days <= 0:
        return float("nan")
    gross = price_pnl + dividend_pnl
    return 1e4 * gross / (short_notional * holding_days / 365.0)


def _aggregate_capacity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    import numpy as np

    def arr(key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)

    turn = arr("traded_notional")
    snd = arr("short_notional") * arr("holding_days") / 365.0
    adv20 = arr("adv_ratio_20d")
    adv60 = arr("adv_ratio_60d")
    hd = arr("holding_days")
    mae = arr("mae")
    mfe = arr("mfe")
    return {
        "n": len(rows),
        "turnover_total": float(np.nansum(turn)),
        "turnover_median": float(np.nanmedian(turn)),
        "short_notional_days_total": float(np.nansum(snd)),
        "adv_ratio_20d_median": float(np.nanmedian(adv20))
        if np.isfinite(adv20).any()
        else float("nan"),
        "adv_ratio_20d_p90": float(np.nanpercentile(adv20, 90))
        if np.isfinite(adv20).any()
        else float("nan"),
        "adv_ratio_60d_median": float(np.nanmedian(adv60))
        if np.isfinite(adv60).any()
        else float("nan"),
        "holding_days_median": float(np.nanmedian(hd))
        if np.isfinite(hd).any()
        else float("nan"),
        "mae_median": float(np.nanmedian(mae))
        if np.isfinite(mae).any()
        else float("nan"),
        "mfe_median": float(np.nanmedian(mfe))
        if np.isfinite(mfe).any()
        else float("nan"),
    }


def _summarise_residual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    diags = [
        r
        for r in rows
        if r.get("residual_market_beta") is not None
        or r.get("residual_sector_beta") is not None
    ]
    if not diags:
        return {"n": 0}
    mb = np.array([d.get("residual_market_beta", np.nan) for d in diags], dtype=float)
    sb = np.array([d.get("residual_sector_beta", np.nan) for d in diags], dtype=float)
    return {
        "n": len(diags),
        "market_beta_net_mean": float(np.nanmean(mb)),
        "market_beta_net_abs_median": float(np.nanmedian(np.abs(mb))),
        "sector_beta_net_mean": float(np.nanmean(sb)),
        "sector_beta_net_abs_median": float(np.nanmedian(np.abs(sb))),
    }


def _apply_scenario(
    record: NormalizedTradePathRecord, scen: TradePathScenarioParams
) -> dict[str, Any]:
    base = record.model_dump(mode="json")
    status = record.status
    reason = record.cancellation_reason
    if status == "closed" and not scen.short_available and record.short_notional > 0.0:
        status = "cancelled"
        reason = "short_unavailable"
    if status != "closed":
        return {
            **base,
            "status": status,
            "cancellation_reason": reason,
            "entry_cost": 0.0,
            "exit_cost": 0.0,
            "borrow_cost": 0.0,
            "cost": 0.0,
            "net_pnl": 0.0,
        }
    commission = float(scen.commission_bps_per_side) / 1e4
    slippage = float(scen.slippage_bps_per_side) / 1e4
    borrow = float(scen.borrow_bps_per_year) / 1e4
    entry_cost = record.entry_notional * (commission + slippage)
    exit_cost = record.exit_notional * (commission + slippage)
    borrow_cost = record.short_notional * borrow * (record.holding_days / 365.0)
    cost = entry_cost + exit_cost
    net = record.gross_pnl - borrow_cost - cost
    return {
        **base,
        "status": "closed",
        "cancellation_reason": None,
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "borrow_cost": borrow_cost,
        "cost": cost,
        "net_pnl": net,
    }


def _summarise_event(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rows = [r for r in records if r.get("status") == "closed"]
    cancelled: dict[str, int] = {}
    censored = 0
    for r in records:
        if r.get("status") == "cancelled":
            reason = r.get("cancellation_reason") or "cancelled"
            cancelled[reason] = cancelled.get(reason, 0) + 1
        elif r.get("status") == "censored":
            censored += 1
    gross = (
        np.array([r["gross_pnl"] for r in rows], dtype=float) if rows else np.array([])
    )
    net = np.array([r["net_pnl"] for r in rows], dtype=float) if rows else np.array([])
    per: dict[str, Any] = {}
    keys = sorted({(str(r["model"]), str(r["window"])) for r in rows})
    for model, window in keys:
        sub = [r for r in rows if r["model"] == model and r["window"] == window]
        g = np.array([r["gross_pnl"] for r in sub], dtype=float)
        per[f"{model}/{window}"] = {
            "n": len(sub),
            "gross_pnl_mean": float(g.mean()),
            "gross_pnl_sum": float(g.sum()),
            "hit_rate": float(np.mean(g > 0)),
            "holding_sessions_median": float(
                np.median([r["holding_sessions"] for r in sub])
            ),
        }
    return {
        "n": len(rows),
        "cancelled": cancelled,
        "censored_dev_unresolved": censored,
        "residual_exposure_diagnostics": _summarise_residual(rows),
        "gross_pnl_mean": float(gross.mean()) if gross.size else 0.0,
        "gross_pnl_sum": float(gross.sum()) if gross.size else 0.0,
        "net_pnl_mean": float(net.mean()) if net.size else 0.0,
        "hit_rate": float(np.mean(gross > 0)) if gross.size else 0.0,
        "capacity": _aggregate_capacity(rows),
        "breakeven_round_trip_bps_median": float(
            np.nanmedian(
                [
                    _breakeven_round_trip_bps(r["gross_pnl"], r["traded_notional"])
                    for r in rows
                ]
            )
        )
        if rows
        else float("nan"),
        "breakeven_borrow_bps_per_year_median": float(
            np.nanmedian(
                [
                    _breakeven_borrow_bps_per_year(
                        r["price_pnl"],
                        r["dividend_pnl"],
                        r["short_notional"],
                        r["holding_days"],
                    )
                    for r in rows
                ]
            )
        )
        if rows
        else float("nan"),
        "per_window_model": per,
    }


def execute_trade_path_scenario_evaluate(
    payload: dict[str, Any] | TradePathScenarioEvaluateRequest,
    *,
    encoded_size: int | None = None,
) -> dict[str, Any]:
    model = _validated_request(payload, encoded_size=encoded_size)
    scen = model.scenario
    evaluated = [_apply_scenario(record, scen) for record in model.records]
    evaluated.sort(key=lambda r: (str(r["record_id"]),))
    summary = _summarise_event(evaluated)
    body = json.dumps(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "batch_id": model.batch_id,
            "reference_notional": model.reference_notional,
            "scenario": scen.model_dump(mode="json"),
            "records": evaluated,
            "event_summary": summary,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = f"sha256:{sha256(body).hexdigest()}"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "batch_id": model.batch_id,
        "summary": {
            "record_count": len(evaluated),
            "closed_count": summary["n"],
            "content_identity": identity,
        },
        "records": evaluated,
        "event_summary": summary,
        "result_json_bytes": body,
    }
