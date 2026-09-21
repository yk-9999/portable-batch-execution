from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from portable_batch_execution.contracts import AdapterDescriptor
from portable_batch_execution.packs.replay_eval import ReplayEvalPack

FIXTURE = (
    Path(__file__).parents[4]
    / "fixtures"
    / "public"
    / "replay_eval"
    / "operations.json"
)


class FakeProjectAdapter:
    """Test-only adapter proving the pack never supplies project semantics."""

    def __init__(self, operations: tuple[str, ...]):
        self._descriptor = AdapterDescriptor(
            adapter_id="test-replay-adapter",
            adapter_version="1",
            adapter_digest="sha256:" + "a" * 64,
            supported_operations=operations,
        )
        self.calls: list[tuple] = []

    @property
    def descriptor(self):
        return self._descriptor

    def validate_job(self, job):
        self.calls.append(("validate_job", job.operation))

    def execute(self, job, shard, params, context):
        self.calls.append(("execute", job.operation, shard, params, context))
        return {"adapter": "attempt", "operation": job.operation}

    def finalize(self, job, canonical_attempts, context):
        self.calls.append(("finalize", job.operation, canonical_attempts, context))
        return {"adapter": "final", "attempts": len(canonical_attempts)}


def _job(operation: str, profile: str = "offline"):
    return SimpleNamespace(
        pack="replay-eval-batch", operation=operation, security_profile=profile
    )


def _pack() -> tuple[ReplayEvalPack, FakeProjectAdapter]:
    operations = tuple(json.loads(FIXTURE.read_text())["closed_operations"])
    adapter = FakeProjectAdapter(operations)
    return ReplayEvalPack(adapter), adapter


@pytest.mark.parametrize(
    "operation", json.loads(FIXTURE.read_text())["closed_operations"]
)
def test_every_closed_operation_delegates_replay_evaluation(operation):
    pack, adapter = _pack()
    profile = (
        "external-api" if operation.endswith("external_api_evaluation") else "offline"
    )

    result = pack.execute(
        _job(operation, profile), "shard-0", {"scenario": "public"}, {"x": 1}
    )

    assert result == {"adapter": "attempt", "operation": operation}
    assert adapter.calls[-1] == (
        "execute",
        operation,
        "shard-0",
        {"scenario": "public"},
        {"x": 1},
    )


def test_compare_benchmark_sweep_walk_forward_oos_and_backtest_remain_adapter_owned():
    pack, adapter = _pack()
    operations = (
        "replay_eval.benchmark",
        "replay_eval.compare",
        "replay_eval.parameter_sweep",
        "replay_eval.walk_forward",
        "replay_eval.oos",
        "replay_eval.backtest",
        "replay_eval.control",
        "replay_eval.robustness",
    )

    for operation in operations:
        assert pack.execute(_job(operation), "s", {}, None)["adapter"] == "attempt"

    assert [call[1] for call in adapter.calls if call[0] == "execute"] == list(
        operations
    )


def test_external_api_operation_is_the_only_external_api_boundary():
    pack, _ = _pack()

    pack.execute(
        _job("replay_eval.external_api_evaluation", "external-api"), "s", {}, None
    )
    with pytest.raises(ValueError, match="requires external-api"):
        pack.execute(_job("replay_eval.external_api_evaluation"), "s", {}, None)
    with pytest.raises(ValueError, match="limited"):
        pack.execute(_job("replay_eval.backtest", "external-api"), "s", {}, None)


def test_finalize_is_delegated_after_boundary_validation():
    pack, adapter = _pack()

    assert pack.finalize(
        _job("replay_eval.replay"), ("attempt-a",), {"summary": True}
    ) == {"adapter": "final", "attempts": 1}
    assert adapter.calls[-1] == (
        "finalize",
        "replay_eval.replay",
        ("attempt-a",),
        {"summary": True},
    )


@pytest.mark.parametrize(
    "params",
    [
        {"callback": "module.call"},
        {"nested": {"import_path": "module"}},
        {"nested": {"handler": "run"}},
        {"n": float("nan")},
    ],
)
def test_params_cannot_carry_import_or_callback_dispatch(params):
    pack, _ = _pack()

    with pytest.raises(ValueError):
        pack.validate_params("replay_eval.replay", params)


def test_unknown_operation_and_adapter_capability_are_rejected():
    pack, _ = _pack()
    with pytest.raises(ValueError, match="unsupported"):
        pack.validate_params("replay_eval.unknown", {})

    adapter = FakeProjectAdapter(("replay_eval.replay",))
    with pytest.raises(ValueError, match="adapter does not support"):
        ReplayEvalPack(adapter).execute(_job("replay_eval.backtest"), "s", {}, None)
