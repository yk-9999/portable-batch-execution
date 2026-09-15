"""Replay evaluation pack boundary.

The pack deliberately does not interpret replay inputs or compute evaluation
metrics.  Those meanings are project-specific and consequently remain behind a
trusted :class:`ProjectAdapter`.  This module only exposes the portable, closed
operation names and enforces the network boundary for external evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from portable_batch_execution.adapters.base import ProjectAdapter


class ReplayEvalPack:
    """Closed replay/evaluation operations for a single project adapter.

    An adapter is required at construction time so there is no import-path,
    callback, or JobSpec-provided executable dispatch path.  The adapter owns
    all replay, benchmark, and external-service semantics.
    """

    pack_id = "replay-eval-batch"
    supported_operations = (
        "replay_eval.replay",
        "replay_eval.benchmark",
        "replay_eval.compare",
        "replay_eval.parameter_sweep",
        "replay_eval.regression",
        "replay_eval.walk_forward",
        "replay_eval.oos",
        "replay_eval.backtest",
        "replay_eval.control",
        "replay_eval.robustness",
        "replay_eval.external_api_evaluation",
    )

    _RESERVED_PARAMETER_KEYS = frozenset(
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
            "callback",
            "callable",
            "function",
            "handler",
        }
    )

    def __init__(self, adapter: ProjectAdapter):
        self._adapter = adapter

    def validate_params(self, operation: str, params: dict) -> dict:
        """Accept JSON-only data for a known operation, never executable hooks."""
        self._require_operation(operation)
        if not isinstance(params, dict):
            raise TypeError("replay evaluation parameters must be a dictionary")
        self._validate_json(params)
        return deepcopy(params)

    def validate_job(self, job: object) -> None:
        """Validate portable pack and security boundaries before adapter dispatch."""
        if getattr(job, "pack", None) != self.pack_id:
            raise ValueError("job does not belong to replay-eval-batch")
        operation = getattr(job, "operation", None)
        self._require_operation(operation)
        profile = getattr(job, "security_profile", None)
        if operation == "replay_eval.external_api_evaluation":
            if profile != "external-api":
                raise ValueError("external API evaluation requires external-api security")
        elif profile == "external-api":
            raise ValueError("external-api security is limited to external API evaluation")
        if operation not in self._adapter.descriptor.supported_operations:
            raise ValueError("adapter does not support replay evaluation operation")
        self._adapter.validate_job(job)

    def execute(self, job: object, shard: object, params: dict, context: object):
        """Delegate project semantics and attempt creation to the trusted adapter."""
        self.validate_job(job)
        validated = self.validate_params(getattr(job, "operation", None), params)
        return self._adapter.execute(job, shard, validated, context)

    def finalize(self, job: object, canonical_attempts: tuple, context: object):
        """Delegate project-specific aggregation to the trusted adapter."""
        self.validate_job(job)
        return self._adapter.finalize(job, canonical_attempts, context)

    def _require_operation(self, operation: object) -> None:
        if operation not in self.supported_operations:
            raise ValueError("unsupported replay evaluation operation")

    @classmethod
    def _validate_json(cls, value: Any) -> None:
        if isinstance(value, dict):
            forbidden = cls._RESERVED_PARAMETER_KEYS.intersection(value)
            if forbidden:
                raise ValueError("executable replay evaluation parameters are forbidden")
            for item in value.values():
                cls._validate_json(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._validate_json(item)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("replay evaluation parameters must be JSON compatible")
        elif isinstance(value, float) and not isfinite(value):
            raise ValueError("replay evaluation numeric parameters must be finite")
