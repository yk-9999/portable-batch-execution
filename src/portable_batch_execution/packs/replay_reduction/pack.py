"""Replay reduction pack: structural canonicalize and event-window extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonicalize import execute_structural_canonicalize
from .event_window import execute_event_window_extract
from .models import PARAM_MODELS


class ReplayReductionPack:
    pack_id = "replay-batch"
    supported_operations = tuple(PARAM_MODELS)

    def validate_params(self, operation: str, params: dict) -> dict:
        try:
            model = PARAM_MODELS[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported replay operation: {operation}") from exc
        return model.model_validate(params).model_dump(mode="json")

    def execute(self, job, shard, params, context):
        operation = getattr(job, "operation", None) or context["operation"]
        if operation == "replay.structural_canonicalize":
            paths = context.get("parquet_paths") or context.get("paths")
            if paths is None:
                raise TypeError("structural canonicalize requires parquet_paths")
            return execute_structural_canonicalize(paths, params)
        if operation == "replay.event_window_extract":
            records = context["records"]
            request = context["request"]
            return execute_event_window_extract(records, request)
        raise ValueError(f"unsupported replay operation: {operation}")

    def finalize(self, job, canonical_attempts, context):
        return canonical_attempts

    def run(
        self,
        operation: str,
        *,
        paths: list[str | Path] | None = None,
        records: list[dict[str, Any]] | None = None,
        request: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ):
        if operation == "replay.structural_canonicalize":
            return execute_structural_canonicalize(paths or [], params or {})
        if operation == "replay.event_window_extract":
            return execute_event_window_extract(records or [], request or {})
        raise ValueError(f"unsupported replay operation: {operation}")
