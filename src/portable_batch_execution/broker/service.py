"""Broker execution, polling, and durable request orchestration."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.backends.base import BackendExecutionRef
from portable_batch_execution.backends.github_actions import (
    GitHubActionsAPIError,
    GitHubActionsBackend,
)
from portable_batch_execution.contracts import PACK_OPS, ShardAttemptRecord
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.kernel import completeness, exhausted_shards

from .config import BrokerConfig
from .planning import (
    broker_execution_fingerprint,
    broker_input_digest,
    canonical_operation_params,
    register_broker_private_run,
)
from .protocol import BrokerExecuteRequest, BrokerExecuteResponse, parse_request
from .state import BrokerRequestState, BrokerRequestStore, RequestBinding


class UnixBrokerService:
    """Synchronous closed execution broker for local Unix clients."""

    def __init__(
        self,
        *,
        state_root: Path,
        config: BrokerConfig,
        controller: A1Controller,
        poll_interval_seconds: float = 1.0,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.state_root = state_root.resolve()
        self.config = config
        self.controller = controller
        self.poll_interval_seconds = poll_interval_seconds
        self._sleep = sleeper or time.sleep
        self._store = BrokerRequestStore(self.state_root / "controller")

    def handle_payload(self, peer_uid: int, payload: object) -> BrokerExecuteResponse:
        try:
            request = parse_request(payload)
        except (TypeError, ValueError):
            return BrokerExecuteResponse(
                request_id=_request_id_or_unknown(payload),
                status="failed",
                error_code="request_invalid",
            )
        if request.pack not in {"tabular-batch", "ml-batch", "media-batch"}:
            return self._failed(request.request_id, "operation_not_allowed")
        if request.operation not in PACK_OPS[request.pack]:
            return self._failed(request.request_id, "operation_not_allowed")
        if not self.config.authorize(peer_uid, request.pack, request.operation):
            return self._failed(request.request_id, "peer_not_authorized")
        try:
            input_bytes = request.decode_input(max_bytes=self.config.max_input_bytes)
        except ValueError:
            return self._failed(request.request_id, "input_invalid")
        try:
            validated_params = canonical_operation_params(
                request.pack, request.operation, request.operation_params
            )
        except ValueError:
            return self._failed(request.request_id, "operation_params_invalid")
        binding = RequestBinding(
            input_digest=broker_input_digest(input_bytes),
            pack=request.pack,
            operation=request.operation,
            operation_params=validated_params,
            execution_fingerprint=broker_execution_fingerprint(
                request_id=request.request_id,
                input_digest=broker_input_digest(input_bytes),
                pack=request.pack,
                operation=request.operation,
                operation_params=validated_params,
                public_sha=self.config.public_sha,
            ),
            public_sha=self.config.public_sha,
        )
        state = self._store.load(request.request_id)
        if state is None:
            state = self._register_request(request, binding, input_bytes)
        elif not state.binding.matches(binding):
            return self._failed(request.request_id, "request_id_conflict")
        if state.status == "succeeded":
            return self._success_from_state(state)
        if state.status == "exhausted":
            return self._exhausted(state)
        return self._drive_to_terminal(request.request_id, state)

    def _register_request(
        self,
        request: BrokerExecuteRequest,
        binding: RequestBinding,
        input_bytes: bytes,
    ) -> BrokerRequestState:
        _, wave, shard, _ = register_broker_private_run(
            state_root=self.state_root,
            request_id=request.request_id,
            pack=request.pack,
            operation=request.operation,
            operation_params=request.operation_params,
            input_bytes=input_bytes,
            input_media_type=request.input_media_type,
            public_sha=self.config.public_sha,
        )
        state = BrokerRequestState(
            request_id=request.request_id,
            binding=binding,
            logical_run_id=wave.logical_run_id,
            wave_id=wave.wave_id,
            shard_id=shard.shard_id,
            status="active",
        )
        self._store.save(state)
        return state

    def _drive_to_terminal(
        self, request_id: str, state: BrokerRequestState
    ) -> BrokerExecuteResponse:
        if self.controller.backend is None:
            return self._failed(request_id, "backend_unavailable")
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        if len(shards) != 1:
            return self._failed(request_id, "broker_internal_error")
        shard = shards[0]
        job_payload = self.controller.registry.resolve_wave(
            state.logical_run_id, state.wave_id
        )
        from portable_batch_execution.contracts import JobSpec

        execution_policy = JobSpec.model_validate(job_payload["job"]).execution
        while True:
            attempts = list(self.controller.data_plane.read_attempts(state.logical_run_id))
            exhausted = exhausted_shards((shard,), attempts, execution_policy)
            if exhausted:
                state.status = "exhausted"
                self._store.save(state)
                return self._exhausted(state)
            canonical, missing, duplicate = completeness(
                {shard.shard_id}, attempts, (shard,)
            )
            if canonical and not missing and not duplicate:
                attempt = canonical[0]
                output_bytes, media_type, digest = self._read_attempt_output(attempt)
                state.status = "succeeded"
                state.output_sha256 = digest
                state.output_media_type = media_type
                self._store.save(state)
                return BrokerExecuteResponse(
                    request_id=state.request_id,
                    status="succeeded",
                    logical_run_id=state.logical_run_id,
                    wave_id=state.wave_id,
                    execution_id=state.execution_id,
                    input_digest=state.binding.input_digest,
                    execution_fingerprint=state.binding.execution_fingerprint,
                    output_media_type=media_type,
                    output_sha256=digest,
                    output_b64=base64.b64encode(output_bytes).decode("ascii"),
                )
            matching_failures = _matching_failed_attempts(shard, attempts)
            if len(matching_failures) >= execution_policy.max_attempts_per_shard:
                state.status = "exhausted"
                self._store.save(state)
                return self._exhausted(state)
            backend_status = self._backend_status(state)
            if state.dispatch_count == 0:
                try:
                    execution = self.controller.dispatch_private_wave(
                        state.logical_run_id, state.wave_id
                    )
                except (ValueError, GitHubActionsAPIError):
                    return self._failed(request_id, "dispatch_failed")
                state.execution_id = execution.execution_id
                state.dispatch_count = 1
                self._store.save(state)
                self._sleep(self.poll_interval_seconds)
                continue
            if backend_status == "running":
                self._sleep(self.poll_interval_seconds)
                continue
            self._sleep(self.poll_interval_seconds)

    def _backend_status(self, state: BrokerRequestState) -> str | None:
        if state.execution_id is None or self.controller.backend is None:
            return None
        status = self.controller.backend.get_run(
            BackendExecutionRef("github-actions", state.execution_id, None)
        )
        return status.status

    def _read_attempt_output(
        self, attempt: ShardAttemptRecord
    ) -> tuple[bytes, str, str]:
        if not attempt.output_refs:
            raise ValueError("successful attempt missing output")
        ref = attempt.output_refs[0]
        payload = self.controller.data_plane.read(ref)
        digest = sha256(payload).hexdigest()
        media_type = ref.media_type or "application/octet-stream"
        return payload, media_type, f"sha256:{digest}"

    def _success_from_state(self, state: BrokerRequestState) -> BrokerExecuteResponse:
        attempts = self.controller.data_plane.read_attempts(state.logical_run_id)
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        canonical, missing, duplicate = completeness(
            {state.shard_id}, list(attempts), shards
        )
        if not canonical or missing or duplicate:
            return self._failed(state.request_id, "broker_internal_error")
        output_bytes, media_type, digest = self._read_attempt_output(canonical[0])
        return BrokerExecuteResponse(
            request_id=state.request_id,
            status="succeeded",
            logical_run_id=state.logical_run_id,
            wave_id=state.wave_id,
            execution_id=state.execution_id,
            input_digest=state.binding.input_digest,
            execution_fingerprint=state.binding.execution_fingerprint,
            output_media_type=media_type,
            output_sha256=digest,
            output_b64=base64.b64encode(output_bytes).decode("ascii"),
        )

    @staticmethod
    def _exhausted(state: BrokerRequestState) -> BrokerExecuteResponse:
        return BrokerExecuteResponse(
            request_id=state.request_id,
            status="exhausted",
            logical_run_id=state.logical_run_id,
            wave_id=state.wave_id,
            execution_id=state.execution_id,
            input_digest=state.binding.input_digest,
            execution_fingerprint=state.binding.execution_fingerprint,
            error_code="attempt_budget_exhausted",
        )

    @staticmethod
    def _failed(request_id: str, error_code: str) -> BrokerExecuteResponse:
        return BrokerExecuteResponse(
            request_id=request_id,
            status="failed",
            error_code=error_code,
        )


def _matching_failed_attempts(shard, attempts) -> list[ShardAttemptRecord]:
    return [
        item
        for item in attempts
        if item.shard_id == shard.shard_id
        and item.input_digest == shard.input_digest
        and item.execution_fingerprint == shard.execution_fingerprint
        and item.status == "failed"
    ]


def _request_id_or_unknown(payload: object) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("request_id"), str):
        return payload["request_id"]
    return "unknown"


def build_service_from_environment(
    *,
    state_root: Path,
    backend: GitHubActionsBackend | None,
    poll_interval_seconds: float,
) -> UnixBrokerService:
    config = BrokerConfig.load(Path(_required_env("PBE_BROKER_CONFIG")))
    controller = A1Controller(state_root, backend=backend)
    return UnixBrokerService(
        state_root=state_root,
        config=config,
        controller=controller,
        poll_interval_seconds=poll_interval_seconds,
    )


def _required_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value
