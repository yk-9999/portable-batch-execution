"""Broker execution, polling, and durable request orchestration."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.backends.base import BackendExecutionRef
from portable_batch_execution.backends.github_actions import (
    GitHubActionsAPIError,
    GitHubActionsBackend,
)
from portable_batch_execution.contracts import PACK_OPS, ShardAttemptRecord, ShardSpec
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.kernel import completeness, exhausted_shards

from .config import BrokerConfig
from .planning import (
    broker_execution_fingerprint,
    broker_shard_input_digest,
    canonical_operation_params,
    opaque_run_id,
    opaque_wave_id,
    register_broker_private_run,
    validate_registered_wave_binding,
)
from .protocol import BrokerExecuteRequest, BrokerExecuteResponse, parse_request
from .state import BrokerRequestState, BrokerRequestStore, RequestBinding

_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled"})
_ALLOWED_BROKER_PACKS = frozenset(
    {"tabular-batch", "ml-batch", "media-batch", "replay-eval-batch"}
)
_BROKER_REPLAY_EVAL_OPERATION = "replay_eval.external_api_evaluation"


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
        if request.pack not in _ALLOWED_BROKER_PACKS:
            return self._failed(request.request_id, "operation_not_allowed")
        if request.pack == "replay-eval-batch":
            if request.operation != _BROKER_REPLAY_EVAL_OPERATION:
                return self._failed(request.request_id, "operation_not_allowed")
        elif request.operation not in PACK_OPS[request.pack]:
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
        static_input_refs = self.config.static_input_refs_for(
            request.pack, request.operation
        )
        input_digest = broker_shard_input_digest(input_bytes, static_input_refs)
        binding = RequestBinding(
            input_digest=input_digest,
            pack=request.pack,
            operation=request.operation,
            operation_params=validated_params,
            execution_fingerprint=broker_execution_fingerprint(
                request_id=request.request_id,
                input_digest=input_digest,
                pack=request.pack,
                operation=request.operation,
                operation_params=validated_params,
                public_sha=self.config.public_sha,
            ),
            public_sha=self.config.public_sha,
        )
        state = self._store.load(request.request_id)
        if state is None:
            run_id = opaque_run_id(request.request_id)
            wave_id = opaque_wave_id(request.request_id)
            try:
                self.controller.registry.resolve_wave(run_id, wave_id)
            except KeyError:
                try:
                    state = self._register_request(
                        request, binding, input_bytes, static_input_refs
                    )
                except ValueError:
                    return self._failed(request.request_id, "broker_internal_error")
            else:
                try:
                    state = self.recover_request_state(request, binding)
                except ValueError as exc:
                    if str(exc) == "request_binding_conflict":
                        return self._failed(request.request_id, "request_id_conflict")
                    raise
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
        static_input_refs,
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
            static_input_refs=static_input_refs,
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

    def _reconcile_if_attempts_changed(
        self, state: BrokerRequestState, shard: ShardSpec, attempts: list[ShardAttemptRecord]
    ) -> None:
        marker = _attempt_marker(shard, attempts)
        if marker == state.last_reconciled_attempt_marker:
            return
        if not _matching_attempts(shard, attempts):
            state.last_reconciled_attempt_marker = marker
            self._store.save(state)
            return
        manifest = self.controller.data_plane.read_manifest(state.logical_run_id)
        if manifest is None:
            return
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        if not shards:
            return
        self.controller.reconcile_run(state.logical_run_id)
        state.last_reconciled_attempt_marker = marker
        self._store.save(state)

    def _sync_dispatch_from_controller(
        self, state: BrokerRequestState, shard: ShardSpec
    ) -> None:
        history = self.controller.read_wave_dispatch_history(
            state.logical_run_id, state.wave_id
        )
        if not history:
            return
        recorded_count = len(history)
        if recorded_count <= state.dispatch_count:
            return
        attempts = list(self.controller.data_plane.read_attempts(state.logical_run_id))
        terminal_count = len(_matching_terminal_attempts(shard, attempts))
        state.dispatch_count = recorded_count
        state.execution_id = history[-1]["execution_id"]
        state.last_dispatched_failure_count = max(
            state.last_dispatched_failure_count,
            min(terminal_count, recorded_count),
        )
        self._store.save(state)

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
        max_dispatches = execution_policy.max_attempts_per_shard
        self._sync_dispatch_from_controller(state, shard)
        while True:
            attempts = list(self.controller.data_plane.read_attempts(state.logical_run_id))
            self._reconcile_if_attempts_changed(state, shard, attempts)
            terminal_count = len(_matching_terminal_attempts(shard, attempts))
            if terminal_count >= execution_policy.max_attempts_per_shard:
                state.status = "exhausted"
                self._store.save(state)
                return self._exhausted(state)
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
                self._reconcile_if_attempts_changed(state, shard, attempts)
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
            try:
                backend_status = self._backend_status(state)
            except GitHubActionsAPIError:
                return self._failed(request_id, "backend_transient")
            backend_terminal = backend_status in {"failed", "cancelled", "succeeded"}
            if backend_status == "running":
                self._sleep(self.poll_interval_seconds)
                continue
            if (
                state.dispatch_count > 0
                and terminal_count <= state.last_dispatched_failure_count
                and not backend_terminal
            ):
                self._sleep(self.poll_interval_seconds)
                continue
            if backend_terminal and state.dispatch_count >= max_dispatches:
                state.status = "exhausted"
                self._store.save(state)
                return self._exhausted(state)
            should_dispatch = (
                state.dispatch_count == 0
                or (
                    state.dispatch_count < max_dispatches
                    and (
                        terminal_count > state.last_dispatched_failure_count
                        or backend_terminal
                    )
                )
            )
            if not should_dispatch:
                self._sleep(self.poll_interval_seconds)
                continue
            try:
                execution = self.controller.dispatch_private_wave(
                    state.logical_run_id, state.wave_id
                )
            except GitHubActionsAPIError:
                return self._failed(request_id, "backend_transient")
            except ValueError:
                return self._failed(request_id, "dispatch_failed")
            state.execution_id = execution.execution_id
            state.last_dispatched_failure_count = terminal_count
            state.dispatch_count += 1
            self._store.save(state)
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
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        attempts = list(self.controller.data_plane.read_attempts(state.logical_run_id))
        if len(shards) == 1:
            self._reconcile_if_attempts_changed(state, shards[0], attempts)
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

    def recover_request_state(self, request: BrokerExecuteRequest, binding: RequestBinding) -> BrokerRequestState:
        """Rebuild durable broker state when registration exists but state file is missing."""
        from portable_batch_execution.contracts import JobSpec

        run_id = opaque_run_id(request.request_id)
        wave_id = opaque_wave_id(request.request_id)
        payload = self.controller.registry.resolve_wave(run_id, wave_id)
        job = JobSpec.model_validate(payload["job"])
        shard = ShardSpec.model_validate(payload["shards"][0])
        static_input_refs = self.config.static_input_refs_for(
            binding.pack, binding.operation
        )
        try:
            validate_registered_wave_binding(
                request_id=request.request_id,
                binding_input_digest=binding.input_digest,
                binding_pack=binding.pack,
                binding_operation=binding.operation,
                binding_operation_params=binding.operation_params,
                binding_execution_fingerprint=binding.execution_fingerprint,
                job=job,
                shard=shard,
                expected_static_input_refs=static_input_refs,
            )
        except ValueError as exc:
            if str(exc) == "request_binding_conflict":
                raise
            raise ValueError("request_binding_conflict") from exc
        state = BrokerRequestState(
            request_id=request.request_id,
            binding=binding,
            logical_run_id=run_id,
            wave_id=wave_id,
            shard_id=shard.shard_id,
            status="active",
        )
        self._sync_dispatch_from_controller(state, shard)
        self._store.save(state)
        return state


def _matching_attempts(
    shard: ShardSpec, attempts: list[ShardAttemptRecord]
) -> list[ShardAttemptRecord]:
    return [
        item
        for item in attempts
        if item.shard_id == shard.shard_id
        and item.input_digest == shard.input_digest
        and item.execution_fingerprint == shard.execution_fingerprint
    ]


def _attempt_marker(shard: ShardSpec, attempts: list[ShardAttemptRecord]) -> str:
    material = [
        (
            item.attempt_id,
            item.status,
            item.input_digest,
            item.execution_fingerprint,
            item.shard_id,
        )
        for item in _matching_attempts(shard, attempts)
    ]
    material.sort()
    return sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _matching_terminal_attempts(shard, attempts) -> list[ShardAttemptRecord]:
    return [
        item
        for item in attempts
        if item.shard_id == shard.shard_id
        and item.input_digest == shard.input_digest
        and item.execution_fingerprint == shard.execution_fingerprint
        and item.status in _TERMINAL_FAILURE_STATUSES
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
