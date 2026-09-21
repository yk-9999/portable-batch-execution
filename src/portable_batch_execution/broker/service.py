"""Broker execution, polling, and durable request orchestration."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Union

from portable_batch_execution.backends.base import BackendExecutionRef
from portable_batch_execution.backends.github_actions import (
    GitHubActionsAPIError,
    GitHubActionsBackend,
)
from portable_batch_execution.contracts import ShardAttemptRecord, ShardSpec
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.kernel import completeness, exhausted_shards
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SET_OPERATION,
)
from portable_batch_execution.transport.hf_bucket import (
    artifact_ref_to_hf_bucket_ref,
    validate_hf_bucket_ref,
)
from portable_batch_execution.worker.hf_direct import artifact_ref_is_hf_bucket

from .artifact_reader import BrokerArtifactReader, build_broker_artifact_reader
from .config import BrokerConfig
from .hf_direct import (
    BrokerHfDirectExecuteRequest,
    BrokerHfDirectExecuteResponse,
    parse_hf_direct_request,
    reject_legacy_byte_fields,
)
from .planning import (
    broker_allows_operation,
    broker_execution_fingerprint,
    broker_shard_input_digest,
    canonical_operation_params,
    opaque_run_id,
    opaque_wave_id,
    register_broker_hf_direct_run,
    register_broker_private_run,
    validate_registered_wave_binding,
)
from .protocol import BrokerExecuteRequest, BrokerExecuteResponse, parse_request
from .state import BrokerRequestState, BrokerRequestStore, RequestBinding

_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled"})
_BrokerResponse = Union[BrokerExecuteResponse, BrokerHfDirectExecuteResponse]


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
        artifact_reader: BrokerArtifactReader | None = None,
    ):
        self.state_root = state_root.resolve()
        self.config = config
        self.controller = controller
        self.poll_interval_seconds = poll_interval_seconds
        self._sleep = sleeper or time.sleep
        self._store = BrokerRequestStore(self.state_root / "controller")
        self._artifact_reader = artifact_reader or build_broker_artifact_reader(
            controller.data_plane
        )

    def handle_payload(self, peer_uid: int, payload: object) -> _BrokerResponse:
        if isinstance(payload, dict) and payload.get("schema_version") == (
            "pbe.a1-unix-broker.hf-direct-request.v1"
        ):
            return self._handle_hf_direct_payload(peer_uid, payload)
        try:
            reject_legacy_byte_fields(payload)
            request = parse_request(payload)
        except (TypeError, ValueError):
            return BrokerExecuteResponse(
                request_id=_request_id_or_unknown(payload),
                status="failed",
                error_code="request_invalid",
            )
        if request.operation == FIXED_SET_OPERATION:
            return self._failed(request.request_id, "request_invalid")
        if not broker_allows_operation(request.pack, request.operation):
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
                    state = self.recover_request_state(request.request_id, binding)
                except ValueError as exc:
                    if str(exc) == "request_binding_conflict":
                        return self._failed(request.request_id, "request_id_conflict")
                    raise
        elif not state.binding.matches(binding):
            return self._failed(request.request_id, "request_id_conflict")
        if state.status == "succeeded":
            return self._success_from_state(state, hf_direct=False)
        if state.status == "exhausted":
            return self._exhausted(state, hf_direct=False)
        return self._drive_to_terminal(request.request_id, state, hf_direct=False)

    def _handle_hf_direct_payload(
        self, peer_uid: int, payload: object
    ) -> BrokerHfDirectExecuteResponse:
        try:
            reject_legacy_byte_fields(payload)
            request = parse_hf_direct_request(payload)
        except (TypeError, ValueError):
            return self._failed_hf(_request_id_or_unknown(payload), "request_invalid")
        if not broker_allows_operation(request.pack, request.operation):
            return self._failed_hf(request.request_id, "operation_not_allowed")
        if not self.config.authorize(peer_uid, request.pack, request.operation):
            return self._failed_hf(request.request_id, "peer_not_authorized")
        try:
            validated_params = canonical_operation_params(
                request.pack, request.operation, request.operation_params
            )
        except ValueError:
            return self._failed_hf(request.request_id, "operation_params_invalid")
        validated_ref = validate_hf_bucket_ref(request.input_hf_ref)
        input_bytes = json.dumps(
            validated_ref, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        input_digest = broker_shard_input_digest(input_bytes)
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
                    state = self._register_hf_direct_request(request, binding)
                except ValueError:
                    return self._failed_hf(request.request_id, "broker_internal_error")
            else:
                try:
                    state = self.recover_request_state(request.request_id, binding)
                except ValueError as exc:
                    if str(exc) == "request_binding_conflict":
                        return self._failed_hf(request.request_id, "request_id_conflict")
                    raise
        elif not state.binding.matches(binding):
            return self._failed_hf(request.request_id, "request_id_conflict")
        if state.status == "succeeded":
            return self._success_from_state(state, hf_direct=True)
        if state.status == "exhausted":
            return self._exhausted(state, hf_direct=True)
        return self._drive_to_terminal(request.request_id, state, hf_direct=True)

    def _register_hf_direct_request(
        self,
        request: BrokerHfDirectExecuteRequest,
        binding: RequestBinding,
    ) -> BrokerRequestState:
        _, wave, shard, _ = register_broker_hf_direct_run(
            state_root=self.state_root,
            request_id=request.request_id,
            pack=request.pack,
            operation=request.operation,
            operation_params=binding.operation_params,
            input_hf_ref=dict(request.input_hf_ref),
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
        self, request_id: str, state: BrokerRequestState, *, hf_direct: bool
    ) -> _BrokerResponse:
        if self.controller.backend is None:
            return (
                self._failed_hf(request_id, "backend_unavailable")
                if hf_direct
                else self._failed(request_id, "backend_unavailable")
            )
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        if len(shards) != 1:
            return (
                self._failed_hf(request_id, "broker_internal_error")
                if hf_direct
                else self._failed(request_id, "broker_internal_error")
            )
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
                return self._exhausted(state, hf_direct=hf_direct)
            exhausted = exhausted_shards((shard,), attempts, execution_policy)
            if exhausted:
                state.status = "exhausted"
                self._store.save(state)
                return self._exhausted(state, hf_direct=hf_direct)
            canonical, missing, duplicate = completeness(
                {shard.shard_id}, attempts, (shard,)
            )
            if canonical and not missing and not duplicate:
                attempt = canonical[0]
                state.status = "succeeded"
                self._store.save(state)
                self._reconcile_if_attempts_changed(state, shard, attempts)
                return self._success_from_attempt(state, attempt, hf_direct=hf_direct)
            try:
                backend_status = self._backend_status(state)
            except GitHubActionsAPIError:
                return (
                    self._failed_hf(request_id, "backend_transient")
                    if hf_direct
                    else self._failed(request_id, "backend_transient")
                )
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
                return self._exhausted(state, hf_direct=hf_direct)
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
                return (
                    self._failed_hf(request_id, "backend_transient")
                    if hf_direct
                    else self._failed(request_id, "backend_transient")
                )
            except ValueError:
                return (
                    self._failed_hf(request_id, "dispatch_failed")
                    if hf_direct
                    else self._failed(request_id, "dispatch_failed")
                )
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
        payload = self._artifact_reader.read(ref)
        digest = sha256(payload).hexdigest()
        media_type = ref.media_type or "application/octet-stream"
        return payload, media_type, f"sha256:{digest}"

    def _hf_output_metadata(
        self, attempt: ShardAttemptRecord
    ) -> tuple[dict[str, object], str]:
        if not attempt.output_refs:
            raise ValueError("successful attempt missing output")
        ref = attempt.output_refs[0]
        if not artifact_ref_is_hf_bucket(ref):
            raise ValueError("hf-direct output must be an HF bucket artifact ref")
        return artifact_ref_to_hf_bucket_ref(ref), ref.media_type or "application/json"

    def _success_from_attempt(
        self,
        state: BrokerRequestState,
        attempt: ShardAttemptRecord,
        *,
        hf_direct: bool,
    ) -> _BrokerResponse:
        if hf_direct:
            output_hf_ref, media_type = self._hf_output_metadata(attempt)
            state.output_media_type = media_type
            self._store.save(state)
            return BrokerHfDirectExecuteResponse(
                request_id=state.request_id,
                status="succeeded",
                logical_run_id=state.logical_run_id,
                wave_id=state.wave_id,
                execution_id=state.execution_id,
                input_digest=state.binding.input_digest,
                execution_fingerprint=state.binding.execution_fingerprint,
                output_media_type=media_type,
                output_hf_ref=output_hf_ref,
            )
        output_bytes, media_type, digest = self._read_attempt_output(attempt)
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

    def _success_from_state(
        self, state: BrokerRequestState, *, hf_direct: bool
    ) -> _BrokerResponse:
        shards = self.controller.registry.load_shards_for_run(state.logical_run_id)
        attempts = list(self.controller.data_plane.read_attempts(state.logical_run_id))
        if len(shards) == 1:
            self._reconcile_if_attempts_changed(state, shards[0], attempts)
        canonical, missing, duplicate = completeness(
            {state.shard_id}, list(attempts), shards
        )
        if not canonical or missing or duplicate:
            return (
                self._failed_hf(state.request_id, "broker_internal_error")
                if hf_direct
                else self._failed(state.request_id, "broker_internal_error")
            )
        return self._success_from_attempt(
            state, canonical[0], hf_direct=hf_direct
        )

    @staticmethod
    def _exhausted(
        state: BrokerRequestState, *, hf_direct: bool
    ) -> _BrokerResponse:
        if hf_direct:
            return BrokerHfDirectExecuteResponse(
                request_id=state.request_id,
                status="exhausted",
                logical_run_id=state.logical_run_id,
                wave_id=state.wave_id,
                execution_id=state.execution_id,
                input_digest=state.binding.input_digest,
                execution_fingerprint=state.binding.execution_fingerprint,
                error_code="attempt_budget_exhausted",
            )
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

    @staticmethod
    def _failed_hf(request_id: str, error_code: str) -> BrokerHfDirectExecuteResponse:
        return BrokerHfDirectExecuteResponse(
            request_id=request_id,
            status="failed",
            error_code=error_code,
        )

    def recover_request_state(
        self, request_id: str, binding: RequestBinding
    ) -> BrokerRequestState:
        """Rebuild durable broker state when registration exists but state file is missing."""
        from portable_batch_execution.contracts import JobSpec

        run_id = opaque_run_id(request_id)
        wave_id = opaque_wave_id(request_id)
        payload = self.controller.registry.resolve_wave(run_id, wave_id)
        job = JobSpec.model_validate(payload["job"])
        shard = ShardSpec.model_validate(payload["shards"][0])
        static_input_refs = self.config.static_input_refs_for(
            binding.pack, binding.operation
        )
        try:
            validate_registered_wave_binding(
                request_id=request_id,
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
            request_id=request_id,
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
    artifact_read_base_url: str | None = None,
    artifact_read_token_file: Path | None = None,
) -> UnixBrokerService:
    config = BrokerConfig.load(Path(_required_env("PBE_BROKER_CONFIG")))
    controller = A1Controller(state_root, backend=backend)
    artifact_reader = build_broker_artifact_reader(
        controller.data_plane,
        artifact_read_base_url=artifact_read_base_url,
        artifact_read_token_file=artifact_read_token_file,
    )
    return UnixBrokerService(
        state_root=state_root,
        config=config,
        controller=controller,
        poll_interval_seconds=poll_interval_seconds,
        artifact_reader=artifact_reader,
    )


def _required_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value
