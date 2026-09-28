"""Reachability index from current recoverable authoritative state."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portable_batch_execution.broker.state import (
    BROKER_TERMINAL_REQUEST_STATUSES,
    BROKER_TRANSPORT_OUTPUT_STATUSES,
    BrokerRequestState,
)
from portable_batch_execution.contracts import (
    ArtifactRef,
    JobSpec,
    RunManifest,
    ShardAttemptRecord,
    ShardSpec,
)
from portable_batch_execution.controller.closed_wave_registry import safe_file_component
from portable_batch_execution.lifecycle.deletion import digest_hex
from portable_batch_execution.lifecycle.delivery import DeliveryRecord, DeliveryRecordStore
from portable_batch_execution.lifecycle.external_references import apply_policy_external_references
from portable_batch_execution.lifecycle.holds import HoldRecord
from portable_batch_execution.lifecycle.paths import deliveries_dir, holds_dir
from portable_batch_execution.lifecycle.policy import LifecyclePolicy

_TERMINAL_MANIFEST = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class ReachabilityIndex:
    protected_by_digest: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    unknown_digests: set[str] = field(default_factory=set)
    unknown_messages: list[str] = field(default_factory=list)
    broker_outputs: dict[str, str] = field(default_factory=dict)
    run_ids_by_digest: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    digest_last_reference_at: dict[str, datetime] = field(default_factory=dict)

    def is_protected(self, digest: str) -> bool:
        if digest in self.unknown_digests:
            return True
        return bool(self.protected_by_digest.get(digest))

    def protection_reasons(self, digest: str) -> set[str]:
        return set(self.protected_by_digest.get(digest, ()))


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _touch(index: ReachabilityIndex, digest: str, when: datetime | None) -> None:
    if when is None:
        return
    when = _coerce_utc(when)
    previous = index.digest_last_reference_at.get(digest)
    if previous is None or when > previous:
        index.digest_last_reference_at[digest] = when


def _artifact_digests_from_refs(refs: tuple[ArtifactRef, ...]) -> set[str]:
    return {ref.sha256 for ref in refs}


def _attempt_refs(record: ShardAttemptRecord) -> set[str]:
    if not record.output_refs:
        return set()
    return _artifact_digests_from_refs(record.output_refs)


def _load_attempt(path: Path) -> ShardAttemptRecord | None:
    try:
        return ShardAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unreadable attempt record: {path}") from exc


def _load_manifest(path: Path) -> RunManifest | None:
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unreadable manifest: {path}") from exc


def _load_deliveries(state_root: Path, index: ReachabilityIndex) -> tuple[DeliveryRecord, ...]:
    records: list[DeliveryRecord] = []
    root = deliveries_dir(state_root)
    if not root.is_dir():
        return ()
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("invalid delivery record")
            records.append(DeliveryRecord.from_json(payload))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            index.unknown_messages.append(f"delivery record unreadable: {path.name}: {exc}")
    return tuple(records)


def _load_holds(state_root: Path, index: ReachabilityIndex) -> tuple[HoldRecord, ...]:
    records: list[HoldRecord] = []
    root = holds_dir(state_root)
    if not root.is_dir():
        return ()
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("invalid hold record")
            records.append(HoldRecord.from_json(payload))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            index.unknown_messages.append(f"hold record unreadable: {path.name}: {exc}")
    return tuple(records)


def _scan_closed_waves(controller_root: Path, index: ReachabilityIndex) -> None:
    waves_root = controller_root / "closed_waves"
    if not waves_root.is_dir():
        return
    for run_dir in sorted(waves_root.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            safe_file_component(run_dir.name, "run_id")
        except ValueError:
            index.unknown_messages.append(f"invalid closed wave run directory: {run_dir.name}")
            continue
        for wave_path in sorted(run_dir.glob("*.wave.json")):
            try:
                payload = json.loads(wave_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                index.unknown_messages.append(f"closed wave unreadable: {wave_path.name}: {exc}")
                continue
            if not isinstance(payload, dict):
                index.unknown_messages.append(f"closed wave invalid: {wave_path.name}")
                continue
            try:
                job = JobSpec.model_validate(payload["job"])
            except (KeyError, ValueError, TypeError) as exc:
                index.unknown_messages.append(f"closed wave job invalid: {wave_path.name}: {exc}")
                continue
            run_id = job.logical_run_id
            when = _coerce_utc(job.provenance.created_at)
            digest = job.input_manifest_ref.sha256
            index.run_ids_by_digest[digest].add(run_id)
            _touch(index, digest, when)
            for shard_payload in payload.get("shards", ()):
                try:
                    shard = ShardSpec.model_validate(shard_payload)
                except (ValueError, TypeError) as exc:
                    index.unknown_messages.append(
                        f"closed wave shard invalid: {wave_path.name}: {exc}"
                    )
                    continue
                for ref in shard.input_refs:
                    index.run_ids_by_digest[ref.sha256].add(run_id)
                    _touch(index, ref.sha256, when)


def build_reachability_index(
    state_root: Path,
    policy: LifecyclePolicy | None = None,
) -> ReachabilityIndex:
    state_root = state_root.resolve()
    index = ReachabilityIndex()
    controller_root = state_root / "controller"

    deliveries = _load_deliveries(state_root, index)
    delivery_by_request = {record.request_id: record for record in deliveries}
    for record in deliveries:
        index.run_ids_by_digest[record.artifact_digest].add(record.logical_run_id)
        _touch(index, record.artifact_digest, record.delivered_at)

    broker_terminal_by_run: dict[str, str] = {}
    broker_runs: set[str] = set()
    for path in sorted((controller_root / "broker" / "requests").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = BrokerRequestState.from_json(payload)
            state_mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            index.unknown_messages.append(f"broker request state unreadable: {path.name}: {exc}")
            continue
        broker_runs.add(state.logical_run_id)
        if state.status not in BROKER_TERMINAL_REQUEST_STATUSES:
            index.protected_by_digest.setdefault("__broker_request__", set()).add(
                f"broker:{state.request_id}:{state.status}"
            )
            if state.output_sha256:
                index.protected_by_digest[state.output_sha256].add(
                    f"broker_request:{state.request_id}"
                )
        if state.output_sha256 and state.status in BROKER_TRANSPORT_OUTPUT_STATUSES:
            digest = state.output_sha256
            index.broker_outputs[state.request_id] = digest
            index.run_ids_by_digest[digest].add(state.logical_run_id)
            _touch(index, digest, state_mtime)
            if (
                state.status == "succeeded"
                and state.request_id not in delivery_by_request
            ):
                index.protected_by_digest.setdefault("__broker_request__", set()).add(
                    f"broker:{state.request_id}:{state.status}"
                )
                index.protected_by_digest[digest].add(f"broker_request:{state.request_id}")
        if state.status in BROKER_TERMINAL_REQUEST_STATUSES:
            broker_terminal_by_run[state.logical_run_id] = state.status

    for record in _load_holds(state_root, index):
        for digest in record.artifact_digests:
            index.protected_by_digest[digest].add(f"hold:{record.hold_id}")
            _touch(index, digest, record.created_at)

    runs_root = state_root / "runs"
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            try:
                safe_file_component(run_id, "run_id")
            except ValueError:
                index.unknown_messages.append(f"invalid run directory: {run_id}")
                continue
            manifest_path = run_dir / "latest.json"
            manifest: RunManifest | None = None
            if manifest_path.is_file():
                try:
                    manifest = _load_manifest(manifest_path)
                except ValueError as exc:
                    index.unknown_messages.append(str(exc))
                    index.unknown_digests.update(_digests_from_unreadable_run(run_dir))
                    continue
            broker_status = broker_terminal_by_run.get(run_id)
            if manifest is not None:
                live_manifest = manifest.status not in _TERMINAL_MANIFEST
                if (
                    live_manifest
                    and broker_status not in BROKER_TERMINAL_REQUEST_STATUSES
                ):
                    index.protected_by_digest.setdefault("__run__", set()).add(
                        f"manifest:{run_id}:{manifest.status}"
                    )
                manifest_when = max(
                    _coerce_utc(manifest.updated_at),
                    _coerce_utc(manifest.created_at),
                )
                terminal_manifest = manifest.status in _TERMINAL_MANIFEST
                broker_terminal = (
                    broker_status in BROKER_TERMINAL_REQUEST_STATUSES
                    if broker_status is not None
                    else False
                )
                broker_owned = run_id in broker_runs
                for ref in manifest.final_output_refs:
                    index.run_ids_by_digest[ref.sha256].add(run_id)
                    _touch(index, ref.sha256, manifest_when)
                    if not (
                        broker_owned
                        and terminal_manifest
                        and broker_terminal
                    ):
                        index.protected_by_digest[ref.sha256].add(
                            f"manifest_final:{run_id}"
                        )
            attempts_dir = run_dir / "attempts"
            if attempts_dir.is_dir():
                for attempt_path in sorted(attempts_dir.glob("*.json")):
                    try:
                        attempt = _load_attempt(attempt_path)
                    except ValueError as exc:
                        index.unknown_messages.append(str(exc))
                        continue
                    if attempt is None:
                        continue
                    attempt_when = max(
                        _coerce_utc(attempt.finished_at),
                        _coerce_utc(attempt.started_at),
                    )
                    for digest in _attempt_refs(attempt):
                        index.run_ids_by_digest[digest].add(run_id)
                        _touch(index, digest, attempt_when)
                        if (
                            manifest is not None
                            and manifest.status not in _TERMINAL_MANIFEST
                            and broker_status not in BROKER_TERMINAL_REQUEST_STATUSES
                        ):
                            index.protected_by_digest[digest].add(
                                f"attempt:{attempt.attempt_id}"
                            )

    _scan_closed_waves(controller_root, index)

    if policy is not None:
        apply_policy_external_references(index, policy)

    return index


def _digests_from_unreadable_run(run_dir: Path) -> set[str]:
    digests: set[str] = set()
    attempts_dir = run_dir / "attempts"
    if not attempts_dir.is_dir():
        return digests
    for attempt_path in attempts_dir.glob("*.json"):
        try:
            attempt = _load_attempt(attempt_path)
        except ValueError:
            continue
        if attempt is not None:
            digests.update(_attempt_refs(attempt))
    return digests


def artifact_payload_path(state_root: Path, artifact_digest: str) -> Path:
    return state_root / "artifacts" / digest_hex(artifact_digest)
