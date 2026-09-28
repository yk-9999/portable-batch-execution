"""Mark-and-sweep garbage collection for artifact payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from portable_batch_execution.lifecycle.deletion import (
    DeletionIntent,
    DeletionIntentStore,
    DeletionReceipt,
    DeletionReceiptStore,
)
from portable_batch_execution.lifecycle.delivery import DeliveryRecordStore
from portable_batch_execution.lifecycle.paths import artifacts_dir
from portable_batch_execution.lifecycle.policy import LifecyclePolicy
from portable_batch_execution.broker.state import (
    BROKER_TRANSPORT_OUTPUT_STATUSES,
    BrokerRequestState,
)
from portable_batch_execution.lifecycle.lock import lifecycle_state_lock
from portable_batch_execution.lifecycle.reachability import (
    ReachabilityIndex,
    artifact_payload_path,
    build_reachability_index,
)

GcMode = Literal["normal", "legacy"]


@dataclass
class GcProtection:
    digest: str
    size_bytes: int
    reasons: tuple[str, ...]


@dataclass
class GcCandidate:
    digest: str
    size_bytes: int
    payload_mtime: datetime | None = None
    effective_last_use: datetime | None = None
    effective_age_seconds: float | None = None


@dataclass
class GcReport:
    mode: GcMode
    artifact_count: int
    artifact_bytes: int
    candidate_count: int
    candidate_bytes: int
    protected: tuple[GcProtection, ...] = ()
    candidates: tuple[GcCandidate, ...] = ()
    deleted: tuple[GcCandidate, ...] = ()
    unknown_references: tuple[str, ...] = ()
    run_reference_pairs: tuple[tuple[str, str], ...] = ()
    blocked: bool = False
    planned_at: datetime | None = None

    def to_dict(self) -> dict:
        return self.to_review_dict()

    def to_review_dict(self) -> dict:
        def _iso(value: datetime | None) -> str | None:
            if value is None:
                return None
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.isoformat()

        sorted_candidates = sorted(self.candidates, key=lambda item: item.digest)
        sorted_protected = sorted(self.protected, key=lambda item: item.digest)
        reason_counts: dict[str, int] = {}
        reason_bytes: dict[str, int] = {}
        for item in sorted_protected:
            for reason in item.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                reason_bytes[reason] = reason_bytes.get(reason, 0) + item.size_bytes
        protection_reason_aggregates = [
            {
                "reason": reason,
                "count": reason_counts[reason],
                "bytes": reason_bytes[reason],
            }
            for reason in sorted(reason_counts)
        ]

        return {
            "mode": self.mode,
            "planned_at": _iso(self.planned_at),
            "artifact_count": self.artifact_count,
            "artifact_bytes": self.artifact_bytes,
            "candidate_count": self.candidate_count,
            "candidate_bytes": self.candidate_bytes,
            "protected_count": len(sorted_protected),
            "protected_bytes": sum(item.size_bytes for item in sorted_protected),
            "deleted_count": len(self.deleted),
            "blocked": self.blocked,
            "unknown_references": sorted(self.unknown_references),
            "protection_reason_aggregates": protection_reason_aggregates,
            "candidates": [
                {
                    "digest": item.digest,
                    "size_bytes": item.size_bytes,
                    "payload_mtime": _iso(item.payload_mtime),
                    "effective_last_use": _iso(item.effective_last_use),
                    "effective_age_seconds": item.effective_age_seconds,
                }
                for item in sorted_candidates
            ],
            "protected": [
                {
                    "digest": item.digest,
                    "size_bytes": item.size_bytes,
                    "reasons": sorted(item.reasons),
                }
                for item in sorted_protected
            ],
            "run_reference_pairs": [
                {"logical_run_id": run_id, "artifact_digest": digest}
                for run_id, digest in sorted(self.run_reference_pairs)
            ],
            "deleted": [
                {"digest": item.digest, "size_bytes": item.size_bytes}
                for item in sorted(self.deleted, key=lambda item: item.digest)
            ],
        }


@dataclass
class _SweepState:
    state_root: Path
    index: ReachabilityIndex
    delivery_by_request: dict
    policy: LifecyclePolicy
    now: datetime
    mode: GcMode
    receipt_store: DeletionReceiptStore
    protected: list[GcProtection] = field(default_factory=list)
    candidates: list[GcCandidate] = field(default_factory=list)


def _policy_identity(policy: LifecyclePolicy) -> str:
    files = ",".join(policy.authoritative_reference_files)
    roots = ",".join(policy.authoritative_reference_roots)
    return (
        f"{policy.schema_version}:grace={policy.delivery_grace_seconds}:"
        f"legacy={policy.legacy_retention_seconds}:refs={files}:roots={roots}"
    )


def plan_gc(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    now: datetime | None = None,
) -> GcReport:
    state_root = state_root.resolve()
    now = now or datetime.now(UTC)
    index = build_reachability_index(state_root, policy)
    delivery_store = DeliveryRecordStore(state_root)
    deliveries = delivery_store.list_all()
    delivery_by_request = {item.request_id: item for item in deliveries}

    artifact_paths = sorted(artifacts_dir(state_root).glob("*"))
    artifact_paths = [p for p in artifact_paths if p.is_file() and not p.name.endswith(".tmp")]

    total_bytes = sum(p.stat().st_size for p in artifact_paths)
    sweep = _SweepState(
        state_root=state_root,
        index=index,
        delivery_by_request=delivery_by_request,
        policy=policy,
        now=now,
        mode=mode,
        receipt_store=DeletionReceiptStore(state_root),
    )

    blocked = bool(index.unknown_messages)
    run_pairs: set[tuple[str, str]] = set()
    for digest, runs in index.run_ids_by_digest.items():
        for run_id in runs:
            run_pairs.add((run_id, digest))

    for path in artifact_paths:
        digest = f"sha256:{path.name}"
        size = path.stat().st_size
        if digest in index.unknown_digests:
            blocked = True
            sweep.protected.append(
                GcProtection(
                    digest=digest,
                    size_bytes=size,
                    reasons=("unknown_authoritative_state",),
                )
            )
            continue
        if index.is_protected(digest):
            sweep.protected.append(
                GcProtection(
                    digest=digest,
                    size_bytes=size,
                    reasons=tuple(sorted(index.protection_reasons(digest))),
                )
            )
            continue
        if sweep.receipt_store.load(digest) is not None:
            continue
        if mode == "normal":
            if _normal_eligible(state_root, sweep, digest, path):
                sweep.candidates.append(
                    _candidate_detail(sweep, digest=digest, size_bytes=size, path=path)
                )
            else:
                sweep.protected.append(
                    GcProtection(
                        digest=digest,
                        size_bytes=size,
                        reasons=_normal_protection_reasons(state_root, sweep, digest),
                    )
                )
        elif _legacy_eligible(state_root, sweep, digest, path):
            sweep.candidates.append(
                _candidate_detail(sweep, digest=digest, size_bytes=size, path=path)
            )
        else:
            sweep.protected.append(
                GcProtection(
                    digest=digest,
                    size_bytes=size,
                    reasons=_legacy_protection_reasons(state_root, sweep, digest, path),
                )
            )

    return GcReport(
        mode=mode,
        artifact_count=len(artifact_paths),
        artifact_bytes=total_bytes,
        candidate_count=len(sweep.candidates),
        candidate_bytes=sum(item.size_bytes for item in sweep.candidates),
        protected=tuple(sweep.protected),
        candidates=tuple(sweep.candidates),
        unknown_references=tuple(index.unknown_messages),
        run_reference_pairs=tuple(sorted(run_pairs)),
        blocked=blocked,
        planned_at=now,
    )


def _candidate_detail(
    sweep: _SweepState, *, digest: str, size_bytes: int, path: Path
) -> GcCandidate:
    payload_mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    effective_last_use = (
        _legacy_effective_last_use(sweep, digest, path)
        if sweep.mode == "legacy"
        else payload_mtime
    )
    age_seconds = (sweep.now - effective_last_use).total_seconds()
    if effective_last_use.tzinfo is None:
        effective_last_use = effective_last_use.replace(tzinfo=UTC)
    return GcCandidate(
        digest=digest,
        size_bytes=size_bytes,
        payload_mtime=payload_mtime,
        effective_last_use=effective_last_use,
        effective_age_seconds=age_seconds,
    )


def apply_gc(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    now: datetime | None = None,
    crash_after_unlink: set[str] | None = None,
) -> GcReport:
    """Apply planned deletions; optional crash simulation for tests."""
    report = plan_gc(state_root, policy, mode=mode, now=now)
    if report.blocked:
        return report
    now = now or datetime.now(UTC)
    receipt_store = DeletionReceiptStore(state_root)
    intent_store = DeletionIntentStore(state_root)
    delivery_store = DeliveryRecordStore(state_root)
    deleted: list[GcCandidate] = []
    with lifecycle_state_lock(state_root):
        deleted.extend(
            _converge_pending_deletion_intents(
                state_root,
                policy,
                mode=mode,
                now=now,
                receipt_store=receipt_store,
                intent_store=intent_store,
                delivery_store=delivery_store,
            )
        )
        deleted.extend(
            _apply_gc_candidates(
                state_root,
                policy,
                mode=mode,
                now=now,
                candidates=report.candidates,
                receipt_store=receipt_store,
                intent_store=intent_store,
                delivery_store=delivery_store,
                crash_after_unlink=crash_after_unlink,
            )
        )
        deleted.extend(
            _converge_receipts_for_missing_payloads(
                state_root,
                policy,
                mode=mode,
                now=now,
                receipt_store=receipt_store,
                delivery_store=delivery_store,
            )
        )
    report.deleted = tuple(deleted)
    return report


def _apply_gc_candidates(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    now: datetime,
    candidates: tuple[GcCandidate, ...],
    receipt_store: DeletionReceiptStore,
    intent_store: DeletionIntentStore,
    delivery_store: DeliveryRecordStore,
    crash_after_unlink: set[str] | None,
) -> list[GcCandidate]:
    deleted: list[GcCandidate] = []
    for candidate in candidates:
        digest = candidate.digest
        if receipt_store.load(digest) is not None:
            continue
        if intent_store.load(digest) is not None:
            continue
        fresh = build_reachability_index(state_root, policy)
        if fresh.is_protected(digest) or digest in fresh.unknown_digests:
            continue
        if fresh.unknown_messages:
            continue
        path = artifact_payload_path(state_root, digest)
        if not path.is_file():
            receipt = _receipt_for_digest(
                state_root,
                digest,
                candidate.size_bytes,
                mode=mode,
                policy=policy,
                now=now,
                delivery_store=delivery_store,
                path=path,
            )
            receipt_store.save(receipt)
            deleted.append(candidate)
            continue
        if not _still_eligible(state_root, policy, mode=mode, digest=digest, now=now):
            continue
        intent = _deletion_intent_for_digest(
            state_root,
            digest,
            candidate.size_bytes,
            mode=mode,
            policy=policy,
            now=now,
            delivery_store=delivery_store,
            path=path,
        )
        intent_store.save(intent)
        path.unlink(missing_ok=True)
        if crash_after_unlink and digest in crash_after_unlink:
            raise RuntimeError("simulated crash during gc sweep")
        receipt = intent.to_receipt(now)
        receipt_store.save(receipt)
        intent_store.finalize(digest)
        deleted.append(candidate)
    return deleted


def _converge_pending_deletion_intents(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    now: datetime,
    receipt_store: DeletionReceiptStore,
    intent_store: DeletionIntentStore,
    delivery_store: DeliveryRecordStore,
) -> list[GcCandidate]:
    converged: list[GcCandidate] = []
    for intent in intent_store.list_all():
        digest = intent.artifact_digest
        if receipt_store.load(digest) is not None:
            intent_store.finalize(digest)
            continue
        path = artifact_payload_path(state_root, digest)
        if not path.is_file():
            receipt = intent.to_receipt(now)
            receipt_store.save(receipt)
            intent_store.finalize(digest)
            converged.append(
                GcCandidate(digest=digest, size_bytes=intent.artifact_size_bytes)
            )
            continue
        fresh = build_reachability_index(state_root, policy)
        if fresh.is_protected(digest) or digest in fresh.unknown_digests:
            intent_store.finalize(digest)
            continue
        if fresh.unknown_messages:
            continue
        if not _still_eligible(
            state_root, policy, mode=intent.mode, digest=digest, now=now
        ):
            intent_store.finalize(digest)
            continue
        path.unlink(missing_ok=True)
        receipt = intent.to_receipt(now)
        receipt_store.save(receipt)
        intent_store.finalize(digest)
        converged.append(GcCandidate(digest=digest, size_bytes=intent.artifact_size_bytes))
    return converged


def _still_eligible(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    digest: str,
    now: datetime,
) -> bool:
    delivery_store = DeliveryRecordStore(state_root)
    sweep = _SweepState(
        state_root=state_root,
        index=build_reachability_index(state_root, policy),
        delivery_by_request={item.request_id: item for item in delivery_store.list_all()},
        policy=policy,
        now=now,
        mode=mode,
        receipt_store=DeletionReceiptStore(state_root),
    )
    path = artifact_payload_path(state_root, digest)
    if mode == "normal":
        return _normal_eligible(state_root, sweep, digest, path)
    return _legacy_eligible(state_root, sweep, digest, path)


def _converge_receipts_for_missing_payloads(
    state_root: Path,
    policy: LifecyclePolicy,
    *,
    mode: GcMode,
    now: datetime,
    receipt_store: DeletionReceiptStore,
    delivery_store: DeliveryRecordStore,
) -> list[GcCandidate]:
    converged: list[GcCandidate] = []
    sweep = _SweepState(
        state_root=state_root,
        index=build_reachability_index(state_root, policy),
        delivery_by_request={item.request_id: item for item in delivery_store.list_all()},
        policy=policy,
        now=now,
        mode=mode,
        receipt_store=receipt_store,
    )
    seen: set[str] = set()
    for record in delivery_store.list_all():
        digest = record.artifact_digest
        if digest in seen or receipt_store.load(digest) is not None:
            continue
        seen.add(digest)
        path = artifact_payload_path(state_root, digest)
        if path.is_file():
            continue
        if sweep.index.is_protected(digest):
            continue
        if mode == "normal" and not _normal_eligible(state_root, sweep, digest, path):
            continue
        if mode == "legacy" and not _legacy_eligible(state_root, sweep, digest, path):
            continue
        size = record.artifact_size_bytes
        receipt = _receipt_for_digest(
            state_root,
            digest,
            size,
            mode=mode,
            policy=policy,
            now=now,
            delivery_store=delivery_store,
            path=path,
        )
        receipt_store.save(receipt)
        converged.append(GcCandidate(digest=digest, size_bytes=size))
    return converged


def _deletion_intent_for_digest(
    state_root: Path,
    digest: str,
    size_bytes: int,
    *,
    mode: GcMode,
    policy: LifecyclePolicy,
    now: datetime,
    delivery_store: DeliveryRecordStore,
    path: Path,
) -> DeletionIntent:
    metadata = _deletion_metadata_for_digest(
        state_root,
        digest,
        size_bytes,
        mode=mode,
        policy=policy,
        delivery_store=delivery_store,
        path=path,
    )
    return DeletionIntent(
        artifact_digest=digest,
        artifact_size_bytes=size_bytes,
        logical_run_ids=metadata["logical_run_ids"],
        producer_uid=metadata["producer_uid"],
        consumer_uid=metadata["consumer_uid"],
        provenance=metadata["provenance"],
        created_at=metadata["created_at"],
        delivered_at=metadata["delivered_at"],
        intended_at=now,
        mode=mode,
        policy_identity=_policy_identity(policy),
    )


def _deletion_metadata_for_digest(
    state_root: Path,
    digest: str,
    size_bytes: int,
    *,
    mode: GcMode,
    policy: LifecyclePolicy,
    delivery_store: DeliveryRecordStore,
    path: Path,
) -> dict:
    index = build_reachability_index(state_root, policy)
    run_ids = tuple(sorted(index.run_ids_by_digest.get(digest, ())))
    delivery = next(
        (item for item in delivery_store.list_all() if item.artifact_digest == digest),
        None,
    )
    created_at = delivery.created_at if delivery else None
    delivered_at = delivery.delivered_at if delivery else None
    if created_at is None and path.is_file():
        created_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return {
        "logical_run_ids": run_ids,
        "producer_uid": delivery.producer_uid if delivery else None,
        "consumer_uid": delivery.consumer_uid if delivery else None,
        "provenance": delivery.provenance if delivery else None,
        "created_at": created_at,
        "delivered_at": delivered_at,
    }


def _receipt_for_digest(
    state_root: Path,
    digest: str,
    size_bytes: int,
    *,
    mode: GcMode,
    policy: LifecyclePolicy,
    now: datetime,
    delivery_store: DeliveryRecordStore,
    path: Path | None = None,
) -> DeletionReceipt:
    if path is None:
        path = artifact_payload_path(state_root, digest)
    metadata = _deletion_metadata_for_digest(
        state_root,
        digest,
        size_bytes,
        mode=mode,
        policy=policy,
        delivery_store=delivery_store,
        path=path,
    )
    return DeletionReceipt(
        artifact_digest=digest,
        artifact_size_bytes=size_bytes,
        logical_run_ids=metadata["logical_run_ids"],
        producer_uid=metadata["producer_uid"],
        consumer_uid=metadata["consumer_uid"],
        provenance=metadata["provenance"],
        created_at=metadata["created_at"],
        delivered_at=metadata["delivered_at"],
        deleted_at=now,
        mode=mode,
        policy_identity=_policy_identity(policy),
    )


def _broker_requests_for_digest(
    state_root: Path, index: ReachabilityIndex, digest: str
) -> list[str]:
    request_ids = {
        request_id
        for request_id, output_digest in index.broker_outputs.items()
        if output_digest == digest
    }
    controller_root = state_root / "controller"
    for path in (controller_root / "broker" / "requests").glob("*.json"):
        try:
            state = BrokerRequestState.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if state.output_sha256 == digest and state.status in BROKER_TRANSPORT_OUTPUT_STATUSES:
            request_ids.add(state.request_id)
    return sorted(request_ids)


def _normal_eligible(
    state_root: Path, sweep: _SweepState, digest: str, path: Path
) -> bool:
    request_ids = _broker_requests_for_digest(state_root, sweep.index, digest)
    if not request_ids:
        return False
    grace = timedelta(seconds=sweep.policy.delivery_grace_seconds)
    for request_id in request_ids:
        delivery = sweep.delivery_by_request.get(request_id)
        if delivery is None:
            return False
        delivered_at = delivery.delivered_at
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=UTC)
        if sweep.now < delivered_at + grace:
            return False
    return True


def _normal_protection_reasons(
    state_root: Path, sweep: _SweepState, digest: str
) -> tuple[str, ...]:
    request_ids = _broker_requests_for_digest(state_root, sweep.index, digest)
    if not request_ids:
        return ("no_broker_transport_ref",)
    reasons: list[str] = []
    grace = timedelta(seconds=sweep.policy.delivery_grace_seconds)
    for request_id in request_ids:
        delivery = sweep.delivery_by_request.get(request_id)
        if delivery is None:
            reasons.append(f"undelivered:{request_id}")
            continue
        delivered_at = delivery.delivered_at
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=UTC)
        if sweep.now < delivered_at + grace:
            reasons.append(f"grace_pending:{request_id}")
    return tuple(reasons) or ("not_eligible",)


def _legacy_effective_last_use(sweep: _SweepState, digest: str, path: Path) -> datetime:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    referenced = sweep.index.digest_last_reference_at.get(digest)
    if referenced is None:
        return mtime
    if referenced.tzinfo is None:
        referenced = referenced.replace(tzinfo=UTC)
    return max(mtime, referenced)


def _legacy_eligible(
    state_root: Path, sweep: _SweepState, digest: str, path: Path
) -> bool:
    request_ids = _broker_requests_for_digest(state_root, sweep.index, digest)
    for request_id in request_ids:
        if request_id in sweep.delivery_by_request:
            return False
    retention = timedelta(seconds=sweep.policy.legacy_retention_seconds)
    last_use = _legacy_effective_last_use(sweep, digest, path)
    return sweep.now >= last_use + retention


def _legacy_protection_reasons(
    state_root: Path, sweep: _SweepState, digest: str, path: Path
) -> tuple[str, ...]:
    request_ids = _broker_requests_for_digest(state_root, sweep.index, digest)
    reasons: list[str] = []
    for request_id in request_ids:
        if request_id in sweep.delivery_by_request:
            reasons.append(f"has_delivery_record:{request_id}")
    retention = timedelta(seconds=sweep.policy.legacy_retention_seconds)
    last_use = _legacy_effective_last_use(sweep, digest, path)
    if sweep.now < last_use + retention:
        reasons.append("legacy_retention_pending")
    return tuple(reasons) or ("not_eligible",)
