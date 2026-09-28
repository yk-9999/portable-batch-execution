"""CLI for lifecycle policy validation, GC, and hold management."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from portable_batch_execution.lifecycle.gc import apply_gc, plan_gc
from portable_batch_execution.lifecycle.holds import HoldRecord, HoldStore
from portable_batch_execution.lifecycle.lock import LifecycleStateLock, lifecycle_state_lock
from portable_batch_execution.lifecycle.policy import (
    load_lifecycle_policy,
    validate_lifecycle_policy_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PBE artifact lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-policy", help="Validate lifecycle policy file")
    validate.add_argument("--policy-file", required=True)
    validate.set_defaults(command="validate-policy")

    for name, mode in (
        ("gc-dry-run", "normal"),
        ("gc-apply", "normal"),
        ("legacy-gc-dry-run", "legacy"),
        ("legacy-gc-apply", "legacy"),
    ):
        cmd = sub.add_parser(name.replace("_", "-"), help=f"{mode} GC")
        cmd.add_argument("--state-root", required=True)
        cmd.add_argument("--policy-file")
        cmd.set_defaults(command=name, gc_mode=mode, apply=name.endswith("apply"))

    hold_add = sub.add_parser("hold-add", help="Add preserve/hold record")
    hold_add.add_argument("--state-root", required=True)
    hold_add.add_argument("--hold-id", required=True)
    hold_add.add_argument("--kind", choices=["preserve", "hold"], default="hold")
    hold_add.add_argument("--digest", action="append", required=True)
    hold_add.add_argument("--reason", default="")
    hold_add.set_defaults(command="hold-add")

    hold_rm = sub.add_parser("hold-remove", help="Remove hold record")
    hold_rm.add_argument("--state-root", required=True)
    hold_rm.add_argument("--hold-id", required=True)
    hold_rm.set_defaults(command="hold-remove")

    args = parser.parse_args(argv)
    if args.command == "validate-policy":
        policy = validate_lifecycle_policy_file(Path(args.policy_file))
        print(json.dumps(policy.to_json(), indent=2))
        return 0

    state_root = Path(args.state_root)
    policy_path = Path(args.policy_file) if args.policy_file else state_root / "lifecycle-policy.json"
    policy = (
        validate_lifecycle_policy_file(policy_path)
        if args.policy_file
        else load_lifecycle_policy(state_root)
    )

    if args.command == "hold-add":
        with lifecycle_state_lock(state_root):
            HoldStore(state_root).put(
                HoldRecord(
                    hold_id=args.hold_id,
                    kind=args.kind,
                    artifact_digests=tuple(args.digest),
                    reason=args.reason,
                    created_at=datetime.now(UTC),
                )
            )
        return 0

    if args.command == "hold-remove":
        with lifecycle_state_lock(state_root):
            removed = HoldStore(state_root).remove(args.hold_id)
        print(json.dumps({"removed": removed}))
        return 0

    if args.command in {"gc-dry-run", "gc-apply", "legacy-gc-dry-run", "legacy-gc-apply"}:
        mode = args.gc_mode
        with LifecycleStateLock(state_root):
            if args.apply:
                report = apply_gc(state_root, policy, mode=mode)
            else:
                report = plan_gc(state_root, policy, mode=mode)
        print(json.dumps(report.to_dict(), indent=2))
        return 1 if report.blocked else 0

    raise AssertionError(f"unknown command {args.command}")
