"""Authoritative artifact references outside the PBE state root (policy-configured)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from portable_batch_execution.lifecycle.policy import LifecyclePolicy

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def configured_reference_paths(policy: LifecyclePolicy) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in policy.authoritative_reference_files:
        paths.append(Path(raw))
    for raw in policy.authoritative_reference_roots:
        root = Path(raw)
        if not root.is_dir():
            raise FileNotFoundError(f"authoritative reference root missing: {root}")
        paths.extend(sorted(p for p in root.rglob("*.json") if p.is_file()))
    return tuple(paths)


def iter_artifact_ref_digests(node: Any) -> Iterator[str]:
    """Yield sha256 digests from ArtifactRef-shaped JSON objects."""
    if isinstance(node, dict):
        object_id = node.get("object_id")
        sha = node.get("sha256")
        if (
            isinstance(object_id, str)
            and object_id
            and isinstance(sha, str)
            and _DIGEST.fullmatch(sha)
        ):
            yield sha
        for value in node.values():
            yield from iter_artifact_ref_digests(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_artifact_ref_digests(item)


def load_reference_digests(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"authoritative reference file missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable authoritative reference file: {path}") from exc
    if not isinstance(payload, (dict, list)):
        raise TypeError(f"authoritative reference file must be JSON object or array: {path}")
    return set(iter_artifact_ref_digests(payload))


def apply_policy_external_references(
    index: Any,
    policy: LifecyclePolicy,
) -> None:
    """Mark digests referenced by configured external JSON; record failures on index."""
    try:
        paths = configured_reference_paths(policy)
    except (OSError, ValueError, TypeError) as exc:
        index.unknown_messages.append(str(exc))
        return
    for path in paths:
        try:
            digests = load_reference_digests(path)
        except (OSError, ValueError, TypeError) as exc:
            index.unknown_messages.append(str(exc))
            continue
        label = path.as_posix()
        for digest in digests:
            index.protected_by_digest[digest].add(f"external_reference:{label}")
