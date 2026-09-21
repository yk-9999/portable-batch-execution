"""Composed HF-direct data plane: A1 control metadata plus HF artifact payloads.

In HF-direct mode the controller-owned A1 data plane is used for control
metadata only -- closed-wave resolution, attempt records, and manifest/status
calls.  Artifact bytes never traverse ``/v1/artifacts``: reads, writes,
existence checks, and verification are delegated to a generic content-addressed
Hugging Face bucket store.  Control methods are delegated verbatim; artifact
methods are never routed to the control client.
"""

from __future__ import annotations

from typing import Any

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)

from .hf import HfBucketArtifactStore, HfBucketIdentity
from .http import HttpPrivateDataPlane


class HfDirectDataPlane:
    """Control metadata on A1; artifact read/write/exists/verify on HF."""

    def __init__(
        self,
        control: HttpPrivateDataPlane,
        artifacts: HfBucketArtifactStore,
    ) -> None:
        self.control = control
        self.artifacts = artifacts
        self.identity: HfBucketIdentity = artifacts.identity

    @classmethod
    def from_environment(cls) -> HfDirectDataPlane:
        control = HttpPrivateDataPlane.from_environment()
        artifacts = HfBucketArtifactStore.from_environment()
        return cls(control, artifacts)

    # -- control metadata (A1 only) -------------------------------------------

    def resolve_wave(self, run_id: str, wave_id: str) -> dict[str, Any]:
        return self.control.resolve_wave(run_id, wave_id)

    def read_attempts(self, run_id: str) -> tuple[ShardAttemptRecord, ...]:
        return self.control.read_attempts(run_id)

    def append_attempt(self, record: ShardAttemptRecord) -> None:
        self.control.append_attempt(record)

    def read_manifest(self, run_id: str) -> RunManifest | None:
        return self.control.read_manifest(run_id)

    def write_next_manifest(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest:
        return self.control.write_next_manifest(manifest, expected_revision)

    # -- artifacts (HF bucket only) -------------------------------------------

    def read(self, ref: ArtifactRef) -> bytes:
        return self.artifacts.read(ref)

    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef:
        return self.artifacts.write(data, media_type)

    def exists(self, ref: ArtifactRef) -> bool:
        return self.artifacts.exists(ref)

    def verify(self, ref: ArtifactRef) -> bool:
        return self.artifacts.verify(ref)


__all__ = ["HfDirectDataPlane"]
