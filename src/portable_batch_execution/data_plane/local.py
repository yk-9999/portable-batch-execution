from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse
from urllib.request import url2pathname

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)
from portable_batch_execution.controller.closed_wave_registry import safe_file_component
from portable_batch_execution.lifecycle.lock import lifecycle_state_lock

from .base import ArtifactContentStream, RevisionConflictError

_ARTIFACT_CHUNK_BYTES = 64 * 1024


class LocalFilesystemDataPlane:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @property
    def _artifacts(self) -> Path:
        path = self.root / "artifacts"
        path.mkdir(exist_ok=True)
        return path

    @property
    def _runs(self) -> Path:
        path = self.root / "runs"
        path.mkdir(exist_ok=True)
        return path

    def _local_payload_path(self, ref: ArtifactRef) -> Path | None:
        """Return the local payload path when ref targets this plane; None if external."""
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file":
            return None
        if parsed.netloc:
            raise ValueError("artifact ref must be a local file URI")
        path = Path(url2pathname(parsed.path)).resolve()
        artifacts = self._artifacts.resolve()
        if path.parent != artifacts or path.name != ref.object_id:
            raise ValueError("artifact ref is outside this data plane")
        expected_hex = ref.sha256.removeprefix("sha256:")
        if ref.object_id != expected_hex:
            raise ValueError("artifact ref object_id does not match digest")
        return path

    def _validate_local_refs(self, refs: tuple[ArtifactRef, ...]) -> None:
        for ref in refs:
            path = self._local_payload_path(ref)
            if path is None:
                continue
            if not path.is_file():
                raise FileNotFoundError(
                    f"local artifact payload missing for {ref.sha256}"
                )
            if ref.size_bytes is not None and path.stat().st_size != ref.size_bytes:
                raise ValueError(f"local artifact size mismatch for {ref.sha256}")

    def validate_static_ref(self, ref: ArtifactRef) -> bool:
        """Validate a static input ref; external refs are accepted without local checks."""
        try:
            path = self._local_payload_path(ref)
        except ValueError:
            return False
        if path is None:
            return True
        if not path.is_file():
            return False
        if ref.size_bytes is not None and path.stat().st_size != ref.size_bytes:
            return False
        digest = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
        return digest == ref.sha256

    def _artifact_ref_for_payload(
        self, digest: str, data: bytes, media_type: str | None
    ) -> ArtifactRef:
        p = self._artifacts / digest
        return ArtifactRef(
            object_id=digest,
            uri=p.as_uri(),
            sha256=f"sha256:{digest}",
            media_type=media_type,
            size_bytes=len(data),
        )

    def _persist_payload_unlocked(self, digest: str, data: bytes) -> None:
        p = self._artifacts / digest
        if p.exists():
            return
        temp = p.with_suffix(".tmp")
        temp.write_bytes(data)
        temp.replace(p)

    def write(
        self,
        data: bytes,
        media_type: str | None = None,
        *,
        _caller_holds_lifecycle_lock: bool = False,
    ) -> ArtifactRef:
        digest = sha256(data).hexdigest()
        p = self._artifacts / digest
        with self._lock:
            if not p.exists():
                if _caller_holds_lifecycle_lock:
                    self._persist_payload_unlocked(digest, data)
                else:
                    with lifecycle_state_lock(self.root):
                        self._persist_payload_unlocked(digest, data)
        return self._artifact_ref_for_payload(digest, data, media_type)

    def write_next_manifest_with_caller_lifecycle_lock(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest:
        """Publish a manifest revision; caller must already hold lifecycle_state_lock."""
        with self._lock:
            return self._write_next_manifest_unlocked(manifest, expected_revision)

    @staticmethod
    def _artifact_path(ref: ArtifactRef) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file" or parsed.netloc:
            raise ValueError("artifact ref must be a local file URI")
        return Path(url2pathname(parsed.path))

    def read(self, ref: ArtifactRef) -> bytes:
        path = self._artifact_path(ref)
        if path.parent != self._artifacts or path.name != ref.object_id:
            raise ValueError("artifact ref is outside this data plane")
        return path.read_bytes()

    def open_content(self, ref: ArtifactRef) -> ArtifactContentStream:
        """Stream artifact bytes in bounded chunks without whole-object reads."""
        path = self._artifact_path(ref)
        if path.parent != self._artifacts or path.name != ref.object_id:
            raise ValueError("artifact ref is outside this data plane")
        return ArtifactContentStream(
            size_bytes=path.stat().st_size,
            chunks=self._iter_artifact_chunks(path),
        )

    @staticmethod
    def _iter_artifact_chunks(path: Path) -> Iterator[bytes]:
        with path.open("rb") as handle:
            while chunk := handle.read(_ARTIFACT_CHUNK_BYTES):
                yield chunk

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            path = self._artifact_path(ref)
            return (
                path.parent == self._artifacts
                and path.name == ref.object_id
                and path.is_file()
            )
        except ValueError:
            return False

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            data = self.read(ref)
        except (OSError, ValueError):
            return False
        return f"sha256:{sha256(data).hexdigest()}" == ref.sha256 and (
            ref.size_bytes is None or len(data) == ref.size_bytes
        )

    def _run_directory(self, run_id: str) -> Path:
        safe_file_component(run_id, "run_id")
        path = self._runs / run_id
        path.mkdir(exist_ok=True)
        return path

    def _append_attempt_unlocked(self, record: ShardAttemptRecord) -> None:
        run = self._run_directory(record.logical_run_id)
        safe_file_component(record.attempt_id, "attempt_id")
        attempts = run / "attempts"
        attempts.mkdir(exist_ok=True)
        path = attempts / f"{record.attempt_id}.json"
        if path.exists():
            existing = ShardAttemptRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if existing != record:
                raise ValueError("attempt_id already belongs to a different record")
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        temporary.replace(path)

    def append_attempt(self, record: ShardAttemptRecord) -> None:
        """Persist an immutable attempt record; duplicate IDs are rejected."""
        with self._lock, lifecycle_state_lock(self.root):
            if record.output_refs:
                self._validate_local_refs(record.output_refs)
            self._append_attempt_unlocked(record)

    def read_attempts(self, run_id: str) -> tuple[ShardAttemptRecord, ...]:
        run = self._run_directory(run_id)
        attempts = run / "attempts"
        if not attempts.exists():
            return ()
        records = [
            ShardAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in attempts.glob("*.json")
        ]
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.finished_at,
                    record.started_at,
                    record.attempt_id,
                ),
            )
        )

    def read_manifest(self, run_id: str) -> RunManifest | None:
        path = self._run_directory(run_id) / "latest.json"
        return (
            RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )

    def _write_next_manifest_unlocked(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest:
        manifest = RunManifest.model_validate(manifest.model_dump())
        run = self._run_directory(manifest.logical_run_id)
        current = self.read_manifest(manifest.logical_run_id)
        current_revision = current.revision if current else -1
        if current_revision != expected_revision:
            raise RevisionConflictError(
                f"expected revision {expected_revision}, found {current_revision}"
            )
        if manifest.revision != expected_revision + 1:
            raise ValueError("next manifest revision must increment by one")
        if manifest.final_output_refs:
            self._validate_local_refs(manifest.final_output_refs)
        history = run / "manifests"
        history.mkdir(exist_ok=True)
        revision_path = history / f"{manifest.revision:020d}.json"
        if revision_path.exists():
            raise RevisionConflictError("manifest revision already exists")
        encoded = manifest.model_dump_json() + "\n"
        temporary = revision_path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(revision_path)
        latest = run / "latest.json"
        latest_temp = latest.with_suffix(".tmp")
        latest_temp.write_text(encoded, encoding="utf-8")
        latest_temp.replace(latest)
        return manifest

    def write_next_manifest(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest:
        """Compare-and-swap latest manifest and retain every immutable revision."""
        with self._lock, lifecycle_state_lock(self.root):
            return self._write_next_manifest_unlocked(manifest, expected_revision)
