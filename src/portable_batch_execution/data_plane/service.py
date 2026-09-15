"""Authenticated private data plane service backed by local persistence."""

from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)
from portable_batch_execution.controller.closed_wave_registry import (
    ClosedWaveRegistry,
    opaque_identifier,
)
from portable_batch_execution.data_plane.base import RevisionConflictError
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane


class PrivateDataPlaneService:
    """Controller-owned data plane: closed-wave resolution plus immutable run state."""

    def __init__(self, state_root: Path, bearer_token: str):
        if not bearer_token:
            raise ValueError("bearer token is required")
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._expected_token = bearer_token.encode("utf-8")
        self.store = LocalFilesystemDataPlane(self.state_root)
        self.registry = ClosedWaveRegistry(self.state_root / "controller")

    def authorize(self, authorization: str | None) -> bool:
        if not authorization or not authorization.startswith("Bearer "):
            return False
        provided = authorization[7:].encode("utf-8")
        return hmac.compare_digest(provided, self._expected_token)

    def resolve_wave(self, run_id: str, wave_id: str) -> dict[str, Any]:
        return self.registry.resolve_wave(run_id, wave_id)

    def dispatch(
        self,
        method: str,
        path: str,
        *,
        authorization: str | None,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes | None]:
        if not self.authorize(authorization):
            return 401, {"Content-Type": "application/json"}, b'{"error":"unauthorized"}'
        headers = {key.lower(): value for key, value in (headers or {}).items()}
        normalized = path.split("?", 1)[0].rstrip("/") or "/"
        segments = [unquote(part) for part in normalized.split("/") if part]
        try:
            if method == "GET" and len(segments) == 5 and segments[:2] == ["v1", "runs"] and segments[3] == "waves":
                payload = self.resolve_wave(segments[2], segments[4])
                return 200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8")
            if method in {"GET", "HEAD"} and len(segments) == 4 and segments[:2] == ["v1", "artifacts"] and segments[3] == "content":
                object_id = opaque_identifier(segments[2], "artifact object_id")
                artifact_path = (self.store.root / "artifacts" / object_id).resolve()
                artifacts_root = (self.store.root / "artifacts").resolve()
                if artifact_path.parent != artifacts_root or not artifact_path.is_file():
                    return 404, {"Content-Type": "application/json"}, b'{"error":"not found"}'
                ref = ArtifactRef(
                    object_id=object_id,
                    uri=artifact_path.as_uri(),
                    sha256=f"sha256:{object_id}",
                )
                data = self.store.read(ref)
                if method == "HEAD":
                    return 200, {"Content-Length": str(len(data))}, b""
                return 200, {"Content-Type": "application/octet-stream"}, data
            if method == "POST" and segments == ["v1", "artifacts"]:
                ref = self.store.write(body or b"", headers.get("content-type"))
                return 200, {"Content-Type": "application/json"}, ref.model_dump_json().encode("utf-8")
            if method == "GET" and len(segments) == 4 and segments[:2] == ["v1", "runs"] and segments[3] == "attempts":
                run_id = opaque_identifier(segments[2], "run_id")
                payload = [
                    item.model_dump(mode="json")
                    for item in self.store.read_attempts(run_id)
                ]
                return 200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8")
            if method == "POST" and len(segments) == 4 and segments[:2] == ["v1", "runs"] and segments[3] == "attempts":
                run_id = opaque_identifier(segments[2], "run_id")
                record = ShardAttemptRecord.model_validate_json(body or b"{}")
                if record.logical_run_id != run_id:
                    return 400, {"Content-Type": "application/json"}, b'{"error":"run mismatch"}'
                try:
                    self.store.append_attempt(record)
                except ValueError:
                    return 409, {"Content-Type": "application/json"}, b'{"error":"attempt conflict"}'
                return 204, {}, b""
            if method == "GET" and len(segments) == 4 and segments[:2] == ["v1", "runs"] and segments[3] == "manifest":
                run_id = opaque_identifier(segments[2], "run_id")
                manifest = self.store.read_manifest(run_id)
                if manifest is None:
                    return 404, {"Content-Type": "application/json"}, b'{"error":"not found"}'
                return 200, {"Content-Type": "application/json"}, manifest.model_dump_json().encode("utf-8")
            if method == "PUT" and len(segments) == 4 and segments[:2] == ["v1", "runs"] and segments[3] == "manifest":
                run_id = opaque_identifier(segments[2], "run_id")
                if_match = headers.get("if-match")
                if if_match is None:
                    return 428, {"Content-Type": "application/json"}, b'{"error":"if-match required"}'
                try:
                    expected_revision = int(if_match)
                except ValueError:
                    return 400, {"Content-Type": "application/json"}, b'{"error":"invalid if-match"}'
                manifest = RunManifest.model_validate_json(body or b"{}")
                if manifest.logical_run_id != run_id:
                    return 400, {"Content-Type": "application/json"}, b'{"error":"run mismatch"}'
                try:
                    written = self.store.write_next_manifest(manifest, expected_revision)
                except RevisionConflictError:
                    return 409, {"Content-Type": "application/json"}, b'{"error":"revision conflict"}'
                except ValueError:
                    return 400, {"Content-Type": "application/json"}, b'{"error":"invalid manifest"}'
                return 200, {"Content-Type": "application/json"}, written.model_dump_json().encode("utf-8")
        except KeyError:
            return 404, {"Content-Type": "application/json"}, b'{"error":"not found"}'
        except ValueError:
            return 400, {"Content-Type": "application/json"}, b'{"error":"bad request"}'
        except OSError:
            return 500, {"Content-Type": "application/json"}, b'{"error":"internal"}'
        return 404, {"Content-Type": "application/json"}, b'{"error":"not found"}'
