"""Hugging Face Storage Bucket transport (not Hub dataset/model repos)."""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from portable_batch_execution.contracts import ArtifactRef

HF_BUCKET_ID = "yamauchiJP/system-trading-data"
HF_BUCKET_RESOURCE = f"buckets/{HF_BUCKET_ID}"
HF_BUCKET_REF_SCHEMA = "pbe.hf-bucket-artifact-ref.v1"
HF_BUCKET_REF_MEDIA_TYPE = "application/vnd.pbe.hf-bucket-ref.v1"
HF_OBJECT_URI_PREFIX = f"hf://{HF_BUCKET_RESOURCE}/"

_TOKEN_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{10,}|Bearer\s+\S+)", re.IGNORECASE)


class HfBucketTransportError(RuntimeError):
    """Fail-closed HF bucket transport error."""


def redact_secrets(message: str) -> str:
    return _TOKEN_PATTERN.sub("<redacted>", message)


def hf_object_uri(object_path: str) -> str:
    normalized = object_path.lstrip("/")
    if not normalized or ".." in normalized.split("/"):
        raise HfBucketTransportError("invalid bucket object path")
    return f"{HF_OBJECT_URI_PREFIX}{normalized}"


def parse_hf_object_uri(uri: str) -> str:
    if not uri.startswith(HF_OBJECT_URI_PREFIX):
        raise HfBucketTransportError("uri is not an HF bucket object uri")
    tail = uri[len(HF_OBJECT_URI_PREFIX) :]
    if not tail or ".." in tail.split("/"):
        raise HfBucketTransportError("invalid bucket object path in uri")
    return tail


def validate_hf_bucket_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    if ref.get("schema_version") != HF_BUCKET_REF_SCHEMA:
        raise HfBucketTransportError("hf bucket ref schema mismatch")
    bucket_id = str(ref.get("bucket_id") or "")
    if bucket_id != HF_BUCKET_ID:
        raise HfBucketTransportError("unexpected bucket_id")
    object_path = str(ref.get("object_path") or "")
    if not object_path or object_path.startswith("/") or ".." in object_path.split("/"):
        raise HfBucketTransportError("invalid object_path")
    digest = str(ref.get("sha256") or "").removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HfBucketTransportError("sha256 required")
    size_bytes = ref.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise HfBucketTransportError("size_bytes invalid")
    media_type = str(ref.get("media_type") or "application/octet-stream")
    if not media_type or len(media_type) > 128 or "/" not in media_type:
        raise HfBucketTransportError("media_type invalid")
    return {
        "schema_version": HF_BUCKET_REF_SCHEMA,
        "bucket_id": bucket_id,
        "object_path": object_path,
        "sha256": digest,
        "size_bytes": size_bytes,
        "media_type": media_type,
    }


def artifact_ref_from_hf_object(
    *,
    object_path: str,
    sha256_hex: str,
    size_bytes: int,
    media_type: str,
) -> ArtifactRef:
    digest = sha256_hex.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HfBucketTransportError("sha256 required")
    return ArtifactRef(
        object_id=digest,
        uri=hf_object_uri(object_path),
        sha256=f"sha256:{digest}",
        media_type=media_type,
        size_bytes=size_bytes,
    )


def artifact_ref_to_hf_bucket_ref(ref: ArtifactRef) -> dict[str, Any]:
    return validate_hf_bucket_ref(
        {
            "schema_version": HF_BUCKET_REF_SCHEMA,
            "bucket_id": HF_BUCKET_ID,
            "object_path": parse_hf_object_uri(ref.uri),
            "sha256": ref.sha256.removeprefix("sha256:"),
            "size_bytes": int(ref.size_bytes or 0),
            "media_type": str(ref.media_type or "application/octet-stream"),
        }
    )


def verify_bytes_identity(data: bytes, *, sha256_hex: str, size_bytes: int) -> None:
    digest = hashlib.sha256(data).hexdigest()
    expected = sha256_hex.removeprefix("sha256:")
    if digest != expected or len(data) != size_bytes:
        raise HfBucketTransportError("downloaded bytes failed identity verification")


class HfBucketStoragePort(Protocol):
    def upload_append_only(self, uploads: Sequence[tuple[Path, str]]) -> None: ...

    def download_to(self, remote_path: str, local_path: Path) -> None: ...

    def remote_exists(self, remote_path: str) -> bool: ...


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 4.0


class HfBucketTransport:
    """Verified read/write against one HF Storage Bucket."""

    def __init__(
        self,
        storage: HfBucketStoragePort,
        *,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._storage = storage
        self._retry = retry or RetryPolicy()

    @staticmethod
    def token_from_environment() -> str:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise HfBucketTransportError("HF_TOKEN is required")
        return token

    def read_verified(self, ref: Mapping[str, Any]) -> bytes:
        validated = validate_hf_bucket_ref(ref)
        object_path = validated["object_path"]
        expected_digest = validated["sha256"]
        expected_size = validated["size_bytes"]

        def attempt() -> bytes:
            tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"pbe-hf-read-{expected_digest}.bin"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._storage.download_to(object_path, tmp)
                data = tmp.read_bytes()
            finally:
                if tmp.is_file():
                    tmp.unlink()
            verify_bytes_identity(
                data, sha256_hex=expected_digest, size_bytes=expected_size
            )
            return data

        return self._with_retry(attempt)

    def write_verified_append_only(
        self,
        *,
        object_path: str,
        data: bytes,
        media_type: str,
    ) -> dict[str, Any]:
        normalized = object_path.lstrip("/")
        if not normalized or ".." in normalized.split("/"):
            raise HfBucketTransportError("invalid upload object_path")
        digest = hashlib.sha256(data).hexdigest()
        size_bytes = len(data)

        def attempt() -> dict[str, Any]:
            if self._storage.remote_exists(normalized):
                raise HfBucketTransportError("refusing overwrite of existing bucket object")
            tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"pbe-hf-upload-{digest}.bin"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp.write_bytes(data)
                self._storage.upload_append_only([(tmp, normalized)])
            finally:
                if tmp.is_file():
                    tmp.unlink()
            if not self._storage.remote_exists(normalized):
                raise HfBucketTransportError("upload existence check failed")
            roundtrip = self.read_verified(
                {
                    "schema_version": HF_BUCKET_REF_SCHEMA,
                    "bucket_id": HF_BUCKET_ID,
                    "object_path": normalized,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "media_type": media_type,
                }
            )
            verify_bytes_identity(roundtrip, sha256_hex=digest, size_bytes=size_bytes)
            return validate_hf_bucket_ref(
                {
                    "schema_version": HF_BUCKET_REF_SCHEMA,
                    "bucket_id": HF_BUCKET_ID,
                    "object_path": normalized,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "media_type": media_type,
                }
            )

        return self._with_retry(attempt)

    def _with_retry(self, fn):
        delay = self._retry.initial_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return fn()
            except HfBucketTransportError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry.max_attempts:
                    break
                time.sleep(delay)
                delay = min(delay * self._retry.backoff_multiplier, self._retry.max_backoff_seconds)
        message = redact_secrets(str(last_exc) if last_exc else "hf bucket operation failed")
        raise HfBucketTransportError(message) from last_exc


class InMemoryHfBucketStorage:
    """Test double without network or real HF token."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_append_only(self, uploads: Sequence[tuple[Path, str]]) -> None:
        destinations = [dest for _, dest in uploads]
        if len(destinations) != len(set(destinations)):
            raise ValueError("duplicate destination path in upload batch")
        for local_path, remote_path in uploads:
            if remote_path in self.objects:
                raise FileExistsError(f"refusing overwrite: {remote_path}")
            self.objects[remote_path] = local_path.read_bytes()

    def download_to(self, remote_path: str, local_path: Path) -> None:
        if remote_path not in self.objects:
            raise FileNotFoundError(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[remote_path])

    def remote_exists(self, remote_path: str) -> bool:
        return remote_path in self.objects


class HfApiBucketStorage:
    """Production adapter using huggingface_hub bucket APIs."""

    def __init__(self, api, bucket_id: str = HF_BUCKET_ID) -> None:
        self._api = api
        self._bucket_id = bucket_id

    def upload_append_only(self, uploads: Sequence[tuple[Path, str]]) -> None:
        destinations = [destination for _, destination in uploads]
        if len(destinations) != len(set(destinations)):
            raise ValueError("duplicate destination path in upload batch")
        existing = list(self._api.get_bucket_paths_info(self._bucket_id, destinations))
        if existing:
            found = ", ".join(sorted(item.path for item in existing))
            raise FileExistsError(f"refusing to overwrite existing bucket path(s): {found}")
        self._api.batch_bucket_files(self._bucket_id, add=list(uploads))

    def download_to(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._api.download_bucket_files(
            self._bucket_id,
            [(remote_path, local_path)],
            raise_on_missing_files=True,
        )

    def remote_exists(self, remote_path: str) -> bool:
        existing = list(self._api.get_bucket_paths_info(self._bucket_id, [remote_path]))
        return bool(existing)


def build_hf_api_storage_from_token(token: str) -> HfApiBucketStorage:
    from huggingface_hub import HfApi

    return HfApiBucketStorage(HfApi(token=token))
