"""Generic content-addressed Hugging Face bucket artifact store.

In HF-direct mode artifact payload bytes travel directly between a GitHub
Actions worker and a private Hugging Face bucket; the controller-owned A1 data
plane carries control metadata only.  This store owns the artifact half:

* store identity (bucket, prefix, object layout) is bounded metadata/config and
  is validated fail-closed before any remote call;
* ``sha256-flat.v1`` maps ``object_id`` (lowercase SHA-256 hex) to
  ``<prefix>/objects/sha256/<object_id>``;
* reads derive the object solely from the closed identity plus ``ref.object_id``
  and verify the expected SHA-256 and exact size before returning bytes;
* writes content-address the bytes, idempotently reuse an existing exact-size
  object, otherwise upload directly, re-check persistence, and return an
  ``ArtifactRef`` with an ``hf://buckets/...`` reference.

The Hugging Face token is read only from the environment variable injected by
GitHub Actions from the fixed repository secret.  It is never placed on argv,
in logs, in returned metadata, or in a persisted file.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from portable_batch_execution.contracts import ArtifactRef

HF_DIRECT_KIND = "hf-buckets-direct"
OBJECTS_KEY = "objects/sha256"
OBJECT_LAYOUT_VERSION = "sha256-flat.v1"
REQUIRED_HF_CLI_VERSION = "1.8.0"

#: Environment variable holding the token injected from the fixed GH secret.
HF_TOKEN_ENV = "HF_SYSTEM_TRADING_DATA_RW_TOKEN"
#: Process variable ``HfApi`` reads when the workflow maps the fixed secret.
HF_CLI_TOKEN_ENV = "HF_TOKEN"

PBE_HF_BUCKET_ENV = "PBE_HF_BUCKET"
PBE_HF_PREFIX_ENV = "PBE_HF_PREFIX"
PBE_HF_OBJECT_LAYOUT_ENV = "PBE_HF_OBJECT_LAYOUT"

DEFAULT_MAX_OBJECT_BYTES = 1024 * 1024 * 1024

_BUCKET_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_PREFIX_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[0-9a-f]{64}$")
_BARE_VERSION_LINE = re.compile(r"^([0-9][0-9A-Za-z.+-]*)$")
_PREFIXED_VERSION_LINE = re.compile(r"^version:\s*([0-9][0-9A-Za-z.+-]*)$", re.IGNORECASE)


class HfBucketApi(Protocol):
    def get_bucket_paths_info(self, bucket_id: str, paths: list[str]): ...

    def download_bucket_files(
        self,
        bucket_id: str,
        path_pairs: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool = ...,
    ) -> None: ...

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[Path, str]] | None = ...,
    ) -> None: ...


class HfBucketStoreError(RuntimeError):
    """Fail-closed HF bucket store error; never carries token material."""


def validate_bucket(bucket: Any) -> str:
    """Fail closed unless ``bucket`` is a bounded ``owner/name`` pair."""
    if not isinstance(bucket, str) or bucket.count("/") != 1:
        raise HfBucketStoreError("HF bucket identity must be an owner/name pair")
    owner, name = bucket.split("/", 1)
    if not _BUCKET_SEGMENT.fullmatch(owner) or not _BUCKET_SEGMENT.fullmatch(name):
        raise HfBucketStoreError("HF bucket identity is not a bounded identifier")
    return bucket


def normalize_prefix(prefix: Any) -> str:
    """Fail closed on unsafe/out-of-scope prefixes and return the closed form."""
    if not isinstance(prefix, str) or not prefix or prefix != prefix.strip():
        raise HfBucketStoreError("HF store prefix must be a non-empty bounded path")
    if len(prefix) > 512:
        raise HfBucketStoreError("HF store prefix is too long")
    if "\\" in prefix or "\x00" in prefix:
        raise HfBucketStoreError("HF store prefix must not contain backslashes")
    if prefix.startswith("/"):
        raise HfBucketStoreError("HF store prefix must be relative")
    segments = prefix.rstrip("/").split("/")
    if not segments or any(not _PREFIX_SEGMENT.fullmatch(segment) for segment in segments):
        raise HfBucketStoreError("HF store prefix must be a bounded relative path")
    return "/".join(segments) + "/"


def validate_object_id(object_id: Any) -> str:
    """Fail closed unless ``object_id`` is a lowercase SHA-256 hex digest."""
    if not isinstance(object_id, str) or not _OBJECT_ID.fullmatch(object_id):
        raise HfBucketStoreError("HF object_id must be a lowercase SHA-256 hex digest")
    return object_id


def parse_hf_cli_version_output(stdout: str | None) -> str:
    """Parse legacy ``hf version`` stdout: one bare semver or one ``version:`` line."""
    text = (stdout or "").strip()
    if not text:
        raise HfBucketStoreError("could not parse hf CLI version output")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HfBucketStoreError("could not parse hf CLI version output")
    line = lines[0]
    prefixed = _PREFIXED_VERSION_LINE.match(line)
    if prefixed is not None:
        return prefixed.group(1)
    bare = _BARE_VERSION_LINE.match(line)
    if bare is not None:
        return bare.group(1)
    raise HfBucketStoreError("could not parse hf CLI version output")


@dataclass(frozen=True)
class HfBucketIdentity:
    """Bounded, fail-closed identity of one content-addressed HF bucket store."""

    bucket: str
    prefix: str
    object_layout_version: str = OBJECT_LAYOUT_VERSION
    kind: str = HF_DIRECT_KIND
    required_hf_cli_version: str = REQUIRED_HF_CLI_VERSION

    @classmethod
    def from_metadata(cls, section: Any) -> HfBucketIdentity:
        if not isinstance(section, Mapping) or not section:
            raise HfBucketStoreError("HF store identity section required")
        if section.get("kind", HF_DIRECT_KIND) != HF_DIRECT_KIND:
            raise HfBucketStoreError("HF store identity kind is unsupported")
        if section.get("object_layout_version") != OBJECT_LAYOUT_VERSION:
            raise HfBucketStoreError(
                f"object_layout_version must equal {OBJECT_LAYOUT_VERSION!r}"
            )
        if section.get("required_hf_cli_version", REQUIRED_HF_CLI_VERSION) != (
            REQUIRED_HF_CLI_VERSION
        ):
            raise HfBucketStoreError(
                f"required_hf_cli_version must equal {REQUIRED_HF_CLI_VERSION!r}"
            )
        return cls(
            bucket=validate_bucket(section.get("bucket")),
            prefix=normalize_prefix(section.get("prefix")),
        )

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> HfBucketIdentity:
        source = os.environ if env is None else env
        return cls.from_metadata(
            {
                "kind": HF_DIRECT_KIND,
                "bucket": source.get(PBE_HF_BUCKET_ENV),
                "prefix": source.get(PBE_HF_PREFIX_ENV),
                "object_layout_version": source.get(
                    PBE_HF_OBJECT_LAYOUT_ENV, OBJECT_LAYOUT_VERSION
                ),
            }
        )

    @property
    def objects_prefix(self) -> str:
        return f"{self.prefix}{OBJECTS_KEY}"

    @property
    def objects_uri(self) -> str:
        return f"hf://buckets/{self.bucket}/{self.objects_prefix}"

    def remote_path(self, object_id: str) -> str:
        return f"{self.objects_prefix}/{validate_object_id(object_id)}"

    def object_uri(self, object_id: str) -> str:
        return f"{self.objects_uri}/{validate_object_id(object_id)}"

    def to_metadata(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "object_layout_version": self.object_layout_version,
            "required_hf_cli_version": self.required_hf_cli_version,
        }


def resolve_hf_direct_token(
    *,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the HF-direct token from explicit injection or bounded env vars."""
    if token is not None:
        return token
    source = os.environ if env is None else env
    return source.get(HF_TOKEN_ENV) or source.get(HF_CLI_TOKEN_ENV)


def build_hf_api(token: str) -> HfBucketApi:
    from huggingface_hub import HfApi

    return HfApi(token=token)


class HfBucketArtifactStore:
    """Artifact read/write/exists/verify over one closed HF bucket identity."""

    def __init__(
        self,
        identity: HfBucketIdentity,
        *,
        api: HfBucketApi | None = None,
        token: str | None = None,
        spool_root: Path | None = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self.identity = identity
        self._api = api
        self._lazy_api: HfBucketApi | None = None
        self._token = token
        self._spool_root = Path(spool_root) if spool_root is not None else None
        if max_object_bytes <= 0:
            raise HfBucketStoreError("max_object_bytes must be positive")
        self._max_object_bytes = max_object_bytes

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None, **kwargs: Any
    ) -> HfBucketArtifactStore:
        return cls(HfBucketIdentity.from_environment(env), **kwargs)

    def _resolve_token(self) -> str | None:
        return resolve_hf_direct_token(token=self._token)

    def _client(self) -> HfBucketApi:
        if self._api is not None:
            return self._api
        if self._lazy_api is None:
            token = self._resolve_token()
            if not token:
                raise HfBucketStoreError("HF token is required for bucket access")
            self._lazy_api = build_hf_api(token)
        return self._lazy_api

    def huggingface_hub_version(self) -> str:
        try:
            import huggingface_hub
        except ImportError as exc:
            raise HfBucketStoreError("huggingface_hub is not available") from exc
        return huggingface_hub.__version__

    def preflight(self) -> None:
        """Fail closed unless pinned ``huggingface_hub`` is importable and exact."""
        version = self.huggingface_hub_version()
        if version != REQUIRED_HF_CLI_VERSION:
            raise HfBucketStoreError(
                f"huggingface_hub version {version!r} does not match required "
                f"{REQUIRED_HF_CLI_VERSION!r}"
            )

    def _spool_directory(self) -> Path:
        if self._spool_root is not None:
            self._spool_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="hf-direct-", dir=self._spool_root))

    def _parse_path_info(self, entries: list[Any], object_id: str) -> int | None:
        if not entries:
            return None
        if len(entries) != 1:
            raise HfBucketStoreError("ambiguous remote object listing")
        entry = entries[0]
        path = getattr(entry, "path", None)
        if path is None and isinstance(entry, Mapping):
            path = entry.get("path") or entry.get("name")
        if path is None:
            raise HfBucketStoreError("hf bucket path info entry missing path")
        if str(path).replace("\\", "/") != self.identity.remote_path(object_id):
            raise HfBucketStoreError("remote object path does not match the closed identity")
        size_raw = getattr(entry, "size", None)
        if size_raw is None and isinstance(entry, Mapping):
            size_raw = entry.get("size") or entry.get("size_bytes") or entry.get("Size")
        if size_raw is None:
            raise HfBucketStoreError("hf bucket path info entry missing size")
        try:
            size_bytes = int(size_raw)
        except (TypeError, ValueError) as exc:
            raise HfBucketStoreError("hf bucket path info entry has an invalid size") from exc
        if size_bytes < 0:
            raise HfBucketStoreError("hf bucket path info entry has an invalid size")
        if size_bytes > self._max_object_bytes:
            raise HfBucketStoreError("remote object exceeds the bounded maximum size")
        return size_bytes

    def remote_size(self, object_id: str) -> int | None:
        """Return the exact remote size, or ``None`` when the object is absent."""
        object_id = validate_object_id(object_id)
        remote_path = self.identity.remote_path(object_id)
        try:
            entries = list(
                self._client().get_bucket_paths_info(self.identity.bucket, [remote_path])
            )
        except (OSError, RuntimeError, ValueError):
            return None
        return self._parse_path_info(entries, object_id)

    def _resolve_object_id(self, ref: ArtifactRef) -> str:
        object_id = validate_object_id(ref.object_id)
        if ref.sha256 != f"sha256:{object_id}":
            raise HfBucketStoreError("artifact ref does not match the closed object layout")
        return object_id

    def read(self, ref: ArtifactRef) -> bytes:
        """Download by object_id, then verify the expected SHA-256 and size."""
        object_id = self._resolve_object_id(ref)
        size_bytes = self.remote_size(object_id)
        if size_bytes is None:
            raise HfBucketStoreError("artifact object is missing")
        if ref.size_bytes is not None and ref.size_bytes != size_bytes:
            raise HfBucketStoreError("artifact object size mismatch")
        spool = self._spool_directory()
        destination = spool / object_id
        remote_path = self.identity.remote_path(object_id)
        try:
            try:
                self._client().download_bucket_files(
                    self.identity.bucket,
                    [(remote_path, destination)],
                    raise_on_missing_files=True,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise HfBucketStoreError("hf bucket download failed") from exc
            if not destination.is_file():
                raise HfBucketStoreError("hf bucket download produced no local object")
            data = destination.read_bytes()
        finally:
            shutil.rmtree(spool, ignore_errors=True)
        if len(data) != size_bytes:
            raise HfBucketStoreError("downloaded object size mismatch")
        if sha256(data).hexdigest() != object_id:
            raise HfBucketStoreError("downloaded object digest mismatch")
        if ref.sha256 != f"sha256:{sha256(data).hexdigest()}":
            raise HfBucketStoreError("downloaded object sha256 mismatch")
        return data

    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef:
        """Content-address, idempotently upload, re-check, and return a reference."""
        object_id = sha256(data).hexdigest()
        size_bytes = len(data)
        if size_bytes > self._max_object_bytes:
            raise HfBucketStoreError("artifact exceeds the bounded maximum size")
        existing = self.remote_size(object_id)
        if existing is not None:
            if existing != size_bytes:
                raise HfBucketStoreError("existing remote object size mismatch")
        else:
            spool = self._spool_directory()
            remote_path = self.identity.remote_path(object_id)
            try:
                spool_path = spool / object_id
                spool_path.write_bytes(data)
                try:
                    self._client().batch_bucket_files(
                        self.identity.bucket,
                        add=[(spool_path, remote_path)],
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    raise HfBucketStoreError("hf bucket upload failed") from exc
            finally:
                shutil.rmtree(spool, ignore_errors=True)
            persisted = self.remote_size(object_id)
            if persisted != size_bytes:
                raise HfBucketStoreError("uploaded object was not persisted with exact size")
        return ArtifactRef(
            object_id=object_id,
            uri=self.identity.object_uri(object_id),
            sha256=f"sha256:{object_id}",
            media_type=media_type,
            size_bytes=size_bytes,
        )

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            object_id = self._resolve_object_id(ref)
        except HfBucketStoreError:
            return False
        try:
            return self.remote_size(object_id) is not None
        except HfBucketStoreError:
            return False

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            data = self.read(ref)
        except HfBucketStoreError:
            return False
        return (
            f"sha256:{sha256(data).hexdigest()}" == ref.sha256
            and (ref.size_bytes is None or len(data) == ref.size_bytes)
        )
