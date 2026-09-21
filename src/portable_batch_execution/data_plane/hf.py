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

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from portable_batch_execution.contracts import ArtifactRef

HF_DIRECT_KIND = "hf-buckets-direct"
OBJECTS_KEY = "objects/sha256"
OBJECT_LAYOUT_VERSION = "sha256-flat.v1"
REQUIRED_HF_CLI_VERSION = "1.8.0"

#: Environment variable holding the token injected from the fixed GH secret.
HF_TOKEN_ENV = "HF_SYSTEM_TRADING_DATA_RW_TOKEN"
#: Child-process variable the ``hf`` CLI reads the token from (never argv/files).
HF_CLI_TOKEN_ENV = "HF_TOKEN"

PBE_HF_BUCKET_ENV = "PBE_HF_BUCKET"
PBE_HF_PREFIX_ENV = "PBE_HF_PREFIX"
PBE_HF_OBJECT_LAYOUT_ENV = "PBE_HF_OBJECT_LAYOUT"
PBE_HF_CLI_BINARY_ENV = "PBE_HF_CLI_BINARY"

DEFAULT_HF_CLI_BINARY = "hf"
DEFAULT_MAX_OBJECT_BYTES = 1024 * 1024 * 1024

_BUCKET_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_PREFIX_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[0-9a-f]{64}$")
_BARE_VERSION_LINE = re.compile(r"^([0-9][0-9A-Za-z.+-]*)$")
_PREFIXED_VERSION_LINE = re.compile(r"^version:\s*([0-9][0-9A-Za-z.+-]*)$", re.IGNORECASE)

HfRunner = Callable[..., Any]


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
    """Parse ``hf version`` stdout: one bare semver or one ``version:`` line."""
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


def run_hf_command(argv: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv), env=dict(env), capture_output=True, text=True, check=False
    )


class HfBucketArtifactStore:
    """Artifact read/write/exists/verify over one closed HF bucket identity."""

    def __init__(
        self,
        identity: HfBucketIdentity,
        *,
        runner: HfRunner | None = None,
        token: str | None = None,
        spool_root: Path | None = None,
        hf_binary: str | None = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self.identity = identity
        self._run: HfRunner = runner or run_hf_command
        self._token = token
        self._spool_root = Path(spool_root) if spool_root is not None else None
        self._hf_binary = hf_binary or os.environ.get(
            PBE_HF_CLI_BINARY_ENV, DEFAULT_HF_CLI_BINARY
        )
        if max_object_bytes <= 0:
            raise HfBucketStoreError("max_object_bytes must be positive")
        self._max_object_bytes = max_object_bytes

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None, **kwargs: Any
    ) -> HfBucketArtifactStore:
        return cls(HfBucketIdentity.from_environment(env), **kwargs)

    # -- invocation ------------------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        token = self._token if self._token is not None else os.environ.get(HF_TOKEN_ENV)
        if token:
            env[HF_CLI_TOKEN_ENV] = token
        return env

    def _invoke(self, argv: Sequence[str], *, operation: str) -> subprocess.CompletedProcess:
        completed = self._run(list(argv), env=self._child_env())
        if completed.returncode != 0:
            raise HfBucketStoreError(
                f"hf {operation} failed with exit code {completed.returncode}"
            )
        return completed

    def cli_version(self) -> str:
        completed = self._invoke([self._hf_binary, "version"], operation="version")
        return parse_hf_cli_version_output(completed.stdout or "")

    def preflight(self) -> None:
        """Fail closed unless the pinned HF CLI is present and exactly versioned."""
        if shutil.which(self._hf_binary) is None:
            raise HfBucketStoreError("hf CLI binary is not available on PATH")
        version = self.cli_version()
        if version != REQUIRED_HF_CLI_VERSION:
            raise HfBucketStoreError(
                f"hf CLI version {version!r} does not match required "
                f"{REQUIRED_HF_CLI_VERSION!r}"
            )

    def _spool_directory(self) -> Path:
        if self._spool_root is not None:
            self._spool_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="hf-direct-", dir=self._spool_root))

    # -- remote metadata -------------------------------------------------------

    def _parse_list(self, stdout: str | None, object_id: str) -> int | None:
        raw = stdout if stdout is not None else ""
        text = raw.strip()
        if not text or text in ("[]", "null", "(empty)"):
            return None
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise HfBucketStoreError("invalid hf buckets list JSON response") from exc
        if isinstance(payload, dict):
            entries = [payload]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise HfBucketStoreError("ambiguous hf buckets list response")
        if not entries:
            return None
        if len(entries) != 1:
            raise HfBucketStoreError("ambiguous remote object listing")
        entry = entries[0]
        if not isinstance(entry, dict):
            raise HfBucketStoreError("invalid hf buckets list entry")
        path = entry.get("path") or entry.get("name")
        if path is None:
            raise HfBucketStoreError("hf buckets list entry missing path")
        if str(path).replace("\\", "/") != self.identity.remote_path(object_id):
            raise HfBucketStoreError("remote object path does not match the closed identity")
        size_raw = entry.get("size") or entry.get("size_bytes") or entry.get("Size")
        if size_raw is None:
            raise HfBucketStoreError("hf buckets list entry missing size")
        try:
            size_bytes = int(size_raw)
        except (TypeError, ValueError) as exc:
            raise HfBucketStoreError("hf buckets list entry has an invalid size") from exc
        if size_bytes < 0:
            raise HfBucketStoreError("hf buckets list entry has an invalid size")
        if size_bytes > self._max_object_bytes:
            raise HfBucketStoreError("remote object exceeds the bounded maximum size")
        return size_bytes

    def remote_size(self, object_id: str) -> int | None:
        """Return the exact remote size, or ``None`` when the object is absent."""
        object_id = validate_object_id(object_id)
        completed = self._run(
            [
                self._hf_binary,
                "buckets",
                "list",
                self.identity.object_uri(object_id),
                "--format",
                "json",
            ],
            env=self._child_env(),
        )
        if completed.returncode != 0:
            return None
        return self._parse_list(completed.stdout, object_id)

    # -- artifact operations ---------------------------------------------------

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
        try:
            self._invoke(
                [
                    self._hf_binary,
                    "buckets",
                    "cp",
                    self.identity.object_uri(object_id),
                    str(destination),
                ],
                operation="buckets cp",
            )
            if not destination.is_file():
                raise HfBucketStoreError("hf buckets cp produced no local object")
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
            try:
                spool_path = spool / object_id
                spool_path.write_bytes(data)
                self._invoke(
                    [
                        self._hf_binary,
                        "buckets",
                        "cp",
                        str(spool_path),
                        self.identity.object_uri(object_id),
                    ],
                    operation="buckets cp",
                )
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
