"""Broker-side artifact output readers (local data plane or optional loopback HTTP)."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlsplit

import httpx

from portable_batch_execution.contracts import ArtifactRef

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


class BrokerArtifactReader(Protocol):
    def read(self, ref: ArtifactRef) -> bytes: ...


class LocalDataPlaneArtifactReader:
    """Read artifact bytes from the controller-local data plane (default)."""

    def __init__(self, data_plane) -> None:
        self._data_plane = data_plane

    def read(self, ref: ArtifactRef) -> bytes:
        return self._data_plane.read(ref)


class HttpBrokerArtifactReader:
    """Fetch artifact bytes from a private data plane HTTP content endpoint."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = validate_artifact_read_base_url(base_url)
        if not bearer_token:
            raise ValueError("artifact read bearer token is required")
        self._client = client or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._client.headers.setdefault("Authorization", f"Bearer {bearer_token}")

    def read(self, ref: ArtifactRef) -> bytes:
        path = (
            f"/v1/artifacts/{_opaque_part(ref.object_id, 'artifact object_id')}/content"
        )
        try:
            response = self._client.get(f"{self._base_url}{path}")
        except httpx.HTTPError as exc:
            raise ValueError("artifact_read_failed") from exc
        if response.status_code >= 400:
            raise ValueError("artifact_read_failed")
        payload = response.content
        if f"sha256:{sha256(payload).hexdigest()}" != ref.sha256:
            raise ValueError("artifact_digest_mismatch")
        if ref.size_bytes is not None and len(payload) != ref.size_bytes:
            raise ValueError("artifact_size_mismatch")
        return payload


def validate_artifact_read_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("artifact read base URL must be an origin URL")
    if parsed.scheme == "https":
        return base_url.rstrip("/")
    if parsed.scheme == "http":
        host = parsed.hostname
        if host is None or host.lower() not in _LOOPBACK_HOSTS:
            raise ValueError("artifact read base URL must use loopback host for http")
        return base_url.rstrip("/")
    raise ValueError("artifact read base URL must use http or https")


def _read_bearer_token_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("artifact read token file is not readable") from exc
    token = raw.strip()
    if not token:
        raise ValueError("artifact read token file is empty")
    return token


def _opaque_part(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(c in value for c in "/\\?#")
    ):
        raise ValueError(f"{name} must be an opaque identifier")
    return quote(value, safe="")


def build_broker_artifact_reader(
    data_plane,
    *,
    artifact_read_base_url: str | None = None,
    artifact_read_token_file: Path | None = None,
    client: httpx.Client | None = None,
) -> BrokerArtifactReader:
    if artifact_read_base_url is None and artifact_read_token_file is None:
        return LocalDataPlaneArtifactReader(data_plane)
    if artifact_read_base_url is None or artifact_read_token_file is None:
        raise ValueError(
            "artifact read base URL and token file must both be configured"
        )
    token = _read_bearer_token_file(artifact_read_token_file.resolve())
    return HttpBrokerArtifactReader(
        artifact_read_base_url,
        token,
        client=client,
    )
