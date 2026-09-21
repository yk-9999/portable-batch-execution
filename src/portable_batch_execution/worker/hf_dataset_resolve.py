"""Pinned public Hugging Face dataset resolve URL downloads (runner-local only)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.transport.hf_bucket import verify_bytes_identity

_HF_DATASET_RESOLVE = re.compile(
    r"^https://huggingface\.co/(?:datasets|spaces)/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+$"
)
_PARQUET_MEDIA_TYPES = frozenset(
    {
        "application/vnd.apache.parquet",
        "application/x-parquet",
    }
)


class HfDatasetResolveError(RuntimeError):
    """Fail-closed pinned HF dataset resolve error."""


def artifact_ref_is_pinned_hf_dataset_resolve(ref: ArtifactRef) -> bool:
    try:
        validate_pinned_hf_dataset_resolve_uri(ref.uri)
        return True
    except HfDatasetResolveError:
        return False


def validate_pinned_hf_dataset_resolve_uri(uri: str) -> str:
    if not uri.startswith("https://huggingface.co/"):
        raise HfDatasetResolveError("HF resolve URL must use huggingface.co")
    if re.search(r"://[^/]*@", uri):
        raise HfDatasetResolveError("HF resolve URL must not embed credentials")
    if "?" in uri or "#" in uri:
        raise HfDatasetResolveError("HF resolve URL must not include query or fragment")
    if not _HF_DATASET_RESOLVE.fullmatch(uri):
        raise HfDatasetResolveError(
            "HF resolve URL must be a pinned dataset resolve path"
        )
    return uri


def validate_parquet_hf_resolve_ref(ref: ArtifactRef) -> ArtifactRef:
    validate_pinned_hf_dataset_resolve_uri(ref.uri)
    media_type = str(ref.media_type or "")
    if media_type not in _PARQUET_MEDIA_TYPES:
        raise HfDatasetResolveError("parquet input requires parquet media type")
    if ref.size_bytes is None or ref.size_bytes < 0:
        raise HfDatasetResolveError("size_bytes required")
    digest = ref.sha256.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HfDatasetResolveError("sha256 required")
    return ref


def download_verified_pinned_resolve(
    ref: ArtifactRef,
    dest: Path,
    *,
    opener: Callable[[str], object] | None = None,
) -> None:
    validated = validate_parquet_hf_resolve_ref(ref)
    dest.parent.mkdir(parents=True, exist_ok=True)
    open_fn = opener or (lambda url: urlopen(url, timeout=120))
    with open_fn(validated.uri) as response:
        data = response.read()
    expected_digest = validated.sha256.removeprefix("sha256:")
    try:
        verify_bytes_identity(
            data,
            sha256_hex=expected_digest,
            size_bytes=int(validated.size_bytes or 0),
        )
    except Exception as exc:
        raise HfDatasetResolveError(
            "pinned resolve bytes failed identity verification"
        ) from exc
    dest.write_bytes(data)


def pinned_resolve_input_digest(refs: tuple[ArtifactRef, ...]) -> str:
    material = []
    for ref in refs:
        validated = validate_parquet_hf_resolve_ref(ref)
        material.append(
            {
                "uri": validated.uri,
                "sha256": validated.sha256,
                "size_bytes": validated.size_bytes,
                "media_type": validated.media_type,
            }
        )
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()
