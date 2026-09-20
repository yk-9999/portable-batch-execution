"""Resolve execute-wave CI dependency profile from private closed-wave metadata."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen as _default_urlopen

_REPLAY_PACK = "replay-batch"
_FALLBACK_PROFILE = "fallback"
_REPLAY_PROFILE = "replay-batch"


def _opaque_part(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or any(c in value for c in "/\\?#"):
        raise ValueError(f"{name} must be an opaque identifier")
    return quote(value, safe="")


def _valid_https_origin(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    return base_url.rstrip("/")


def resolve_profile(
    *,
    mode: str,
    run_id: str,
    wave_id: str,
    base_url: str,
    bearer_token: str,
    urlopen: Callable[..., Any] = _default_urlopen,
) -> str:
    if mode != "private":
        return _FALLBACK_PROFILE
    if not run_id or not wave_id:
        return _FALLBACK_PROFILE
    origin = _valid_https_origin(base_url)
    if not origin or not bearer_token:
        return _FALLBACK_PROFILE
    try:
        run_part = _opaque_part(run_id, "run_id")
        wave_part = _opaque_part(wave_id, "wave_id")
    except ValueError:
        return _FALLBACK_PROFILE
    url = f"{origin}/v1/runs/{run_part}/waves/{wave_part}"
    request = Request(url, headers={"Authorization": f"Bearer {bearer_token}"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        return _FALLBACK_PROFILE
    if not isinstance(payload, dict):
        return _FALLBACK_PROFILE
    job = payload.get("job")
    if not isinstance(job, dict):
        return _FALLBACK_PROFILE
    if job.get("pack") == _REPLAY_PACK:
        return _REPLAY_PROFILE
    return _FALLBACK_PROFILE


def main() -> int:
    profile = resolve_profile(
        mode=os.environ.get("PBE_MODE", ""),
        run_id=os.environ.get("PBE_RUN_ID", ""),
        wave_id=os.environ.get("PBE_WAVE_ID", ""),
        base_url=os.environ.get("PBE_PRIVATE_DATA_PLANE_BASE_URL", ""),
        bearer_token=os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", ""),
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"profile={profile}\n")
    else:
        sys.stdout.write(f"{profile}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
