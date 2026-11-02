"""Ephemeral Hugging Face model download via GitHub Actions OIDC.

Never accepts a static HF credential. No token or remote error body is logged.
All downloads live inside the caller's per-attempt TemporaryDirectory.
"""

from __future__ import annotations

import os
import re
import subprocess
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.model_delivery import HFPrivateModelDelivery


class HFModelDownloadError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _failure_from_status(code: int | None) -> str:
    if code in (401, 403):
        return "hf_model_auth_failed"
    if code == 429:
        return "hf_model_rate_limited"
    return "hf_model_download_failed"


def _mint_oidc_token(repo_id: str, root: Path) -> str:
    # Refuse all ambient credentials. The HF CLI obtains and exchanges a
    # fresh GitHub OIDC identity, scoped to the configured repo publisher.
    if not os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN") or not os.environ.get(
        "ACTIONS_ID_TOKEN_REQUEST_URL"
    ):
        raise HFModelDownloadError("hf_model_oidc_unavailable")
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env.update(
        {
            "HF_OIDC_RESOURCE": repo_id,
            "HF_HOME": str(root / "hf-home"),
            "HF_HUB_CACHE": str(root / "hf-home" / "hub"),
            "HF_XET_CACHE": str(root / "hf-home" / "xet"),
        }
    )
    try:
        result = subprocess.run(
            ["hf", "auth", "token"],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise HFModelDownloadError("hf_model_oidc_unavailable") from None
    if result.returncode != 0:
        # The response body could contain identifiers; do not return it.
        detail = result.stderr
        match = re.search(r"\b(401|403|429)\b", detail)
        status = int(match.group(1)) if match else None
        if status is None and re.search(r"\b(?:invalid_grant|invalid_request)\b", detail, re.I):
            raise HFModelDownloadError("hf_model_auth_failed")
        raise HFModelDownloadError(_failure_from_status(status))
    token = result.stdout.strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9_.-]{12,8192}", token):
        raise HFModelDownloadError("hf_model_auth_failed")
    return token


def _verify_download(path: Path, ref: ArtifactRef) -> None:
    try:
        if not path.is_file() or path.is_symlink() or ref.size_bytes is None:
            raise HFModelDownloadError("hf_model_digest_mismatch")
        if path.stat().st_size != ref.size_bytes:
            raise HFModelDownloadError("hf_model_digest_mismatch")
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if f"sha256:{digest.hexdigest()}" != ref.sha256:
            raise HFModelDownloadError("hf_model_digest_mismatch")
    except OSError:
        raise HFModelDownloadError("hf_model_download_failed") from None


def stage_private_hf_models(
    source: HFPrivateModelDelivery,
    refs: tuple[ArtifactRef, ...],
    root: Path,
) -> tuple[Path, Path]:
    """Stage pinned models; return validated directories for local-only inference."""
    if len(refs) != 4 or any(ref.size_bytes is None for ref in refs):
        raise HFModelDownloadError("hf_model_reference_invalid")
    token = _mint_oidc_token(source.repo_id, root)
    # Local per-job dir only: no home-scoped persistent SDK/Xet caches.
    env_keys = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE")
    previous = {key: os.environ.get(key) for key in env_keys}
    os.environ["HF_HOME"] = str(root / "hf-home")
    os.environ["HF_HUB_CACHE"] = str(root / "hf-home" / "hub")
    os.environ["HF_XET_CACHE"] = str(root / "hf-home" / "xet")
    try:
        from huggingface_hub import hf_hub_download

        model_dirs = (root / "model_a", root / "model_b")
        for index, (filename, ref) in enumerate(zip(source.filenames, refs, strict=True)):
            try:
                downloaded = hf_hub_download(
                    repo_id=source.repo_id,
                    filename=filename,
                    revision=source.revision,
                    repo_type="model",
                    token=token,
                    local_dir=root / "hf-download",
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                raise HFModelDownloadError(_failure_from_status(status)) from None
            # The SDK must not return an outside path or redirected file.
            source_path = Path(downloaded)
            if not source_path.resolve().is_relative_to((root / "hf-download").resolve()):
                raise HFModelDownloadError("hf_model_download_failed")
            _verify_download(source_path, ref)
            target_dir = model_dirs[0 if index < 2 else 1]
            target_dir.mkdir(exist_ok=True)
            target = target_dir / ("config.json" if index % 2 == 0 else "model.safetensors")
            # Both paths are under this attempt root: atomic rename avoids an
            # additional 816 MB disk copy and keeps storage ephemeral.
            source_path.replace(target)
        return model_dirs
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
