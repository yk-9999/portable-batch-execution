"""Model downloads must be repository/OIDC scoped, ephemeral and content pinned."""

from __future__ import annotations

import sys
import types
from hashlib import sha256
from types import SimpleNamespace

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.model_delivery import HFPrivateModelDelivery
from portable_batch_execution.worker import hf_private_models as hf

_FILENAMES = (
    "model_a/config.json",
    "model_a/model.safetensors",
    "model_b/config.json",
    "model_b/model.safetensors",
)
_SOURCE = HFPrivateModelDelivery(
    kind="hf-private-repo-oidc",
    repo_id="sample/private-model",
    revision="a" * 40,
    filenames=_FILENAMES,
)
_PAYLOADS = (b'{"model_type":"distilbert"}', b"a" * 900, b'{"model_type":"distilbert"}', b"b" * 500)


def _refs(payloads=_PAYLOADS):
    return tuple(
        ArtifactRef(
            object_id=f"model-{i}",
            uri=f"pbe://private/model-{i}",
            sha256="sha256:" + sha256(data).hexdigest(),
            size_bytes=len(data),
        )
        for i, data in enumerate(payloads)
    )


def _fake_hub(monkeypatch, *, fail_status=None, corrupt=None):
    calls = []
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://github.example/oidc")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "github-internal")
    def cli(argv, **kwargs):
        assert argv == ["hf", "auth", "token"]
        assert kwargs["env"]["HF_OIDC_RESOURCE"] == _SOURCE.repo_id
        assert "HF_TOKEN" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="hf_jwt_fake-very-long-ephemeral-credential", stderr="")
    monkeypatch.setattr(hf.subprocess, "run", cli)
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == _SOURCE.repo_id
        assert kwargs["revision"] == "a" * 40
        assert kwargs["token"] == "hf_jwt_fake-very-long-ephemeral-credential"
        assert kwargs["local_dir"].name == "hf-download"
        if fail_status is not None:
            error = RuntimeError("secret response body should not escape")
            error.response = SimpleNamespace(status_code=fail_status)
            raise error
        file_index = _FILENAMES.index(kwargs["filename"])
        data = (_PAYLOADS[file_index] if corrupt is None or corrupt != file_index else b"wrong")
        dest = kwargs["local_dir"] / kwargs["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return str(dest)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_hf_download_uses_pinned_repo_files_and_ephemeral_paths(tmp_path, monkeypatch):
    calls = _fake_hub(monkeypatch)
    a, b = hf.stage_private_hf_models(_SOURCE, _refs(), tmp_path)
    assert (a / "config.json").read_bytes() == _PAYLOADS[0]
    assert (a / "model.safetensors").read_bytes() == _PAYLOADS[1]
    assert (b / "config.json").read_bytes() == _PAYLOADS[2]
    assert (b / "model.safetensors").read_bytes() == _PAYLOADS[3]
    assert [c["filename"] for c in calls] == list(_FILENAMES)
    assert not list(tmp_path.glob("hf-download/*/model.safetensors"))


@pytest.mark.parametrize("status,code", [
    (401, "hf_model_auth_failed"),
    (403, "hf_model_auth_failed"),
    (429, "hf_model_rate_limited"),
    (503, "hf_model_download_failed"),
])
def test_http_failures_are_sanitized(tmp_path, monkeypatch, status, code):
    _fake_hub(monkeypatch, fail_status=status)
    with pytest.raises(hf.HFModelDownloadError) as result:
        hf.stage_private_hf_models(_SOURCE, _refs(), tmp_path)
    assert result.value.code == code
    assert "secret" not in str(result.value)


@pytest.mark.parametrize("bad_index", [0, 1, 2, 3])
def test_corrupt_model_is_never_accepted(tmp_path, monkeypatch, bad_index):
    _fake_hub(monkeypatch, corrupt=bad_index)
    with pytest.raises(hf.HFModelDownloadError, match="hf_model_digest_mismatch"):
        hf.stage_private_hf_models(_SOURCE, _refs(), tmp_path)


def test_oidc_unavailable_never_falls_back_to_static_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "a-permanent-token-must-not-be-used")
    with pytest.raises(hf.HFModelDownloadError, match="hf_model_oidc_unavailable"):
        hf.stage_private_hf_models(_SOURCE, _refs(), tmp_path)


def test_oidc_exchange_errors_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://github.example")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "oidc-identity")
    monkeypatch.setattr(hf.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr="403 Unauthorized token=PRIVATE"
    ))
    with pytest.raises(hf.HFModelDownloadError, match="hf_model_auth_failed"):
        hf.stage_private_hf_models(_SOURCE, _refs(), tmp_path)


def test_same_content_different_expected_sha_fails(tmp_path, monkeypatch):
    _fake_hub(monkeypatch)
    refs = list(_refs())
    refs[2] = refs[2].model_copy(update={"sha256": "sha256:" + "0" * 64})
    with pytest.raises(hf.HFModelDownloadError, match="hf_model_digest_mismatch"):
        hf.stage_private_hf_models(_SOURCE, tuple(refs), tmp_path)
