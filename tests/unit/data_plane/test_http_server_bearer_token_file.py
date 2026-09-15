from unittest.mock import patch

import pytest

from portable_batch_execution.data_plane.http_server import (
    serve_private_data_plane_from_environment,
)

_FAKE_BEARER = "fake-data-plane-bearer-for-unit-test"
_PATH_MARKER = "bearer-token-file-marker-9c2e"


def test_serve_from_environment_loads_bearer_token_file(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    state_root.mkdir()
    token_file = tmp_path / "bearer.token"
    token_file.write_text(f"  {_FAKE_BEARER}  \n", encoding="utf-8")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_STATE_ROOT", str(state_root))
    monkeypatch.delenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BIND_PORT", "0")

    with patch(
        "portable_batch_execution.data_plane.http_server.PrivateDataPlaneService"
    ) as service_cls:
        server = serve_private_data_plane_from_environment()
        try:
            service_cls.assert_called_once_with(state_root, _FAKE_BEARER)
        finally:
            server.server_close()


def test_serve_from_environment_literal_bearer_token_unchanged(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_STATE_ROOT", str(state_root))
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", _FAKE_BEARER)
    monkeypatch.delenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE", raising=False)
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BIND_PORT", "0")

    with patch(
        "portable_batch_execution.data_plane.http_server.PrivateDataPlaneService"
    ) as service_cls:
        server = serve_private_data_plane_from_environment()
        try:
            service_cls.assert_called_once_with(state_root, _FAKE_BEARER)
        finally:
            server.server_close()


def test_serve_from_environment_rejects_dual_bearer_sources_without_leaking(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "state"
    state_root.mkdir()
    token_file = tmp_path / _PATH_MARKER
    token_file.write_text(_FAKE_BEARER, encoding="utf-8")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_STATE_ROOT", str(state_root))
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", _FAKE_BEARER)
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE", str(token_file))

    with pytest.raises(
        ValueError, match="private data plane server environment is not configured"
    ) as exc:
        serve_private_data_plane_from_environment()

    message = str(exc.value)
    assert _FAKE_BEARER not in message
    assert _PATH_MARKER not in message


@pytest.mark.parametrize("contents", ("", "   \n"))
def test_serve_from_environment_rejects_empty_bearer_token_file_without_leaking(
    tmp_path, contents, monkeypatch
):
    state_root = tmp_path / "state"
    state_root.mkdir()
    token_file = tmp_path / _PATH_MARKER
    token_file.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_STATE_ROOT", str(state_root))
    monkeypatch.delenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE", str(token_file))

    with pytest.raises(
        ValueError, match="private data plane server environment is not configured"
    ) as exc:
        serve_private_data_plane_from_environment()

    assert _FAKE_BEARER not in str(exc.value)
    assert _PATH_MARKER not in str(exc.value)


def test_serve_from_environment_rejects_missing_bearer_token_file_without_leaking(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "state"
    state_root.mkdir()
    missing = tmp_path / _PATH_MARKER
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_STATE_ROOT", str(state_root))
    monkeypatch.delenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE", str(missing))

    with pytest.raises(
        ValueError, match="private data plane server environment is not configured"
    ) as exc:
        serve_private_data_plane_from_environment()

    assert _PATH_MARKER not in str(exc.value)
