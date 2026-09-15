from unittest.mock import MagicMock, patch

import pytest

from portable_batch_execution.backends.github_actions import BackendExecutionRef
from portable_batch_execution.controller import cli

_FAKE_TOKEN = "fake-github-pat-for-unit-test"
_PATH_MARKER = "token-file-path-marker-7f3a"


def _dispatch_argv(state_root: str, token_file: str | None = None) -> list[str]:
    argv = [
        "dispatch",
        "--state-root",
        state_root,
        "--run-id",
        "opaque-run",
        "--wave-id",
        "opaque-wave",
        "--github-owner",
        "owner",
        "--github-repo",
        "repo",
        "--github-workflow",
        "execute-wave.yml",
    ]
    if token_file is not None:
        argv.extend(["--github-token-file", token_file])
    return argv


def test_dispatch_github_token_file_passes_token_without_leaking(tmp_path, capsys):
    token_file = tmp_path / "github.token"
    token_file.write_text(f"  {_FAKE_TOKEN}  \n", encoding="utf-8")
    execution = BackendExecutionRef("github-actions", "99", "https://example/run")

    with (
        patch.object(cli, "GitHubActionsBackend") as backend_cls,
        patch.object(cli, "A1Controller") as controller_cls,
    ):
        controller_cls.return_value.dispatch_private_wave.return_value = execution
        rc = cli.main(_dispatch_argv(str(tmp_path), str(token_file)))

    assert rc == 0
    backend_cls.assert_called_once()
    assert backend_cls.call_args.kwargs["token"] == _FAKE_TOKEN
    captured = capsys.readouterr()
    assert _FAKE_TOKEN not in captured.out
    assert _FAKE_TOKEN not in captured.err
    assert str(token_file) not in captured.out
    assert str(token_file) not in captured.err


def test_inspect_github_token_file_passes_token_without_leaking(tmp_path, capsys):
    token_file = tmp_path / "github.token"
    token_file.write_text(_FAKE_TOKEN, encoding="utf-8")

    with (
        patch.object(cli, "GitHubActionsBackend") as backend_cls,
        patch.object(cli, "A1Controller") as controller_cls,
    ):
        controller_cls.return_value.inspect_run.return_value = {"run_id": "opaque-run"}
        rc = cli.main(
            [
                "inspect",
                "--state-root",
                str(tmp_path),
                "--run-id",
                "opaque-run",
                "--github-owner",
                "owner",
                "--github-repo",
                "repo",
                "--github-workflow",
                "execute-wave.yml",
                "--github-token-file",
                str(token_file),
            ]
        )

    assert rc == 0
    backend_cls.assert_called_once()
    assert backend_cls.call_args.kwargs["token"] == _FAKE_TOKEN
    captured = capsys.readouterr()
    assert _FAKE_TOKEN not in captured.out
    assert _FAKE_TOKEN not in captured.err


def test_dispatch_without_github_token_file_leaves_backend_env_default(tmp_path):
    with (
        patch.object(cli, "GitHubActionsBackend") as backend_cls,
        patch.object(cli, "A1Controller") as controller_cls,
    ):
        controller_cls.return_value.dispatch_private_wave.return_value = MagicMock(
            backend_id="github-actions",
            execution_id="1",
        )
        cli.main(_dispatch_argv(str(tmp_path)))

    backend_cls.assert_called_once()
    assert "token" not in backend_cls.call_args.kwargs


@pytest.mark.parametrize("contents", ("", "   \n"))
def test_dispatch_rejects_empty_github_token_file_without_leaking(
    tmp_path, contents, capsys
):
    token_file = tmp_path / _PATH_MARKER
    token_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="github token file is not available") as exc:
        cli.main(_dispatch_argv(str(tmp_path), str(token_file)))

    message = str(exc.value)
    assert _FAKE_TOKEN not in message
    assert _PATH_MARKER not in message
    captured = capsys.readouterr()
    assert _FAKE_TOKEN not in captured.out + captured.err


def test_dispatch_rejects_missing_github_token_file_without_leaking(tmp_path, capsys):
    missing = tmp_path / _PATH_MARKER

    with pytest.raises(ValueError, match="github token file is not available") as exc:
        cli.main(_dispatch_argv(str(tmp_path), str(missing)))

    assert _PATH_MARKER not in str(exc.value)
    captured = capsys.readouterr()
    assert _PATH_MARKER not in captured.out + captured.err
