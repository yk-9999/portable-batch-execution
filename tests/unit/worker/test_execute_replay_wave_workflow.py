import importlib.util
import json
import tomllib
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "execute-wave.yml"
RESOLVE_SCRIPT = (
    REPO_ROOT / ".github" / "scripts" / "resolve_execute_wave_dependency_profile.py"
)
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_resolve_module():
    spec = importlib.util.spec_from_file_location(
        "resolve_execute_wave_dependency_profile", RESOLVE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture
def replay_optional_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["replay"]


def test_execute_wave_remains_dispatch_workflow(workflow):
    assert WORKFLOW.name == "execute-wave.yml"
    assert "workflow_dispatch:" in workflow
    assert "execute-replay-wave" not in workflow
    assert not (
        REPO_ROOT / ".github" / "workflows" / "execute-replay-wave.yml"
    ).exists()


def test_execute_wave_conditional_replay_dependency_install(workflow):
    assert "resolve_execute_wave_dependency_profile.py" in workflow
    assert "steps.dependency_profile.outputs.profile == 'replay-batch'" in workflow
    assert "uv sync --extra replay" in workflow
    assert "if: steps.dependency_profile.outputs.profile != 'replay-batch'" in workflow
    assert "uv sync --dev --extra distilbert" in workflow


def test_execute_wave_replay_path_skips_heavy_setup(workflow):
    replay_install = workflow.split("Install dependencies (replay-batch)", 1)[1].split(
        "Install dependencies (default)", 1
    )[0]
    assert "distilbert" not in replay_install
    assert "--dev" not in replay_install

    ffmpeg_block = workflow.split("name: Install FFmpeg", 1)[1].split(
        "uv run python", 1
    )[0]
    assert "steps.dependency_profile.outputs.profile != 'replay-batch'" in ffmpeg_block
    assert "ffmpeg" in ffmpeg_block.lower()
    assert "apt-get" in ffmpeg_block


def test_execute_wave_worker_invocation_and_secrets_unchanged(workflow):
    assert "contents: read" in workflow
    assert "python -m portable_batch_execution.worker.execute_wave" in workflow
    assert '"$PBE_WAVE_ID"' in workflow
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in workflow
    assert "PBE_RUN_ID: ${{ inputs.run_id }}" in workflow
    assert (
        "PBE_MODE: ${{ inputs.jpx_hf_direct && 'hf-direct' || (inputs.private && 'private' || 'public') }}"
        in workflow
    )
    assert (
        "PBE_PRIVATE_DATA_PLANE_BASE_URL: ${{ inputs.private && !inputs.jpx_hf_direct && secrets.PBE_PRIVATE_DATA_PLANE_BASE_URL || '' }}"
        in workflow
    )
    assert (
        "PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN: ${{ inputs.private && !inputs.jpx_hf_direct && secrets.PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN || '' }}"
        in workflow
    )
    assert "hf_wave_descriptor_ref" in workflow
    assert "PBE_HF_WAVE_DESCRIPTOR_REF" in workflow
    assert "PBE_EXPECTED_PUBLIC_REVISION" in workflow
    assert "github.sha" in workflow
    assert "pytest" not in workflow


def test_replay_optional_dependency_profile_covers_worker_imports(
    replay_optional_dependencies,
):
    joined = " ".join(replay_optional_dependencies).lower()
    assert "tabular" in joined
    assert "numpy" in joined
    assert "distilbert" not in joined
    assert "torch" not in joined
    assert "transformers" not in joined


def test_replay_worker_entry_imports_without_ml_or_torch():
    module = importlib.import_module("portable_batch_execution.worker.execute_wave")
    assert module.__name__ == "portable_batch_execution.worker.execute_wave"


def test_resolve_profile_uses_private_run_wave_endpoint_with_bearer():
    resolve = _load_resolve_module()
    captured: dict[str, object] = {}
    secret = "super-secret-bearer-token-value"

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"job": {"pack": "replay-batch"}}).encode("utf-8")

    def fake_urlopen(request, timeout=60):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        return _Response()

    profile = resolve.resolve_profile(
        mode="private",
        run_id="opaque-run",
        wave_id="opaque-wave",
        base_url="https://plane.example",
        bearer_token=secret,
        urlopen=fake_urlopen,
    )

    assert profile == "replay-batch"
    assert (
        captured["url"] == "https://plane.example/v1/runs/opaque-run/waves/opaque-wave"
    )
    assert captured["authorization"] == f"Bearer {secret}"


def test_resolve_profile_falls_back_when_pack_is_not_replay_batch():
    resolve = _load_resolve_module()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"job": {"pack": "ml-batch"}}).encode("utf-8")

    profile = resolve.resolve_profile(
        mode="private",
        run_id="opaque-run",
        wave_id="opaque-wave",
        base_url="https://plane.example",
        bearer_token="token",
        urlopen=lambda request, timeout=60: _Response(),
    )
    assert profile == "fallback"


@pytest.mark.parametrize(
    ("mode", "run_id", "wave_id"),
    [
        ("public", "opaque-run", "opaque-wave"),
        ("private", "", "opaque-wave"),
        ("private", "opaque-run", ""),
    ],
)
def test_resolve_profile_falls_back_without_private_run_wave(mode, run_id, wave_id):
    resolve = _load_resolve_module()
    profile = resolve.resolve_profile(
        mode=mode,
        run_id=run_id,
        wave_id=wave_id,
        base_url="https://plane.example",
        bearer_token="token",
        urlopen=MagicMock(),
    )
    assert profile == "fallback"


def test_resolve_profile_main_does_not_print_bearer_token(tmp_path, monkeypatch):
    secret = "must-not-appear-in-output-7f3a"
    github_output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setenv("PBE_MODE", "private")
    monkeypatch.setenv("PBE_RUN_ID", "opaque-run")
    monkeypatch.setenv("PBE_WAVE_ID", "opaque-wave")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BASE_URL", "https://plane.example")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", secret)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"job": {"pack": "tabular-batch"}}).encode("utf-8")

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=60: _Response()
    )
    resolve = _load_resolve_module()
    assert resolve.main() == 0
    assert secret not in github_output.read_text(encoding="utf-8")
    assert github_output.read_text(encoding="utf-8") == "profile=fallback\n"
