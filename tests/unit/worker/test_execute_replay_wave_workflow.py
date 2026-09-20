import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "execute-replay-wave.yml"
EXECUTE_WAVE = REPO_ROOT / ".github" / "workflows" / "execute-wave.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture
def replay_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture
def replay_optional_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["replay"]


def test_replay_workflow_dispatch_contract_matches_execute_wave(replay_workflow):
    execute_wave = EXECUTE_WAVE.read_text(encoding="utf-8")

    for marker in (
        "wave_id:",
        "run_id:",
        "private:",
        'description: Planned wave identifier',
        "required: true",
        "type: boolean",
    ):
        assert marker in replay_workflow
        assert marker in execute_wave


def test_replay_workflow_least_permissions_and_worker_invocation(replay_workflow):
    assert "contents: read" in replay_workflow
    assert "python -m portable_batch_execution.worker.execute_wave" in replay_workflow
    assert '"$PBE_WAVE_ID"' in replay_workflow
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in replay_workflow
    assert "PBE_RUN_ID: ${{ inputs.run_id }}" in replay_workflow
    assert "PBE_MODE: ${{ inputs.private && 'private' || 'public' }}" in replay_workflow
    assert (
        "PBE_PRIVATE_DATA_PLANE_BASE_URL: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BASE_URL || '' }}"
        in replay_workflow
    )
    assert (
        "PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN || '' }}"
        in replay_workflow
    )
    assert "pytest" not in replay_workflow


def test_replay_workflow_avoids_heavy_execute_wave_setup(replay_workflow):
    execute_wave = EXECUTE_WAVE.read_text(encoding="utf-8")

    assert "distilbert" not in replay_workflow
    assert "ffmpeg" not in replay_workflow.lower()
    assert "apt-get" not in replay_workflow
    assert "uv sync --extra replay" in replay_workflow
    assert "--dev" not in replay_workflow

    assert "distilbert" in execute_wave
    assert "ffmpeg" in execute_wave.lower()
    assert "uv sync --dev --extra distilbert" in execute_wave


def test_replay_optional_dependency_profile_covers_worker_imports(replay_optional_dependencies):
    joined = " ".join(replay_optional_dependencies).lower()
    assert "tabular" in joined
    assert "numpy" in joined
    assert "distilbert" not in joined
    assert "torch" not in joined
    assert "transformers" not in joined


def test_replay_worker_entry_imports_without_ml_or_torch():
    import importlib

    module = importlib.import_module("portable_batch_execution.worker.execute_wave")
    assert module.__name__ == "portable_batch_execution.worker.execute_wave"
