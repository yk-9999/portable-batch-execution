from pathlib import Path

import pytest

from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.worker import execute_public_wave


def test_public_worker_executes_committed_wave_and_only_appends_attempts(tmp_path):
    attempts = execute_public_wave("wave-0000", state_root=tmp_path)

    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].output_refs
    state = LocalFilesystemDataPlane(tmp_path)
    assert state.read_attempts("public-synthetic-rolling-v1") == attempts
    assert state.read_manifest("public-synthetic-rolling-v1") is None


@pytest.mark.parametrize("wave_id", ["", "wave-0001", "../wave-0000", "wave-0000; echo x", "$(whoami)"])
def test_public_worker_rejects_non_allowlisted_or_executable_wave_input(wave_id):
    with pytest.raises(ValueError):
        execute_public_wave(wave_id)


def test_execute_wave_workflow_invokes_worker_with_environment_boundary():
    workflow = (Path(__file__).parents[3] / ".github" / "workflows" / "execute-wave.yml").read_text()

    assert "python -m portable_batch_execution.worker.execute_wave" in workflow
    assert '"$PBE_WAVE_ID"' in workflow
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in workflow
    assert "pytest" not in workflow


def test_execute_wave_workflow_resolves_profile_before_conditionally_installing_ffmpeg():
    workflow = (Path(__file__).parents[3] / ".github" / "workflows" / "execute-wave.yml").read_text()

    assert "uv sync --dev --extra distilbert" in workflow
    profile_step = workflow.index("python -m portable_batch_execution.worker.runtime_profile")
    ffmpeg_step = workflow.index("- name: Install FFmpeg")
    assert profile_step < ffmpeg_step
    resolver_config = workflow.split("- name: Resolve runtime profile", 1)[1].split(
        "- name: Install FFmpeg", 1
    )[0]
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in resolver_config
    assert "PBE_RUN_ID: ${{ inputs.run_id }}" in resolver_config
    assert "PBE_MODE: ${{ inputs.private && 'private' || 'public' }}" in resolver_config
    assert "PBE_PRIVATE_DATA_PLANE_BASE_URL: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BASE_URL || '' }}" in resolver_config
    assert "PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN || '' }}" in resolver_config
    assert "PBE_EXTERNAL_API_NVIDIA_API_KEY" not in resolver_config

    worker_config = workflow.split(
        "- run: uv run python -m portable_batch_execution.worker.execute_wave", 1
    )[1]
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in worker_config
    assert "PBE_RUN_ID: ${{ inputs.run_id }}" in worker_config
    assert "PBE_MODE: ${{ inputs.private && 'private' || 'public' }}" in worker_config
    assert "PBE_PRIVATE_DATA_PLANE_BASE_URL: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BASE_URL || '' }}" in worker_config
    assert "PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN: ${{ inputs.private && secrets.PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN || '' }}" in worker_config
    assert "PBE_EXTERNAL_API_NVIDIA_API_KEY: ${{ inputs.private && secrets.PBE_EXTERNAL_API_NVIDIA_API_KEY || '' }}" in worker_config

    ffmpeg_install = workflow[ffmpeg_step:]
    assert "if: steps.profile.outputs.profile == 'media'" in ffmpeg_install
    assert "timeout-minutes: 15" in ffmpeg_install
    assert "sudo apt-get install -y --no-install-recommends ffmpeg" in ffmpeg_install
    assert workflow.count("sudo apt-get install") == 1
