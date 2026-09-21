import json
from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest

from portable_batch_execution.backends.github_actions import (
    BackendExecutionRef,
    GitHubActionsBackend,
)
from portable_batch_execution.contracts import (
    ShardAttemptRecord,
)
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.data_plane import (
    HttpPrivateDataPlane,
    PrivateDataPlaneService,
)
from portable_batch_execution.worker.execute_wave import (
    PrivateWaveExecutionError,
    execute_private_wave,
)


def _mock_backend(handler):
    return GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(handler)
        ),
    )


def test_prepare_registers_closed_wave_and_manifest(tmp_path):
    controller = A1Controller(tmp_path)
    prepared = controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    assert prepared.logical_run_id == "opaque-run"
    payload = controller.registry.resolve_wave("opaque-run", "opaque-wave")
    assert payload["job"]["operation"] == "tabular.rolling"
    manifest = controller.data_plane.read_manifest("opaque-run")
    assert manifest is not None and manifest.revision == 0


def test_dispatch_state_rejects_path_like_run_id(tmp_path):
    controller = A1Controller(tmp_path)
    with pytest.raises(ValueError, match="opaque identifier"):
        controller.inspect_run("../escape")
    with pytest.raises(ValueError, match="opaque identifier"):
        controller.inspect_run("..")
    sentinel = tmp_path / "controller" / "dispatch" / "escape.json"
    assert not sentinel.exists()


def test_dispatch_persists_exact_backend_execution_id(tmp_path):
    def handler(request):
        return httpx.Response(
            201, json={"workflow_run_id": 424242, "html_url": "https://run"}
        )

    controller = A1Controller(tmp_path, backend=_mock_backend(handler))
    prepared = controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    execution = controller.dispatch_private_wave(
        prepared.logical_run_id, prepared.wave_id
    )
    assert execution == BackendExecutionRef("github-actions", "424242", "https://run")
    state = json.loads(
        (tmp_path / "controller" / "dispatch" / "opaque-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["waves"]["opaque-wave"]["execution_id"] == "424242"


def test_reconcile_refreshes_manifest_from_immutable_attempts(tmp_path):
    controller = A1Controller(tmp_path)
    prepared = controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    now = datetime.now(UTC)
    attempt = ShardAttemptRecord(
        logical_run_id="opaque-run",
        shard_id=prepared.shards[0].shard_id,
        attempt_id="opaque-wave-shard-000000-abc-1",
        status="succeeded",
        input_digest=prepared.shards[0].input_digest,
        execution_fingerprint=prepared.shards[0].execution_fingerprint,
        started_at=now,
        finished_at=now,
        wave_id="opaque-wave",
        output_refs=(),
    )
    controller.data_plane.append_attempt(attempt)
    manifest = controller.reconcile_run("opaque-run")
    assert manifest.revision == 1
    assert manifest.completed_shard_ids == (prepared.shards[0].shard_id,)


def test_private_client_round_trips_through_service_dispatch(tmp_path):
    service = PrivateDataPlaneService(tmp_path, "plane-token")
    controller = A1Controller(tmp_path)
    prepared = controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        status, headers, body = service.dispatch(
            request.method,
            request.url.path,
            authorization=request.headers.get("Authorization"),
            headers={key.lower(): value for key, value in request.headers.items()},
            body=request.content,
        )
        return httpx.Response(status, headers=headers, content=body or b"")

    plane = HttpPrivateDataPlane(
        "https://plane.example",
        "plane-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    payload = plane.resolve_wave("opaque-run", "opaque-wave")
    assert payload["wave"]["wave_id"] == prepared.wave_id
    with (
        patch(
            "portable_batch_execution.worker.execute_wave.TabularPack.execute",
            side_effect=RuntimeError("fail"),
        ),
        pytest.raises(PrivateWaveExecutionError),
    ):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
