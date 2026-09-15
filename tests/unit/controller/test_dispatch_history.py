import json
from unittest.mock import MagicMock

import httpx
import pytest

from portable_batch_execution.backends.base import BackendExecutionRef, BackendRunStatus
from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.controller.a1_controller import A1Controller


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


def test_two_dispatches_same_wave_retain_execution_ids_in_order(tmp_path):
    ids = [424242, 525252]
    idx = 0

    def counting_handler(request):
        nonlocal idx
        run_id = ids[idx]
        idx += 1
        return httpx.Response(
            201, json={"workflow_run_id": run_id, "html_url": f"https://run/{run_id}"}
        )

    controller = A1Controller(tmp_path, backend=_mock_backend(counting_handler))
    prepared = controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    first = controller.dispatch_private_wave(prepared.logical_run_id, prepared.wave_id)
    second = controller.dispatch_private_wave(prepared.logical_run_id, prepared.wave_id)
    assert first.execution_id == "424242"
    assert second.execution_id == "525252"

    state = json.loads(
        (tmp_path / "controller" / "dispatch" / "opaque-run.json").read_text(
            encoding="utf-8"
        )
    )
    history = state["waves"]["opaque-wave"]["dispatches"]
    assert [entry["execution_id"] for entry in history] == ["424242", "525252"]
    assert state["waves"]["opaque-wave"]["execution_id"] == "525252"


def test_legacy_single_dict_dispatch_state_normalizes_on_inspect(tmp_path):
    dispatch_path = tmp_path / "controller" / "dispatch" / "opaque-run.json"
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    dispatch_path.write_text(
        json.dumps(
            {
                "logical_run_id": "opaque-run",
                "waves": {
                    "opaque-wave": {
                        "backend_id": "github-actions",
                        "execution_id": "999001",
                        "web_url": "https://legacy",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controller = A1Controller(tmp_path)
    inspected = controller.inspect_run("opaque-run")
    wave = inspected["dispatch"]["opaque-wave"]
    assert wave["dispatches"] == [
        {
            "backend_id": "github-actions",
            "execution_id": "999001",
            "web_url": "https://legacy",
        }
    ]
    assert wave["latest_execution_id"] == "999001"


def test_duplicate_execution_id_is_idempotent(tmp_path):
    controller = A1Controller(tmp_path)
    state = {"logical_run_id": "opaque-run", "waves": {}}
    execution = BackendExecutionRef("github-actions", "424242", "https://run")
    controller._record_wave_dispatch(state, "opaque-wave", execution)
    controller._record_wave_dispatch(state, "opaque-wave", execution)
    history = state["waves"]["opaque-wave"]["dispatches"]
    assert len(history) == 1
    assert history[0]["execution_id"] == "424242"


def test_conflicting_execution_id_metadata_rejected(tmp_path):
    controller = A1Controller(tmp_path)
    state = {"logical_run_id": "opaque-run", "waves": {}}
    controller._record_wave_dispatch(
        state,
        "opaque-wave",
        BackendExecutionRef("github-actions", "424242", "https://run/a"),
    )
    with pytest.raises(ValueError, match="conflicting dispatch metadata"):
        controller._record_wave_dispatch(
            state,
            "opaque-wave",
            BackendExecutionRef("github-actions", "424242", "https://run/b"),
        )


def test_inspect_queries_latest_execution_id_only(tmp_path):
    backend = MagicMock(spec=GitHubActionsBackend)
    backend.get_run.return_value = BackendRunStatus(
        execution_id="525252",
        status="running",
    )

    controller = A1Controller(tmp_path, backend=backend)
    controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    dispatch_path = tmp_path / "controller" / "dispatch" / "opaque-run.json"
    dispatch_path.write_text(
        json.dumps(
            {
                "logical_run_id": "opaque-run",
                "waves": {
                    "opaque-wave": {
                        "dispatches": [
                            {
                                "backend_id": "github-actions",
                                "execution_id": "424242",
                                "web_url": "https://run/424242",
                            },
                            {
                                "backend_id": "github-actions",
                                "execution_id": "525252",
                                "web_url": "https://run/525252",
                            },
                        ],
                        "backend_id": "github-actions",
                        "execution_id": "525252",
                        "web_url": "https://run/525252",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inspected = controller.inspect_run("opaque-run")
    backend.get_run.assert_called_once()
    queried = backend.get_run.call_args.args[0]
    assert queried.execution_id == "525252"
    assert inspected["backend_status"]["queried_execution_id"] == "525252"
    assert inspected["backend_status"]["execution_id"] == "525252"
