import inspect
import json

import httpx
import pytest

from portable_batch_execution.backends.base import ExecutionBackend, WaveSubmission
from portable_batch_execution.backends.github_actions import (
    BackendExecutionRef,
    GitHubActionsAPIError,
    GitHubActionsBackend,
    map_run_status,
)
from portable_batch_execution.contracts import WaveSpec


def client(handler):
    return httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )


def _reject_http(_request):
    raise AssertionError("no HTTP")


def submission() -> WaveSubmission:
    return WaveSubmission(
        WaveSpec(
            logical_run_id="run",
            wave_id="wave-0000",
            ordinal=0,
            shard_ids=("shard-0",),
            max_parallel=1,
        )
    )


def test_submit_wave_private_data_plane_dispatches_opaque_run_and_wave():
    def handler(request):
        assert json.loads(request.content) == {
            "ref": "main",
            "inputs": {
                "wave_id": "opaque-wave",
                "run_id": "opaque-run",
                "private": True,
            },
            "return_run_details": True,
        }
        return httpx.Response(
            201, json={"workflow_run_id": 9, "html_url": "https://run/9"}
        )

    backend = GitHubActionsBackend(
        "o",
        "r",
        "execute-wave.yml",
        private_data_plane=True,
        client=client(handler),
    )
    submission = WaveSubmission(
        WaveSpec(
            logical_run_id="opaque-run",
            wave_id="opaque-wave",
            ordinal=0,
            shard_ids=("opaque-shard",),
            max_parallel=1,
        )
    )
    assert backend.submit_wave(submission) == BackendExecutionRef(
        "github-actions", "9", "https://run/9"
    )


@pytest.mark.parametrize(
    "field",
    ("logical_run_id", "wave_id"),
)
def test_submit_wave_private_data_plane_rejects_unsafe_identifiers(field):
    values = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
    }
    values[field] = "../escape"
    submission = WaveSubmission(
        WaveSpec(
            logical_run_id=values["logical_run_id"],
            wave_id=values["wave_id"],
            ordinal=0,
            shard_ids=("opaque-shard",),
            max_parallel=1,
        )
    )
    backend = GitHubActionsBackend(
        "o",
        "r",
        "w",
        private_data_plane=True,
        client=client(_reject_http),
    )
    with pytest.raises(ValueError, match="must be an opaque identifier"):
        backend.submit_wave(submission)


def test_submit_wave_uses_env_token_and_returns_run_details(monkeypatch):
    monkeypatch.setenv("PBE_GITHUB_TOKEN", "test-token")

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.url.path.endswith("/dispatches")
        assert json.loads(request.content) == {
            "ref": "feature/test",
            "inputs": {"wave_id": "wave-0000"},
            "return_run_details": True,
        }
        return httpx.Response(
            201, json={"workflow_run_id": 42, "html_url": "https://run"}
        )

    backend = GitHubActionsBackend(
        "o",
        "r",
        "execute-wave.yml",
        dispatch_ref="feature/test",
        client=client(handler),
    )
    assert backend.submit_wave(submission()) == BackendExecutionRef(
        "github-actions", "42", "https://run"
    )


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ({"status": "queued", "conclusion": None}, "running"),
        ({"status": "in_progress", "conclusion": None}, "running"),
        ({"status": "completed", "conclusion": "success"}, "succeeded"),
        ({"status": "completed", "conclusion": "cancelled"}, "cancelled"),
        ({"status": "completed", "conclusion": "failure"}, "failed"),
    ],
)
def test_status_mapping(run, expected):
    assert map_run_status(run) == expected


def test_get_evidence_and_cancel():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "conclusion": "success",
                    "html_url": "https://run/7",
                    "created_at": "2026-01-01T00:00:00Z",
                },
            )
        return httpx.Response(202)

    backend = GitHubActionsBackend("o", "r", "w", client=client(handler))
    execution = BackendExecutionRef("github-actions", "7")
    assert backend.collect_execution_evidence(execution).status == "succeeded"
    assert backend.cancel_run(execution) is None
    assert requests == [
        ("GET", "/repos/o/r/actions/runs/7"),
        ("POST", "/repos/o/r/actions/runs/7/cancel"),
    ]


def test_api_error_is_structured():
    backend = GitHubActionsBackend(
        "o",
        "r",
        "w",
        client=client(lambda request: httpx.Response(403, json={"message": "denied"})),
    )
    with pytest.raises(
        GitHubActionsAPIError, match=r"get run failed \(403\): denied"
    ) as error:
        backend.get_run(BackendExecutionRef("github-actions", "7"))
    assert error.value.status_code == 403


def test_github_backend_structurally_satisfies_shared_protocol_signatures():
    backend = GitHubActionsBackend(
        "o", "r", "w", client=client(lambda _request: httpx.Response(200))
    )

    assert isinstance(backend, ExecutionBackend)
    assert tuple(inspect.signature(GitHubActionsBackend.submit_wave).parameters) == (
        "self",
        "request",
    )
    assert tuple(inspect.signature(GitHubActionsBackend.get_run).parameters) == (
        "self",
        "execution",
    )
    assert tuple(inspect.signature(GitHubActionsBackend.cancel_run).parameters) == (
        "self",
        "execution",
    )
    assert tuple(
        inspect.signature(GitHubActionsBackend.collect_execution_evidence).parameters
    ) == ("self", "execution")
