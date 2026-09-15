import json

import httpx
import pytest

from portable_batch_execution.backends.github_actions import (
    BackendExecutionRef,
    GitHubActionsAPIError,
    GitHubActionsBackend,
    map_run_status,
)


def client(handler):
    return httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )


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
        return httpx.Response(201, json={"workflow_run_id": 42, "html_url": "https://run"})

    backend = GitHubActionsBackend("o", "r", "execute-wave.yml", client=client(handler))
    assert backend.submit_wave("feature/test", {"wave_id": "wave-0000"}) == BackendExecutionRef(
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
    assert backend.collect_execution_evidence(execution)["status"] == "succeeded"
    assert backend.cancel_run(execution) is True
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
    with pytest.raises(GitHubActionsAPIError, match=r"get run failed \(403\): denied") as error:
        backend.get_run(BackendExecutionRef("github-actions", "7"))
    assert error.value.status_code == 403
