from __future__ import annotations

import json

import httpx

from portable_batch_execution.backends.github_actions import GitHubActionsBackend


def test_github_backend_dispatches_a_wave_and_returns_the_backend_run_reference():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204, json={"workflow_run_id": 42, "html_url": "https://github.test/runs/42"})

    client = httpx.Client(
        base_url="https://api.github.test", transport=httpx.MockTransport(handler)
    )
    backend = GitHubActionsBackend(
        "public-owner",
        "public-repo",
        "wave.yml",
        token="public-test-token",
        client=client,
    )

    execution = backend.submit_wave("main", {"run_id": "run-1", "wave_id": "wave-0000"})

    assert requests[0].url.path == "/repos/public-owner/public-repo/actions/workflows/wave.yml/dispatches"
    assert requests[0].headers["Authorization"] == "Bearer public-test-token"
    assert json.loads(requests[0].content) == {
        "ref": "main",
        "inputs": {"run_id": "run-1", "wave_id": "wave-0000"},
        "return_run_details": True,
    }
    assert execution.execution_id == "42"
    assert execution.web_url == "https://github.test/runs/42"
    assert backend.capabilities().max_shards_per_wave == 256


def test_github_backend_collects_and_cancels_the_same_execution_id():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"id": 42, "status": "completed"})
        return httpx.Response(202)

    client = httpx.Client(
        base_url="https://api.github.test", transport=httpx.MockTransport(handler)
    )
    backend = GitHubActionsBackend("owner", "repo", "wave.yml", client=client)
    execution = backend.submit_wave("main", {}) if False else type("Execution", (), {"execution_id": "42"})()

    assert backend.collect_execution_evidence(execution) == {"id": 42, "status": "completed"}
    assert backend.cancel_run(execution)
    assert paths == [
        ("GET", "/repos/owner/repo/actions/runs/42"),
        ("POST", "/repos/owner/repo/actions/runs/42/cancel"),
    ]
