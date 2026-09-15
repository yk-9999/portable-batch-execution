from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str = "github-actions"
    max_shards_per_wave: int = 256
    supports_cancel: bool = True
    returns_execution_id_on_submit: bool = True


@dataclass(frozen=True)
class BackendExecutionRef:
    backend_id: str
    execution_id: str
    web_url: str | None = None


class GitHubActionsAPIError(RuntimeError):
    """A GitHub Actions API request failed without exposing credential material."""

    def __init__(self, operation: str, response: httpx.Response):
        self.operation = operation
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-github-request-id")
        try:
            body = response.json()
            detail = str(body.get("message", "")) if isinstance(body, dict) else ""
        except ValueError:
            detail = response.text[:200]
        suffix = f": {detail}" if detail else ""
        super().__init__(f"GitHub Actions {operation} failed ({response.status_code}){suffix}")


def map_run_status(run: dict[str, Any]) -> str:
    """Map GitHub's status/conclusion pair to the portable execution status."""
    status = run.get("status")
    conclusion = run.get("conclusion")
    if status in {"queued", "requested", "waiting", "pending", "in_progress"}:
        return "running"
    if status != "completed":
        return "running"
    if conclusion == "success":
        return "succeeded"
    if conclusion in {"cancelled", "skipped", "stale"}:
        return "cancelled"
    return "failed"


class GitHubActionsBackend:
    def __init__(
        self,
        owner: str,
        repo: str,
        workflow_id: str,
        token: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.owner = owner
        self.repo = repo
        self.workflow_id = workflow_id
        self.token = token if token is not None else os.environ.get("PBE_GITHUB_TOKEN")
        self.client = client or httpx.Client(
            base_url="https://api.github.com", timeout=30.0
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self, operation: str, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        response = self.client.request(method, path, headers=self._headers(), **kwargs)
        if response.is_error:
            raise GitHubActionsAPIError(operation, response)
        return response

    def submit_wave(self, ref: str, inputs: dict[str, str]) -> BackendExecutionRef:
        response = self._request(
            "submit wave",
            "POST",
            f"/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/dispatches",
            json={"ref": ref, "inputs": inputs, "return_run_details": True},
        )
        try:
            d = response.json()
            run_id = d["workflow_run_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GitHubActionsAPIError(
                "submit wave response missing workflow_run_id", response
            ) from exc
        return BackendExecutionRef(
            "github-actions",
            str(run_id),
            d.get("html_url") or d.get("run_url"),
        )

    def get_run(self, execution: BackendExecutionRef) -> dict[str, Any]:
        run = self._request(
            "get run",
            "GET",
            f"/repos/{self.owner}/{self.repo}/actions/runs/{execution.execution_id}",
        ).json()
        if not isinstance(run, dict):
            raise TypeError("GitHub Actions get run response must be an object")
        return {**run, "portable_status": map_run_status(run)}

    def cancel_run(self, execution: BackendExecutionRef) -> bool:
        self._request(
            "cancel run",
            "POST",
            f"/repos/{self.owner}/{self.repo}/actions/runs/{execution.execution_id}/cancel",
        )
        return True

    def collect_execution_evidence(self, execution: BackendExecutionRef) -> dict[str, Any]:
        run = self.get_run(execution)
        return {
            "backend_id": "github-actions",
            "execution_id": execution.execution_id,
            "status": run["portable_status"],
            "github_status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "web_url": run.get("html_url") or execution.web_url,
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "run_started_at": run.get("run_started_at"),
        }
