from __future__ import annotations

import os
from typing import Any

import httpx

from .base import (
    BackendCapabilities,
    BackendEvidence,
    BackendExecutionRef,
    BackendRunStatus,
    WaveSubmission,
)


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
        dispatch_ref: str = "main",
        token: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.owner = owner
        self.repo = repo
        self.workflow_id = workflow_id
        self.dispatch_ref = dispatch_ref
        self.token = token if token is not None else os.environ.get("PBE_GITHUB_TOKEN")
        self.client = client or httpx.Client(
            base_url="https://api.github.com", timeout=30.0
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("github-actions", 256, True, True)

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

    def submit_wave(self, request: WaveSubmission) -> BackendExecutionRef:
        response = self._request(
            "submit wave",
            "POST",
            f"/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/dispatches",
            json={
                "ref": self.dispatch_ref,
                "inputs": {"wave_id": request.wave.wave_id},
                "return_run_details": True,
            },
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

    def get_run(self, execution: BackendExecutionRef) -> BackendRunStatus:
        run = self._request(
            "get run",
            "GET",
            f"/repos/{self.owner}/{self.repo}/actions/runs/{execution.execution_id}",
        ).json()
        if not isinstance(run, dict):
            raise TypeError("GitHub Actions get run response must be an object")
        return BackendRunStatus(
            execution_id=execution.execution_id,
            status=map_run_status(run),
            started_at=run.get("run_started_at"),
            updated_at=run.get("updated_at"),
        )

    def cancel_run(self, execution: BackendExecutionRef) -> None:
        self._request(
            "cancel run",
            "POST",
            f"/repos/{self.owner}/{self.repo}/actions/runs/{execution.execution_id}/cancel",
        )

    def collect_execution_evidence(self, execution: BackendExecutionRef) -> BackendEvidence:
        raw = self._request(
            "get run",
            "GET",
            f"/repos/{self.owner}/{self.repo}/actions/runs/{execution.execution_id}",
        ).json()
        if not isinstance(raw, dict):
            raise TypeError("GitHub Actions get run response must be an object")
        return BackendEvidence(
            backend_id="github-actions",
            execution_id=execution.execution_id,
            status=map_run_status(raw),
            web_url=raw.get("html_url") or execution.web_url,
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            run_started_at=raw.get("run_started_at"),
            details={"github_status": raw.get("status"), "conclusion": raw.get("conclusion")},
        )
