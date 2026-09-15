from __future__ import annotations

from dataclasses import dataclass

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


class GitHubActionsBackend:
    def __init__(
        self,
        owner: str,
        repo: str,
        workflow_id: str,
        token: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.owner, self.repo, self.workflow_id, self.token, self.client = (
            owner,
            repo,
            workflow_id,
            token,
            client or httpx.Client(base_url="https://api.github.com"),
        )

    def capabilities(self):
        return BackendCapabilities()

    def submit_wave(self, ref: str, inputs: dict[str, str]):
        r = self.client.post(
            f"/repos/{self.owner}/{self.repo}/actions/workflows/{self.workflow_id}/dispatches",
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            json={"ref": ref, "inputs": inputs, "return_run_details": True},
        )
        r.raise_for_status()
        d = r.json()
        return BackendExecutionRef(
            "github-actions",
            str(d["workflow_run_id"]),
            d.get("html_url") or d.get("run_url"),
        )

    def get_run(self, e):
        return self.client.get(
            f"/repos/{self.owner}/{self.repo}/actions/runs/{e.execution_id}"
        ).json()

    def cancel_run(self, e):
        return self.client.post(
            f"/repos/{self.owner}/{self.repo}/actions/runs/{e.execution_id}/cancel"
        ).is_success

    def collect_execution_evidence(self, e):
        return self.get_run(e)
