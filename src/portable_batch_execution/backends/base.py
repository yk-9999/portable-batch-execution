from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from portable_batch_execution.contracts import WaveSpec


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    max_shards_per_wave: int
    supports_cancel: bool
    returns_execution_id_on_submit: bool


@dataclass(frozen=True)
class BackendExecutionRef:
    backend_id: str
    execution_id: str
    web_url: str | None = None


@dataclass(frozen=True)
class WaveSubmission:
    """A portable request to submit one already-planned wave."""

    wave: WaveSpec


@dataclass(frozen=True)
class BackendRunStatus:
    execution_id: str
    status: Literal["running", "succeeded", "failed", "cancelled"]
    started_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class BackendEvidence:
    backend_id: str
    execution_id: str
    status: Literal["running", "succeeded", "failed", "cancelled"]
    web_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    run_started_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def submit_wave(self, request: WaveSubmission) -> BackendExecutionRef: ...
    def get_run(self, execution: BackendExecutionRef) -> BackendRunStatus: ...
    def cancel_run(self, execution: BackendExecutionRef) -> None: ...
    def collect_execution_evidence(
        self, execution: BackendExecutionRef
    ) -> BackendEvidence: ...
