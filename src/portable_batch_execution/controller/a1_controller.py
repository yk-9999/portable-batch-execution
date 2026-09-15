"""A1 single-writer controller for private closed-wave runs."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.backends.base import BackendExecutionRef, WaveSubmission
from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.contracts import (
    JobSpec,
    Provenance,
    RunManifest,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.controller.closed_wave_registry import ClosedWaveRegistry
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane
from portable_batch_execution.kernel import RunController


@dataclass(frozen=True)
class PreparedPrivateRun:
    logical_run_id: str
    wave_id: str
    job: JobSpec
    wave: WaveSpec
    shards: tuple[ShardSpec, ...]
    manifest: RunManifest


def job_spec_digest(job: JobSpec) -> str:
    return sha256(job.model_dump_json().encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_public_synthetic_plan() -> dict:
    plan_path = _repository_root() / "fixtures" / "public" / "synthetic" / "wave-0000.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError("public synthetic wave plan must be an object")
    return plan


class A1Controller:
    """Prepare private runs, dispatch waves, and reconcile manifests as sole writer."""

    def __init__(
        self,
        state_root: Path,
        backend: GitHubActionsBackend | None = None,
    ):
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.data_plane = LocalFilesystemDataPlane(self.state_root)
        self.registry = ClosedWaveRegistry(self.state_root / "controller")
        self.run_controller = RunController(self.data_plane)
        self.backend = backend
        self._dispatch_root = self.state_root / "controller" / "dispatch"
        self._dispatch_root.mkdir(parents=True, exist_ok=True)

    def _dispatch_path(self, run_id: str) -> Path:
        return self._dispatch_root / f"{run_id}.json"

    def _read_dispatch_state(self, run_id: str) -> dict:
        path = self._dispatch_path(run_id)
        if not path.is_file():
            return {"logical_run_id": run_id, "waves": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid dispatch state")
        payload.setdefault("waves", {})
        return payload

    def _write_dispatch_state(self, run_id: str, payload: dict) -> None:
        path = self._dispatch_path(run_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(path)

    def prepare_private_synthetic_run(
        self,
        *,
        logical_run_id: str | None = None,
        wave_id: str | None = None,
    ) -> PreparedPrivateRun:
        plan = _load_public_synthetic_plan()
        fixture_root = (_repository_root() / "fixtures" / "public").resolve()
        data_name = plan.get("input_fixture")
        if not isinstance(data_name, str) or Path(data_name).name != data_name:
            raise ValueError("public synthetic input fixture is invalid")
        data_path = (fixture_root / data_name).resolve()
        if fixture_root not in data_path.parents or not data_path.is_file():
            raise ValueError("public synthetic input fixture is unavailable")

        run_id = logical_run_id or f"run-{secrets.token_hex(8)}"
        wave = wave_id or f"wave-{secrets.token_hex(4)}"
        input_ref = self.data_plane.write(data_path.read_bytes(), "application/json")
        job = JobSpec.model_validate(
            {
                **plan["job"],
                "logical_run_id": run_id,
                "input_manifest_ref": input_ref.model_dump(mode="json"),
                "execution": {
                    "max_parallel": 1,
                    "max_attempts_per_shard": 4,
                    "resume_enabled": True,
                },
            }
        )
        shard_template = plan["shards"][0]
        shard = ShardSpec.model_validate(
            {
                **shard_template,
                "logical_run_id": run_id,
                "correctness": job.sharding.model_dump(mode="json"),
                "input_refs": [input_ref.model_dump(mode="json")],
            }
        )
        wave_spec = WaveSpec(
            logical_run_id=run_id,
            wave_id=wave,
            ordinal=0,
            shard_ids=(shard.shard_id,),
            max_parallel=1,
        )
        shards = (shard,)
        self.registry.register_closed_wave(job, wave_spec, shards)
        now = datetime.now(UTC)
        manifest = RunManifest(
            logical_run_id=run_id,
            revision=0,
            job_spec_digest=job_spec_digest(job),
            status="planned",
            expected_shard_ids=(shard.shard_id,),
            created_at=now,
            updated_at=now,
            provenance=Provenance(
                producer="a1-controller",
                revision="private-synthetic-v1",
                created_at=now,
            ),
            waves=(wave_spec,),
        )
        self.data_plane.write_next_manifest(manifest, -1)
        return PreparedPrivateRun(
            logical_run_id=run_id,
            wave_id=wave,
            job=job,
            wave=wave_spec,
            shards=shards,
            manifest=manifest,
        )

    def dispatch_private_wave(self, run_id: str, wave_id: str) -> BackendExecutionRef:
        if self.backend is None:
            raise ValueError("GitHub backend is not configured")
        payload = self.registry.resolve_wave(run_id, wave_id)
        wave = WaveSpec.model_validate(payload["wave"])
        if wave.logical_run_id != run_id or wave.wave_id != wave_id:
            raise ValueError("resolved wave does not match dispatch request")
        execution = self.backend.submit_wave(WaveSubmission(wave))
        state = self._read_dispatch_state(run_id)
        state["waves"][wave_id] = {
            "backend_id": execution.backend_id,
            "execution_id": execution.execution_id,
            "web_url": execution.web_url,
        }
        self._write_dispatch_state(run_id, state)
        return execution

    def inspect_run(self, run_id: str) -> dict:
        manifest = self.data_plane.read_manifest(run_id)
        dispatch = self._read_dispatch_state(run_id)
        backend_status = None
        if self.backend is not None:
            waves = dispatch.get("waves", {})
            if isinstance(waves, dict) and len(waves) == 1:
                entry = next(iter(waves.values()))
                if isinstance(entry, dict) and entry.get("execution_id"):
                    backend_status = self.backend.get_run(
                        BackendExecutionRef(
                            entry.get("backend_id", "github-actions"),
                            str(entry["execution_id"]),
                            entry.get("web_url"),
                        )
                    )
        return {
            "logical_run_id": run_id,
            "manifest_revision": manifest.revision if manifest else None,
            "manifest_status": manifest.status if manifest else None,
            "dispatch": dispatch.get("waves", {}),
            "backend_status": None
            if backend_status is None
            else {
                "execution_id": backend_status.execution_id,
                "status": backend_status.status,
            },
        }

    def reconcile_run(self, run_id: str) -> RunManifest:
        manifest = self.data_plane.read_manifest(run_id)
        if manifest is None:
            raise ValueError("run manifest not found")
        shards = self.registry.load_shards_for_run(run_id)
        if not shards:
            raise ValueError("planned shards not found")
        return self.run_controller.refresh(manifest, manifest.revision, shards)
