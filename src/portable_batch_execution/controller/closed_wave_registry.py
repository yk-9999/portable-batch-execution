"""Private closed-wave registry under a controller-owned state root."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from portable_batch_execution.contracts import JobSpec, ShardSpec, WaveSpec

_FORBIDDEN = "/\\?#"


def opaque_identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character in value for character in _FORBIDDEN)
    ):
        raise ValueError(f"{name} must be an opaque identifier")
    return value


class ClosedWaveRegistry:
    """Persist JobSpec + WaveSpec + ShardSpecs and resolve one opaque run/wave pair."""

    def __init__(self, controller_root: Path):
        self._root = controller_root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._waves = self._root / "closed_waves"
        self._waves.mkdir(exist_ok=True)
        self._lock = RLock()

    def _run_dir(self, run_id: str) -> Path:
        run_id = opaque_identifier(run_id, "run_id")
        path = self._waves / run_id
        path.mkdir(exist_ok=True)
        return path

    def register_closed_wave(
        self,
        job: JobSpec,
        wave: WaveSpec,
        shards: tuple[ShardSpec, ...],
    ) -> None:
        run_id = opaque_identifier(job.logical_run_id, "run_id")
        wave_id = opaque_identifier(wave.wave_id, "wave_id")
        if wave.logical_run_id != run_id:
            raise ValueError("wave logical_run_id must match job")
        if any(shard.logical_run_id != run_id for shard in shards):
            raise ValueError("all shards must belong to the run")
        if tuple(shard.shard_id for shard in shards) != wave.shard_ids:
            raise ValueError("closed wave shard plan mismatch")
        bundle = {
            "job": job.model_dump(mode="json"),
            "wave": wave.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
        }
        with self._lock:
            run_dir = self._run_dir(run_id)
            job_path = run_dir / "job.json"
            if job_path.exists():
                existing = JobSpec.model_validate_json(job_path.read_text(encoding="utf-8"))
                if existing != job:
                    raise ValueError("run already registered with a different job")
            else:
                job_path.write_text(job.model_dump_json() + "\n", encoding="utf-8")
            wave_path = run_dir / f"{wave_id}.wave.json"
            if wave_path.exists():
                raise ValueError("wave_id already registered for run")
            temporary = wave_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(bundle) + "\n", encoding="utf-8")
            temporary.replace(wave_path)

    def resolve_wave(self, run_id: str, wave_id: str) -> dict[str, object]:
        run_id = opaque_identifier(run_id, "run_id")
        wave_id = opaque_identifier(wave_id, "wave_id")
        wave_path = self._waves / run_id / f"{wave_id}.wave.json"
        if not wave_path.is_file():
            raise KeyError("closed wave not found")
        payload = json.loads(wave_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("invalid closed wave bundle")
        return payload

    def load_shards_for_run(self, run_id: str) -> tuple[ShardSpec, ...]:
        run_id = opaque_identifier(run_id, "run_id")
        run_dir = self._waves / run_id
        if not run_dir.is_dir():
            return ()
        shards: list[ShardSpec] = []
        for wave_path in sorted(run_dir.glob("*.wave.json")):
            payload = json.loads(wave_path.read_text(encoding="utf-8"))
            for item in payload.get("shards", ()):
                shards.append(ShardSpec.model_validate(item))
        return tuple(sorted(shards, key=lambda shard: (shard.ordinal, shard.shard_id)))
