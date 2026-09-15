"""Closed public-synthetic worker entry point for one already planned wave."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.contracts import (
    JobSpec,
    ShardAttemptRecord,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.packs import TabularPack

_WAVE_ID = re.compile(r"wave-[0-9]{4}")
_PUBLIC_WAVES = frozenset({"wave-0000"})


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plan_path(wave_id: str, repository_root: Path) -> Path:
    if not isinstance(wave_id, str) or not _WAVE_ID.fullmatch(wave_id):
        raise ValueError("wave_id must be a closed planned wave identifier")
    if wave_id not in _PUBLIC_WAVES:
        raise ValueError("wave_id is not an approved public synthetic wave")
    return repository_root / "fixtures" / "public" / "synthetic" / f"{wave_id}.json"


def _load_plan(wave_id: str, repository_root: Path) -> dict:
    path = _plan_path(wave_id, repository_root)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("public synthetic wave plan is invalid") from exc
    if not isinstance(plan, dict):
        raise TypeError("public synthetic wave plan must be an object")
    return plan


def execute_public_wave(
    wave_id: str,
    *,
    repository_root: Path | None = None,
    state_root: Path | None = None,
) -> tuple[ShardAttemptRecord, ...]:
    """Execute a committed public plan and append attempts without touching manifests."""
    root = (repository_root or _repository_root()).resolve()
    plan = _load_plan(wave_id, root)
    fixture_root = (root / "fixtures" / "public").resolve()
    data_name = plan.get("input_fixture")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise ValueError("public synthetic input fixture is invalid")
    data_path = (fixture_root / data_name).resolve()
    if fixture_root not in data_path.parents or not data_path.is_file():
        raise ValueError("public synthetic input fixture is unavailable")

    context = tempfile.TemporaryDirectory() if state_root is None else nullcontext()
    with context as temporary:
        plane = LocalFilesystemDataPlane(state_root or Path(temporary))
        input_ref = plane.write(data_path.read_bytes(), "application/json")
        job = JobSpec.model_validate(
            {**plan["job"], "input_manifest_ref": input_ref.model_dump(mode="json")}
        )
        wave = WaveSpec.model_validate(plan["wave"])
        if wave.wave_id != wave_id or wave.logical_run_id != job.logical_run_id:
            raise ValueError("public synthetic wave does not match its job")
        shards = tuple(
            ShardSpec.model_validate(
                {
                    **item,
                    "logical_run_id": job.logical_run_id,
                    "correctness": job.sharding.model_dump(mode="json"),
                    "input_refs": [input_ref.model_dump(mode="json")],
                }
            )
            for item in plan["shards"]
        )
        if tuple(shard.shard_id for shard in shards) != wave.shard_ids:
            raise ValueError("public synthetic wave shard plan is invalid")
        if job.pack != "tabular-batch" or job.operation != "tabular.rolling":
            raise ValueError("public synthetic plan uses an unsupported pack operation")
        rows = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("public synthetic input must be a record array")
        pack = TabularPack()
        attempts: list[ShardAttemptRecord] = []
        for shard in shards:
            started_at = datetime.now(UTC)
            result = pack.execute(job, shard, job.operation_params, {"data": rows})
            output = json.dumps(result.to_dicts(), sort_keys=True).encode("utf-8")
            output_ref = plane.write(output, "application/json")
            attempt = ShardAttemptRecord(
                logical_run_id=job.logical_run_id,
                shard_id=shard.shard_id,
                attempt_id=f"{wave.wave_id}-{shard.shard_id}",
                status="succeeded",
                input_digest=shard.input_digest,
                execution_fingerprint=shard.execution_fingerprint,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                wave_id=wave.wave_id,
                output_refs=(output_ref,),
                output_digest=sha256(output).hexdigest(),
                counts={"input_rows": len(rows), "output_rows": result.height},
            )
            plane.append_attempt(attempt)
            attempts.append(attempt)
        return tuple(attempts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one approved public synthetic wave.")
    parser.add_argument("--wave-id", required=True)
    args = parser.parse_args(argv)
    attempts = execute_public_wave(args.wave_id)
    print(json.dumps({"wave_id": args.wave_id, "attempt_ids": [item.attempt_id for item in attempts]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
