"""Broker registration under shared lifecycle writer lock contention."""

from __future__ import annotations

import base64
import errno
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    opaque_run_id,
    opaque_wave_id,
    register_broker_private_run,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.broker.state import BrokerRequestStore
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.lifecycle.lock import (
    LifecycleLockError,
    LifecycleLockTimeoutError,
    LifecycleStateLock,
)

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000
_SORT_PARAMS = {"by": [{"column": "id"}]}


def _register_kwargs(
    state_root: Path,
    request_id: str,
    *,
    input_bytes: bytes = b"[1,2]",
) -> dict[str, object]:
    return {
        "state_root": state_root,
        "request_id": request_id,
        "pack": "tabular-batch",
        "operation": "tabular.sort",
        "operation_params": _SORT_PARAMS,
        "input_bytes": input_bytes,
        "input_media_type": "application/json",
        "public_sha": _PUBLIC_SHA,
    }


def _register(state_root: Path, request_id: str, **kwargs: object):
    return register_broker_private_run(**_register_kwargs(state_root, request_id, **kwargs))


def _config(tmp_path: Path) -> BrokerConfig:
    path = tmp_path / "broker-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [["tabular-batch", "tabular.sort"]]
                },
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def _wave_count(state_root: Path) -> int:
    waves_root = state_root / "controller" / "closed_waves"
    if not waves_root.is_dir():
        return 0
    return len(list(waves_root.glob("*/*.wave.json")))


def _artifact_count(state_root: Path) -> int:
    artifacts = state_root / "artifacts"
    if not artifacts.is_dir():
        return 0
    return len(list(artifacts.iterdir()))


def test_concurrent_distinct_request_ids_register_without_lock_errors(tmp_path):
    state_root = tmp_path / "shared-state"
    state_root.mkdir()
    request_ids = [f"req-concurrent-{index}" for index in range(8)]

    def run(request_id: str):
        job, wave, shard, manifest = _register(state_root, request_id)
        return request_id, job.job_id, wave.wave_id, shard.shard_id, manifest.revision

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, request_ids))

    assert len(results) == 8
    assert len({item[1] for item in results}) == 8
    assert _wave_count(state_root) == 8


def test_concurrent_same_request_id_identical_binding(tmp_path):
    state_root = tmp_path / "shared-state"
    state_root.mkdir()
    barrier = threading.Barrier(4)

    def run():
        barrier.wait(timeout=5)
        return _register(state_root, "req-same-id")

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda _: run(), range(4)))

    job_ids = {outcome[0].job_id for outcome in outcomes}
    wave_ids = {outcome[1].wave_id for outcome in outcomes}
    assert len(job_ids) == 1
    assert len(wave_ids) == 1
    assert _wave_count(state_root) == 1


def test_concurrent_same_request_id_conflicting_binding(tmp_path):
    state_root = tmp_path / "shared-state"
    state_root.mkdir()
    barrier = threading.Barrier(4)
    errors: list[Exception] = []

    def run(payload: bytes):
        try:
            barrier.wait(timeout=5)
            _register(state_root, "req-conflict", input_bytes=payload)
        except ValueError as exc:
            errors.append(exc)

    payloads = [b"[1]", b"[2]", b"[3]", b"[4]"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, payloads))

    assert len(errors) == 3
    assert all(str(exc) == "request_binding_conflict" for exc in errors)
    assert _wave_count(state_root) == 1


def test_lifecycle_lock_timeout_is_distinct_from_immediate_contention(tmp_path):
    holder = LifecycleStateLock(tmp_path)
    holder.acquire()
    try:
        with pytest.raises(LifecycleLockTimeoutError):
            LifecycleStateLock(tmp_path).acquire(timeout=0.15)
    finally:
        holder.release()


def test_lifecycle_lock_does_not_retry_unrelated_oserror(tmp_path, monkeypatch):
    def failing_open(*args, **kwargs):
        raise OSError(errno.EIO, "simulated disk fault")

    monkeypatch.setattr(os, "open", failing_open)
    with pytest.raises(LifecycleLockError) as exc_info:
        LifecycleStateLock(tmp_path).acquire(timeout=1.0)
    assert not isinstance(exc_info.value, LifecycleLockTimeoutError)
    assert exc_info.value.__cause__ is not None
    assert exc_info.value.__cause__.errno == errno.EIO


def test_register_broker_busy_when_lifecycle_lock_held(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    config = _config(tmp_path)
    controller = A1Controller(state_root, backend=None)
    service = UnixBrokerService(
        state_root=state_root,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )
    holder = LifecycleStateLock(state_root)
    holder.acquire()
    try:
        with patch(
            "portable_batch_execution.broker.planning._BROKER_REGISTRATION_LOCK_ACQUIRE_TIMEOUT_S",
            0.2,
        ):
            response = service.handle_payload(
                _UID,
                {
                    "schema_version": "pbe.a1-unix-broker.request.v1",
                    "request_id": "req-busy",
                    "pack": "tabular-batch",
                    "operation": "tabular.sort",
                    "operation_params": _SORT_PARAMS,
                    "input_media_type": "application/json",
                    "input_b64": "WzEsMl0=",
                },
            )
    finally:
        holder.release()
    assert response.status == "failed"
    assert response.error_code == "broker_busy"


def test_service_register_after_resolve_miss_reports_request_id_conflict(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    request_id = "req-service-binding-race"
    winner_input = b"[1,2,3]"
    loser_input = b"[9,8,7]"
    winner_job, _, winner_shard, _ = _register(
        state_root, request_id, input_bytes=winner_input
    )
    artifacts_before = _artifact_count(state_root)
    run_id = opaque_run_id(request_id)
    wave_id = opaque_wave_id(request_id)

    config = _config(tmp_path)
    controller = A1Controller(state_root, backend=None)
    service = UnixBrokerService(
        state_root=state_root,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )

    def resolve_optimistic_miss(_run_id: str, _wave_id: str):
        raise KeyError(_run_id, _wave_id)

    with patch.object(controller.registry, "resolve_wave", side_effect=resolve_optimistic_miss):
        response = service.handle_payload(
            _UID,
            {
                "schema_version": "pbe.a1-unix-broker.request.v1",
                "request_id": request_id,
                "pack": "tabular-batch",
                "operation": "tabular.sort",
                "operation_params": _SORT_PARAMS,
                "input_media_type": "application/json",
                "input_b64": base64.b64encode(loser_input).decode("ascii"),
            },
        )

    assert response.status == "failed"
    assert response.error_code == "request_id_conflict"
    assert _wave_count(state_root) == 1
    assert _artifact_count(state_root) == artifacts_before
    payload = controller.registry.resolve_wave(run_id, wave_id)
    assert payload["job"]["job_id"] == winner_job.job_id
    assert payload["shards"][0]["input_digest"] == winner_shard.input_digest
    assert BrokerRequestStore(state_root / "controller").load(request_id) is None


def _process_register(state_root_str: str, request_id: str, queue: multiprocessing.Queue) -> None:
    state_root = Path(state_root_str)
    try:
        job, _, _, _ = register_broker_private_run(**_register_kwargs(state_root, request_id))
        queue.put(("ok", job.job_id))
    except Exception as exc:  # noqa: BLE001 - child must report failure
        queue.put(("err", type(exc).__name__, str(exc)))


def test_cross_process_concurrent_registrations(tmp_path):
    state_root = tmp_path / "mp-state"
    state_root.mkdir()
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    ids = [f"req-mp-{index}" for index in range(4)]
    processes = [
        ctx.Process(
            target=_process_register,
            args=(str(state_root.resolve()), request_id, queue),
        )
        for request_id in ids
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes = [queue.get(timeout=5) for _ in ids]
    assert all(item[0] == "ok" for item in outcomes)
    assert len({item[1] for item in outcomes}) == 4
    assert _wave_count(state_root) == 4


def test_concurrent_four_new_ids_match_public_broker_race_shape(tmp_path):
    """Regression for four simultaneous AF_UNIX new registrations on one state_root."""
    state_root = tmp_path / "race-state"
    state_root.mkdir()
    ids = [f"req-race-{index}" for index in range(4)]
    errors: list[str] = []

    def run(request_id: str):
        try:
            _register(state_root, request_id, input_bytes=f"[{request_id}]".encode())
        except LifecycleLockError as exc:
            errors.append(type(exc).__name__)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, request_id) for request_id in ids]
        for future in as_completed(futures):
            future.result()

    assert errors == []
    assert _wave_count(state_root) == 4
