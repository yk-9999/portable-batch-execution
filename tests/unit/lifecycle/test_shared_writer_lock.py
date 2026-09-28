"""Writer lock coordination compatible with common artifact GC."""

from __future__ import annotations

import threading
from hashlib import sha256
from unittest.mock import patch

from portable_batch_execution.broker.planning import register_broker_private_run
from portable_batch_execution.controller.closed_wave_registry import ClosedWaveRegistry
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane
from portable_batch_execution.lifecycle.lock import (
    LifecycleLockError,
    LifecycleStateLock,
)

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"


def test_register_broker_holds_lifecycle_lock_for_input_and_manifest(tmp_path):
    with patch.object(LifecycleStateLock, "acquire", autospec=True) as acquire:
        register_broker_private_run(
            state_root=tmp_path,
            request_id="req-lock-once",
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params={"by": [{"column": "id"}]},
            input_bytes=b"[1,2]",
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
        )
    assert acquire.call_count == 1


def test_register_broker_reuses_digest_atomically_against_competing_lock(tmp_path):
    shared = b"shared-broker-input"
    digest_hex = sha256(shared).hexdigest()
    plane = LocalFilesystemDataPlane(tmp_path)
    plane.write(shared, "application/json")

    gc_outcome: list[str] = []
    original_register = ClosedWaveRegistry.register_closed_wave

    def register_with_concurrent_lock_attempt(self, job, wave, shards):
        def competitor():
            try:
                with LifecycleStateLock(tmp_path):
                    gc_outcome.append("ran")
            except LifecycleLockError:
                gc_outcome.append("blocked")

        thread = threading.Thread(target=competitor)
        thread.start()
        thread.join(timeout=10)
        assert gc_outcome == ["blocked"]
        return original_register(self, job, wave, shards)

    with patch.object(
        ClosedWaveRegistry, "register_closed_wave", register_with_concurrent_lock_attempt
    ):
        _, _, shard, manifest = register_broker_private_run(
            state_root=tmp_path,
            request_id="req-new-shared",
            pack="tabular-batch",
            operation="tabular.sort",
            operation_params={"by": [{"column": "id"}]},
            input_bytes=shared,
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
        )
    payload = tmp_path / "artifacts" / digest_hex
    assert payload.is_file()
    assert shard.input_refs[0].sha256 == f"sha256:{digest_hex}"
    assert manifest is not None
