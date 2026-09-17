import io
import json
from datetime import UTC, datetime
from hashlib import sha256

import polars as pl

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.worker.execute_wave import execute_private_wave


def _parquet_payload(rows):
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    return buffer.getvalue()


def _ref(name: str, payload: bytes) -> ArtifactRef:
    return ArtifactRef(
        object_id=name,
        uri=f"pbe://private/{name}",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type="application/vnd.apache.parquet",
    )


def _plane_for_replay(*, operation, input_refs, operation_params):
    payloads = {ref.object_id: None for ref in input_refs}

    class Plane:
        def __init__(self):
            self.appended = []
            self._payloads = dict(payloads)

        def resolve_wave(self, run_id, wave_id):
            now = datetime.now(UTC).isoformat()
            return {
                "job": {
                    "job_id": "job",
                    "logical_run_id": "opaque-run",
                    "pack": "replay-batch",
                    "operation": operation,
                    "input_manifest_ref": input_refs[0].model_dump(mode="json"),
                    "sharding": {},
                    "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
                    "security_profile": "offline",
                    "provenance": {"producer": "test", "revision": "1", "created_at": now},
                    "operation_params": operation_params,
                },
                "wave": {
                    "logical_run_id": "opaque-run",
                    "wave_id": "opaque-wave",
                    "ordinal": 0,
                    "shard_ids": ["opaque-shard"],
                    "max_parallel": 1,
                },
                "shards": [
                    {
                        "logical_run_id": "opaque-run",
                        "shard_id": "opaque-shard",
                        "ordinal": 0,
                        "correctness": {},
                        "input_refs": [ref.model_dump(mode="json") for ref in input_refs],
                        "input_digest": "current",
                        "execution_fingerprint": "fixed",
                    }
                ],
            }

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            return self._payloads[ref.object_id]

        def open_content(self, ref):
            from portable_batch_execution.data_plane.base import ArtifactContentStream

            payload = self._payloads[ref.object_id]

            def chunks():
                for offset in range(0, len(payload), 13):
                    yield payload[offset : offset + 13]

            return ArtifactContentStream(size_bytes=len(payload), chunks=chunks())

        def write(self, data, media_type):
            self.last_written = data
            return ArtifactRef(
                object_id="output",
                uri="pbe://private/output",
                sha256="sha256:" + sha256(data).hexdigest(),
                size_bytes=len(data),
            )

        def append_attempt(self, record):
            self.appended.append(record)

    plane = Plane()
    for ref in input_refs:
        plane._payloads[ref.object_id] = None
    return plane


def test_private_wave_materializes_multiple_parquet_inputs_for_canonicalize():
    payload_a = _parquet_payload(
        [{"identity": 1, "identity_norm": "1", "price": 1.0, "symbol": "AAA", "block": 1}]
    )
    payload_b = _parquet_payload(
        [{"identity": 2, "identity_norm": "2", "price": 2.0, "symbol": "AAA", "block": 2}]
    )
    refs = [_ref("a", payload_a), _ref("b", payload_b)]
    plane = _plane_for_replay(
        operation="replay.structural_canonicalize",
        input_refs=refs,
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
        },
    )
    plane._payloads["a"] = payload_a
    plane._payloads["b"] = payload_b

    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    body = json.loads(plane.last_written.decode())
    assert body["schema_version"] == "pbe.replay.structural-canonicalize-result.v2"
    assert len(body["identity_profiles"]) == 2
    assert attempts[0].counts["input_rows"] == 2
