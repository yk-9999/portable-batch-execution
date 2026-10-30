"""Private weight transfer stays streamed, SHA-bound and attempt-isolated."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.data_plane.base import ArtifactContentStream
from portable_batch_execution.worker import execute_wave


def _ref(name: str, data: bytes) -> ArtifactRef:
    return ArtifactRef(
        object_id=name,
        uri=f"pbe://private/{name}",
        sha256="sha256:" + sha256(data).hexdigest(),
        size_bytes=len(data),
    )


class _StreamingPlane:
    def __init__(self, *, tamper: str | None = None, failure: str | None = None):
        tensor = json.dumps(
            {
                "schema_version": "pbe.ml.distilbert-pair-binary-scores.v1",
                "row_ids": ["row-2", "row-1"],
                "model_a": {"input_ids": [[2], [4]], "attention_mask": [[1], [1]]},
                "model_b": {"input_ids": [[2], [4]], "attention_mask": [[1], [1]]},
            }
        ).encode()
        self.contents = {
            "tensor": tensor,
            "config-a": b'{"model_type":"distilbert"}',
            "weight-a": bytes(range(256)) * 128,
            "config-b": b'{"model_type":"distilbert"}',
            "weight-b": bytes(range(255, -1, -1)) * 128,
        }
        self.refs = [_ref(n, v) for n, v in self.contents.items()]
        self.tamper = tamper
        self.failure = failure
        self.read_calls = []
        self.stream_calls = []
        self.attempts = []
        self.output = b""
        now = datetime.now(UTC).isoformat()
        self.job = {
            "job_id": "job",
            "logical_run_id": "opaque-run",
            "pack": "ml-batch",
            "operation": "ml.distilbert_pair_binary_scores",
            "input_manifest_ref": self.refs[0].model_dump(mode="json"),
            "sharding": {},
            "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
            "security_profile": "offline",
            "provenance": {"producer": "test", "revision": "1", "created_at": now},
            "operation_params": {},
        }
        self.shard = {
            "logical_run_id": "opaque-run",
            "shard_id": "opaque-shard",
            "ordinal": 0,
            "correctness": {},
            "input_refs": [r.model_dump(mode="json") for r in self.refs],
            "input_digest": "current",
            "execution_fingerprint": "fixed",
        }
        self.wave = {
            "logical_run_id": "opaque-run",
            "wave_id": "opaque-wave",
            "ordinal": 0,
            "shard_ids": ["opaque-shard"],
            "max_parallel": 1,
        }

    def resolve_wave(self, run_id, wave_id):
        return {"job": self.job, "wave": self.wave, "shards": [self.shard]}

    def read_attempts(self, run_id):
        return tuple(self.attempts)

    def read(self, ref):
        self.read_calls.append(ref.object_id)
        if ref.object_id != "tensor":
            raise AssertionError("models must not be fetched as full bytes")
        return self.contents[ref.object_id]

    def open_content(self, ref):
        self.stream_calls.append(ref.object_id)
        if self.failure == f"read-error:{ref.object_id}":
            raise OSError("simulated transient stream open failure")
        payload = self.contents[ref.object_id]
        if self.tamper == ref.object_id:
            payload = b"x" + payload[1:]
        if self.failure == ref.object_id:
            payload = payload[:-1]

        def chunks():
            for offset in range(0, len(payload), 1024):
                yield payload[offset:offset + 1024]

        return ArtifactContentStream(size_bytes=ref.size_bytes, chunks=chunks())

    def append_attempt(self, attempt):
        self.attempts.append(attempt)

    def write(self, data, media_type):
        self.output = data
        return _ref("output", data)


def test_models_stream_to_temporary_files_without_full_read(monkeypatch):
    plane = _StreamingPlane()
    temporary_paths = []

    def simulated_inference(payload, *, model_a_dir: Path, model_b_dir: Path):
        assert [p.name for p in (model_a_dir, model_b_dir)] == ["model_a", "model_b"]
        assert (model_a_dir / "config.json").read_bytes() == plane.contents["config-a"]
        assert (model_a_dir / "model.safetensors").read_bytes() == plane.contents["weight-a"]
        assert (model_b_dir / "model.safetensors").read_bytes() == plane.contents["weight-b"]
        temporary_paths.extend([model_a_dir, model_b_dir])
        return {
            "schema_version": "pbe.ml.distilbert-pair-binary-scores-output.v1",
            "row_ids": payload["row_ids"],
            "model_a_scores": [0.2, 0.3],
            "model_b_scores": [0.4, 0.5],
        }

    monkeypatch.setattr(
        execute_wave, "execute_distilbert_pair_binary_scores_from_staged_models",
        simulated_inference,
    )
    attempts = execute_wave.execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert [a.status for a in attempts] == ["succeeded"]
    assert plane.read_calls == ["tensor"]
    assert plane.stream_calls == ["config-a", "weight-a", "config-b", "weight-b"]
    assert json.loads(plane.output)["row_ids"] == ["row-2", "row-1"]
    assert all(not path.exists() for path in temporary_paths)


@pytest.mark.parametrize(
    ("tamper", "failure", "expected"),
    [
        ("weight-a", None, "input_artifact_mismatch"),
        ("weight-b", None, "input_artifact_mismatch"),
        (None, "weight-a", "input_artifact_mismatch"),
        (None, "weight-b", "input_artifact_mismatch"),
        (None, "read-error:weight-a", "input_artifact_read_failed"),
        (None, "read-error:weight-b", "input_artifact_read_failed"),
    ],
)
def test_weight_stream_integrity_fails_closed(monkeypatch, tamper, failure, expected):
    plane = _StreamingPlane(tamper=tamper, failure=failure)
    monkeypatch.setattr(
        execute_wave, "execute_distilbert_pair_binary_scores_from_staged_models",
        lambda *_a, **_kw: pytest.fail("inference must not run with unverified weights"),
    )
    with pytest.raises(execute_wave.PrivateWaveExecutionError):
        execute_wave.execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.attempts[0].status == "failed"
    assert plane.attempts[0].failure == expected
    assert plane.output == b""
