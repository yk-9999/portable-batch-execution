import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    INPUT_SCHEMA_VERSION,
)
from portable_batch_execution.worker.execute_wave import (
    PrivateWaveExecutionError,
    execute_private_wave,
)

torch = pytest.importorskip("torch")
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
)


def _artifact_ref(name: str, payload: bytes) -> dict:
    return ArtifactRef(
        object_id=name,
        uri=f"pbe://private/{name}",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
    ).model_dump(mode="json")


def _tiny_model_bundle(tmp_path: Path, seed: int) -> tuple[bytes, bytes]:
    torch.manual_seed(seed)
    config = DistilBertConfig(
        vocab_size=128,
        dim=32,
        hidden_dim=64,
        n_heads=2,
        n_layers=1,
        max_position_embeddings=256,
        num_labels=2,
        architectures=["DistilBertForSequenceClassification"],
    )
    model = DistilBertForSequenceClassification(config)
    directory = tmp_path / f"model-{seed}"
    directory.mkdir()
    model.save_pretrained(directory, safe_serialization=True)
    return (directory / "config.json").read_bytes(), (directory / "model.safetensors").read_bytes()


def _distilbert_plane(tmp_path, *, input_refs=None, read_map=None):
    tensor_payload = json.dumps(
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "row_ids": ["r1"],
            "model_a": {"input_ids": [[3, 4, 0]], "attention_mask": [[1, 1, 0]]},
            "model_b": {"input_ids": [[3, 4, 0]], "attention_mask": [[1, 1, 0]]},
        }
    ).encode()
    model_a = _tiny_model_bundle(tmp_path, 1)
    model_b = _tiny_model_bundle(tmp_path, 2)
    default_refs = [
        _artifact_ref("tensor", tensor_payload),
        _artifact_ref("model-a-config", model_a[0]),
        _artifact_ref("model-a-weights", model_a[1]),
        _artifact_ref("model-b-config", model_b[0]),
        _artifact_ref("model-b-weights", model_b[1]),
    ]
    refs = input_refs if input_refs is not None else default_refs
    now = datetime.now(UTC).isoformat()
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "ml-batch",
        "operation": "ml.distilbert_pair_binary_scores",
        "input_manifest_ref": refs[0],
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {},
    }
    shard = {
        "logical_run_id": "opaque-run",
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": refs,
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": ["opaque-shard"],
        "max_parallel": 1,
    }
    if read_map is not None:
        payloads = read_map
    elif len(refs) == 5:
        payloads = {
            refs[0]["object_id"]: tensor_payload,
            refs[1]["object_id"]: model_a[0],
            refs[2]["object_id"]: model_a[1],
            refs[3]["object_id"]: model_b[0],
            refs[4]["object_id"]: model_b[1],
        }
    else:
        payloads = {}

    class Plane:
        def __init__(self):
            self.appended = []
            self.last_written = b""

        def resolve_wave(self, run_id, wave_id):
            return {"job": job, "wave": wave, "shards": [shard]}

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            key = ref.object_id if hasattr(ref, "object_id") else ref["object_id"]
            if key not in payloads:
                raise OSError("missing")
            return payloads[key]

        def write(self, payload, media_type):
            self.last_written = payload
            return ArtifactRef(
                object_id="output",
                uri="pbe://private/output",
                sha256="sha256:" + sha256(payload).hexdigest(),
                media_type=media_type,
                size_bytes=len(payload),
            )

        def append_attempt(self, record):
            self.appended.append(record)

    return Plane()


def test_distilbert_private_wave_routes_five_input_refs(tmp_path):
    plane = _distilbert_plane(tmp_path)
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    output = json.loads(plane.last_written.decode())
    assert output["row_ids"] == ["r1"]
    assert len(output["model_a_scores"]) == 1
    assert len(output["model_b_scores"]) == 1


def test_distilbert_rejects_wrong_input_ref_count(tmp_path):
    plane = _distilbert_plane(tmp_path, input_refs=[_artifact_ref("only", b"{}")])
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "input_artifact_invalid"


def test_distilbert_rejects_digest_mismatch(tmp_path):
    plane = _distilbert_plane(tmp_path)
    bad_ref = _artifact_ref("tensor", b"{}")
    bad_ref["sha256"] = "sha256:" + ("0" * 64)
    plane_shards = plane.resolve_wave("opaque-run", "opaque-wave")["shards"]
    plane_shards[0]["input_refs"][0] = bad_ref
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "input_artifact_mismatch"
