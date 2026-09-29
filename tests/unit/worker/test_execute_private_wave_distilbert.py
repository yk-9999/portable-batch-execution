import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    INPUT_SCHEMA_VERSION,
    DistilbertStaticModelPair,
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


def _tensor_payload(row_id: str) -> bytes:
    return json.dumps(
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "row_ids": [row_id],
            "model_a": {"input_ids": [[3, 4, 0]], "attention_mask": [[1, 1, 0]]},
            "model_b": {"input_ids": [[3, 4, 0]], "attention_mask": [[1, 1, 0]]},
        }
    ).encode()


def _multi_shard_distilbert_plane(tmp_path, shard_specs, *, prior_attempts=()):
    model_a = _tiny_model_bundle(tmp_path, 1)
    model_b = _tiny_model_bundle(tmp_path, 2)
    static_refs = [
        _artifact_ref("model-a-config", model_a[0]),
        _artifact_ref("model-a-weights", model_a[1]),
        _artifact_ref("model-b-config", model_b[0]),
        _artifact_ref("model-b-weights", model_b[1]),
    ]
    static_payloads = {
        "model-a-config": model_a[0],
        "model-a-weights": model_a[1],
        "model-b-config": model_b[0],
        "model-b-weights": model_b[1],
    }
    payloads = dict(static_payloads)
    shards = []
    for ordinal, spec in enumerate(shard_specs):
        tensor_ref = _artifact_ref(f"tensor-{spec['shard_id']}", spec["tensor"])
        refs = [tensor_ref, *static_refs]
        shards.append(
            {
                "logical_run_id": "opaque-run",
                "shard_id": spec["shard_id"],
                "ordinal": ordinal,
                "correctness": {},
                "input_refs": refs,
                "input_digest": spec.get("input_digest", f"digest-{spec['shard_id']}"),
                "execution_fingerprint": spec.get(
                    "execution_fingerprint", "fixed"
                ),
            }
        )
        payloads[tensor_ref["object_id"]] = spec["tensor"]

    now = datetime.now(UTC).isoformat()
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "ml-batch",
        "operation": "ml.distilbert_pair_binary_scores",
        "input_manifest_ref": shards[0]["input_refs"][0],
        "sharding": {},
        "execution": {"max_parallel": 2, "max_attempts_per_shard": 4},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {},
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": [item["shard_id"] for item in shards],
        "max_parallel": 2,
    }

    class Plane:
        def __init__(self):
            self.appended = []
            self.written_payloads = []
            self.read_counts = {}

        def resolve_wave(self, run_id, wave_id):
            return {"job": job, "wave": wave, "shards": shards}

        def read_attempts(self, run_id):
            return tuple(prior_attempts) + tuple(self.appended)

        def read(self, ref):
            key = ref.object_id if hasattr(ref, "object_id") else ref["object_id"]
            self.read_counts[key] = self.read_counts.get(key, 0) + 1
            if key not in payloads:
                raise OSError("missing")
            return payloads[key]

        def write(self, payload, media_type):
            digest = sha256(payload).hexdigest()
            ref = ArtifactRef(
                object_id=f"output-{digest[:8]}",
                uri=f"pbe://private/output-{digest[:8]}",
                sha256="sha256:" + digest,
                media_type=media_type,
                size_bytes=len(payload),
            )
            self.written_payloads.append(payload)
            return ref

        def append_attempt(self, record):
            self.appended.append(record)

    plane = Plane()
    plane.payloads = payloads
    plane.shards = shards
    return plane, static_refs


def test_distilbert_multi_shard_reads_static_models_once(tmp_path, monkeypatch):
    shard_specs = [
        {"shard_id": "shard-0", "tensor": _tensor_payload("r0")},
        {"shard_id": "shard-1", "tensor": _tensor_payload("r1")},
        {"shard_id": "shard-2", "tensor": _tensor_payload("r2")},
    ]
    plane, static_refs = _multi_shard_distilbert_plane(tmp_path, shard_specs)
    load_calls = {"count": 0}
    original = DistilbertStaticModelPair.from_verified_artifact_bytes

    def counting_from_verified(*args, **kwargs):
        load_calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        DistilbertStaticModelPair,
        "from_verified_artifact_bytes",
        counting_from_verified,
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 3
    assert all(item.status == "succeeded" for item in attempts)
    assert load_calls["count"] == 1
    for ref in static_refs:
        assert plane.read_counts[ref["object_id"]] == 1
    assert plane.read_counts["tensor-shard-0"] == 1
    assert plane.read_counts["tensor-shard-1"] == 1
    assert plane.read_counts["tensor-shard-2"] == 1
    attempt_ids = {item.attempt_id for item in attempts}
    assert len(attempt_ids) == 3
    shard_ids = {item.shard_id for item in attempts}
    assert shard_ids == {"shard-0", "shard-1", "shard-2"}


def test_distilbert_multi_shard_rejects_invalid_static_ref_metadata(tmp_path):
    shard_specs = [
        {"shard_id": "shard-0", "tensor": _tensor_payload("r0")},
        {"shard_id": "shard-1", "tensor": _tensor_payload("r1")},
    ]
    plane, static_refs = _multi_shard_distilbert_plane(tmp_path, shard_specs)
    config_ref = plane.shards[1]["input_refs"][1]
    tampered_config_ref = {
        **config_ref,
        "size_bytes": config_ref["size_bytes"] + 1,
    }
    plane.shards[1]["input_refs"] = [
        plane.shards[1]["input_refs"][0],
        tampered_config_ref,
        plane.shards[1]["input_refs"][2],
        plane.shards[1]["input_refs"][3],
        plane.shards[1]["input_refs"][4],
    ]
    with pytest.raises(PrivateWaveExecutionError) as exc_info:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    attempts = exc_info.value.attempts
    by_shard = {item.shard_id: item for item in attempts}
    assert by_shard["shard-0"].status == "succeeded"
    assert by_shard["shard-1"].status == "failed"
    assert by_shard["shard-1"].failure == "input_artifact_mismatch"
    assert plane.read_counts[static_refs[0]["object_id"]] == 2


def test_distilbert_multi_shard_isolated_outputs_and_partial_failure(tmp_path):
    good = _tensor_payload("good")
    bad = json.dumps({"schema_version": INPUT_SCHEMA_VERSION, "row_ids": []}).encode()
    shard_specs = [
        {"shard_id": "ok-a", "tensor": good},
        {"shard_id": "bad", "tensor": bad},
        {"shard_id": "ok-b", "tensor": good},
    ]
    plane, _ = _multi_shard_distilbert_plane(tmp_path, shard_specs)
    with pytest.raises(PrivateWaveExecutionError) as exc_info:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    attempts = exc_info.value.attempts
    by_shard = {item.shard_id: item for item in attempts}
    assert by_shard["ok-a"].status == "succeeded"
    assert by_shard["ok-b"].status == "succeeded"
    assert by_shard["bad"].status == "failed"
    assert by_shard["bad"].failure == "shard_pack_execution_failed"
    outputs = [json.loads(item.decode()) for item in plane.written_payloads]
    assert len(outputs) == 2
    assert all(output["row_ids"] == ["good"] for output in outputs)


def test_distilbert_multi_shard_retry_budget_is_per_shard(tmp_path):
    good = _tensor_payload("retry-row")
    bad = json.dumps({"schema_version": INPUT_SCHEMA_VERSION, "row_ids": []}).encode()
    shard_specs = [
        {
            "shard_id": "flaky",
            "tensor": bad,
            "input_digest": "flaky-digest",
            "execution_fingerprint": "fp",
        }
    ]
    plane, _ = _multi_shard_distilbert_plane(tmp_path, shard_specs)
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(plane.appended) == 1
    failed_attempt = plane.appended[0]
    assert failed_attempt.status == "failed"

    plane.payloads["tensor-flaky"] = good
    plane.shards[0]["input_refs"][0] = _artifact_ref("tensor-flaky", good)
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].attempt_id != failed_attempt.attempt_id
