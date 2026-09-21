import json
from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pytest
from support.hf_bucket_api import FakeHfApi

from portable_batch_execution.contracts import ArtifactRef, ShardAttemptRecord
from portable_batch_execution.data_plane.hf import (
    HF_DIRECT_KIND,
    HF_TOKEN_ENV,
    OBJECT_LAYOUT_VERSION,
    HfBucketArtifactStore,
    HfBucketIdentity,
)
from portable_batch_execution.data_plane.hf_direct import HfDirectDataPlane
from portable_batch_execution.data_plane.http import HttpPrivateDataPlane
from portable_batch_execution.worker.execute_wave import execute_private_wave

_TOKEN = "hf_SENTINEL_HF_DIRECT_TOKEN"
_IDENTITY = HfBucketIdentity.from_metadata(
    {
        "kind": HF_DIRECT_KIND,
        "bucket": "yamauchiJP/system-trading-data",
        "prefix": "live-stream-news/hf-direct-20260921/",
        "object_layout_version": OBJECT_LAYOUT_VERSION,
        "required_hf_cli_version": "1.8.0",
    }
)


def _payloads():
    rows = [{"group": "a", "value": 1}, {"group": "a", "value": 3}]
    input_bytes = json.dumps(rows).encode()
    object_id = sha256(input_bytes).hexdigest()
    return rows, input_bytes, object_id


def _resolved(input_bytes, object_id):
    now = datetime.now(UTC).isoformat()
    input_ref = ArtifactRef(
        object_id=object_id,
        uri=f"{_IDENTITY.objects_uri}/{object_id}",
        sha256=f"sha256:{object_id}",
        size_bytes=len(input_bytes),
    )
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": "tabular-batch",
        "operation": "tabular.rolling",
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 2},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {"column": "value", "window_size": 2, "output_column": "rolling"},
    }
    shard = {
        "logical_run_id": "opaque-run",
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [input_ref.model_dump(mode="json")],
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
    return {"job": job, "wave": wave, "shards": [shard]}, input_ref


def _hf_direct_plane(monkeypatch, resolved, appended):
    monkeypatch.setenv(HF_TOKEN_ENV, _TOKEN)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/waves/opaque-wave"):
            return httpx.Response(200, json=resolved)
        if request.url.path.endswith("/attempts") and request.method == "GET":
            return httpx.Response(200, json=[item.model_dump(mode="json") for item in appended])
        if request.url.path.endswith("/attempts") and request.method == "POST":
            appended.append(
                ShardAttemptRecord.model_validate_json(request.content)
            )
            return httpx.Response(204)
        return httpx.Response(404, json={"error": "not found"})

    control = HttpPrivateDataPlane(
        "https://plane.example",
        "control-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    api = FakeHfApi()
    store = HfBucketArtifactStore(identity=_IDENTITY, api=api)
    return HfDirectDataPlane(control, store), seen, api


def test_hf_direct_worker_never_issues_a1_artifact_endpoints(monkeypatch):
    _rows, input_bytes, object_id = _payloads()
    resolved, _ref = _resolved(input_bytes, object_id)
    appended: list = []
    plane, seen, api = _hf_direct_plane(monkeypatch, resolved, appended)
    api.objects[_IDENTITY.remote_path(object_id)] = input_bytes

    attempts = execute_private_wave(
        "opaque-run", "opaque-wave", plane=plane, mode="hf-direct"
    )

    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    artifact_calls = [
        (method, path)
        for method, path in seen
        if "/v1/artifacts" in path
    ]
    assert artifact_calls == []
    assert all(path.startswith("/v1/runs/") for _method, path in seen)
    output_ref = attempts[0].output_refs[0]
    assert output_ref.uri.startswith(
        "hf://buckets/yamauchiJP/system-trading-data/live-stream-news/hf-direct-20260921/objects/sha256/"
    )
    assert output_ref.object_id in {
        path.rsplit("/", 1)[-1] for path in api.objects
    }
    assert _TOKEN not in json.dumps(attempts[0].model_dump(mode="json"))


def test_hf_direct_worker_fails_closed_on_missing_remote_input(monkeypatch):
    _rows, input_bytes, object_id = _payloads()
    resolved, _ref = _resolved(input_bytes, object_id)
    plane, seen, _api = _hf_direct_plane(monkeypatch, resolved, [])

    with pytest.raises(Exception) as error:
        execute_private_wave(
            "opaque-run", "opaque-wave", plane=plane, mode="hf-direct"
        )
    assert all("/v1/artifacts" not in path for _method, path in seen)
    assert _TOKEN not in str(error.value)
