from datetime import UTC, datetime
from hashlib import sha256

import httpx
from support.hf_bucket_api import FakeHfApi

from portable_batch_execution.contracts import ArtifactRef, ShardAttemptRecord
from portable_batch_execution.data_plane.hf import (
    HF_DIRECT_KIND,
    OBJECT_LAYOUT_VERSION,
    HfBucketArtifactStore,
    HfBucketIdentity,
)
from portable_batch_execution.data_plane.hf_direct import HfDirectDataPlane
from portable_batch_execution.data_plane.http import HttpPrivateDataPlane

_IDENTITY = HfBucketIdentity.from_metadata(
    {
        "kind": HF_DIRECT_KIND,
        "bucket": "yamauchiJP/system-trading-data",
        "prefix": "live-stream-news/hf-direct-20260921/",
        "object_layout_version": OBJECT_LAYOUT_VERSION,
        "required_hf_cli_version": "1.8.0",
    }
)


def _recording_control():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/attempts") and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/attempts") and request.method == "POST":
            return httpx.Response(204)
        if "/waves/" in request.url.path:
            return httpx.Response(200, json={"job": {}, "wave": {}, "shards": []})
        return httpx.Response(404, json={"error": "not found"})

    control = HttpPrivateDataPlane(
        "https://plane.example",
        "token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return control, seen


def _plane():
    control, seen = _recording_control()
    api = FakeHfApi()
    store = HfBucketArtifactStore(identity=_IDENTITY, api=api)
    return HfDirectDataPlane(control, store), seen, api


def test_control_metadata_is_delegated_to_a1():
    plane, seen, _api = _plane()
    plane.resolve_wave("run", "wave")
    assert [method for method, _ in seen] == ["GET"]
    assert seen[0][1] == "/v1/runs/run/waves/wave"


def test_artifact_reads_and_writes_never_touch_a1_artifact_endpoints():
    plane, seen, api = _plane()
    data = b"direct-bytes"
    object_id = sha256(data).hexdigest()
    api.objects[_IDENTITY.remote_path(object_id)] = data
    ref = ArtifactRef(
        object_id=object_id,
        uri=f"{_IDENTITY.objects_uri}/{object_id}",
        sha256=f"sha256:{object_id}",
        size_bytes=len(data),
    )
    assert plane.read(ref) == data
    written = plane.write(b"other-bytes")
    assert plane.exists(written) is True
    assert plane.verify(written) is True
    assert seen == []
    assert api.get_paths_calls or api.batch_calls


def test_attempt_and_manifest_calls_stay_on_control_plane():
    plane, seen, _cli = _plane()
    now = datetime.now(UTC)
    record = ShardAttemptRecord(
        logical_run_id="run",
        shard_id="shard",
        attempt_id="run-shard-1",
        status="failed",
        input_digest="d",
        execution_fingerprint="f",
        started_at=now,
        finished_at=now,
        failure="shard_execution_failed",
    )
    plane.append_attempt(record)
    assert plane.read_attempts("run") == ()
    assert seen == [
        ("POST", "/v1/runs/run/attempts"),
        ("GET", "/v1/runs/run/attempts"),
    ]
    assert all("/v1/artifacts" not in path for _method, path in seen)
