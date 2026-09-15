from hashlib import sha256

import httpx
import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.data_plane import (
    HttpPrivateDataPlane,
    PrivateDataPlaneError,
)


def test_http_plane_resolves_exact_opaque_pair_and_uses_bearer(monkeypatch):
    seen = {}
    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"job": {}, "wave": {}, "shards": []})
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BASE_URL", "https://plane.example")
    monkeypatch.setenv("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN", "not-for-output")
    plane = HttpPrivateDataPlane.from_environment()
    plane._client = httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer not-for-output"})
    assert plane.resolve_wave("run-opaque", "wave-opaque")["shards"] == []
    assert seen == {"path": "/v1/runs/run-opaque/waves/wave-opaque", "auth": "Bearer not-for-output"}


def test_http_plane_never_leaks_response_body_or_token_in_error():
    plane = HttpPrivateDataPlane("https://plane.example", "super-secret", client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403, text="private body super-secret"))))
    with pytest.raises(PrivateDataPlaneError) as error:
        plane.resolve_wave("run", "wave")
    assert str(error.value) == "private data plane resolve wave failed with HTTP 403"
    assert "secret" not in str(error.value)
    with pytest.raises(ValueError):
        plane.resolve_wave("../path", "wave")


def test_http_plane_reads_by_object_id_not_artifact_uri():
    data = b"[]"
    ref = ArtifactRef(object_id="opaque-object", uri="pbe://private/opaque-object", sha256="sha256:" + sha256(data).hexdigest(), size_bytes=len(data))
    seen = []
    plane = HttpPrivateDataPlane("https://plane.example", "token", client=httpx.Client(transport=httpx.MockTransport(lambda request: (seen.append(request.url.path), httpx.Response(200, content=data))[1])))
    assert plane.read(ref) == data
    assert seen == ["/v1/artifacts/opaque-object/content"]

