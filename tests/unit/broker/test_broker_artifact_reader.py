from hashlib import sha256

import httpx
import pytest

from portable_batch_execution.broker.artifact_reader import (
    HttpBrokerArtifactReader,
    LocalDataPlaneArtifactReader,
    build_broker_artifact_reader,
    validate_artifact_read_base_url,
)
from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.controller.a1_controller import A1Controller


def _ref(payload: bytes, *, object_id: str = "obj-1") -> ArtifactRef:
    digest = sha256(payload).hexdigest()
    return ArtifactRef(
        object_id=object_id,
        uri=f"file:///tmp/{object_id}",
        sha256=f"sha256:{digest}",
        media_type="application/json",
        size_bytes=len(payload),
    )


def test_default_local_reader_uses_data_plane(tmp_path):
    controller = A1Controller(tmp_path, backend=None)
    payload = b'[{"id":1}]'
    ref = controller.data_plane.write(payload, "application/json")
    reader = build_broker_artifact_reader(controller.data_plane)
    assert isinstance(reader, LocalDataPlaneArtifactReader)
    assert reader.read(ref) == payload


def test_http_reader_returns_verified_bytes(tmp_path):
    payload = b"broker-output"
    ref = _ref(payload)
    token_path = tmp_path / "token"
    token_path.write_text("secret-token\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret-token"
        assert request.url.path == f"/v1/artifacts/{ref.object_id}/content"
        return httpx.Response(200, content=payload)

    client = httpx.Client(
        base_url="http://127.0.0.1:18090",
        transport=httpx.MockTransport(handler),
    )
    reader = build_broker_artifact_reader(
        data_plane=None,
        artifact_read_base_url="http://127.0.0.1:18090",
        artifact_read_token_file=token_path,
        client=client,
    )
    assert reader.read(ref) == payload


def test_http_reader_digest_mismatch_fails_closed(tmp_path):
    payload = b"expected"
    ref = _ref(b"different")
    token_path = tmp_path / "token"
    token_path.write_text("t", encoding="utf-8")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )
    reader = HttpBrokerArtifactReader(
        "http://127.0.0.1:1",
        "t",
        client=client,
    )
    with pytest.raises(ValueError, match="artifact_digest_mismatch"):
        reader.read(ref)


def test_http_reader_size_mismatch_fails_closed(tmp_path):
    payload = b"12345"
    ref = _ref(payload, object_id="size-mismatch")
    ref = ArtifactRef(
        object_id=ref.object_id,
        uri=ref.uri,
        sha256=ref.sha256,
        media_type=ref.media_type,
        size_bytes=1,
    )
    token_path = tmp_path / "token"
    token_path.write_text("t", encoding="utf-8")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=payload))
    )
    reader = HttpBrokerArtifactReader("http://127.0.0.1:1", "t", client=client)
    with pytest.raises(ValueError, match="artifact_size_mismatch"):
        reader.read(ref)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com:18090",
        "http://10.0.0.1:1",
    ],
)
def test_rejects_non_loopback_http_base_url(base_url):
    with pytest.raises(ValueError, match="loopback"):
        validate_artifact_read_base_url(base_url)


def test_accepts_loopback_http_and_https_origins():
    assert (
        validate_artifact_read_base_url("http://127.0.0.1:18090")
        == "http://127.0.0.1:18090"
    )
    assert (
        validate_artifact_read_base_url("https://data.example.test")
        == "https://data.example.test"
    )


def test_token_never_appears_in_http_reader_errors(tmp_path, caplog):
    token_path = tmp_path / "token"
    token_path.write_text("super-secret-token", encoding="utf-8")
    ref = _ref(b"x")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500)),
    )
    reader = build_broker_artifact_reader(
        data_plane=None,
        artifact_read_base_url="http://127.0.0.1:9",
        artifact_read_token_file=token_path,
        client=client,
    )
    with pytest.raises(ValueError, match="artifact_read_failed"):
        reader.read(ref)
    combined = caplog.text + str(caplog.records)
    assert "super-secret-token" not in combined
