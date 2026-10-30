import threading
from io import BytesIO

import httpx
import pytest

from portable_batch_execution.data_plane.http_server import (
    require_loopback_bind_host,
    serve_private_data_plane,
)


def test_large_artifact_get_streams_over_loopback(tmp_path):
    server = serve_private_data_plane(tmp_path, "plane-token", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        headers = {"Authorization": "Bearer plane-token"}
        data = bytes(range(256)) * 8192
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            written = client.post("/v1/artifacts", content=data, headers=headers)
            assert written.status_code == 200
            object_id = written.json()["object_id"]
            served = client.get(f"/v1/artifacts/{object_id}/content", headers=headers)
        assert served.status_code == 200
        assert served.headers["Content-Type"] == "application/octet-stream"
        assert served.headers["Content-Length"] == str(len(data))
        assert served.content == data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_loopback_artifact_head_reports_stream_content_length(tmp_path):
    server = serve_private_data_plane(tmp_path, "plane-token", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        headers = {"Authorization": "Bearer plane-token"}
        data = b"loopback-stream-bound"
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            written = client.post("/v1/artifacts", content=data, headers=headers)
            object_id = written.json()["object_id"]
            head = client.head(
                f"/v1/artifacts/{object_id}/content",
                headers=headers,
            )
            streamed = client.get(
                f"/v1/artifacts/{object_id}/content",
                headers=headers,
            )
        assert head.status_code == 200
        assert head.headers["Content-Length"] == str(len(data))
        assert streamed.status_code == 200
        assert streamed.headers["Content-Length"] == str(len(data))
        assert streamed.content == data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unauthorized_request_does_not_read_request_body(tmp_path):
    server = serve_private_data_plane(tmp_path, "plane-token", port=0)
    handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
    payload = b"x" * 4096
    handler.rfile = BytesIO(payload)
    handler.wfile = BytesIO()
    handler.path = "/v1/artifacts"
    handler.headers = {"Content-Length": str(len(payload)), "Authorization": "Bearer wrong"}
    handler.request_version = "HTTP/1.1"
    handler.protocol_version = "HTTP/1.1"
    handler.close_connection = True
    handler.requestline = "POST /v1/artifacts HTTP/1.1"

    handler._dispatch("POST")

    assert handler.rfile.read() == payload
    assert b"unauthorized" in handler.wfile.getvalue()


def test_rejects_non_loopback_bind_host(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        serve_private_data_plane(tmp_path, "plane-token", host="0.0.0.0")


@pytest.mark.parametrize("host", ("127.0.0.1", "::1", "localhost"))
def test_accepts_loopback_bind_hosts(host):
    assert require_loopback_bind_host(host) == host


def test_sendfile_get_is_byte_exact_and_requires_authorization(tmp_path, monkeypatch):
    """Loopback optimized response sends only authenticated immutable bytes."""
    import socket

    calls = []
    original = socket.socket.sendfile

    def tracked(self, file, offset=0, count=None):
        calls.append(count)
        return original(self, file, offset=offset, count=count)

    monkeypatch.setattr(socket.socket, "sendfile", tracked)
    server = serve_private_data_plane(tmp_path, "plane-token", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        data = bytes(range(256)) * 8192
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            write = client.post(
                "/v1/artifacts", content=data,
                headers={"Authorization": "Bearer plane-token"}
            )
            object_id = write.json()["object_id"]
            path = f"/v1/artifacts/{object_id}/content"
            unauthorized = client.get(path)
            assert unauthorized.status_code == 401
            assert not calls
            auth = {"Authorization": "Bearer plane-token"}
            head = client.head(path, headers=auth)
            assert head.status_code == 200
            assert not calls
            response = client.get(path, headers=auth)
            assert response.status_code == 200
            assert response.content == data
            assert response.headers["Content-Length"] == str(len(data))
            assert calls == [len(data)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_artifact_get_retains_streamed_fallback_without_local_file_path(tmp_path):
    from portable_batch_execution.data_plane.base import ArtifactContentStream

    server = serve_private_data_plane(tmp_path, "plane-token", port=0)
    store = server.RequestHandlerClass.service.fget(
        server.RequestHandlerClass.__new__(server.RequestHandlerClass)
    ).store
    original = store.open_content

    def without_sendfile_path(ref):
        stream = original(ref)
        return ArtifactContentStream(size_bytes=stream.size_bytes, chunks=stream.chunks)

    store.open_content = without_sendfile_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        headers = {"Authorization": "Bearer plane-token"}
        data = bytes(range(256)) * 1024
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            written = client.post("/v1/artifacts", content=data, headers=headers)
            object_id = written.json()["object_id"]
            response = client.get(f"/v1/artifacts/{object_id}/content", headers=headers)
            assert response.status_code == 200
            assert response.content == data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
