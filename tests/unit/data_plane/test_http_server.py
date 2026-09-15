from io import BytesIO

import pytest

from portable_batch_execution.data_plane.http_server import (
    require_loopback_bind_host,
    serve_private_data_plane,
)


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
