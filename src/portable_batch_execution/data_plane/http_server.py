"""Loopback HTTP front-end for the authenticated private data plane service."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .service import PrivateDataPlaneService


def _response_bytes(body: bytes | None) -> bytes:
    return body if body is not None else b""


class _PrivateDataPlaneHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        status, headers, payload = self.service.dispatch(
            method,
            self.path,
            authorization=self.headers.get("Authorization"),
            headers={key: value for key, value in self.headers.items()},
            body=body,
        )
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        data = _response_bytes(payload)
        if data:
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if method != "HEAD" and data:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")


def serve_private_data_plane(
    state_root: Path,
    bearer_token: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    service = PrivateDataPlaneService(state_root, bearer_token)

    class Handler(_PrivateDataPlaneHandler):
        @property
        def service(self) -> PrivateDataPlaneService:
            return service

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def serve_private_data_plane_from_environment() -> ThreadingHTTPServer:
    import os

    state_root = os.environ.get("PBE_PRIVATE_DATA_PLANE_STATE_ROOT")
    bearer_token = os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN")
    host = os.environ.get("PBE_PRIVATE_DATA_PLANE_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PBE_PRIVATE_DATA_PLANE_BIND_PORT", "8765"))
    if not state_root or not bearer_token:
        raise ValueError("private data plane server environment is not configured")
    return serve_private_data_plane(Path(state_root), bearer_token, host=host, port=port)
