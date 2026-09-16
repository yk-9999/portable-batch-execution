"""AF_UNIX broker server loop."""

from __future__ import annotations

import json
import socket
from pathlib import Path

from .peers import read_peer_credentials
from .protocol import BrokerExecuteResponse, response_to_json
from .service import UnixBrokerService

_MAX_FRAME_BYTES = 8_388_608


def serve_unix_broker(
    *,
    socket_path: Path,
    service: UnixBrokerService,
) -> None:
    socket_path = socket_path.resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(8)
    while True:
        connection, _address = listener.accept()
        with connection:
            _handle_connection(connection, service)


def _handle_connection(connection: socket.socket, service: UnixBrokerService) -> None:
    try:
        _, uid, _gid = read_peer_credentials(connection)
        frame = _read_frame(connection)
        payload = json.loads(frame.decode("utf-8"))
        response = service.handle_payload(uid, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        response = BrokerExecuteResponse(
            request_id="unknown",
            status="failed",
            error_code="broker_internal_error",
        )
    connection.sendall(response_to_json(response))


def _read_frame(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        part = connection.recv(65536)
        if not part:
            break
        total += len(part)
        if total > _MAX_FRAME_BYTES:
            raise ValueError("request frame exceeds broker bound")
        chunks.append(part)
        if part.endswith(b"\n"):
            break
    if not chunks:
        raise ValueError("empty request frame")
    return b"".join(chunks)
