"""AF_UNIX broker server loop."""

from __future__ import annotations

import json
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .peers import read_peer_credentials
from .protocol import BrokerExecuteResponse, response_to_json
from .service import UnixBrokerService

MAX_ACTIVE_CONNECTIONS = 8


def serve_unix_broker(
    *,
    socket_path: Path,
    service: UnixBrokerService,
    socket_mode: int | None = None,
) -> None:
    socket_path = socket_path.resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, socket_mode or service.config.socket_mode)
    listener.listen(MAX_ACTIVE_CONNECTIONS)
    slots = threading.BoundedSemaphore(MAX_ACTIVE_CONNECTIONS)
    with ThreadPoolExecutor(max_workers=MAX_ACTIVE_CONNECTIONS) as workers:
        while True:
            # Block before accept: backpressure rather than rejecting live callers.
            # This also keeps the executor queue bounded by the active slot count.
            slots.acquire()
            try:
                connection, _address = listener.accept()
            except BaseException:
                slots.release()
                raise
            try:
                workers.submit(_serve_connection, connection, service, slots)
            except BaseException:
                connection.close()
                slots.release()
                raise


def _serve_connection(connection: socket.socket, service: UnixBrokerService, slots) -> None:
    try:
        with connection:
            _handle_connection(connection, service)
    finally:
        slots.release()


def _handle_connection(connection: socket.socket, service: UnixBrokerService) -> None:
    try:
        _, uid, _gid = read_peer_credentials(connection)
        frame = _read_frame(connection, service.config.max_request_frame_bytes)
        payload = json.loads(frame.decode("utf-8"))
        response_bytes = service.handle_message(uid, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        response_bytes = response_to_json(
            BrokerExecuteResponse(
                request_id="unknown",
                status="failed",
                error_code="broker_internal_error",
            )
        )
    connection.sendall(response_bytes)


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        part = connection.recv(65536)
        if not part:
            break
        total += len(part)
        if total > max_frame_bytes:
            raise ValueError("request frame exceeds broker bound")
        chunks.append(part)
        if part.endswith(b"\n"):
            break
    if not chunks:
        raise ValueError("empty request frame")
    return b"".join(chunks)
