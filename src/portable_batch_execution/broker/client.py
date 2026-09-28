"""Reusable Unix broker consumer client with durable delivery handoff."""

from __future__ import annotations

import base64
import json
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portable_batch_execution.broker.protocol import (
    BrokerDeliveryCommitRequest,
    BrokerDeliveryCommitResponse,
    BrokerExecuteResponse,
    _REQUEST_SCHEMA,
)
from portable_batch_execution.lifecycle.durable import persist_verified_bytes


@dataclass(frozen=True)
class DurableBrokerConsumeResult:
    response: BrokerExecuteResponse
    destination: Path
    delivery: BrokerDeliveryCommitResponse | None


class UnixBrokerClient:
    """AF_UNIX JSON-line client; inject ``sender`` in tests."""

    def __init__(
        self,
        socket_path: Path | None = None,
        *,
        sender: Callable[[bytes], bytes] | None = None,
        max_frame_bytes: int = 64 * 1024 * 1024,
    ):
        self._socket_path = socket_path.resolve() if socket_path else None
        self._sender = sender
        self._max_frame_bytes = max_frame_bytes

    def _roundtrip(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = (json.dumps(payload) + "\n").encode("utf-8")
        if self._sender is not None:
            raw = self._sender(frame)
        else:
            if self._socket_path is None:
                raise ValueError("socket_path or sender required")
            raw = self._send_socket(frame)
        return json.loads(raw.decode("utf-8"))

    def _send_socket(self, frame: bytes) -> bytes:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self._socket_path))
            connection.sendall(frame)
            return self._read_frame(connection)

    def _read_frame(self, connection: socket.socket) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            part = connection.recv(65536)
            if not part:
                break
            total += len(part)
            if total > self._max_frame_bytes:
                raise ValueError("response frame exceeds bound")
            chunks.append(part)
            if part.endswith(b"\n"):
                break
        return b"".join(chunks)

    def execute(self, request: dict[str, Any]) -> BrokerExecuteResponse:
        if request.get("schema_version", _REQUEST_SCHEMA) != _REQUEST_SCHEMA:
            raise ValueError("unsupported execute request schema")
        payload = self._roundtrip(request)
        return BrokerExecuteResponse.model_validate(payload)

    def commit_delivery(
        self, request_id: str, output_sha256: str, output_size_bytes: int
    ) -> BrokerDeliveryCommitResponse:
        commit = BrokerDeliveryCommitRequest(
            request_id=request_id,
            output_sha256=output_sha256,
            output_size_bytes=output_size_bytes,
        )
        payload = self._roundtrip(commit.model_dump())
        return BrokerDeliveryCommitResponse.model_validate(payload)

    def execute_and_durably_persist(
        self,
        request: dict[str, Any],
        destination: Path,
    ) -> DurableBrokerConsumeResult:
        response = self.execute(request)
        if response.status != "succeeded":
            return DurableBrokerConsumeResult(
                response=response, destination=destination, delivery=None
            )
        if not response.output_b64 or not response.output_sha256:
            raise ValueError("successful broker response missing output")
        if response.output_size_bytes is None:
            raise ValueError("successful broker response missing authoritative size")
        output = base64.b64decode(response.output_b64, validate=True)
        persist_verified_bytes(
            output,
            destination,
            expected_sha256=response.output_sha256,
            expected_size_bytes=response.output_size_bytes,
        )
        delivery = self.commit_delivery(
            response.request_id,
            response.output_sha256,
            response.output_size_bytes,
        )
        if delivery.status != "accepted":
            raise ValueError(f"delivery commit failed: {delivery.error_code}")
        return DurableBrokerConsumeResult(
            response=response,
            destination=destination,
            delivery=delivery,
        )
