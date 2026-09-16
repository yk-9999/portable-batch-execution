"""Linux peer credential helpers for AF_UNIX connections."""

from __future__ import annotations

import socket
import struct
import sys
from typing import Protocol


class PeerCredentials(Protocol):
    pid: int
    uid: int
    gid: int


def peer_credentials_available() -> bool:
    return sys.platform == "linux" and hasattr(socket, "SO_PEERCRED")


def read_peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    if not peer_credentials_available():
        raise OSError("SO_PEERCRED is only available on Linux AF_UNIX sockets")
    data = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", data)
