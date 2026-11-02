"""Interprocess exclusive lock for lifecycle writers and GC."""

from __future__ import annotations

import errno
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from portable_batch_execution.lifecycle.paths import lifecycle_root, writer_lock_path

_LIFECYCLE_LOCK_CONTENTION_DELAYS_S = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)


class LifecycleLockError(OSError):
    pass


class LifecycleLockTimeoutError(LifecycleLockError):
    """Lifecycle writer lock was contended until the acquire deadline expired."""


def _lock_contention_errno() -> frozenset[int]:
    if sys.platform == "win32":
        return frozenset({errno.EACCES, errno.EAGAIN})
    return frozenset({errno.EAGAIN, errno.EWOULDBLOCK})


def _is_lock_contention(exc: OSError) -> bool:
    return exc.errno in _lock_contention_errno()


class LifecycleStateLock:
    def __init__(self, state_root: Path):
        self._state_root = state_root.resolve()
        self._path = writer_lock_path(state_root)
        self._handle: int | None = None

    def _open_writer_lock_handle(self) -> int:
        try:
            lifecycle_root(self._state_root).mkdir(parents=True, exist_ok=True)
            return os.open(self._path, os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            raise LifecycleLockError("lifecycle state root is locked") from exc

    def _try_acquire_handle(self, handle: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def acquire(self, *, timeout: float | None = None) -> None:
        if timeout is None:
            self._acquire_once()
            return
        deadline = time.monotonic() + timeout
        delay_index = 0
        while True:
            try:
                self._acquire_once()
                return
            except LifecycleLockError as exc:
                cause = exc.__cause__
                if not isinstance(cause, OSError) or not _is_lock_contention(cause):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LifecycleLockTimeoutError(
                        "lifecycle state root lock acquire timed out"
                    ) from exc
                delay = _LIFECYCLE_LOCK_CONTENTION_DELAYS_S[
                    min(delay_index, len(_LIFECYCLE_LOCK_CONTENTION_DELAYS_S) - 1)
                ]
                time.sleep(min(delay, remaining))
                delay_index += 1

    def _acquire_once(self) -> None:
        handle = self._open_writer_lock_handle()
        try:
            self._try_acquire_handle(handle)
        except OSError as exc:
            os.close(handle)
            raise LifecycleLockError("lifecycle state root is locked") from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(handle)
            return
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


@contextmanager
def lifecycle_state_lock(state_root: Path, *, acquire_timeout: float | None = None):
    lock = LifecycleStateLock(state_root)
    lock.acquire(timeout=acquire_timeout)
    try:
        yield lock
    finally:
        lock.release()
