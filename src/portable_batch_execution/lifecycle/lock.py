"""Interprocess exclusive lock for lifecycle writers and GC."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from portable_batch_execution.lifecycle.paths import lifecycle_root, writer_lock_path


class LifecycleLockError(OSError):
    pass


class LifecycleStateLock:
    def __init__(self, state_root: Path):
        self._state_root = state_root.resolve()
        self._path = writer_lock_path(state_root)
        self._handle: int | None = None

    def acquire(self) -> None:
        lifecycle_root(self._state_root).mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            import msvcrt

            handle = os.open(self._path, os.O_CREAT | os.O_RDWR)
            try:
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                os.close(handle)
                raise LifecycleLockError("lifecycle state root is locked") from exc
            self._handle = handle
            return
        import fcntl

        handle = os.open(self._path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
def lifecycle_state_lock(state_root: Path):
    lock = LifecycleStateLock(state_root)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
