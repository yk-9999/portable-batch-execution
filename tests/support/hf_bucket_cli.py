"""In-memory model of the ``hf buckets list`` / ``hf buckets cp`` CLI contract."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess


def completed(stdout: str = "", returncode: int = 0) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class FakeBucketCli:
    """Records every invocation and models content-addressed object storage."""

    def __init__(self, prefix: str = "live-stream-news/hf-direct-20260921/objects/sha256"):
        self.prefix = prefix
        self.objects: dict[str, bytes] = {}
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []

    def __call__(self, argv, *, env):
        argv = [str(item) for item in argv]
        self.calls.append(argv)
        self.envs.append(dict(env))
        subcommand = argv[2] if len(argv) > 2 else ""
        if subcommand == "list":
            return self._list(argv[3])
        if subcommand == "cp":
            return self._cp(argv[3], argv[4])
        raise AssertionError(f"unexpected hf invocation: {argv}")

    def _list(self, uri: str) -> CompletedProcess:
        object_id = uri.rstrip("/").rsplit("/", 1)[-1]
        data = self.objects.get(object_id)
        if data is None:
            return completed("[]")
        entry = {"path": f"{self.prefix}/{object_id}", "size": len(data)}
        return completed(json.dumps([entry]))

    def _cp(self, source: str, destination: str) -> CompletedProcess:
        if source.startswith("hf://"):
            object_id = source.rstrip("/").rsplit("/", 1)[-1]
            if object_id not in self.objects:
                return completed("", returncode=1)
            Path(destination).write_bytes(self.objects[object_id])
            return completed("")
        data = Path(source).read_bytes()
        object_id = destination.rstrip("/").rsplit("/", 1)[-1]
        self.objects[object_id] = data
        return completed("")
