"""In-memory model of ``huggingface_hub.HfApi`` bucket file APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FakeBucketPathInfo:
    path: str
    size: int


class FakeHfApi:
    """Records invocations and models content-addressed bucket object storage."""

    def __init__(self, *, token: str | None = None) -> None:
        self.token = token
        self.objects: dict[str, bytes] = {}
        self.get_paths_calls: list[tuple[str, list[str]]] = []
        self.download_calls: list[tuple[str, list[tuple[str, Path]]]] = []
        self.batch_calls: list[tuple[str, list[tuple[Path, str]]]] = []

    def get_bucket_paths_info(self, bucket_id: str, paths: list[str]):
        self.get_paths_calls.append((bucket_id, list(paths)))
        for path in paths:
            data = self.objects.get(path)
            if data is not None:
                yield FakeBucketPathInfo(path=path, size=len(data))

    def download_bucket_files(
        self,
        bucket_id: str,
        path_pairs: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool = True,
    ) -> None:
        self.download_calls.append((bucket_id, list(path_pairs)))
        for remote_path, local_path in path_pairs:
            data = self.objects.get(remote_path)
            if data is None:
                if raise_on_missing_files:
                    raise FileNotFoundError(remote_path)
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[Path, str]] | None = None,
    ) -> None:
        uploads = list(add or [])
        self.batch_calls.append((bucket_id, uploads))
        for local_path, remote_path in uploads:
            self.objects[remote_path] = local_path.read_bytes()
