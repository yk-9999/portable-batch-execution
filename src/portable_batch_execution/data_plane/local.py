from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from portable_batch_execution.contracts import ArtifactRef


class LocalFilesystemDataPlane:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef:
        digest = sha256(data).hexdigest()
        p = self.root / digest
        p.write_bytes(data)
        return ArtifactRef(
            object_id=digest,
            uri=p.as_uri(),
            sha256=f"sha256:{digest}",
            media_type=media_type,
            size_bytes=len(data),
        )

    def read(self, ref: ArtifactRef) -> bytes:
        return Path(ref.uri.removeprefix("file:///")).read_bytes()

    def verify(self, ref: ArtifactRef) -> bool:
        return f"sha256:{sha256(self.read(ref)).hexdigest()}" == ref.sha256
