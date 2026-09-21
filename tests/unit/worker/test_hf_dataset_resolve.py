from __future__ import annotations

import io
import unittest
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.worker.hf_dataset_resolve import (
    HfDatasetResolveError,
    download_verified_pinned_resolve,
    validate_pinned_hf_dataset_resolve_uri,
)


class TestHfDatasetResolve(unittest.TestCase):
    def test_rejects_non_pinned_and_query_urls(self):
        with self.assertRaises(HfDatasetResolveError):
            validate_pinned_hf_dataset_resolve_uri(
                "https://huggingface.co/datasets/org/name/resolve/main/file.parquet"
            )
        with self.assertRaises(HfDatasetResolveError):
            validate_pinned_hf_dataset_resolve_uri(
                "https://huggingface.co/datasets/org/name/resolve/"
                + ("a" * 40)
                + "/file.parquet?download=1"
            )

    def test_download_verifies_sha_and_size(self):
        payload = b"PAR1mock"
        digest = sha256(payload).hexdigest()
        revision = "b" * 40
        uri = (
            "https://huggingface.co/datasets/example/set/resolve/"
            f"{revision}/inputs/shard.parquet"
        )
        ref = ArtifactRef(
            object_id=digest,
            uri=uri,
            sha256=f"sha256:{digest}",
            size_bytes=len(payload),
            media_type="application/vnd.apache.parquet",
        )
        dest = Path(self._tmp) / "shard.parquet"
        download_verified_pinned_resolve(
            ref,
            dest,
            opener=lambda _url: io.BytesIO(payload),
        )
        self.assertEqual(dest.read_bytes(), payload)

    def test_download_mismatch_fails(self):
        payload = b"PAR1mock"
        digest = sha256(payload).hexdigest()
        revision = "c" * 40
        uri = (
            "https://huggingface.co/datasets/example/set/resolve/"
            f"{revision}/inputs/shard.parquet"
        )
        ref = ArtifactRef(
            object_id=digest,
            uri=uri,
            sha256=f"sha256:{digest}",
            size_bytes=len(payload),
            media_type="application/vnd.apache.parquet",
        )
        dest = Path(self._tmp) / "bad.parquet"
        with self.assertRaises(HfDatasetResolveError):
            download_verified_pinned_resolve(
                ref,
                dest,
                opener=lambda _url: io.BytesIO(b"other"),
            )

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.mkdtemp()


if __name__ == "__main__":
    unittest.main()
