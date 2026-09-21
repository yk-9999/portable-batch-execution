from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from portable_batch_execution.transport.hf_bucket import (
    HfBucketTransport,
    HfBucketTransportError,
    InMemoryHfBucketStorage,
    RetryPolicy,
    redact_secrets,
    validate_hf_bucket_ref,
)


class TestHfBucketTransport(unittest.TestCase):
    def test_write_read_verify_and_retry(self):
        storage = InMemoryHfBucketStorage()

        class FlakyStorage(InMemoryHfBucketStorage):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def download_to(self, remote_path: str, local_path: Path) -> None:
                self.attempts += 1
                if self.attempts == 1:
                    raise TimeoutError("transient")
                return super().download_to(remote_path, local_path)

        transport = HfBucketTransport(
            FlakyStorage(),
            retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.01, max_backoff_seconds=0.01),
        )
        uploaded = transport.write_verified_append_only(
            object_path="pair-trading/v1/public-eval/x.json",
            data=b"{}",
            media_type="application/json",
        )
        ref = validate_hf_bucket_ref(uploaded)
        payload = transport.read_verified(ref)
        self.assertEqual(payload, b"{}")

    def test_redacts_token_like_substrings(self):
        text = redact_secrets("failure for hf_abcdefghijklmnopqrstuv")
        self.assertNotIn("hf_abcdefghijklmnopqrstuv", text)

    def test_refusing_overwrite(self):
        storage = InMemoryHfBucketStorage()
        transport = HfBucketTransport(storage)
        transport.write_verified_append_only(
            object_path="pair-trading/v1/public-eval/a.json",
            data=b"a",
            media_type="application/json",
        )
        with self.assertRaises(HfBucketTransportError):
            transport.write_verified_append_only(
                object_path="pair-trading/v1/public-eval/a.json",
                data=b"b",
                media_type="application/json",
            )

    def test_idempotent_reuse_on_matching_bytes(self):
        storage = InMemoryHfBucketStorage()
        transport = HfBucketTransport(storage)
        first = transport.write_verified_append_only(
            object_path="pair-trading/v1/public-eval/a.json",
            data=b"a",
            media_type="application/json",
        )
        second = transport.write_verified_append_only(
            object_path="pair-trading/v1/public-eval/a.json",
            data=b"a",
            media_type="application/json",
        )
        self.assertEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
