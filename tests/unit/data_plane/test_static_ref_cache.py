"""Model static artifact hashes are cached only while the inode version is unchanged."""

import os
import time
from unittest.mock import patch

from portable_batch_execution.data_plane.local import (
    _STATIC_VALIDATED_CACHE,
    _STATIC_VALIDATED_CACHE_LIMIT,
    LocalFilesystemDataPlane,
)


def test_same_immutable_static_artifact_is_reused_between_plane_instances(tmp_path):
    first = LocalFilesystemDataPlane(tmp_path)
    ref = first.write(b"a" * 512000, "application/octet-stream")
    second = LocalFilesystemDataPlane(tmp_path)
    _STATIC_VALIDATED_CACHE.clear()
    assert first.validate_static_ref(ref)
    assert second.validate_static_ref(ref)
    with patch("portable_batch_execution.data_plane.local.sha256", side_effect=AssertionError("rehash")):
        assert second.validate_static_ref(ref)
        assert first.validate_static_ref(ref)
    assert len(_STATIC_VALIDATED_CACHE) == 1


def test_same_size_overwrite_with_restored_mtime_is_rejected(tmp_path):
    store = LocalFilesystemDataPlane(tmp_path)
    ref = store.write(b"correct" * 20000, "application/octet-stream")
    _STATIC_VALIDATED_CACHE.clear()
    assert store.validate_static_ref(ref)
    path = tmp_path / "artifacts" / ref.object_id
    before = path.stat()
    # ctime clocks on some hosts need a separate tick for an observable change.
    time.sleep(0.02)
    path.write_bytes(b"corrupt" * 20000)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    assert not store.validate_static_ref(ref)


def test_replaced_inode_with_same_digest_revalidates(tmp_path):
    store = LocalFilesystemDataPlane(tmp_path)
    content = b"same-content" * 15000
    ref = store.write(content, "application/octet-stream")
    _STATIC_VALIDATED_CACHE.clear()
    assert store.validate_static_ref(ref)
    path = tmp_path / "artifacts" / ref.object_id
    replacement = tmp_path / "replacement"
    replacement.write_bytes(content)
    replacement.replace(path)
    assert store.validate_static_ref(ref)
    assert len(_STATIC_VALIDATED_CACHE) == 2


def test_cache_size_stays_bounded(tmp_path):
    _STATIC_VALIDATED_CACHE.clear()
    store = LocalFilesystemDataPlane(tmp_path)
    for i in range(_STATIC_VALIDATED_CACHE_LIMIT + 5):
        ref = store.write(f"bounded-{i}".encode())
        assert store.validate_static_ref(ref)
    assert len(_STATIC_VALIDATED_CACHE) == _STATIC_VALIDATED_CACHE_LIMIT
