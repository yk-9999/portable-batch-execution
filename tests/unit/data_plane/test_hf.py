import json
import sys
from hashlib import sha256
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from support.hf_bucket_api import FakeHfApi

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.data_plane.hf import (
    HF_CLI_TOKEN_ENV,
    HF_DIRECT_KIND,
    HF_TOKEN_ENV,
    OBJECT_LAYOUT_VERSION,
    HfBucketArtifactStore,
    HfBucketIdentity,
    HfBucketStoreError,
    build_hf_api,
    resolve_hf_direct_token,
)

_TOKEN = "hf_SENTINEL_TOKEN_DO_NOT_LEAK"
_OBJECT_ID = sha256(b"payload-bytes").hexdigest()
_REMOTE_PREFIX = "live-stream-news/hf-direct-20260921/objects/sha256"


def _identity(bucket="yamauchiJP/system-trading-data", prefix="live-stream-news/hf-direct-20260921/"):
    return HfBucketIdentity.from_metadata(
        {
            "kind": HF_DIRECT_KIND,
            "bucket": bucket,
            "prefix": prefix,
            "object_layout_version": OBJECT_LAYOUT_VERSION,
            "required_hf_cli_version": "1.8.0",
        }
    )


def _remote_path(object_id: str) -> str:
    return f"{_REMOTE_PREFIX}/{object_id}"


def _ref(data: bytes, **overrides):
    object_id = sha256(data).hexdigest()
    fields = {
        "object_id": object_id,
        "uri": f"hf://buckets/yamauchiJP/system-trading-data/live-stream-news/hf-direct-20260921/objects/sha256/{object_id}",
        "sha256": f"sha256:{object_id}",
        "size_bytes": len(data),
    }
    fields.update(overrides)
    return ArtifactRef(**fields)


def _store(api: FakeHfApi | None = None, **kwargs):
    fake = api or FakeHfApi()
    return HfBucketArtifactStore(identity=_identity(), api=fake, **kwargs), fake


def test_identity_maps_sha256_flat_object_layout():
    identity = _identity()
    assert identity.object_layout_version == OBJECT_LAYOUT_VERSION == "sha256-flat.v1"
    assert identity.objects_uri == (
        "hf://buckets/yamauchiJP/system-trading-data/"
        "live-stream-news/hf-direct-20260921/objects/sha256"
    )
    assert identity.remote_path(_OBJECT_ID) == (
        "live-stream-news/hf-direct-20260921/objects/sha256/" + _OBJECT_ID
    )
    assert identity.object_uri(_OBJECT_ID).endswith("/objects/sha256/" + _OBJECT_ID)


@pytest.mark.parametrize(
    "section",
    (
        {},
        {"bucket": "yamauchiJP/system-trading-data"},
        {"prefix": "ok/"},
        {
            "kind": HF_DIRECT_KIND,
            "bucket": "yamauchiJP/system-trading-data",
            "prefix": "ok/",
            "object_layout_version": "sha256-nested.v2",
        },
        {"kind": "something-else", "bucket": "a/b", "prefix": "ok/",
         "object_layout_version": OBJECT_LAYOUT_VERSION},
    ),
)
def test_identity_fails_closed_on_incomplete_or_wrong_metadata(section):
    with pytest.raises(HfBucketStoreError):
        HfBucketIdentity.from_metadata(section)


@pytest.mark.parametrize(
    ("bucket", "prefix"),
    (
        ("no-slash", "live-stream-news/"),
        ("owner/", "live-stream-news/"),
        ("/name", "live-stream-news/"),
        ("owner/name/extra", "live-stream-news/"),
        ("owner/na me", "live-stream-news/"),
        ("owner/name", ""),
        ("owner/name", "/absolute/"),
        ("owner/name", "../escape/"),
        ("owner/name", "live-stream-news/../../escape/"),
        ("owner/name", "live-stream-news\\windows/"),
        ("owner/name", "live-stream-news/ /bad/"),
        ("owner/name", None),
    ),
)
def test_identity_fails_closed_on_unsafe_or_out_of_scope_paths(bucket, prefix):
    with pytest.raises(HfBucketStoreError):
        HfBucketIdentity.from_metadata(
            {
                "kind": HF_DIRECT_KIND,
                "bucket": bucket,
                "prefix": prefix,
                "object_layout_version": OBJECT_LAYOUT_VERSION,
            }
        )


def test_read_derives_object_from_identity_then_verifies_hash_and_size():
    api = FakeHfApi()
    data = b"payload-bytes"
    api.objects[_remote_path(_OBJECT_ID)] = data
    store, _ = _store(api)
    assert store.read(_ref(data)) == data
    assert api.get_paths_calls[-1] == (
        "yamauchiJP/system-trading-data",
        [_remote_path(_OBJECT_ID)],
    )
    _, pairs = api.download_calls[-1]
    assert pairs[0][0] == _remote_path(_OBJECT_ID)


def test_read_fails_closed_when_object_missing():
    store, _ = _store(FakeHfApi())
    with pytest.raises(HfBucketStoreError):
        store.read(_ref(b"payload-bytes"))


def test_read_fails_closed_on_size_or_digest_mismatch():
    data = b"payload-bytes"
    api = FakeHfApi()
    api.objects[_remote_path(_OBJECT_ID)] = data
    store, _ = _store(api)
    with pytest.raises(HfBucketStoreError):
        store.read(_ref(data, size_bytes=len(data) + 1))

    api_corrupt = FakeHfApi()
    api_corrupt.objects[_remote_path(_OBJECT_ID)] = b"corrupted!!"
    store_corrupt, _ = _store(api_corrupt)
    with pytest.raises(HfBucketStoreError):
        store_corrupt.read(_ref(b"corrupted!!"))


def test_read_rejects_ref_object_id_that_is_not_lowercase_sha256():
    api = FakeHfApi()
    store, _ = _store(api)
    ref = ArtifactRef(
        object_id="OpaqueObject",
        uri="hf://buckets/yamauchiJP/system-trading-data/objects/sha256/OpaqueObject",
        sha256="sha256:" + _OBJECT_ID,
    )
    with pytest.raises(HfBucketStoreError):
        store.read(ref)
    assert api.get_paths_calls == []


def test_write_content_addresses_and_returns_hf_reference():
    store, api = _store()
    ref = store.write(b"payload-bytes", "application/octet-stream")
    assert ref.object_id == _OBJECT_ID
    assert ref.sha256 == f"sha256:{_OBJECT_ID}"
    assert ref.size_bytes == len(b"payload-bytes")
    assert ref.uri.startswith("hf://buckets/yamauchiJP/system-trading-data/")
    assert ref.uri.endswith("/objects/sha256/" + _OBJECT_ID)
    assert api.objects[_remote_path(_OBJECT_ID)] == b"payload-bytes"


def test_write_is_idempotent_and_reuses_existing_exact_size_object():
    api = FakeHfApi()
    store, _ = _store(api)
    first = store.write(b"payload-bytes")
    uploads_after_first = len(api.batch_calls)
    second = store.write(b"payload-bytes")
    uploads_after_second = len(api.batch_calls)
    assert first == second
    assert uploads_after_first == 1
    assert uploads_after_second == 1


def test_write_rechecks_persistence_and_fails_on_size_drift():
    class _WrongSizeApi(FakeHfApi):
        def batch_bucket_files(self, bucket_id, *, add=None):
            super().batch_bucket_files(bucket_id, add=add)
            for _, remote_path in add or []:
                self.objects[remote_path] = b"short"

    store, _ = _store(
        _WrongSizeApi(),
        persistence_verify_attempts=3,
        persistence_verify_delay_seconds=0,
    )
    with pytest.raises(HfBucketStoreError, match="wrong size"):
        store.write(b"payload-bytes")


def test_write_retries_bounded_post_upload_visibility_lag():
    class _DelayedVisibilityApi(FakeHfApi):
        def __init__(self):
            super().__init__()
            self.uploaded = False
            self.hidden_checks = 2

        def batch_bucket_files(self, bucket_id, *, add=None):
            super().batch_bucket_files(bucket_id, add=add)
            self.uploaded = True

        def get_bucket_paths_info(self, bucket_id, paths):
            if self.uploaded and self.hidden_checks:
                self.get_paths_calls.append((bucket_id, list(paths)))
                self.hidden_checks -= 1
                return
            yield from super().get_bucket_paths_info(bucket_id, paths)

    store, api = _store(
        _DelayedVisibilityApi(),
        persistence_verify_attempts=4,
        persistence_verify_delay_seconds=0,
    )
    ref = store.write(b"payload-bytes")
    assert ref.object_id == _OBJECT_ID
    assert api.hidden_checks == 0


def test_write_retries_transient_post_upload_metadata_error():
    class _TransientMetadataApi(FakeHfApi):
        def __init__(self):
            super().__init__()
            self.uploaded = False
            self.failures_left = 2

        def batch_bucket_files(self, bucket_id, *, add=None):
            super().batch_bucket_files(bucket_id, add=add)
            self.uploaded = True

        def get_bucket_paths_info(self, bucket_id, paths):
            if self.uploaded and self.failures_left:
                self.get_paths_calls.append((bucket_id, list(paths)))
                self.failures_left -= 1
                raise RuntimeError("transient metadata failure")
            yield from super().get_bucket_paths_info(bucket_id, paths)

    store, api = _store(
        _TransientMetadataApi(),
        persistence_verify_attempts=4,
        persistence_verify_delay_seconds=0,
    )
    ref = store.write(b"payload-bytes")
    assert ref.object_id == _OBJECT_ID
    assert api.failures_left == 0


def test_write_fails_closed_when_uploaded_object_remains_missing():
    class _MissingAfterUploadApi(FakeHfApi):
        def batch_bucket_files(self, bucket_id, *, add=None):
            self.batch_calls.append((bucket_id, list(add or [])))

    store, _ = _store(
        _MissingAfterUploadApi(),
        persistence_verify_attempts=3,
        persistence_verify_delay_seconds=0,
    )
    with pytest.raises(HfBucketStoreError, match="not persisted with exact size"):
        store.write(b"payload-bytes")


def test_exists_and_verify_are_fail_closed():
    api = FakeHfApi()
    data = b"payload-bytes"
    store, _ = _store(api)
    assert store.exists(_ref(data)) is False
    assert store.verify(_ref(data)) is False
    api.objects[_remote_path(_OBJECT_ID)] = data
    assert store.exists(_ref(data)) is True
    assert store.verify(_ref(data)) is True
    assert store.verify(_ref(data, size_bytes=0)) is False


def test_token_is_passed_to_hf_api_and_never_in_returned_metadata(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV, _TOKEN)
    captured: dict[str, str] = {}

    class _RecordingApi(FakeHfApi):
        def __init__(self, *, token=None):
            super().__init__(token=token)
            captured["token"] = token

    with patch(
        "portable_batch_execution.data_plane.hf.build_hf_api",
        side_effect=lambda token: _RecordingApi(token=token),
    ):
        store = HfBucketArtifactStore(identity=_identity())
        ref = store.write(b"payload-bytes")
        store.read(ref)
    assert captured["token"] == _TOKEN
    assert _TOKEN not in json.dumps(ref.model_dump(mode="json"))


def test_resolve_hf_direct_token_prefers_fixed_secret_then_hf_token_env():
    assert resolve_hf_direct_token(token=_TOKEN) == _TOKEN
    assert resolve_hf_direct_token(env={HF_TOKEN_ENV: "secret-token"}) == "secret-token"
    assert resolve_hf_direct_token(env={HF_CLI_TOKEN_ENV: "mapped-token"}) == "mapped-token"
    assert (
        resolve_hf_direct_token(
            env={HF_TOKEN_ENV: "secret-token", HF_CLI_TOKEN_ENV: "mapped-token"}
        )
        == "secret-token"
    )


def test_build_hf_api_passes_token_to_hf_api_constructor():
    mock_hf_api = MagicMock()
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = mock_hf_api
    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        build_hf_api(_TOKEN)
    mock_hf_api.assert_called_once_with(token=_TOKEN)


def test_identity_from_environment_uses_only_bounded_metadata(monkeypatch):
    monkeypatch.setenv("PBE_HF_BUCKET", "yamauchiJP/system-trading-data")
    monkeypatch.setenv("PBE_HF_PREFIX", "live-stream-news/hf-direct-20260921/")
    monkeypatch.setenv("PBE_HF_OBJECT_LAYOUT", "sha256-flat.v1")
    identity = HfBucketIdentity.from_environment()
    assert identity.bucket == "yamauchiJP/system-trading-data"
    assert identity.prefix == "live-stream-news/hf-direct-20260921/"
    assert identity.to_metadata()["object_layout_version"] == "sha256-flat.v1"


@pytest.mark.parametrize(
    "env",
    (
        {},
        {"PBE_HF_BUCKET": "yamauchiJP/system-trading-data"},
        {
            "PBE_HF_BUCKET": "yamauchiJP/system-trading-data",
            "PBE_HF_PREFIX": "../escape/",
        },
        {
            "PBE_HF_BUCKET": "yamauchiJP/system-trading-data",
            "PBE_HF_PREFIX": "ok/",
            "PBE_HF_OBJECT_LAYOUT": "sha256-nested.v2",
        },
    ),
)
def test_identity_from_environment_fails_closed(env):
    with pytest.raises(HfBucketStoreError):
        HfBucketIdentity.from_environment(env)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    (("1.8.0", "1.8.0"), ("1.8.0\n", "1.8.0"), ("version: 1.8.0\n", "1.8.0")),
)
def test_parse_hf_cli_version_output_accepts_supported_forms(stdout, expected):
    from portable_batch_execution.data_plane.hf import parse_hf_cli_version_output

    assert parse_hf_cli_version_output(stdout) == expected


@pytest.mark.parametrize("stdout", ("", "   ", "1.8.0\n2.0.0", "garbage", "hf 1.8.0"))
def test_parse_hf_cli_version_output_rejects_malformed(stdout):
    from portable_batch_execution.data_plane.hf import parse_hf_cli_version_output

    with pytest.raises(HfBucketStoreError):
        parse_hf_cli_version_output(stdout)


def test_preflight_fails_closed_on_missing_or_wrong_huggingface_hub(monkeypatch):
    store, _ = _store()
    with (
        patch(
            "portable_batch_execution.data_plane.hf.HfBucketArtifactStore.huggingface_hub_version",
            side_effect=HfBucketStoreError("huggingface_hub is not available"),
        ),
        pytest.raises(HfBucketStoreError, match="not available"),
    ):
        store.preflight()

    store_wrong, _ = _store()
    with (
        patch.object(store_wrong, "huggingface_hub_version", return_value="2.0.0"),
        pytest.raises(HfBucketStoreError, match="does not match required"),
    ):
        store_wrong.preflight()


def test_preflight_passes_on_exact_pinned_huggingface_hub():
    store, _ = _store()
    with patch.object(store, "huggingface_hub_version", return_value="1.8.0"):
        store.preflight()
