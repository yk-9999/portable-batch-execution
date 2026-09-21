import json
from hashlib import sha256

import pytest
from support.hf_bucket_cli import FakeBucketCli as _FakeBucketCli
from support.hf_bucket_cli import completed as _completed

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.data_plane.hf import (
    HF_CLI_TOKEN_ENV,
    HF_DIRECT_KIND,
    HF_TOKEN_ENV,
    OBJECT_LAYOUT_VERSION,
    HfBucketArtifactStore,
    HfBucketIdentity,
    HfBucketStoreError,
)

_TOKEN = "hf_SENTINEL_TOKEN_DO_NOT_LEAK"
_OBJECT_ID = sha256(b"payload-bytes").hexdigest()


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
    cli = _FakeBucketCli()
    data = b"payload-bytes"
    cli.objects[_OBJECT_ID] = data
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    assert store.read(_ref(data)) == data
    list_call = next(call for call in cli.calls if call[2] == "list")
    assert list_call[3].endswith("/objects/sha256/" + _OBJECT_ID)


def test_read_fails_closed_when_object_missing():
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    with pytest.raises(HfBucketStoreError):
        store.read(_ref(b"payload-bytes"))


def test_read_fails_closed_on_size_or_digest_mismatch():
    data = b"payload-bytes"
    cli = _FakeBucketCli()
    cli.objects[_OBJECT_ID] = data
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    with pytest.raises(HfBucketStoreError):
        store.read(_ref(data, size_bytes=len(data) + 1))

    cli_corrupt = _FakeBucketCli()
    cli_corrupt.objects[_OBJECT_ID] = b"corrupted!!"
    store_corrupt = HfBucketArtifactStore(identity=_identity(), runner=cli_corrupt)
    with pytest.raises(HfBucketStoreError):
        store_corrupt.read(_ref(b"corrupted!!"))


def test_read_rejects_ref_object_id_that_is_not_lowercase_sha256():
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    ref = ArtifactRef(
        object_id="OpaqueObject",
        uri="hf://buckets/yamauchiJP/system-trading-data/objects/sha256/OpaqueObject",
        sha256="sha256:" + _OBJECT_ID,
    )
    with pytest.raises(HfBucketStoreError):
        store.read(ref)
    assert cli.calls == []


def test_write_content_addresses_and_returns_hf_reference():
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    ref = store.write(b"payload-bytes", "application/octet-stream")
    assert ref.object_id == _OBJECT_ID
    assert ref.sha256 == f"sha256:{_OBJECT_ID}"
    assert ref.size_bytes == len(b"payload-bytes")
    assert ref.uri.startswith("hf://buckets/yamauchiJP/system-trading-data/")
    assert ref.uri.endswith("/objects/sha256/" + _OBJECT_ID)


def test_write_is_idempotent_and_reuses_existing_exact_size_object():
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    first = store.write(b"payload-bytes")
    uploads_after_first = [call for call in cli.calls if call[2] == "cp" and not call[3].startswith("hf://")]
    second = store.write(b"payload-bytes")
    uploads_after_second = [call for call in cli.calls if call[2] == "cp" and not call[3].startswith("hf://")]
    assert first == second
    assert len(uploads_after_first) == 1
    assert len(uploads_after_second) == 1


def test_write_rechecks_persistence_and_fails_on_size_drift():
    class _NonPersistingCli(_FakeBucketCli):
        def _cp(self, source, destination):
            if source.startswith("hf://"):
                return super()._cp(source, destination)
            self.objects[destination.rstrip("/").rsplit("/", 1)[-1]] = b"short"
            return _completed("")

    store = HfBucketArtifactStore(identity=_identity(), runner=_NonPersistingCli())
    with pytest.raises(HfBucketStoreError):
        store.write(b"payload-bytes")


def test_exists_and_verify_are_fail_closed():
    cli = _FakeBucketCli()
    data = b"payload-bytes"
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    assert store.exists(_ref(data)) is False
    assert store.verify(_ref(data)) is False
    cli.objects[_OBJECT_ID] = data
    assert store.exists(_ref(data)) is True
    assert store.verify(_ref(data)) is True
    assert store.verify(_ref(data, size_bytes=0)) is False


def test_token_is_never_on_argv_or_in_returned_metadata(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV, _TOKEN)
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    ref = store.write(b"payload-bytes")
    store.read(ref)
    for call in cli.calls:
        assert all(_TOKEN not in argument for argument in call)
    assert _TOKEN not in json.dumps(ref.model_dump(mode="json"))
    assert all(_TOKEN not in value for env in cli.envs for value in env.values() if value != _TOKEN)
    assert all(env.get(HF_CLI_TOKEN_ENV) == _TOKEN for env in cli.envs)


def test_token_is_absent_from_child_env_when_not_configured(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(identity=_identity(), runner=cli)
    store.write(b"payload-bytes")
    assert all(HF_CLI_TOKEN_ENV not in env for env in cli.envs)


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


def test_preflight_fails_closed_on_absent_or_wrong_cli(monkeypatch):
    cli = _FakeBucketCli()
    store = HfBucketArtifactStore(
        identity=_identity(), runner=cli, hf_binary="definitely-not-hf"
    )
    with pytest.raises(HfBucketStoreError):
        store.preflight()

    def wrong_version(argv, *, env):
        return _completed("2.0.0")

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/hf")
    store_wrong = HfBucketArtifactStore(
        identity=_identity(), runner=wrong_version, hf_binary="hf"
    )
    with pytest.raises(HfBucketStoreError):
        store_wrong.preflight()


def test_preflight_passes_on_exact_pinned_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/hf")

    def exact_version(argv, *, env):
        assert argv == ["hf", "--version"]
        return _completed("version: 1.8.0\n")

    store = HfBucketArtifactStore(
        identity=_identity(), runner=exact_version, hf_binary="hf"
    )
    store.preflight()
