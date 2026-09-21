import json

import httpx
import pytest

from portable_batch_execution.backends.base import WaveSubmission
from portable_batch_execution.backends.github_actions import (
    BackendExecutionRef,
    GitHubActionsBackend,
)
from portable_batch_execution.contracts import WaveSpec
from portable_batch_execution.data_plane.hf import (
    HfBucketIdentity,
    HfBucketStoreError,
)

_TOKEN = "SENTINEL_HF_DIRECT_DISPATCH_TOKEN"
_STORE = {
    "kind": "hf-buckets-direct",
    "bucket": "yamauchiJP/system-trading-data",
    "prefix": "live-stream-news/hf-direct-20260921/",
    "object_layout_version": "sha256-flat.v1",
    "required_hf_cli_version": "1.8.0",
}


def _client(handler):
    return httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )


def _reject_http(_request):
    raise AssertionError("no HTTP")


def _submission(run_id="opaque-run", wave_id="opaque-wave"):
    return WaveSubmission(
        WaveSpec(
            logical_run_id=run_id,
            wave_id=wave_id,
            ordinal=0,
            shard_ids=("opaque-shard",),
            max_parallel=1,
        )
    )


def test_hf_direct_submit_dispatches_exact_bounded_metadata():
    def handler(request):
        assert json.loads(request.content) == {
            "ref": "main",
            "inputs": {
                "wave_id": "opaque-wave",
                "run_id": "opaque-run",
                "hf_direct": True,
                "hf_bucket": "yamauchiJP/system-trading-data",
                "hf_prefix": "live-stream-news/hf-direct-20260921/",
                "hf_object_layout": "sha256-flat.v1",
            },
            "return_run_details": True,
        }
        return httpx.Response(201, json={"workflow_run_id": 11, "html_url": "https://run/11"})

    backend = GitHubActionsBackend(
        "o", "r", "execute-wave.yml", hf_direct_store=_STORE, client=_client(handler)
    )
    assert backend.submit_wave(_submission()) == BackendExecutionRef(
        "github-actions", "11", "https://run/11"
    )


def test_hf_direct_submit_never_sends_private_or_token_or_payload():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"workflow_run_id": 12})

    backend = GitHubActionsBackend(
        "o",
        "r",
        "w",
        hf_direct_store=_STORE,
        token=_TOKEN,
        client=_client(handler),
    )
    backend.submit_wave(_submission())
    inputs = captured["body"]["inputs"]
    assert "private" not in inputs
    assert set(inputs) == {
        "wave_id",
        "run_id",
        "hf_direct",
        "hf_bucket",
        "hf_prefix",
        "hf_object_layout",
    }
    serialized = json.dumps(captured["body"])
    assert _TOKEN not in serialized
    assert "data" not in inputs and "payload" not in inputs and "artifact" not in serialized


def test_hf_direct_store_identity_is_validated_and_frozen_at_construction():
    backend = GitHubActionsBackend("o", "r", "w", hf_direct_store=_STORE)
    assert isinstance(backend.hf_direct_store, HfBucketIdentity)
    assert backend.hf_direct_store.bucket == "yamauchiJP/system-trading-data"
    assert backend.hf_direct_store.prefix == "live-stream-news/hf-direct-20260921/"
    assert backend.hf_direct_store.object_layout_version == "sha256-flat.v1"


@pytest.mark.parametrize(
    "store",
    (
        {"bucket": "yamauchiJP/system-trading-data", "prefix": "ok/"},
        {**_STORE, "bucket": "no-slash"},
        {**_STORE, "prefix": "../escape/"},
        {**_STORE, "prefix": "/absolute/"},
        {**_STORE, "object_layout_version": "sha256-nested.v2"},
        {**_STORE, "kind": "something-else"},
        {},
    ),
)
def test_hf_direct_unsafe_or_invalid_config_fails_closed(store):
    with pytest.raises(HfBucketStoreError):
        GitHubActionsBackend("o", "r", "w", hf_direct_store=store, client=_client(_reject_http))


def test_hf_direct_rejects_ambiguous_dual_configuration():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GitHubActionsBackend(
            "o",
            "r",
            "w",
            private_data_plane=True,
            hf_direct_store=_STORE,
            client=_client(_reject_http),
        )


@pytest.mark.parametrize("field", ("run_id", "wave_id"))
def test_hf_direct_rejects_unsafe_identifiers_before_dispatch(field):
    values = {"run_id": "opaque-run", "wave_id": "opaque-wave"}
    values[field] = "../escape"
    backend = GitHubActionsBackend(
        "o", "r", "w", hf_direct_store=_STORE, client=_client(_reject_http)
    )
    with pytest.raises(ValueError, match="must be an opaque identifier"):
        backend.submit_wave(_submission(**values))


def test_old_private_data_plane_dispatch_is_unchanged():
    def handler(request):
        assert json.loads(request.content) == {
            "ref": "main",
            "inputs": {
                "wave_id": "opaque-wave",
                "run_id": "opaque-run",
                "private": True,
            },
            "return_run_details": True,
        }
        return httpx.Response(201, json={"workflow_run_id": 13})

    backend = GitHubActionsBackend(
        "o", "r", "w", private_data_plane=True, client=_client(handler)
    )
    assert backend.hf_direct_store is None
    assert backend.submit_wave(_submission()) == BackendExecutionRef(
        "github-actions", "13", None
    )


def test_default_and_public_dispatch_are_unchanged():
    def handler(request):
        assert json.loads(request.content)["inputs"] == {"wave_id": "wave-0000"}
        return httpx.Response(201, json={"workflow_run_id": 14})

    backend = GitHubActionsBackend("o", "r", "w", client=_client(handler))
    assert backend.hf_direct_store is None
    assert backend.private_data_plane is False
    assert backend.submit_wave(
        WaveSubmission(
            WaveSpec(
                logical_run_id="run",
                wave_id="wave-0000",
                ordinal=0,
                shard_ids=("shard-0",),
                max_parallel=1,
            )
        )
    ) == BackendExecutionRef("github-actions", "14", None)
