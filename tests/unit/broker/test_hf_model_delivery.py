"""Offline checks for pinned HF model delivery, legacy request identity, and scope."""

from __future__ import annotations

import json

import pytest

from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import register_broker_private_run
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane
from portable_batch_execution.model_delivery import HFPrivateModelDelivery

_MODEL = {
    "kind": "hf-private-repo-oidc",
    "repo_id": "yamauchiJP/candidate-private-model",
    "revision": "a" * 40,
    "filenames": [
        "model_a/config.json",
        "model_a/model.safetensors",
        "model_b/config.json",
        "model_b/model.safetensors",
    ],
}
_PACK = ("ml-batch", "ml.distilbert_pair_binary_scores")


def _refs(root):
    plane = LocalFilesystemDataPlane(root)
    return tuple(plane.write(bytes([n]) * 8, "application/octet-stream") for n in range(4))


def _broker_config(tmp_path, refs, delivery):
    entry = {
        "pack": _PACK[0],
        "operation": _PACK[1],
        "input_refs": [ref.model_dump(mode="json") for ref in refs],
    }
    if delivery is not None:
        entry["model_delivery"] = delivery
    conf = {
        "schema_version": "pbe.a1-unix-broker.config.v1",
        "public_sha": "a" * 40,
        "allowed_operations_by_uid": {"1000": [list(_PACK)]},
        "static_input_bindings": [entry],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(conf))
    return BrokerConfig.load(path)


def test_config_pins_readonly_source_without_exposing_model_to_client(tmp_path):
    refs = _refs(tmp_path / "state")
    config = _broker_config(tmp_path, refs, _MODEL)
    assert config.model_delivery_for(*_PACK).revision == "a" * 40
    assert config.static_input_refs_for(*_PACK) == refs


@pytest.mark.parametrize("override", [
    {"repo_id": "../other/evil"},
    {"repo_id": "some/../evil"},
    {"revision": "main"},
    {"filenames": ["../escape/config.json", "a/model.safetensors", "b/config.json", "b/model.safetensors"]},
    {"filenames": ["a/config.json", "a/model.safetensors", "a/config.json", "b/model.safetensors"]},
    {"extra_secret": "no"},
])
def test_config_rejects_mutable_or_unsafe_source(tmp_path, override):
    refs = _refs(tmp_path / "state")
    candidate = {**_MODEL, **override}
    with pytest.raises((ValueError, TypeError)):
        _broker_config(tmp_path, refs, candidate)


def test_legacy_registration_replay_unchanged_after_delivery_enabled(tmp_path):
    root = tmp_path / "state"
    refs = _refs(root)
    args = {
        "state_root": root,
        "request_id": "req-old-pinned",
        "pack": _PACK[0],
        "operation": _PACK[1],
        "operation_params": {},
        "input_bytes": b'{"rows":[]}',
        "input_media_type": "application/json",
        "public_sha": "f" * 40,
        "static_input_refs": refs,
    }
    old_job, _wave, old_shard, _manifest = register_broker_private_run(**args)
    old_json = old_job.model_dump_json()
    assert "model_delivery" not in old_json
    assert old_job.security_profile == "offline"
    # Changing only the retrieval route must not change model/input identity.
    source = HFPrivateModelDelivery.model_validate(_MODEL)
    same_job, _wave, same_shard, _manifest = register_broker_private_run(
        **args, model_delivery=source
    )
    assert same_job == old_job
    assert same_shard.input_digest == old_shard.input_digest
    assert same_shard.execution_fingerprint == old_shard.execution_fingerprint
    assert "model_delivery" not in same_job.model_dump_json()


def test_fresh_registration_uses_bounded_network_without_changing_fingerprint(tmp_path):
    root = tmp_path / "state"
    refs = _refs(root)
    base = {
        "state_root": root,
        "pack": _PACK[0],
        "operation": _PACK[1],
        "operation_params": {},
        "input_bytes": b'{"rows":[]}',
        "input_media_type": "application/json",
        "public_sha": "f" * 40,
        "static_input_refs": refs,
    }
    source = HFPrivateModelDelivery.model_validate(_MODEL)
    new_job, _, new_shard, _ = register_broker_private_run(
        **base, request_id="req-new-pinned", model_delivery=source
    )
    assert new_job.model_delivery == source
    assert new_job.security_profile == "bounded-network"
    assert new_shard.input_refs[1:] == refs
    assert new_job.operation_params == {}
    new_json = json.loads(new_job.model_dump_json())
    assert new_json["model_delivery"]["repo_id"] == _MODEL["repo_id"]
    assert new_json["model_delivery"]["revision"] == _MODEL["revision"]


def test_source_requires_four_digest_size_pinned_refs(tmp_path):
    refs = _refs(tmp_path / "state")
    entry = {
        "pack": _PACK[0], "operation": _PACK[1],
        "input_refs": [ref.model_dump(mode="json") for ref in refs[:3]],
        "model_delivery": _MODEL,
    }
    config = {
        "schema_version": "pbe.a1-unix-broker.config.v1",
        "public_sha": "a" * 40,
        "allowed_operations_by_uid": {"1000": [list(_PACK)]},
        "static_input_bindings": [entry],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="four size-pinned"):
        BrokerConfig.load(p)


def test_service_uses_source_only_from_private_broker_config(tmp_path, monkeypatch):
    from portable_batch_execution.broker import service as broker_service
    from portable_batch_execution.broker.service import UnixBrokerService
    from portable_batch_execution.controller.a1_controller import A1Controller

    root = tmp_path / "state"
    refs = _refs(root)
    config = _broker_config(tmp_path, refs, _MODEL)
    controller = A1Controller(root, backend=None)
    service = UnixBrokerService(state_root=root, config=config, controller=controller)
    original = broker_service.register_broker_private_run
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(broker_service, "register_broker_private_run", capture)
    response = service.handle_payload(
        1000,
        {
            "schema_version": "pbe.a1-unix-broker.request.v1",
            "request_id": "hf-model-private-config",
            "pack": _PACK[0],
            "operation": _PACK[1],
            "operation_params": {},
            "input_media_type": "application/json",
            "input_b64": "e30=",
        },
    )
    assert captured["model_delivery"] == config.model_delivery_for(*_PACK)
    assert captured["static_input_refs"] == refs
    assert response.status in {"failed", "exhausted", "succeeded"}
    # Source metadata never comes from client-supplied operation_params.
    assert captured["operation_params"] == {}
