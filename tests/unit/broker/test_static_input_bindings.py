import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    broker_composite_input_digest,
    broker_input_digest,
    broker_shard_input_digest,
    register_broker_private_run,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.data_plane.local import LocalFilesystemDataPlane

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000


def _artifact_ref(object_id: str, payload: bytes) -> dict:
    return ArtifactRef(
        object_id=object_id,
        uri=f"pbe://local/{object_id}",
        sha256="sha256:" + sha256(payload).hexdigest(),
        media_type="application/octet-stream",
        size_bytes=len(payload),
    ).model_dump(mode="json")


def _config(tmp_path: Path, *, static_bindings: list | None = None) -> BrokerConfig:
    payload = {
        "schema_version": "pbe.a1-unix-broker.config.v1",
        "public_sha": _PUBLIC_SHA,
        "max_input_bytes": 1_048_576,
        "allowed_operations_by_uid": {
            str(_UID): [
                ["tabular-batch", "tabular.sort"],
                ["ml-batch", "ml.distilbert_pair_binary_scores"],
            ]
        },
    }
    if static_bindings is not None:
        payload["static_input_bindings"] = static_bindings
    path = tmp_path / "broker-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return BrokerConfig.load(path)


def _stage_static_artifacts(state_root: Path, count: int = 2) -> list[dict]:
    plane = LocalFilesystemDataPlane(state_root)
    refs = []
    for index in range(count):
        payload = f"static-{index}".encode()
        ref = plane.write(payload, "application/octet-stream")
        refs.append(ref.model_dump(mode="json"))
    return refs


def test_config_without_static_bindings_defaults_empty(tmp_path):
    config = _config(tmp_path)
    assert config.static_input_bindings == {}
    assert config.static_input_refs_for("tabular-batch", "tabular.sort") == ()


def test_config_parses_static_binding_lookup(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    refs = _stage_static_artifacts(state_root, 2)
    config = _config(
        tmp_path,
        static_bindings=[
            {
                "pack": "ml-batch",
                "operation": "ml.distilbert_pair_binary_scores",
                "input_refs": refs,
            }
        ],
    )
    loaded = config.static_input_refs_for(
        "ml-batch", "ml.distilbert_pair_binary_scores"
    )
    assert len(loaded) == 2
    assert loaded[0].object_id == refs[0]["object_id"]


def test_config_rejects_duplicate_pack_operation(tmp_path):
    refs = _stage_static_artifacts(tmp_path, 1)
    with pytest.raises(ValueError, match="duplicate"):
        _config(
            tmp_path,
            static_bindings=[
                {
                    "pack": "ml-batch",
                    "operation": "ml.distilbert_pair_binary_scores",
                    "input_refs": refs,
                },
                {
                    "pack": "ml-batch",
                    "operation": "ml.distilbert_pair_binary_scores",
                    "input_refs": refs,
                },
            ],
        )


def test_empty_static_refs_preserve_legacy_input_digest():
    payload = b'[{"id":1}]'
    assert broker_shard_input_digest(payload, ()) == broker_input_digest(payload)


def test_composite_digest_changes_with_asset_digest_or_order(tmp_path):
    plane = LocalFilesystemDataPlane(tmp_path)
    first = plane.write(b"a", "application/octet-stream")
    second = plane.write(b"b", "application/octet-stream")
    raw = broker_input_digest(b"client")
    digest_ab = broker_composite_input_digest(raw, (first, second))
    digest_ba = broker_composite_input_digest(raw, (second, first))
    assert digest_ab != digest_ba
    tampered = ArtifactRef(
        object_id=first.object_id,
        uri=first.uri,
        sha256="sha256:" + ("0" * 64),
        media_type=first.media_type,
        size_bytes=first.size_bytes,
    )
    assert broker_composite_input_digest(raw, (tampered, second)) != digest_ab


def test_register_rejects_static_ref_digest_mismatch(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    plane = LocalFilesystemDataPlane(state_root)
    valid = plane.write(b"model", "application/octet-stream")
    bad = ArtifactRef(
        object_id=valid.object_id,
        uri=valid.uri,
        sha256="sha256:" + ("0" * 64),
        media_type=valid.media_type,
        size_bytes=valid.size_bytes,
    )
    with pytest.raises(ValueError, match="static input reference"):
        register_broker_private_run(
            state_root=state_root,
            request_id="req-static-bad-digest",
            pack="ml-batch",
            operation="ml.distilbert_pair_binary_scores",
            operation_params={},
            input_bytes=b"{}",
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
            static_input_refs=(bad,),
        )


def test_register_rejects_missing_static_ref(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    missing = _artifact_ref("missing", b"x")
    with pytest.raises(ValueError, match="static input reference"):
        register_broker_private_run(
            state_root=state_root,
            request_id="req-static-missing",
            pack="ml-batch",
            operation="ml.distilbert_pair_binary_scores",
            operation_params={},
            input_bytes=b"{}",
            input_media_type="application/json",
            public_sha=_PUBLIC_SHA,
            static_input_refs=(ArtifactRef.model_validate(missing),),
        )


def test_register_creates_ordered_multi_ref_shard(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    static_refs = tuple(
        ArtifactRef.model_validate(item)
        for item in _stage_static_artifacts(state_root, 4)
    )
    client_bytes = b'{"schema_version":"pbe.ml.distilbert-pair-binary-scores.v1"}'
    _, _, shard, _ = register_broker_private_run(
        state_root=state_root,
        request_id="req-static-order",
        pack="ml-batch",
        operation="ml.distilbert_pair_binary_scores",
        operation_params={},
        input_bytes=client_bytes,
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
        static_input_refs=static_refs,
    )
    assert len(shard.input_refs) == 5
    assert shard.input_refs[0].sha256 == f"sha256:{sha256(client_bytes).hexdigest()}"
    assert tuple(shard.input_refs[1:]) == static_refs
    assert shard.input_digest == broker_shard_input_digest(client_bytes, static_refs)


def test_service_passes_configured_static_refs_to_planner(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    static_json = _stage_static_artifacts(state_root, 2)
    config = _config(
        tmp_path,
        static_bindings=[
            {
                "pack": "ml-batch",
                "operation": "ml.distilbert_pair_binary_scores",
                "input_refs": static_json,
            }
        ],
    )
    backend = A1Controller(
        state_root,
        backend=None,
    )
    service = UnixBrokerService(
        state_root=state_root,
        config=config,
        controller=backend,
        poll_interval_seconds=0.0,
    )
    captured: dict[str, object] = {}

    def _capture_register(**kwargs):
        captured.update(kwargs)
        return register_broker_private_run(**kwargs)

    request = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": "req-service-static",
        "pack": "ml-batch",
        "operation": "ml.distilbert_pair_binary_scores",
        "operation_params": {},
        "input_media_type": "application/json",
        "input_b64": "e30=",
    }
    with patch(
        "portable_batch_execution.broker.service.register_broker_private_run",
        side_effect=_capture_register,
    ):
        service.handle_payload(_UID, request)
    static_refs = captured.get("static_input_refs")
    assert static_refs is not None
    assert len(static_refs) == 2
    assert static_refs[0].object_id == static_json[0]["object_id"]
