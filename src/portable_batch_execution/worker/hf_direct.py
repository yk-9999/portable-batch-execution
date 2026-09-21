"""HF-direct wave execution helpers (no Private Data Plane byte I/O)."""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from typing import Any, Mapping

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SET_OPERATION,
    FIXED_SET_RESULT_SCHEMA_VERSION,
)
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate import (
    REQUEST_SCHEMA_VERSION,
)
from portable_batch_execution.transport.hf_bucket import (
    HfBucketTransport,
    artifact_ref_from_hf_object,
    artifact_ref_to_hf_bucket_ref,
    parse_hf_object_uri,
    validate_hf_bucket_ref,
)


class HfDirectExecutionError(RuntimeError):
    """Fail-closed HF-direct execution error."""


def jpx_hf_direct_enabled() -> bool:
    return os.environ.get("PBE_JPX_HF_DIRECT", "").strip() == "1"


def assert_no_private_data_plane_dependency() -> None:
    if os.environ.get("PBE_PRIVATE_DATA_PLANE_BASE_URL"):
        raise HfDirectExecutionError(
            "JPX HF-direct mode must not depend on Private Data Plane URLs"
        )
    if os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN"):
        raise HfDirectExecutionError(
            "JPX HF-direct mode must not depend on Private Data Plane credentials"
        )


def artifact_ref_is_hf_bucket(ref: ArtifactRef) -> bool:
    try:
        parse_hf_object_uri(ref.uri)
        return True
    except Exception:
        return False


def hf_bucket_ref_from_artifact(ref: ArtifactRef) -> dict[str, Any]:
    return validate_hf_bucket_ref(
        {
            "schema_version": "pbe.hf-bucket-artifact-ref.v1",
            "bucket_id": "yamauchiJP/system-trading-data",
            "object_path": parse_hf_object_uri(ref.uri),
            "sha256": ref.sha256.removeprefix("sha256:"),
            "size_bytes": int(ref.size_bytes or 0),
            "media_type": str(ref.media_type or "application/json"),
        }
    )


def read_verified_input_batch(
    transport: HfBucketTransport, ref: ArtifactRef
) -> tuple[bytes, dict[str, Any]]:
    payload = transport.read_verified(hf_bucket_ref_from_artifact(ref))
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HfDirectExecutionError("input batch is not valid JSON") from exc
    if parsed.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise HfDirectExecutionError("input batch schema mismatch")
    return payload, parsed


def derive_result_object_path_from_input_object_path(input_object_path: str) -> str:
    pattern = re.compile(
        r"^(?P<prefix>.+/normalized/(?P<digest>[^/]+)/)(?P<batch_id>batch-\d{5})-(?P<tag>[0-9a-f]{16})\.json$"
    )
    match = pattern.match(input_object_path.lstrip("/"))
    if match is None:
        raise HfDirectExecutionError("cannot derive result path from input object path")
    prefix = match.group("prefix").split("/normalized/")[0].rstrip("/")
    manifest_digest = match.group("digest")
    batch_id = match.group("batch_id")
    tag = sha256(f"{manifest_digest}|{batch_id}|fixed-set".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}/results/{manifest_digest}/{batch_id}-{tag}.json".lstrip("/")


def execute_hf_direct_trade_path_fixed_set_shard(
    *,
    transport: HfBucketTransport,
    replay_pack,
    job,
    shard,
    input_ref: ArtifactRef,
    output_object_path: str,
) -> tuple[ArtifactRef, dict[str, Any]]:
    if job.operation != FIXED_SET_OPERATION:
        raise HfDirectExecutionError("unsupported HF-direct operation")
    if job.operation_params.get("transport_profile") != "hf_bucket_direct":
        raise HfDirectExecutionError("transport_profile mismatch")
    assert_no_private_data_plane_dependency()

    batch_bytes, parsed_batch = read_verified_input_batch(transport, input_ref)
    resolved_output_path = output_object_path.lstrip("/")
    if not resolved_output_path:
        resolved_output_path = derive_result_object_path_from_input_object_path(
            parse_hf_object_uri(input_ref.uri)
        )
    result_payload = replay_pack.execute(
        job,
        shard,
        job.operation_params,
        {
            "batch": parsed_batch,
            "encoded_size": len(batch_bytes),
            "operation": job.operation,
        },
    )
    output = bytes(result_payload.get("result_json_bytes") or b"")
    if not output:
        raise HfDirectExecutionError("fixed set result bytes missing")
    if result_payload.get("schema_version") != FIXED_SET_RESULT_SCHEMA_VERSION:
        raise HfDirectExecutionError("fixed set result schema mismatch")

    uploaded = transport.write_verified_append_only(
        object_path=resolved_output_path,
        data=output,
        media_type="application/json",
    )
    out_ref = artifact_ref_from_hf_object(
        object_path=uploaded["object_path"],
        sha256_hex=uploaded["sha256"],
        size_bytes=uploaded["size_bytes"],
        media_type=uploaded["media_type"],
    )
    summary = dict(result_payload.get("summary") or {})
    summary["output_hf_ref"] = artifact_ref_to_hf_bucket_ref(out_ref)
    return out_ref, summary


def broker_input_digest_from_hf_ref(ref: Mapping[str, Any]) -> str:
    validated = validate_hf_bucket_ref(ref)
    material = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    return sha256(material.encode("utf-8")).hexdigest()
