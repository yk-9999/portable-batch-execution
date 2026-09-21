from __future__ import annotations

import io
import json
import os
import unittest
from hashlib import sha256
from unittest import mock

import polars as pl

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    JobSpec,
    Provenance,
    ShardCorrectnessSpec,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SET_OPERATION,
)
from portable_batch_execution.transport.hf_bucket import (
    HF_BUCKET_ID,
    HF_BUCKET_REF_SCHEMA,
    HfBucketTransport,
    HfBucketTransportError,
    InMemoryHfBucketStorage,
    artifact_ref_from_hf_object,
)
from portable_batch_execution.worker.hf_direct_wave import (
    HF_WAVE_DESCRIPTOR_SCHEMA,
    HF_WAVE_RESULT_MANIFEST_SCHEMA,
    HfDirectWaveError,
    SerializedWaveDescriptor,
    execute_hf_direct_wave,
    load_wave_descriptor_from_ref,
    validate_wave_descriptor_payload,
    wave_result_manifest_object_path,
)


def _batch_bytes(batch_id: str = "batch-00000") -> bytes:
    return json.dumps(
        {
            "schema_version": "pbe.replay.trade-path-scenario-evaluate.v1",
            "batch_id": batch_id,
            "reference_notional": 1.0,
            "scenario": {
                "commission_bps_per_side": 0.0,
                "slippage_bps_per_side": 0.0,
                "borrow_bps_per_year": 0.0,
                "short_available": True,
            },
            "records": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hf_direct_env(public_revision: str) -> dict[str, str]:
    return {
        "PBE_MODE": "hf-direct",
        "PBE_JPX_HF_DIRECT": "1",
        "HF_TOKEN": "hf_test",
        "PBE_EXPECTED_PUBLIC_REVISION": public_revision,
    }


def _build_descriptor(storage: InMemoryHfBucketStorage, shard_count: int = 8):
    public_revision = "a" * 40
    manifest_digest = "manifest-digest"
    bucket_prefix = "pair-trading/v1/public-eval/"
    logical_run_id = "run-1"
    wave_id = "wave-0001"
    shards: list[ShardSpec] = []
    shard_ids: list[str] = []
    for ordinal in range(shard_count):
        batch_id = f"batch-{ordinal:05d}"
        batch = _batch_bytes(batch_id)
        digest = sha256(batch).hexdigest()
        input_path = (
            f"{bucket_prefix.rstrip('/')}/normalized/{manifest_digest}/"
            f"{batch_id}-{digest[:16]}.json"
        )
        storage.objects[input_path] = batch
        input_ref = artifact_ref_from_hf_object(
            object_path=input_path,
            sha256_hex=digest,
            size_bytes=len(batch),
            media_type="application/json",
        )
        shard_id = f"{wave_id}-shard-{ordinal:04d}"
        shard_ids.append(shard_id)
        shards.append(
            ShardSpec(
                logical_run_id=logical_run_id,
                shard_id=shard_id,
                ordinal=ordinal,
                correctness=ShardCorrectnessSpec(mode="independent"),
                input_refs=(input_ref,),
                input_digest=digest,
                execution_fingerprint=sha256(
                    f"{public_revision}|{shard_id}".encode()
                ).hexdigest(),
            )
        )
    job = JobSpec(
        job_id="job-1",
        logical_run_id=logical_run_id,
        pack="replay-batch",
        operation=FIXED_SET_OPERATION,
        input_manifest_ref=shards[0].input_refs[0],
        sharding=ShardCorrectnessSpec(mode="independent"),
        execution=ExecutionPolicy(
            max_parallel=min(8, shard_count),
            max_attempts_per_shard=4,
            resume_enabled=True,
        ),
        security_profile="offline",
        provenance=Provenance(
            producer="test",
            revision=public_revision,
            created_at="1970-01-01T00:00:00+00:00",
        ),
        operation_params={
            "schema_version": "pbe.replay.trade-path-scenario-evaluate-fixed-set-job.v1",
            "transport_profile": "hf_bucket_direct",
            "bucket_prefix": bucket_prefix,
        },
    )
    wave = WaveSpec(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=tuple(shard_ids),
        max_parallel=min(8, shard_count),
    )
    serialized = SerializedWaveDescriptor.build(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        manifest_digest=manifest_digest,
        public_revision=public_revision,
        bucket_prefix=bucket_prefix,
        job=job,
        wave=wave,
        shards=tuple(shards),
    )
    storage.objects[serialized.hf_ref["object_path"]] = serialized.bytes
    manifest_path = wave_result_manifest_object_path(
        bucket_prefix=bucket_prefix,
        public_revision=public_revision,
        manifest_digest=manifest_digest,
        wave_id=wave_id,
    )
    return serialized.hf_ref, manifest_path, public_revision, manifest_digest


def _parquet_payload(rows):
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    return buffer.getvalue()


def _pinned_parquet_ref(payload: bytes, revision: str, name: str = "shard.parquet"):
    digest = sha256(payload).hexdigest()
    uri = (
        f"https://huggingface.co/datasets/example/set/resolve/{revision}/inputs/{name}"
    )
    return ArtifactRef(
        object_id=digest,
        uri=uri,
        sha256=f"sha256:{digest}",
        size_bytes=len(payload),
        media_type="application/vnd.apache.parquet",
    )


def _build_paired_fill_descriptor(
    storage: InMemoryHfBucketStorage, public_revision: str
):
    manifest_digest = "paired-fill-manifest"
    bucket_prefix = "replay/public/paired-fill/"
    logical_run_id = "run-paired"
    wave_id = "wave-0002"
    trade_payload = _parquet_payload(
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "pair_role": "A",
                "core_price": 1.0,
                "core_size": 2.0,
                "start_pos": 0.0,
                "signed_qty": 2.0,
            },
            {
                "identity": 1,
                "identity_norm": "1",
                "pair_role": "B",
                "core_price": 1.0,
                "core_size": 2.0,
                "start_pos": 0.0,
                "signed_qty": -2.0,
            },
        ]
    )
    request = {
        "schema_version": "pbe.replay.paired-fill-reduce.v1",
        "request_id": "hf-direct-paired-fill",
        "identity_mapping": {
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
        },
        "pair_mapping": {
            "pair_role_column": "pair_role",
            "aggressor_role_value": "A",
            "passive_role_value": "B",
            "measurement_core_fields": ["core_price", "core_size"],
            "start_position_column": "start_pos",
            "signed_execution_column": "signed_qty",
        },
        "partition": {"terminal": True},
        "max_output_rows": 10,
        "max_exception_rows": 10,
    }
    request_bytes = json.dumps(request, sort_keys=True).encode("utf-8")
    request_digest = sha256(request_bytes).hexdigest()
    request_path = (
        f"{bucket_prefix.rstrip('/')}/paired-fill-requests/{public_revision}/"
        f"{manifest_digest}/{wave_id}-request-{request_digest[:16]}.json"
    )
    storage.objects[request_path] = request_bytes
    request_ref = artifact_ref_from_hf_object(
        object_path=request_path,
        sha256_hex=request_digest,
        size_bytes=len(request_bytes),
        media_type="application/json",
    )
    parquet_ref = _pinned_parquet_ref(trade_payload, public_revision)
    shard_id = f"{wave_id}-shard-0000"
    shards = [
        ShardSpec(
            logical_run_id=logical_run_id,
            shard_id=shard_id,
            ordinal=0,
            correctness=ShardCorrectnessSpec(mode="independent"),
            input_refs=(parquet_ref, request_ref),
            input_digest=sha256(trade_payload).hexdigest(),
            execution_fingerprint=sha256(
                f"{public_revision}|{shard_id}".encode()
            ).hexdigest(),
        )
    ]
    job = JobSpec(
        job_id="job-paired",
        logical_run_id=logical_run_id,
        pack="replay-batch",
        operation="replay.paired_fill_reduce",
        input_manifest_ref=request_ref,
        sharding=ShardCorrectnessSpec(mode="independent"),
        execution=ExecutionPolicy(
            max_parallel=1,
            max_attempts_per_shard=4,
            resume_enabled=True,
        ),
        security_profile="offline",
        provenance=Provenance(
            producer="test",
            revision=public_revision,
            created_at="1970-01-01T00:00:00+00:00",
        ),
        operation_params={
            "schema_version": "pbe.replay.paired-fill-reduce-job.v1",
            "transport_profile": "hf_direct",
            "bucket_prefix": bucket_prefix,
        },
    )
    wave = WaveSpec(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=(shard_id,),
        max_parallel=1,
    )
    serialized = SerializedWaveDescriptor.build(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        manifest_digest=manifest_digest,
        public_revision=public_revision,
        bucket_prefix=bucket_prefix,
        job=job,
        wave=wave,
        shards=tuple(shards),
    )
    storage.objects[serialized.hf_ref["object_path"]] = serialized.bytes
    manifest_path = wave_result_manifest_object_path(
        bucket_prefix=bucket_prefix,
        public_revision=public_revision,
        manifest_digest=manifest_digest,
        wave_id=wave_id,
    )
    return serialized.hf_ref, manifest_path, public_revision, trade_payload, parquet_ref


class TestHfDirectWaveWorker(unittest.TestCase):
    def test_descriptor_roundtrip_and_hash_verification(self):
        storage = InMemoryHfBucketStorage()
        ref, _manifest_path, _rev, _digest = _build_descriptor(storage, shard_count=2)
        transport = HfBucketTransport(storage)
        descriptor = load_wave_descriptor_from_ref(transport, ref)
        self.assertEqual(descriptor["wave"].wave_id, "wave-0001")
        validate_wave_descriptor_payload(
            {
                "schema_version": HF_WAVE_DESCRIPTOR_SCHEMA,
                "public_revision": "a" * 40,
                "logical_run_id": "run-1",
                "wave_id": "wave-0001",
                "manifest_digest": "manifest-digest",
                "job": descriptor["job"].model_dump(mode="json"),
                "wave": descriptor["wave"].model_dump(mode="json"),
                "shards": [
                    item.model_dump(mode="json") for item in descriptor["shards"]
                ],
            }
        )

    def test_eight_shard_wave_writes_manifest(self):
        storage = InMemoryHfBucketStorage()
        ref, manifest_path, public_revision, _digest = _build_descriptor(
            storage, shard_count=8
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(public_revision)
        with mock.patch.dict(os.environ, env, clear=False):
            summary = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        manifest = summary["manifest"]
        self.assertEqual(manifest["schema_version"], HF_WAVE_RESULT_MANIFEST_SCHEMA)
        self.assertEqual(len(manifest["shards"]), 8)
        self.assertTrue(all(row["status"] == "succeeded" for row in manifest["shards"]))
        for row in manifest["shards"]:
            output_ref = row["output_hf_ref"]
            self.assertEqual(output_ref["schema_version"], HF_BUCKET_REF_SCHEMA)
            self.assertEqual(output_ref["bucket_id"], HF_BUCKET_ID)
            self.assertTrue(storage.remote_exists(output_ref["object_path"]))
        self.assertTrue(storage.remote_exists(manifest_path))
        stored = json.loads(storage.objects[manifest_path].decode("utf-8"))
        self.assertEqual(stored["public_revision"], public_revision)

    def test_idempotent_result_reuse(self):
        storage = InMemoryHfBucketStorage()
        ref, manifest_path, public_revision, _digest = _build_descriptor(
            storage, shard_count=1
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(public_revision)
        with mock.patch.dict(os.environ, env, clear=False):
            first = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
            second = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        self.assertEqual(
            first["result_manifest_hf_ref"]["sha256"],
            second["result_manifest_hf_ref"]["sha256"],
        )

    def test_no_private_data_plane_env_required(self):
        storage = InMemoryHfBucketStorage()
        ref, manifest_path, public_revision, _digest = _build_descriptor(
            storage, shard_count=1
        )
        transport = HfBucketTransport(storage)
        env = {
            **_hf_direct_env(public_revision),
            "PBE_PRIVATE_DATA_PLANE_BASE_URL": "",
            "PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN": "",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )

    def test_checked_out_revision_mismatch_fails(self):
        storage = InMemoryHfBucketStorage()
        ref, manifest_path, _public_revision, _digest = _build_descriptor(
            storage, shard_count=1
        )
        transport = HfBucketTransport(storage)
        wrong = "b" * 40
        env = _hf_direct_env(wrong)
        with (
            mock.patch.dict(os.environ, env, clear=False),
            self.assertRaises(HfDirectWaveError),
        ):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )

    @mock.patch(
        "portable_batch_execution.worker.hf_direct.download_verified_pinned_resolve"
    )
    def test_paired_fill_hf_direct_multi_output_manifest(self, mock_download):
        storage = InMemoryHfBucketStorage()
        public_revision = "d" * 40
        ref, manifest_path, revision, trade_payload, _parquet_ref = (
            _build_paired_fill_descriptor(storage, public_revision)
        )

        def _fake_download(_ref, dest, opener=None):
            dest.write_bytes(trade_payload)

        mock_download.side_effect = _fake_download
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        with mock.patch.dict(os.environ, env, clear=False):
            summary = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        row = summary["manifest"]["shards"][0]
        self.assertEqual(row["status"], "succeeded")
        self.assertIn("output_hf_ref", row)
        self.assertIn("output_hf_refs", row)
        self.assertGreaterEqual(len(row["output_hf_refs"]), 1)
        self.assertIn("input_digest", row)
        for output_ref in row["output_hf_refs"]:
            self.assertTrue(storage.remote_exists(output_ref["object_path"]))

    @mock.patch(
        "portable_batch_execution.worker.hf_direct.download_verified_pinned_resolve"
    )
    def test_paired_fill_idempotent_output_paths(self, mock_download):
        storage = InMemoryHfBucketStorage()
        public_revision = "e" * 40
        ref, manifest_path, revision, trade_payload, _parquet_ref = (
            _build_paired_fill_descriptor(storage, public_revision)
        )
        mock_download.side_effect = lambda _ref, dest, opener=None: dest.write_bytes(
            trade_payload
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        with mock.patch.dict(os.environ, env, clear=False):
            first = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
            second = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        self.assertEqual(
            first["manifest"]["shards"][0]["output_hf_ref"],
            second["manifest"]["shards"][0]["output_hf_ref"],
        )

    def test_collision_mismatch_fails(self):
        storage = InMemoryHfBucketStorage()
        ref, manifest_path, public_revision, _digest = _build_descriptor(
            storage, shard_count=1
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(public_revision)
        with mock.patch.dict(os.environ, env, clear=False):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        storage.objects[manifest_path] = b"{}"
        with (
            mock.patch.dict(os.environ, env, clear=False),
            self.assertRaises(HfBucketTransportError),
        ):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )


if __name__ == "__main__":
    unittest.main()
