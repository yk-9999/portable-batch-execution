from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACK_OPS = {
    "tabular-batch": {
        "tabular.normalize",
        "tabular.cast",
        "tabular.sort",
        "tabular.dedup",
        "tabular.join",
        "tabular.pit_join",
        "tabular.window",
        "tabular.rolling",
        "tabular.statistics",
        "tabular.format_migration",
    },
    "acquisition-batch": {
        "acquisition.rest",
        "acquisition.html",
        "acquisition.incremental",
    },
    "replay-eval-batch": {
        "replay_eval.replay",
        "replay_eval.benchmark",
        "replay_eval.compare",
        "replay_eval.parameter_sweep",
        "replay_eval.regression",
        "replay_eval.walk_forward",
        "replay_eval.oos",
        "replay_eval.backtest",
        "replay_eval.control",
        "replay_eval.robustness",
        "replay_eval.external_api_evaluation",
    },
    "ml-batch": {
        "ml.tfidf",
        "ml.hashing_vectorizer",
        "ml.logistic_regression",
        "ml.sgd_classifier",
        "ml.train_test",
        "ml.group_holdout",
        "ml.cross_validate",
        "ml.calibrate",
        "ml.threshold_sweep",
        "ml.batch_inference",
        "ml.embedding",
        "ml.similarity",
        "ml.clustering",
    },
    "media-batch": {
        "media.decode",
        "media.extract_audio",
        "media.resample",
        "media.mono",
        "media.segment",
        "media.merge",
        "media.metadata",
        "media.asr_merge",
        "media.overlap_remove",
    },
}


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRef(Frozen):
    object_id: str
    uri: str
    sha256: str
    media_type: str | None = None
    size_bytes: int | None = Field(None, ge=0)


class Provenance(Frozen):
    producer: str
    revision: str
    created_at: datetime
    parent_run_ids: tuple[str, ...] = ()


class InputManifestEntry(Frozen):
    name: str
    artifact: ArtifactRef
    partition_key: str | None = None


class InputManifest(Frozen):
    schema_version: Literal["1"] = "1"
    manifest_id: str
    entries: tuple[InputManifestEntry, ...]
    provenance: Provenance


class AdapterDescriptor(Frozen):
    adapter_id: str
    adapter_version: str
    adapter_api_version: Literal["1"] = "1"
    adapter_digest: str
    supported_operations: tuple[str, ...]


class CompatibilitySpec(Frozen):
    contract_version: Literal["1"] = "1"
    kernel_api_version: Literal["1"] = "1"
    pack_api_version: Literal["1"] = "1"
    adapter_api_version: Literal["1"] = "1"


class ExecutionPolicy(Frozen):
    max_parallel: int = Field(gt=0)
    max_attempts_per_shard: int = Field(gt=0)
    resume_enabled: bool = True


class Extent(Frozen):
    value: float = Field(ge=0)
    unit: Literal["records", "seconds", "minutes", "hours", "days"]


class RangeSpec(Frozen):
    kind: Literal["index", "time", "key"]
    start: str | int | float | None = None
    end: str | int | float | None = None


class ShardCorrectnessSpec(Frozen):
    mode: Literal["independent", "partition_affinity"] = "independent"
    lookback: Extent | None = None
    lookforward: Extent | None = None
    halo_before: Extent | None = None
    halo_after: Extent | None = None
    global_order_required: bool = False
    global_finalize_required: bool = False


class JobSpec(Frozen):
    schema_version: Literal["1"] = "1"
    job_id: str
    logical_run_id: str
    pack: Literal[
        "tabular-batch",
        "acquisition-batch",
        "replay-eval-batch",
        "ml-batch",
        "media-batch",
    ]
    operation: str
    input_manifest_ref: ArtifactRef
    sharding: ShardCorrectnessSpec
    execution: ExecutionPolicy
    security_profile: Literal["offline", "bounded-network", "external-api"]
    provenance: Provenance
    adapter: AdapterDescriptor | None = None
    operation_params: dict[str, Any] = Field(default_factory=dict)
    compatibility: CompatibilitySpec = Field(default_factory=CompatibilitySpec)

    @model_validator(mode="after")
    def closed(self):
        if self.operation not in PACK_OPS[self.pack] or {
            "shell",
            "command",
            "cmd",
            "python",
            "python_code",
            "script",
            "sql",
            "import_path",
            "entrypoint",
            "executable",
        } & set(self.operation_params):
            raise ValueError("closed operation parameters")
        return self


class ShardSpec(Frozen):
    schema_version: Literal["1"] = "1"
    logical_run_id: str
    shard_id: str
    ordinal: int = Field(ge=0)
    partition_key: str | None = None
    primary_range: RangeSpec | None = None
    correctness: ShardCorrectnessSpec
    input_refs: tuple[ArtifactRef, ...]
    input_digest: str
    execution_fingerprint: str


class WaveSpec(Frozen):
    schema_version: Literal["1"] = "1"
    logical_run_id: str
    wave_id: str
    ordinal: int = Field(ge=0)
    shard_ids: tuple[str, ...]
    max_parallel: int = Field(gt=0)


class ShardAttemptRecord(Frozen):
    schema_version: Literal["1"] = "1"
    logical_run_id: str
    shard_id: str
    attempt_id: str
    status: Literal["succeeded", "failed", "cancelled"]
    input_digest: str
    execution_fingerprint: str
    started_at: datetime
    finished_at: datetime
    wave_id: str | None = None
    output_refs: tuple[ArtifactRef, ...] = ()
    output_digest: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    failure: str | None = None
    backend_execution_id: str | None = None
    backend_job_id: str | None = None


class RunManifest(Frozen):
    schema_version: Literal["1"] = "1"
    logical_run_id: str
    revision: int = Field(ge=0)
    job_spec_digest: str
    status: Literal[
        "planned", "running", "finalizing", "succeeded", "failed", "cancelled"
    ]
    expected_shard_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    provenance: Provenance
    waves: tuple[WaveSpec, ...] = ()
    discovered_attempts: tuple[ShardAttemptRecord, ...] = ()
    canonical_attempts: tuple[ShardAttemptRecord, ...] = ()
    completed_shard_ids: tuple[str, ...] = ()
    missing_shard_ids: tuple[str, ...] = ()
    duplicate_shard_ids: tuple[str, ...] = ()
    finalization_status: Literal["not_started", "ready", "finalized", "failed"] = (
        "not_started"
    )
    final_output_refs: tuple[ArtifactRef, ...] = ()
    backend_executions: tuple[dict[str, Any], ...] = ()
