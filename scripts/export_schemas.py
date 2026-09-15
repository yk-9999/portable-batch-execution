from pathlib import Path

from portable_batch_execution.contracts import (
    InputManifest,
    JobSpec,
    RunManifest,
    ShardAttemptRecord,
    ShardSpec,
    WaveSpec,
)

models = {
    "input-manifest": InputManifest,
    "job-spec": JobSpec,
    "shard-spec": ShardSpec,
    "wave-spec": WaveSpec,
    "shard-attempt-record": ShardAttemptRecord,
    "run-manifest": RunManifest,
}
out = Path("schemas/v1")
out.mkdir(parents=True, exist_ok=True)
import json

for name, model in models.items():
    (out / f"{name}.schema.json").write_text(
        json.dumps(model.model_json_schema(), indent=2) + "\n"
    )
