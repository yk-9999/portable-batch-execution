import json
from pathlib import Path

from portable_batch_execution.contracts import (
    InputManifest,
    JobSpec,
    RunManifest,
    ShardAttemptRecord,
    ShardSpec,
    WaveSpec,
)

MODELS = {
    "input-manifest": InputManifest,
    "job-spec": JobSpec,
    "shard-spec": ShardSpec,
    "wave-spec": WaveSpec,
    "shard-attempt-record": ShardAttemptRecord,
    "run-manifest": RunManifest,
}


def test_checked_in_v1_schemas_match_contract_models():
    schema_dir = Path(__file__).parents[2] / "schemas" / "v1"
    for name, model in MODELS.items():
        actual = json.loads((schema_dir / f"{name}.schema.json").read_text())
        assert actual == model.model_json_schema()
