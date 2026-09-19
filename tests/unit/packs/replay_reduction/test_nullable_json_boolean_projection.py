import json

import polars as pl

from portable_batch_execution.packs.replay_reduction.models import (
    NullableJsonScalarProjection,
)
from portable_batch_execution.packs.replay_reduction.nullable_json_projection import (
    project_nullable_json_scalar_batch,
)

_PROJECTION = NullableJsonScalarProjection.model_validate(
    {
        "schema_version": "pbe.replay.nullable-json-scalar-projection.v1",
        "source_column": "raw_json",
        "key_path": ["flags", "active"],
        "scalar_type": "boolean",
        "output_column": "is_active",
    }
)


def test_nullable_boolean_projection_preserves_true_false():
    batch = pl.DataFrame(
        {
            "raw_json": [
                json.dumps({"flags": {"active": True}}),
                json.dumps({"flags": {"active": False}}),
                None,
            ]
        }
    )
    out = project_nullable_json_scalar_batch(batch, (_PROJECTION,))
    assert out["is_active"].to_list() == [True, False, None]
    assert out["is_active"].dtype == pl.Boolean


def test_nullable_boolean_projection_rejects_numeric_and_string_coercion():
    batch = pl.DataFrame(
        {
            "raw_json": [
                json.dumps({"flags": {"active": 1}}),
                json.dumps({"flags": {"active": "true"}}),
            ]
        }
    )
    out = project_nullable_json_scalar_batch(batch, (_PROJECTION,))
    assert out["is_active"].to_list() == [None, None]
