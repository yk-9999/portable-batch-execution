import json

import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
)
from portable_batch_execution.packs.replay_reduction.models import (
    PAIRED_FILL_MAX_EXCEPTION_ROWS,
    PAIRED_FILL_MAX_INPUT_BYTES,
    PAIRED_FILL_MAX_INPUT_FILES,
    PAIRED_FILL_MAX_OUTPUT_BYTES,
    PAIRED_FILL_MAX_OUTPUT_ROWS,
    AdministrativeRowHandling,
    PairedFillReduceRequest,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_reduce import (
    METADATA_SCHEMA_VERSION,
    build_paired_fill_metadata,
    execute_paired_fill_reduce,
    publish_paired_fill_reduce_artifacts,
)

_ROLE_A = "side_a"
_ROLE_B = "side_b"


def _row(**fields):
    base = {
        "identity": 1,
        "identity_norm": "1",
        "pair_role": _ROLE_A,
        "core_price": 100.0,
        "core_size": 2.0,
        "start_pos": 0.0,
        "signed_qty": 2.0,
    }
    base.update(fields)
    return base


def _request(**overrides):
    base = {
        "schema_version": "pbe.replay.paired-fill-reduce.v1",
        "request_id": "pf-1",
        "identity_mapping": {
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
        },
        "pair_mapping": {
            "pair_role_column": "pair_role",
            "aggressor_role_value": _ROLE_A,
            "passive_role_value": _ROLE_B,
            "measurement_core_fields": ["core_price", "core_size"],
            "start_position_column": "start_pos",
            "signed_execution_column": "signed_qty",
        },
        "partition": {"terminal": True},
        "max_output_rows": 100,
        "max_exception_rows": 10,
    }
    partition = overrides.pop("partition", None)
    if partition is not None:
        base["partition"] = partition
    base.update(overrides)
    return base


def _write(path, rows):
    pl.DataFrame(rows).write_parquet(path)
    return path


def _run(tmp_path, rows, **request_overrides):
    path = _write(tmp_path / "part.parquet", rows)
    return execute_paired_fill_reduce([path], _request(**request_overrides))


def _participant(ledger_row, role):
    return next(item for item in ledger_row["participants"] if item["role"] == role)


def test_complete_pair_open_both(tmp_path):
    result = _run(
        tmp_path,
        [
            _row(identity=10, identity_norm="10", start_pos=0.0, signed_qty=3.0),
            _row(
                identity=10,
                identity_norm="10",
                pair_role=_ROLE_B,
                start_pos=0.0,
                signed_qty=-3.0,
            ),
        ],
    )
    row = result["ledger_rows"][0]
    assert row["classification"] == "complete_pair"
    assert _participant(row, "aggressor")["opening_quantity"] == 3.0
    assert _participant(row, "passive")["opening_quantity"] == 3.0


def test_complete_pair_flip(tmp_path):
    result = _run(
        tmp_path,
        [
            _row(identity=11, identity_norm="11", start_pos=10.0, signed_qty=-15.0),
            _row(
                identity=11,
                identity_norm="11",
                pair_role=_ROLE_B,
                start_pos=-5.0,
                signed_qty=5.0,
            ),
        ],
    )
    aggressor = _participant(result["ledger_rows"][0], "aggressor")
    assert aggressor["closing_quantity"] == 10.0
    assert aggressor["opening_quantity"] == 5.0
    assert aggressor["post_position"] == -5.0


def test_singleton(tmp_path):
    result = _run(
        tmp_path,
        [
            _row(identity=20, identity_norm="20"),
            _row(identity=21, identity_norm="21"),
        ],
    )
    assert len(result["ledger_rows"]) == 2
    assert all(item["classification"] == "singleton" for item in result["ledger_rows"])


def test_boundary_split_pair(tmp_path):
    first = _write(
        tmp_path / "a.parquet",
        [_row(identity=30, identity_norm="30", start_pos=1.0, signed_qty=1.0)],
    )
    second = _write(
        tmp_path / "b.parquet",
        [
            _row(
                identity=30,
                identity_norm="30",
                pair_role=_ROLE_B,
                start_pos=0.0,
                signed_qty=-1.0,
            )
        ],
    )
    result = execute_paired_fill_reduce([first, second], _request())
    assert result["ledger_rows"][0]["classification"] == "complete_pair"


def test_measurement_core_conflict(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        _run(
            tmp_path,
            [
                _row(identity=40, identity_norm="40", core_price=1.0),
                _row(
                    identity=40,
                    identity_norm="40",
                    pair_role=_ROLE_B,
                    core_price=2.0,
                ),
            ],
        )


def test_invalid_pair_role(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        _run(
            tmp_path,
            [
                _row(identity=50, identity_norm="50"),
                _row(identity=50, identity_norm="50", pair_role=_ROLE_A),
            ],
        )


def test_non_adjacent_reappearing_identity(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        _run(
            tmp_path,
            [
                _row(identity=60, identity_norm="60"),
                _row(identity=61, identity_norm="61"),
                _row(identity=60, identity_norm="60"),
            ],
        )


def test_administrative_sentinel(tmp_path):
    request = _request(
        administrative_row_handling={
            "schema_version": "pbe.replay.administrative-row-handling.v1",
            "predicate": {
                "identity_equals": 0,
                "exact_match_fields": {"identity_norm": None},
            },
        }
    )
    path = _write(
        tmp_path / "part.parquet",
        [
            {
                "identity": 0,
                "identity_norm": None,
                "pair_role": _ROLE_A,
                "core_price": 1.0,
                "core_size": 1.0,
                "start_pos": 0.0,
                "signed_qty": 0.0,
            },
            _row(identity=70, identity_norm="70"),
        ],
    )
    result = execute_paired_fill_reduce([path], request)
    assert result["summary"]["administrative_row_count"] == 1
    assert result["ledger_rows"][0]["classification"] == "singleton"


def test_unrecognized_invalid_identity(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        _run(tmp_path, [_row(identity=0, identity_norm="0")])


def test_malformed_sentinel_config():
    with pytest.raises(ValidationError):
        AdministrativeRowHandling.model_validate(
            {
                "schema_version": "pbe.replay.administrative-row-handling.v1",
                "predicate": {"identity_equals": "bad"},
            }
        )


def test_bounds_enforced(tmp_path):
    with pytest.raises(ValueError, match="output row count exceeds limit"):
        _run(
            tmp_path,
            [
                _row(identity=80, identity_norm="80"),
                _row(identity=81, identity_norm="81"),
                _row(identity=82, identity_norm="82"),
            ],
            max_output_rows=1,
        )


def test_deterministic_content_identity(tmp_path):
    rows = [
        _row(identity=90, identity_norm="90"),
        _row(identity=90, identity_norm="90", pair_role=_ROLE_B, signed_qty=-2.0),
    ]
    path = _write(tmp_path / "part.parquet", rows)
    request = _request()
    first = execute_paired_fill_reduce([path], request)
    second = execute_paired_fill_reduce([path], request)
    assert first["summary"]["content_identity"] == second["summary"]["content_identity"]
    assert first["ledger_parquet_identity"] == second["ledger_parquet_identity"]


def test_nullable_json_projection_and_request_parsing(tmp_path):
    request = _request(
        nullable_json_scalar_projections=[
            {
                "schema_version": "pbe.replay.nullable-json-scalar-projection.v1",
                "source_column": "raw_json",
                "key_path": ["meta", "tag"],
                "scalar_type": "string",
                "output_column": "proj_tag",
            }
        ]
    )
    PairedFillReduceRequest.model_validate(request)
    path = _write(
        tmp_path / "part.parquet",
        [
            {
                **_row(identity=100, identity_norm="100"),
                "raw_json": json.dumps({"meta": {"tag": "ok"}}),
            }
        ],
    )
    result = execute_paired_fill_reduce([path], request)
    assert result["ledger_rows"][0]["classification"] == "singleton"


def test_publish_paired_fill_artifacts_metadata_and_parquet_integrity():
    class _Ref:
        def __init__(self, object_id, payload, media_type):
            self.object_id = object_id
            self.uri = f"pbe://private/{object_id}"
            self.sha256 = "sha256:" + __import__("hashlib").sha256(payload).hexdigest()
            self.size_bytes = len(payload)
            self.media_type = media_type

        def model_dump(self, mode="json"):
            return {
                "object_id": self.object_id,
                "uri": self.uri,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "media_type": self.media_type,
            }

    class Plane:
        def __init__(self):
            self.payloads = {}
            self.counter = 0

        def write(self, data, media_type):
            object_id = f"out-{self.counter}"
            self.counter += 1
            self.payloads[object_id] = data
            return _Ref(object_id, data, media_type)

    plane = Plane()
    result = {
        "schema_version": "pbe.replay.paired-fill-reduce-result.v1",
        "summary": {"ledger_row_count": 1, "content_identity": "sha256:abc"},
        "exceptions": [],
        "outgoing_carry": {"schema_version": "pbe.replay.paired-fill-reduce-carry.v1"},
        "ledger_parquet_bytes": b"PAR1",
        "ledger_parquet_identity": "sha256:"
        + __import__("hashlib").sha256(b"PAR1").hexdigest(),
    }

    def _matches(data, ref):
        return ref.sha256 == "sha256:" + __import__("hashlib").sha256(data).hexdigest()

    metadata_bytes, refs = publish_paired_fill_reduce_artifacts(
        plane,
        result,
        artifact_ref_matches_bytes=_matches,
        shard_stage_failure=RuntimeError,
    )
    assert len(refs) == 2
    metadata = __import__("json").loads(metadata_bytes.decode())
    assert metadata["schema_version"] == METADATA_SCHEMA_VERSION
    assert metadata["ledger_parquet_ref"]["object_id"] == refs[1].object_id
    assert (
        build_paired_fill_metadata(result, ledger_parquet_ref=refs[1])["summary"][
            "ledger_row_count"
        ]
        == 1
    )


def test_carry_non_terminal_partition(tmp_path):
    first = _write(tmp_path / "a.parquet", [_row(identity=110, identity_norm="110")])
    second = _write(
        tmp_path / "b.parquet",
        [
            _row(
                identity=110,
                identity_norm="110",
                pair_role=_ROLE_B,
                signed_qty=-2.0,
            )
        ],
    )
    request = _request(partition={"terminal": False})
    mid = execute_paired_fill_reduce([first], request)
    assert mid["outgoing_carry"]["pending_identity"] == 110
    request_cont = _request(
        partition={
            "terminal": True,
            "incoming_carry": mid["outgoing_carry"],
        }
    )
    final = execute_paired_fill_reduce([second], request_cont)
    assert final["ledger_rows"][0]["classification"] == "complete_pair"
    assert mid["outgoing_carry"]["pending_group_key"] == [110]


def test_identity_namespace_separates_same_numeric_id(tmp_path):
    rows = [
        {**_row(identity=5, identity_norm="5"), "book_id": "a"},
        {
            **_row(identity=5, identity_norm="5", pair_role=_ROLE_B),
            "book_id": "b",
        },
    ]
    result = execute_paired_fill_reduce(
        [_write(tmp_path / "p.parquet", rows)],
        _request(
            identity_mapping={
                "identity_source_column": "identity",
                "identity_normalized_column": "identity_norm",
                "namespace_columns": ["book_id"],
            }
        ),
    )
    assert len(result["ledger_rows"]) == 2
    assert all(item["classification"] == "singleton" for item in result["ledger_rows"])
    assert {item["identity_namespace"]["book_id"] for item in result["ledger_rows"]} == {
        "a",
        "b",
    }


def test_side_size_signed_execution_flip(tmp_path):
    pair_mapping = {
        "pair_role_column": "pair_role",
        "aggressor_role_value": _ROLE_A,
        "passive_role_value": _ROLE_B,
        "measurement_core_fields": ["core_price", "core_size"],
        "start_position_column": "start_pos",
        "side_size_signed_execution": {
            "schema_version": "pbe.replay.side-size-signed-execution.v1",
            "side_column": "side",
            "size_column": "trade_size",
            "buy_side_value": "BUY",
            "sell_side_value": "SELL",
        },
    }
    request = _request(pair_mapping=pair_mapping)
    rows = [
        {
            **_row(identity=12, identity_norm="12", start_pos=10.0),
            "side": "SELL",
            "trade_size": 15.0,
        },
        {
            **_row(
                identity=12,
                identity_norm="12",
                pair_role=_ROLE_B,
                start_pos=-5.0,
            ),
            "side": "BUY",
            "trade_size": 5.0,
        },
    ]
    result = execute_paired_fill_reduce(
        [_write(tmp_path / "p.parquet", rows)], request
    )
    aggressor = _participant(result["ledger_rows"][0], "aggressor")
    assert aggressor["signed_execution"] == -15.0
    assert aggressor["closing_quantity"] == 10.0
    assert aggressor["opening_quantity"] == 5.0


def test_participant_field_passthrough_including_null(tmp_path):
    pair_mapping = {
        "pair_role_column": "pair_role",
        "aggressor_role_value": _ROLE_A,
        "passive_role_value": _ROLE_B,
        "measurement_core_fields": ["core_price", "core_size"],
        "start_position_column": "start_pos",
        "signed_execution_column": "signed_qty",
        "participant_field_bindings": [
            {"output_field": "audit_ref", "source_column": "audit_ref"},
        ],
    }
    result = execute_paired_fill_reduce(
        [
            _write(
                tmp_path / "p.parquet",
                [
                    {**_row(identity=13, identity_norm="13"), "audit_ref": "x1"},
                    {
                        **_row(
                            identity=13,
                            identity_norm="13",
                            pair_role=_ROLE_B,
                            signed_qty=-2.0,
                        ),
                        "audit_ref": None,
                    },
                ],
            )
        ],
        _request(pair_mapping=pair_mapping),
    )
    parts = result["ledger_rows"][0]["participants"]
    refs = {item["role"]: item["audit_ref"] for item in parts}
    assert refs["aggressor"] == "x1"
    assert refs["passive"] is None


def test_complete_pair_lineage_and_same_participant_flag(tmp_path):
    first = _write(
        tmp_path / "a.parquet",
        [
            {
                **_row(identity=14, identity_norm="14", start_pos=0.0, signed_qty=1.0),
                "participant_key": "p1",
            }
        ],
    )
    second = _write(
        tmp_path / "b.parquet",
        [
            {
                **_row(
                    identity=14,
                    identity_norm="14",
                    pair_role=_ROLE_B,
                    start_pos=0.0,
                    signed_qty=-1.0,
                ),
                "participant_key": "p1",
            }
        ],
    )
    pair_mapping = {
        "pair_role_column": "pair_role",
        "aggressor_role_value": _ROLE_A,
        "passive_role_value": _ROLE_B,
        "measurement_core_fields": ["core_price", "core_size"],
        "start_position_column": "start_pos",
        "signed_execution_column": "signed_qty",
        "participant_identity_column": "participant_key",
    }
    result = execute_paired_fill_reduce(
        [first, second], _request(pair_mapping=pair_mapping)
    )
    row = result["ledger_rows"][0]
    assert row["first_source_input_index"] == 0
    assert row["first_source_row_offset"] == 0
    assert row["last_source_input_index"] == 1
    assert row["last_source_row_offset"] == 0
    assert row["participants_same_identity"] is True


def test_request_accepts_frozen_upper_bounds():
    PairedFillReduceRequest.model_validate(
        {
            **_request(),
            "max_input_files": PAIRED_FILL_MAX_INPUT_FILES,
            "max_input_bytes": PAIRED_FILL_MAX_INPUT_BYTES,
            "max_output_rows": PAIRED_FILL_MAX_OUTPUT_ROWS,
            "max_output_bytes": PAIRED_FILL_MAX_OUTPUT_BYTES,
            "max_exception_rows": PAIRED_FILL_MAX_EXCEPTION_ROWS,
        }
    )


def test_request_rejects_bounds_above_ceiling():
    with pytest.raises(ValidationError):
        PairedFillReduceRequest.model_validate(
            {**_request(), "max_input_files": PAIRED_FILL_MAX_INPUT_FILES + 1}
        )


def test_input_bytes_limit_enforced(tmp_path):
    path = _write(tmp_path / "p.parquet", [_row(identity=15, identity_norm="15")])
    size = path.stat().st_size
    with pytest.raises(ValueError, match="input bytes exceed limit"):
        execute_paired_fill_reduce(
            [path],
            _request(max_input_bytes=max(1, size - 1)),
        )


def test_output_bytes_limit_enforced(tmp_path):
    with pytest.raises(ValueError, match="parquet output exceeds byte limit"):
        _run(
            tmp_path,
            [
                _row(identity=16, identity_norm="16"),
                _row(
                    identity=16,
                    identity_norm="16",
                    pair_role=_ROLE_B,
                    signed_qty=-2.0,
                ),
            ],
            max_output_bytes=32,
        )


def test_signed_execution_mapping_requires_exactly_one_mode():
    with pytest.raises(ValidationError):
        PairedFillReduceRequest.model_validate(
            _request(
                pair_mapping={
                    "pair_role_column": "pair_role",
                    "aggressor_role_value": _ROLE_A,
                    "passive_role_value": _ROLE_B,
                    "measurement_core_fields": ["core_price", "core_size"],
                    "start_position_column": "start_pos",
                    "signed_execution_column": "signed_qty",
                    "side_size_signed_execution": {
                        "schema_version": "pbe.replay.side-size-signed-execution.v1",
                        "side_column": "side",
                        "size_column": "trade_size",
                        "buy_side_value": "BUY",
                        "sell_side_value": "SELL",
                    },
                }
            )
        )
