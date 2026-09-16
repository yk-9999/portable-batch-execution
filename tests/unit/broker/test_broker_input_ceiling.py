import json

import pytest

from portable_batch_execution.broker.config import (
    BrokerConfig,
    max_request_frame_bytes,
)

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_ABSOLUTE_MAX_INPUT_BYTES = 134_217_728
_DEFAULT_MAX_INPUT_BYTES = 1_048_576
_JSON_FRAME_OVERHEAD_BYTES = 65_536


def _write_config(tmp_path, max_input_bytes: int) -> BrokerConfig:
    path = tmp_path / "broker-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": max_input_bytes,
                "allowed_operations_by_uid": {"1000": [["tabular-batch", "tabular.sort"]]},
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def test_default_max_input_bytes_unchanged(tmp_path):
    path = tmp_path / "broker-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "allowed_operations_by_uid": {"1000": [["tabular-batch", "tabular.sort"]]},
            }
        ),
        encoding="utf-8",
    )
    config = BrokerConfig.load(path)
    assert config.max_input_bytes == _DEFAULT_MAX_INPUT_BYTES


def test_accepts_configured_128_mib_ceiling(tmp_path):
    config = _write_config(tmp_path, _ABSOLUTE_MAX_INPUT_BYTES)
    assert config.max_input_bytes == _ABSOLUTE_MAX_INPUT_BYTES


def test_rejects_above_128_mib_at_load(tmp_path):
    with pytest.raises(ValueError, match="broker absolute bound"):
        _write_config(tmp_path, _ABSOLUTE_MAX_INPUT_BYTES + 1)


def test_rejects_above_128_mib_for_frame_calculation():
    with pytest.raises(ValueError, match="broker absolute bound"):
        max_request_frame_bytes(_ABSOLUTE_MAX_INPUT_BYTES + 1)


def test_request_frame_bytes_bounded_at_128_mib():
    max_bytes = _ABSOLUTE_MAX_INPUT_BYTES
    encoded = ((max_bytes + 2) // 3) * 4
    expected = encoded + _JSON_FRAME_OVERHEAD_BYTES
    assert max_request_frame_bytes(max_bytes) == expected
    assert max_request_frame_bytes(max_bytes) < 200_000_000
