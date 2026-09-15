import pytest

from portable_batch_execution.worker import execute_wave


def test_execute_wave_delegates_closed_wave_identifier():
    assert execute_wave("wave-0000", lambda wave_id: {"wave_id": wave_id}) == {
        "wave_id": "wave-0000"
    }


def test_execute_wave_requires_identifier():
    with pytest.raises(ValueError, match="wave_id is required"):
        execute_wave("", lambda wave_id: wave_id)
