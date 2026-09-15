from __future__ import annotations

from collections.abc import Callable
from typing import Any


def execute_wave(wave_id: str, execute: Callable[[str], Any]) -> Any:
    """Execute one already-planned wave through a caller-supplied closed handler.

    This deliberately accepts only a wave identifier; workflow dispatch never carries
    shell, Python, SQL, or arbitrary executable input.
    """
    if not wave_id or not isinstance(wave_id, str):
        raise ValueError("wave_id is required")
    return execute(wave_id)
