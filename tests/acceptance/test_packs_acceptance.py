from __future__ import annotations

import json
from pathlib import Path

from portable_batch_execution.packs import acquisition, media, ml, replay_eval, tabular

PUBLIC_FIXTURES = Path(__file__).parents[2] / "fixtures" / "public"


def _rolling_rows(rows: list[dict], window: int, boundaries: list[tuple[int, int]]):
    output = []
    for start, end in boundaries:
        halo_start = max(0, start - (window - 1))
        result = tabular("tabular.rolling", rows[halo_start:end], column="value", window=window)
        output.extend(result[start - halo_start :])
    return output


def _pit(observations: list[dict], facts: list[dict]) -> list[dict]:
    """Fixture oracle for a point-in-time join: latest fact at or before each observation."""
    result = []
    for observation in observations:
        matches = [
            fact
            for fact in facts
            if fact["entity"] == observation["entity"] and fact["at"] <= observation["at"]
        ]
        result.append({**observation, "value": max(matches, key=lambda fact: fact["at"])["value"] if matches else None})
    return result


def test_five_domain_packs_have_a_closed_functional_smoke():
    assert tabular("tabular.sort", [{"id": 2}, {"id": 1}], by="id") == [{"id": 1}, {"id": 2}]
    assert acquisition("acquisition.incremental", [{"id": 1}, {"id": 2}], since=2) == [{"id": 2}]
    assert replay_eval("replay_eval.replay", [1.0, 3.0]) == {"count": 2, "mean": 2.0}
    assert ml("ml.hashing_vectorizer", ["alpha", "beta"]) == (2, 16)
    assert media("media.metadata", [{"start": 0, "end": 2}, {"start": 2, "end": 5}]) == {"segments": 2, "duration": 5}


def test_rolling_monolithic_equals_sharded_halo_then_trim():
    rows = json.loads((PUBLIC_FIXTURES / "rolling-input.json").read_text())

    monolithic = tabular("tabular.rolling", rows, column="value", window=3)
    sharded = _rolling_rows(rows, window=3, boundaries=[(0, 3), (3, 6), (6, 9)])

    assert sharded == monolithic


def test_pit_monolithic_equals_partition_sharded_fixture_result():
    fixture = json.loads((PUBLIC_FIXTURES / "pit-input.json").read_text())
    observations = fixture["observations"]
    facts = fixture["facts"]

    monolithic = _pit(observations, facts)
    sharded = [
        row
        for entity in ("a", "b")
        for row in _pit(
            [item for item in observations if item["entity"] == entity],
            [item for item in facts if item["entity"] == entity],
        )
    ]

    assert sharded == monolithic


def test_public_fixtures_do_not_contain_private_or_credential_material():
    prohibited = ("private", "secret", "token", "password", "credential", "sk-")
    files = sorted(PUBLIC_FIXTURES.glob("*.json"))

    assert files
    assert all(not any(word in path.read_text().lower() for word in prohibited) for path in files)
