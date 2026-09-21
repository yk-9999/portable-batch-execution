from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from portable_batch_execution.contracts import AdapterDescriptor
from portable_batch_execution.packs import (
    AcquisitionPack,
    MediaPack,
    MLPack,
    ReplayEvalPack,
    TabularPack,
)

PUBLIC_FIXTURES = Path(__file__).parents[2] / "fixtures" / "public"


def _rolling_rows(rows: list[dict], window: int, boundaries: list[tuple[int, int]]):
    pack = TabularPack()
    output = []
    for start, end in boundaries:
        halo_start = max(0, start - (window - 1))
        result = pack.run(
            "tabular.rolling",
            rows[halo_start:end],
            {
                "column": "value",
                "window_size": window,
                "output_column": "rolling",
            },
        ).to_dicts()
        output.extend(result[start - halo_start :])
    return output


def _pit(observations: list[dict], facts: list[dict]) -> list[dict]:
    """Fixture oracle for a point-in-time join: latest fact at or before each observation."""
    result = []
    for observation in observations:
        matches = [
            fact
            for fact in facts
            if fact["entity"] == observation["entity"]
            and fact["at"] <= observation["at"]
        ]
        result.append(
            {
                **observation,
                "value": max(matches, key=lambda fact: fact["at"])["value"]
                if matches
                else None,
            }
        )
    return result


def test_five_domain_packs_have_a_closed_functional_smoke():
    assert TabularPack().run(
        "tabular.sort", [{"id": 2}, {"id": 1}], {"by": [{"column": "id"}]}
    ).to_dicts() == [{"id": 1}, {"id": 2}]
    assert (
        AcquisitionPack().validate_params(
            "acquisition.incremental",
            {
                "url": "https://example.test/items",
                "incremental": {"cursor_field": "id"},
            },
        )["max_pages"]
        == 100
    )

    class Adapter:
        descriptor = AdapterDescriptor(
            adapter_id="acceptance-adapter",
            adapter_version="1",
            adapter_digest="sha256:" + "a" * 64,
            supported_operations=("replay_eval.replay",),
        )

        def validate_job(self, job):
            return None

        def execute(self, job, shard, params, context):
            return {"operation": job.operation, "params": params}

        def finalize(self, job, canonical_attempts, context):
            return canonical_attempts

    replay_job = SimpleNamespace(
        pack="replay-eval-batch",
        operation="replay_eval.replay",
        security_profile="offline",
    )
    assert ReplayEvalPack(Adapter()).execute(replay_job, "shard-0", {}, None) == {
        "operation": "replay_eval.replay",
        "params": {},
    }
    assert MLPack().execute(
        "ml.hashing_vectorizer", ["alpha", "beta"], n_features=16
    ).shape == (2, 16)
    assert MediaPack().overlap_remove(
        [{"start": 0, "end": 2}, {"start": 1, "end": 3}]
    ) == [{"start": 0, "end": 2}, {"start": 2, "end": 3}]


def test_rolling_monolithic_equals_sharded_halo_then_trim():
    rows = json.loads((PUBLIC_FIXTURES / "rolling-input.json").read_text())

    monolithic = (
        TabularPack()
        .run(
            "tabular.rolling",
            rows,
            {"column": "value", "window_size": 3, "output_column": "rolling"},
        )
        .to_dicts()
    )
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
    assert all(
        not any(word in path.read_text().lower() for word in prohibited)
        for path in files
    )
