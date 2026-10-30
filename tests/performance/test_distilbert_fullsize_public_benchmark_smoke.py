"""Smoke checks for public DistilBERT benchmark helpers (tiny models, <=64 rows)."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from tests.performance.distilbert_fullsize_public_benchmark import (
    BENCHMARK_SYNTHETIC_MODEL_A_SEED,
    BENCHMARK_SYNTHETIC_MODEL_B_SEED,
    TINY_MODEL_VOCAB_SIZE,
    _build_model_bundle,
    _payload_from_partition,
    benchmark_synthetic_model_seed_pair,
    benchmark_synthetic_vocab_size,
    run_benchmark_case,
    run_model_parallel_pair_scores,
    run_sequential_pair_scores,
    scores_digest,
    slice_synthetic_partition,
    validate_benchmark_output,
)


def test_benchmark_synthetic_model_seed_pair_is_fixed():
    assert benchmark_synthetic_model_seed_pair() == (
        BENCHMARK_SYNTHETIC_MODEL_A_SEED,
        BENCHMARK_SYNTHETIC_MODEL_B_SEED,
    )
    assert BENCHMARK_SYNTHETIC_MODEL_B_SEED == BENCHMARK_SYNTHETIC_MODEL_A_SEED + 1


def test_benchmark_synthetic_vocab_size_tiny_vs_full():
    assert benchmark_synthetic_vocab_size(tiny_models=True) == TINY_MODEL_VOCAB_SIZE
    assert benchmark_synthetic_vocab_size(tiny_models=False) == 32000


def test_build_model_bundle_is_deterministic_for_fixed_seed():
    first = _build_model_bundle(seed=BENCHMARK_SYNTHETIC_MODEL_A_SEED, tiny=True)
    second = _build_model_bundle(seed=BENCHMARK_SYNTHETIC_MODEL_A_SEED, tiny=True)
    assert first.weights_bytes == second.weights_bytes
    assert first.config_bytes == second.config_bytes


@pytest.mark.parametrize(
    ("case_id", "mode"),
    [
        ("1x2048_sequential", "sequential"),
        ("2x1024_sequential", "sequential"),
        ("1x2048_model_parallel", "model_parallel"),
        ("4x512_sequential", "sequential"),
    ],
)
def test_model_bundles_ignore_case_id_and_mode(case_id, mode):
    model_a, model_b = benchmark_synthetic_model_seed_pair()
    reference_a = _build_model_bundle(seed=model_a, tiny=True)
    reference_b = _build_model_bundle(seed=model_b, tiny=True)
    report = run_benchmark_case(
        case_id=case_id,
        mode=mode,
        offset=0,
        row_count=4,
        tiny_models=True,
    )
    assert report["status"] == "ok"
    assert report["model_bytes"]["model_a_total"] == reference_a.total_bytes
    assert report["model_bytes"]["model_b_total"] == reference_b.total_bytes


@pytest.mark.parametrize("offset", [0, 64, 512, 1024, 1536])
def test_model_bundles_ignore_partition_offset(offset):
    baseline = run_benchmark_case(
        case_id="offset_probe",
        mode="sequential",
        offset=0,
        row_count=4,
        tiny_models=True,
    )
    report = run_benchmark_case(
        case_id="offset_probe",
        mode="sequential",
        offset=offset,
        row_count=4,
        tiny_models=True,
    )
    assert report["status"] == "ok"
    assert report["model_bytes"]["model_a_total"] == baseline["model_bytes"]["model_a_total"]
    assert report["model_bytes"]["model_b_total"] == baseline["model_bytes"]["model_b_total"]


def test_tiny_synthetic_tokens_stay_within_vocab():
    _, input_ids, _ = slice_synthetic_partition(
        0, 2048, total_rows=2048, vocab_size=TINY_MODEL_VOCAB_SIZE
    )
    for row in input_ids:
        assert all(0 <= token < TINY_MODEL_VOCAB_SIZE for token in row)


def test_run_benchmark_case_tiny_uses_tiny_vocab_not_full():
    report = run_benchmark_case(
        case_id="vocab_probe",
        mode="sequential",
        offset=0,
        row_count=32,
        tiny_models=True,
    )
    assert report["status"] == "ok"
    _, input_ids, _ = slice_synthetic_partition(
        0, 32, total_rows=2048, vocab_size=TINY_MODEL_VOCAB_SIZE
    )
    assert max(max(row) for row in input_ids) < TINY_MODEL_VOCAB_SIZE
    assert report["model_architecture"] == {"tiny_smoke": True}


@pytest.mark.parametrize("offset", [0, 128, 512])
def test_run_benchmark_case_tiny_sequential_ok(offset):
    report = run_benchmark_case(
        case_id="1x2048_sequential",
        mode="sequential",
        offset=offset,
        row_count=16,
        tiny_models=True,
    )
    assert report["status"] == "ok"
    assert report["output"]["row_ids_len"] == 16


@pytest.mark.parametrize("offset", [0, 128, 512])
def test_run_benchmark_case_tiny_model_parallel_ok(offset):
    report = run_benchmark_case(
        case_id="1x2048_model_parallel",
        mode="model_parallel",
        offset=offset,
        row_count=16,
        tiny_models=True,
    )
    assert report["status"] == "ok"
    assert report["output"]["row_ids_len"] == 16


@pytest.mark.parametrize("offset", [0, 256, 768])
def test_run_benchmark_case_modes_match_at_tolerance(offset):
    sequential = run_benchmark_case(
        case_id="1x2048_sequential",
        mode="sequential",
        offset=offset,
        row_count=24,
        tiny_models=True,
    )
    parallel = run_benchmark_case(
        case_id="1x2048_model_parallel",
        mode="model_parallel",
        offset=offset,
        row_count=24,
        tiny_models=True,
    )
    assert sequential["status"] == parallel["status"] == "ok"
    assert (
        sequential["output"]["digest_sha256"]
        == parallel["output"]["digest_sha256"]
    )


def test_sequential_and_model_parallel_scores_match_tiny_partition():
    torch.set_num_threads(4)
    offset = 0
    row_count = 64
    model_a = _build_model_bundle(seed=701, tiny=True)
    model_b = _build_model_bundle(seed=702, tiny=True)
    row_ids, input_ids, attention_mask = slice_synthetic_partition(
        offset, row_count, total_rows=2048, vocab_size=128
    )
    payload = _payload_from_partition(row_ids, input_ids, attention_mask)
    sequential, _ = run_sequential_pair_scores(payload, model_a=model_a, model_b=model_b)
    parallel, _ = run_model_parallel_pair_scores(payload, model_a=model_a, model_b=model_b)
    assert sequential["row_ids"] == parallel["row_ids"] == row_ids
    for index, row_id in enumerate(row_ids):
        assert sequential["model_a_scores"][index] == pytest.approx(
            parallel["model_a_scores"][index], rel=0.0, abs=1e-5
        )
        assert sequential["model_b_scores"][index] == pytest.approx(
            parallel["model_b_scores"][index], rel=0.0, abs=1e-5
        )
    assert scores_digest(sequential) == scores_digest(parallel)


def test_partition_offsets_are_stable_slices():
    full_ids, _, _ = slice_synthetic_partition(0, 2048, total_rows=2048)
    part_a, _, _ = slice_synthetic_partition(0, 1024, total_rows=2048)
    part_b, _, _ = slice_synthetic_partition(1024, 1024, total_rows=2048)
    assert part_a + part_b == full_ids
    quarters = [
        slice_synthetic_partition(0, 512, total_rows=2048)[0],
        slice_synthetic_partition(512, 512, total_rows=2048)[0],
        slice_synthetic_partition(1024, 512, total_rows=2048)[0],
        slice_synthetic_partition(1536, 512, total_rows=2048)[0],
    ]
    merged: list[str] = []
    for chunk in quarters:
        merged.extend(chunk)
    assert merged == full_ids


def test_validate_benchmark_output_rejects_row_id_mismatch():
    row_ids, _, _ = slice_synthetic_partition(0, 4, total_rows=2048, vocab_size=128)
    result = {
        "row_ids": ["wrong-id"],
        "model_a_scores": [0.1],
        "model_b_scores": [0.2],
    }
    with pytest.raises(ValueError, match="row_ids differ"):
        validate_benchmark_output(result, expected_row_ids=row_ids, row_count=4)


def test_validate_benchmark_output_rejects_score_length_mismatch():
    row_ids, _, _ = slice_synthetic_partition(0, 4, total_rows=2048, vocab_size=128)
    result = {
        "row_ids": row_ids,
        "model_a_scores": [0.1, 0.2],
        "model_b_scores": [0.3] * len(row_ids),
    }
    with pytest.raises(ValueError, match="model_a_scores length"):
        validate_benchmark_output(result, expected_row_ids=row_ids, row_count=4)


def test_run_benchmark_case_errors_on_mocked_wrong_output_row_ids():
    bad_result = {
        "schema_version": "output-v1",
        "row_ids": ["bench-row-99999"],
        "model_a_scores": [0.5] * 8,
        "model_b_scores": [0.5] * 8,
    }

    def fake_sequential(*_args, **_kwargs):
        return bad_result, 0.01

    with patch(
        "tests.performance.distilbert_fullsize_public_benchmark.run_sequential_pair_scores",
        side_effect=fake_sequential,
    ):
        report = run_benchmark_case(
            case_id="1x2048_sequential",
            mode="sequential",
            offset=0,
            row_count=8,
            tiny_models=True,
        )
    assert report["status"] == "error"
    assert report["error"] is not None
    assert "ValueError" in report["error"]
    assert "row_ids differ" in report["error"]


def test_active_lengths_within_benchmark_bounds():
    _, input_ids, attention_mask = slice_synthetic_partition(0, 128, total_rows=2048)
    for ids_row, mask_row in zip(input_ids, attention_mask, strict=True):
        active = sum(mask_row)
        assert 24 <= active <= 96
        assert len(ids_row) == len(mask_row) == 128
        assert all(token >= 0 for token in ids_row)
        assert math.isfinite(active)
