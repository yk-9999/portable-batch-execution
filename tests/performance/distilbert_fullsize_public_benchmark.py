"""Public-runner DistilBERT pair-score benchmark (synthetic models and tensors only)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    _infer_class1_probabilities,
    _stage_model_bundle,
    execute_distilbert_pair_binary_scores,
)

GLOBAL_ROW_TOTAL = 2048
_BENCH_SEQ_LEN = 128
TINY_MODEL_VOCAB_SIZE = 128
BENCHMARK_SYNTHETIC_MODEL_A_SEED = 8_100_000
BENCHMARK_SYNTHETIC_MODEL_B_SEED = 8_100_001
_ACTIVE_LEN_MIN = 24
_ACTIVE_LEN_MAX = 96

_FULLSIZE_ARCH: dict[str, int | str] = {
    "family": "distilbert",
    "architectures": "DistilBertForSequenceClassification",
    "n_layers": 6,
    "dim": 768,
    "n_heads": 12,
    "hidden_dim": 3072,
    "vocab_size": 32000,
    "num_labels": 2,
    "max_position_embeddings": 512,
}

Mode = Literal["sequential", "model_parallel"]


def benchmark_synthetic_model_seed_pair() -> tuple[int, int]:
    """Fixed seeds shared by every partition and execution mode for comparable runs."""
    return BENCHMARK_SYNTHETIC_MODEL_A_SEED, BENCHMARK_SYNTHETIC_MODEL_B_SEED


def benchmark_synthetic_vocab_size(*, tiny_models: bool) -> int:
    if tiny_models:
        return TINY_MODEL_VOCAB_SIZE
    return int(_FULLSIZE_ARCH["vocab_size"])


@dataclass(frozen=True)
class ModelBundleMeta:
    config_bytes: bytes
    weights_bytes: bytes
    config_bytes_len: int
    weights_bytes_len: int
    total_bytes: int
    creation_wall_s: float


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _active_length(global_index: int) -> int:
    span = _ACTIVE_LEN_MAX - _ACTIVE_LEN_MIN + 1
    return _ACTIVE_LEN_MIN + (global_index * 37 + 11) % span


def generate_global_synthetic_rows(
    *,
    total_rows: int = GLOBAL_ROW_TOTAL,
    seq_len: int = _BENCH_SEQ_LEN,
    vocab_size: int = 32000,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    token_modulus = vocab_size - 1
    if seq_len > 128:
        raise ValueError("benchmark seq_len must be <= 128 for padded tensors")
    row_ids: list[str] = []
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for global_index in range(total_rows):
        active_len = _active_length(global_index)
        if active_len > seq_len:
            raise ValueError("active length exceeds seq_len")
        row_ids.append(f"bench-row-{global_index:05d}")
        tokens = [
            (global_index * 17 + token * 13 + 5) % token_modulus + 1
            for token in range(active_len)
        ]
        tokens.extend([0] * (seq_len - active_len))
        mask = [1] * active_len + [0] * (seq_len - active_len)
        input_ids.append(tokens)
        attention_mask.append(mask)
    return row_ids, input_ids, attention_mask


def slice_synthetic_partition(
    offset: int,
    row_count: int,
    *,
    total_rows: int = GLOBAL_ROW_TOTAL,
    vocab_size: int = 32000,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    if offset < 0 or row_count < 1 or offset + row_count > total_rows:
        raise ValueError("invalid partition for global synthetic rows")
    row_ids, input_ids, attention_mask = generate_global_synthetic_rows(
        total_rows=total_rows,
        vocab_size=vocab_size,
    )
    end = offset + row_count
    return row_ids[offset:end], input_ids[offset:end], attention_mask[offset:end]


def _build_model_bundle(
    *,
    seed: int,
    tiny: bool,
) -> ModelBundleMeta:
    start = time.perf_counter()
    torch.manual_seed(seed)
    if tiny:
        config = DistilBertConfig(
            vocab_size=128,
            dim=32,
            hidden_dim=64,
            n_heads=2,
            n_layers=1,
            max_position_embeddings=256,
            num_labels=2,
            architectures=["DistilBertForSequenceClassification"],
        )
    else:
        config = DistilBertConfig(
            vocab_size=int(_FULLSIZE_ARCH["vocab_size"]),
            dim=int(_FULLSIZE_ARCH["dim"]),
            hidden_dim=int(_FULLSIZE_ARCH["hidden_dim"]),
            n_heads=int(_FULLSIZE_ARCH["n_heads"]),
            n_layers=int(_FULLSIZE_ARCH["n_layers"]),
            max_position_embeddings=int(_FULLSIZE_ARCH["max_position_embeddings"]),
            num_labels=int(_FULLSIZE_ARCH["num_labels"]),
            architectures=["DistilBertForSequenceClassification"],
        )
    model = DistilBertForSequenceClassification(config)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        model.save_pretrained(directory, safe_serialization=True)
        config_bytes = (directory / "config.json").read_bytes()
        weights_bytes = (directory / "model.safetensors").read_bytes()
    creation_wall_s = time.perf_counter() - start
    return ModelBundleMeta(
        config_bytes=config_bytes,
        weights_bytes=weights_bytes,
        config_bytes_len=len(config_bytes),
        weights_bytes_len=len(weights_bytes),
        total_bytes=len(config_bytes) + len(weights_bytes),
        creation_wall_s=creation_wall_s,
    )


def _payload_from_partition(
    row_ids: list[str],
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
) -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "row_ids": row_ids,
        "model_a": {"input_ids": input_ids, "attention_mask": attention_mask},
        "model_b": {"input_ids": input_ids, "attention_mask": attention_mask},
    }


def run_sequential_pair_scores(
    payload: dict[str, Any],
    *,
    model_a: ModelBundleMeta,
    model_b: ModelBundleMeta,
) -> tuple[dict[str, Any], float]:
    torch.set_num_threads(4)
    start = time.perf_counter()
    result = execute_distilbert_pair_binary_scores(
        payload,
        model_a_config=model_a.config_bytes,
        model_a_weights=model_a.weights_bytes,
        model_b_config=model_b.config_bytes,
        model_b_weights=model_b.weights_bytes,
    )
    infer_wall_s = time.perf_counter() - start
    return result, infer_wall_s


def run_model_parallel_pair_scores(
    payload: dict[str, Any],
    *,
    model_a: ModelBundleMeta,
    model_b: ModelBundleMeta,
) -> tuple[dict[str, Any], float]:
    from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
        validate_distilbert_pair_binary_scores_input,
    )

    validated = validate_distilbert_pair_binary_scores_input(payload)
    model_a_ids, model_a_mask = validated["model_a"]
    model_b_ids, model_b_mask = validated["model_b"]
    torch.set_num_threads(2)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_a_dir = root / "model_a"
        model_b_dir = root / "model_b"
        model_a_dir.mkdir()
        model_b_dir.mkdir()
        _stage_model_bundle(model_a.config_bytes, model_a.weights_bytes, model_a_dir)
        _stage_model_bundle(model_b.config_bytes, model_b.weights_bytes, model_b_dir)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                _infer_class1_probabilities, model_a_dir, model_a_ids, model_a_mask
            )
            future_b = executor.submit(
                _infer_class1_probabilities, model_b_dir, model_b_ids, model_b_mask
            )
            model_a_scores = future_a.result()
            model_b_scores = future_b.result()
    infer_wall_s = time.perf_counter() - start
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "row_ids": list(validated["row_ids"]),
        "model_a_scores": model_a_scores,
        "model_b_scores": model_b_scores,
    }
    return result, infer_wall_s


def validate_benchmark_output(
    result: dict[str, Any],
    *,
    expected_row_ids: list[str],
    row_count: int,
) -> None:
    if result["row_ids"] != expected_row_ids:
        raise ValueError(
            "output row_ids differ from expected partition slice "
            f"(expected {row_count}, got {len(result['row_ids'])})"
        )
    model_a_len = len(result["model_a_scores"])
    if model_a_len != row_count:
        raise ValueError(
            f"model_a_scores length {model_a_len} != row_count {row_count}"
        )
    model_b_len = len(result["model_b_scores"])
    if model_b_len != row_count:
        raise ValueError(
            f"model_b_scores length {model_b_len} != row_count {row_count}"
        )


def scores_digest(result: dict[str, Any]) -> str:
    payload = {
        "row_ids": result["row_ids"],
        "model_a_scores": result["model_a_scores"],
        "model_b_scores": result["model_b_scores"],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runner_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "github_runner_os": os.environ.get("RUNNER_OS"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_job": os.environ.get("GITHUB_JOB"),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource as resource_module
    except ImportError:
        return None
    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def run_benchmark_case(
    *,
    case_id: str,
    mode: Mode,
    offset: int,
    row_count: int,
    tiny_models: bool = False,
) -> dict[str, Any]:
    job_started_at = _utc_now_iso()
    cpu_start = time.process_time()
    total_start = time.perf_counter()
    status = "ok"
    error: str | None = None
    result: dict[str, Any] | None = None
    infer_wall_s = 0.0
    model_a_meta: ModelBundleMeta | None = None
    model_b_meta: ModelBundleMeta | None = None
    try:
        model_a_seed, model_b_seed = benchmark_synthetic_model_seed_pair()
        vocab_size = benchmark_synthetic_vocab_size(tiny_models=tiny_models)
        model_a_meta = _build_model_bundle(seed=model_a_seed, tiny=tiny_models)
        model_b_meta = _build_model_bundle(seed=model_b_seed, tiny=tiny_models)
        row_ids, input_ids, attention_mask = slice_synthetic_partition(
            offset, row_count, vocab_size=vocab_size
        )
        payload = _payload_from_partition(row_ids, input_ids, attention_mask)
        if mode == "sequential":
            result, infer_wall_s = run_sequential_pair_scores(
                payload, model_a=model_a_meta, model_b=model_b_meta
            )
        elif mode == "model_parallel":
            result, infer_wall_s = run_model_parallel_pair_scores(
                payload, model_a=model_a_meta, model_b=model_b_meta
            )
        else:
            raise ValueError(f"unsupported mode: {mode}")
        assert result is not None
        validate_benchmark_output(
            result,
            expected_row_ids=row_ids,
            row_count=row_count,
        )
    except Exception as exc:  # noqa: BLE001 — benchmark boundary logs failure JSON
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    total_wall_s = time.perf_counter() - total_start
    cpu_process_s = time.process_time() - cpu_start
    job_finished_at = _utc_now_iso()
    report: dict[str, Any] = {
        "benchmark": "distilbert_fullsize_public",
        "case_id": case_id,
        "partition_offset": offset,
        "row_count": row_count,
        "mode": mode,
        "torch_num_threads": torch.get_num_threads(),
        "runner": _runner_info(),
        "timestamps": {
            "job_started_at": job_started_at,
            "job_finished_at": job_finished_at,
        },
        "timing_wall_s": {
            "model_creation": (
                (model_a_meta.creation_wall_s + model_b_meta.creation_wall_s)
                if model_a_meta and model_b_meta
                else None
            ),
            "infer": infer_wall_s if status == "ok" else None,
            "total": total_wall_s,
        },
        "process_cpu_s": cpu_process_s,
        "peak_rss_bytes": _peak_rss_bytes(),
        "model_architecture": _FULLSIZE_ARCH if not tiny_models else {"tiny_smoke": True},
        "model_bytes": (
            {
                "model_a_total": model_a_meta.total_bytes,
                "model_b_total": model_b_meta.total_bytes,
                "model_a_creation_wall_s": model_a_meta.creation_wall_s,
                "model_b_creation_wall_s": model_b_meta.creation_wall_s,
            }
            if model_a_meta and model_b_meta
            else None
        ),
        "status": status,
        "error": error,
    }
    if result is not None:
        report["output"] = {
            "schema_version": result.get("schema_version"),
            "row_ids_len": len(result["row_ids"]),
            "model_a_scores_len": len(result["model_a_scores"]),
            "model_b_scores_len": len(result["model_b_scores"]),
            "digest_sha256": scores_digest(result),
        }
    return report


def main() -> None:
    case_id = os.environ.get("PBE_BENCH_CASE_ID", "1x2048_sequential")
    mode_raw = os.environ.get("PBE_BENCH_MODE", "sequential")
    if mode_raw not in ("sequential", "model_parallel"):
        raise SystemExit(f"invalid PBE_BENCH_MODE: {mode_raw}")
    mode: Mode = mode_raw  # type: ignore[assignment]
    offset = int(os.environ.get("PBE_BENCH_OFFSET", "0"))
    row_count = int(os.environ.get("PBE_BENCH_ROW_COUNT", "2048"))
    tiny = os.environ.get("PBE_BENCH_TINY", "").lower() in {"1", "true", "yes"}
    report = run_benchmark_case(
        case_id=case_id,
        mode=mode,
        offset=offset,
        row_count=row_count,
        tiny_models=tiny,
    )
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
