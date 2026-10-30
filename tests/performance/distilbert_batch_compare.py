"""Optional local timing/memory probe for DistilBERT pair-score row bounds (not CI)."""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    INPUT_SCHEMA_VERSION,
    execute_distilbert_pair_binary_scores,
)


def _tiny_model_bundle(tmp_path: Path, seed: int) -> tuple[bytes, bytes]:
    torch.manual_seed(seed)
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
    model = DistilBertForSequenceClassification(config)
    directory = tmp_path / f"model-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    return (directory / "config.json").read_bytes(), (directory / "model.safetensors").read_bytes()


def _synthetic_rows(row_count: int, *, seq_len: int) -> tuple[list[str], list[list[int]], list[list[int]]]:
    row_ids = [f"row-{index:04d}" for index in range(row_count)]
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for index in range(row_count):
        active_len = (index % (seq_len - 1)) + 1
        tokens = [(index + token + 3) % 127 + 1 for token in range(active_len)]
        tokens.extend([0] * (seq_len - active_len))
        mask = [1] * active_len + [0] * (seq_len - active_len)
        input_ids.append(tokens)
        attention_mask.append(mask)
    return row_ids, input_ids, attention_mask


def _run_once(
    row_count: int,
    *,
    model_a: tuple[bytes, bytes],
    model_b: tuple[bytes, bytes],
    seq_len: int,
) -> tuple[float, int]:
    row_ids, input_ids, attention_mask = _synthetic_rows(row_count, seq_len=seq_len)
    payload = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "row_ids": row_ids,
        "model_a": {"input_ids": input_ids, "attention_mask": attention_mask},
        "model_b": {"input_ids": input_ids, "attention_mask": attention_mask},
    }
    tracemalloc.start()
    start = time.perf_counter()
    result = execute_distilbert_pair_binary_scores(
        payload,
        model_a_config=model_a[0],
        model_a_weights=model_a[1],
        model_b_config=model_b[0],
        model_b_weights=model_b[1],
    )
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(result["model_a_scores"]) == row_count
    assert len(result["model_b_scores"]) == row_count
    return elapsed, peak


def main() -> None:
    torch.set_num_threads(int(os.environ.get("PBE_TORCH_NUM_THREADS", "2")))
    tmp = Path(os.environ.get("PBE_PERF_TMP", ".distilbert-perf-tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    model_a = _tiny_model_bundle(tmp, 91)
    model_b = _tiny_model_bundle(tmp, 92)
    seq_len = int(os.environ.get("PBE_PERF_SEQ_LEN", "8"))
    print(f"torch_num_threads={torch.get_num_threads()} seq_len={seq_len}")
    for row_count in (1024, 2048):
        elapsed, peak = _run_once(
            row_count,
            model_a=model_a,
            model_b=model_b,
            seq_len=seq_len,
        )
        print(f"rows={row_count} elapsed_s={elapsed:.3f} peak_tracemalloc_bytes={peak}")


if __name__ == "__main__":
    main()
