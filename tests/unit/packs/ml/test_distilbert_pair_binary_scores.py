import json
from pathlib import Path
from unittest import mock

import pytest

from portable_batch_execution.broker.planning import canonical_operation_params
from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    _INFER_SUBBATCH_MAX_ROWS,
    _MAX_ROWS,
    INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    _effective_subbatch_seq_len,
    execute_distilbert_pair_binary_scores,
    validate_distilbert_classifier_config,
    validate_distilbert_pair_binary_scores_input,
)

torch = pytest.importorskip("torch")
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
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
    directory.mkdir()
    model.save_pretrained(directory, safe_serialization=True)
    return (directory / "config.json").read_bytes(), (directory / "model.safetensors").read_bytes()


def _tensor_request(row_ids: list[str], input_ids: list[list[int]], attention_mask: list[list[int]]):
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "row_ids": row_ids,
        "model_a": {"input_ids": input_ids, "attention_mask": attention_mask},
        "model_b": {"input_ids": input_ids, "attention_mask": attention_mask},
    }


def _reference_scores(
    config_bytes: bytes,
    weights_bytes: bytes,
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
    tmp_path: Path,
) -> list[float]:
    directory = tmp_path / f"reference-model-{abs(hash(config_bytes))}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_bytes(config_bytes)
    (directory / "model.safetensors").write_bytes(weights_bytes)
    model = DistilBertForSequenceClassification.from_pretrained(
        directory, local_files_only=True
    )
    model.eval()
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids_tensor, attention_mask=mask_tensor).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    return [float(value) for value in probabilities.tolist()]


def _reference_scores_subbatched(
    config_bytes: bytes,
    weights_bytes: bytes,
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
    tmp_path: Path,
) -> list[float]:
    scores: list[float] = []
    row_count = len(input_ids)
    for start in range(0, row_count, _INFER_SUBBATCH_MAX_ROWS):
        end = min(start + _INFER_SUBBATCH_MAX_ROWS, row_count)
        batch_mask = attention_mask[start:end]
        effective_len = _effective_subbatch_seq_len(batch_mask)
        batch_ids = [row[:effective_len] for row in input_ids[start:end]]
        batch_mask = [row[:effective_len] for row in batch_mask]
        scores.extend(
            _reference_scores(
                config_bytes,
                weights_bytes,
                batch_ids,
                batch_mask,
                tmp_path,
            )
        )
    return scores


def _synthetic_rows(
    row_count: int,
    *,
    seq_len: int,
    pad_value: int = 0,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    row_ids = [f"row-{index:04d}" for index in range(row_count)]
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for index in range(row_count):
        active_len = (index % (seq_len - 1)) + 1
        tokens = [(index + token + 3) % 127 + 1 for token in range(active_len)]
        tokens.extend([pad_value] * (seq_len - active_len))
        mask = [1] * active_len + [0] * (seq_len - active_len)
        input_ids.append(tokens)
        attention_mask.append(mask)
    return row_ids, input_ids, attention_mask


def test_validate_input_row_bounds_128_129_512_513():
    base_ids = [[1, 2]]
    base_mask = [[1, 1]]
    for count in (128, 129, 512):
        row_ids = [f"r{index}" for index in range(count)]
        payload = _tensor_request(row_ids, base_ids * count, base_mask * count)
        validated = validate_distilbert_pair_binary_scores_input(payload)
        assert len(validated["row_ids"]) == count
    row_ids = [f"r{index}" for index in range(513)]
    payload = _tensor_request(row_ids, base_ids * 513, base_mask * 513)
    with pytest.raises(ValueError, match="row bound"):
        validate_distilbert_pair_binary_scores_input(payload)
    assert _MAX_ROWS == 512


def test_execute_large_requests_preserve_row_order_and_output_shape(tmp_path):
    model_a = _tiny_model_bundle(tmp_path, 31)
    model_b = _tiny_model_bundle(tmp_path, 32)
    for row_count in (129, 257, 512):
        row_ids, input_ids, attention_mask = _synthetic_rows(row_count, seq_len=12)
        payload = _tensor_request(row_ids, input_ids, attention_mask)
        result = execute_distilbert_pair_binary_scores(
            payload,
            model_a_config=model_a[0],
            model_a_weights=model_a[1],
            model_b_config=model_b[0],
            model_b_weights=model_b[1],
        )
        assert result["row_ids"] == row_ids
        assert len(result["model_a_scores"]) == row_count
        assert len(result["model_b_scores"]) == row_count


def test_execute_large_requests_match_subbatched_reference_scores(tmp_path):
    model_a = _tiny_model_bundle(tmp_path, 41)
    model_b = _tiny_model_bundle(tmp_path, 42)
    for row_count in (129, 257, 512):
        row_ids, input_ids, attention_mask = _synthetic_rows(row_count, seq_len=16)
        payload = _tensor_request(row_ids, input_ids, attention_mask)
        result = execute_distilbert_pair_binary_scores(
            payload,
            model_a_config=model_a[0],
            model_a_weights=model_a[1],
            model_b_config=model_b[0],
            model_b_weights=model_b[1],
        )
        expected_a = _reference_scores_subbatched(
            model_a[0], model_a[1], input_ids, attention_mask, tmp_path
        )
        expected_b = _reference_scores_subbatched(
            model_b[0], model_b[1], input_ids, attention_mask, tmp_path
        )
        assert result["model_a_scores"] == expected_a
        assert result["model_b_scores"] == expected_b


def test_effective_subbatch_seq_len_trims_trailing_padding_only():
    mask = [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    assert _effective_subbatch_seq_len(mask) == 3
    internal = [[1, 0, 1, 0, 0]]
    assert _effective_subbatch_seq_len(internal) == 3


def test_execute_global_padding_matches_per_chunk_reference_via_trimming(tmp_path):
    model_a = _tiny_model_bundle(tmp_path, 51)
    row_count = 129
    seq_len = 14
    row_ids, input_ids, attention_mask = _synthetic_rows(row_count, seq_len=seq_len)
    payload = _tensor_request(row_ids, input_ids, attention_mask)
    result = execute_distilbert_pair_binary_scores(
        payload,
        model_a_config=model_a[0],
        model_a_weights=model_a[1],
        model_b_config=model_a[0],
        model_b_weights=model_a[1],
    )
    expected = _reference_scores_subbatched(
        model_a[0], model_a[1], input_ids, attention_mask, tmp_path
    )
    assert result["model_a_scores"] == expected


def test_infer_uses_single_model_load_and_subbatch_forwards_at_most_128(tmp_path):
    model_a = _tiny_model_bundle(tmp_path, 61)
    row_ids, input_ids, attention_mask = _synthetic_rows(512, seq_len=20)
    payload = _tensor_request(row_ids, input_ids, attention_mask)
    forward_batch_sizes: list[int] = []
    original_from_pretrained = DistilBertForSequenceClassification.from_pretrained
    load_counts = {"model_a": 0, "model_b": 0}

    def counting_from_pretrained(directory, *args, **kwargs):
        model = original_from_pretrained(directory, *args, **kwargs)
        directory_name = Path(directory).name
        if directory_name == "model_a":
            load_counts["model_a"] += 1
        elif directory_name == "model_b":
            load_counts["model_b"] += 1

        original_forward = model.forward

        def forward_with_batch_size(*f_args, **f_kwargs):
            batch = f_kwargs.get("input_ids")
            if batch is None and f_args:
                batch = f_args[0]
            if batch is not None:
                forward_batch_sizes.append(int(batch.shape[0]))
            return original_forward(*f_args, **f_kwargs)

        model.forward = forward_with_batch_size
        return model

    with mock.patch.object(
        DistilBertForSequenceClassification,
        "from_pretrained",
        side_effect=counting_from_pretrained,
    ):
        execute_distilbert_pair_binary_scores(
            payload,
            model_a_config=model_a[0],
            model_a_weights=model_a[1],
            model_b_config=model_a[0],
            model_b_weights=model_a[1],
        )

    assert load_counts == {"model_a": 1, "model_b": 1}
    assert forward_batch_sizes
    assert max(forward_batch_sizes) <= _INFER_SUBBATCH_MAX_ROWS
    assert sum(forward_batch_sizes) == 512 * 2


def test_execute_at_128_rows_matches_prior_single_batch_semantics(tmp_path):
    row_ids, input_ids, attention_mask = _synthetic_rows(128, seq_len=10)
    model_a = _tiny_model_bundle(tmp_path, 71)
    model_b = _tiny_model_bundle(tmp_path, 72)
    payload = _tensor_request(row_ids, input_ids, attention_mask)
    result = execute_distilbert_pair_binary_scores(
        payload,
        model_a_config=model_a[0],
        model_a_weights=model_a[1],
        model_b_config=model_b[0],
        model_b_weights=model_b[1],
    )
    expected_a = _reference_scores(model_a[0], model_a[1], input_ids, attention_mask, tmp_path)
    expected_b = _reference_scores(model_b[0], model_b[1], input_ids, attention_mask, tmp_path)
    assert result["model_a_scores"] == expected_a
    assert result["model_b_scores"] == expected_b


def test_closed_operation_params_require_empty_dict():
    assert (
        canonical_operation_params("ml-batch", "ml.distilbert_pair_binary_scores", {})
        == {}
    )
    with pytest.raises(ValueError, match="closed operation parameters"):
        canonical_operation_params(
            "ml-batch", "ml.distilbert_pair_binary_scores", {"x": 1}
        )


def test_validate_input_rejects_extra_top_level_fields():
    payload = _tensor_request(["a"], [[1, 2]], [[1, 1]])
    payload["threshold"] = 0.5
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_distilbert_pair_binary_scores_input(payload)


def test_validate_input_rejects_sequence_length_out_of_bounds():
    long_row = list(range(257))
    payload = _tensor_request(["a"], [long_row], [[1] * 257])
    with pytest.raises(ValueError, match="sequence length"):
        validate_distilbert_pair_binary_scores_input(payload)


def test_validate_classifier_config_rejects_wrong_model_type():
    with pytest.raises(ValueError, match="model_type"):
        validate_distilbert_classifier_config({"model_type": "bert"})


def test_validate_classifier_config_rejects_arbitrary_architecture():
    with pytest.raises(ValueError, match="architectures"):
        validate_distilbert_classifier_config(
            {
                "model_type": "distilbert",
                "architectures": [
                    "DistilBertForSequenceClassification",
                    "BertForSequenceClassification",
                ],
            }
        )


def test_validate_classifier_config_rejects_id2label_with_wrong_label_count():
    with pytest.raises(ValueError, match="id2label"):
        validate_distilbert_classifier_config(
            {
                "model_type": "distilbert",
                "id2label": {"0": "neg", "1": "pos", "2": "extra"},
            }
        )


def test_saved_real_config_json_without_num_labels_passes_raw_validation(tmp_path):
    config_bytes, _ = _tiny_model_bundle(tmp_path, 7)
    raw = json.loads(config_bytes.decode("utf-8"))
    assert "num_labels" not in raw
    validate_distilbert_classifier_config(raw)


def test_execute_matches_reference_scores_and_preserves_row_order(tmp_path):
    row_ids = ["row-b", "row-a"]
    input_ids = [[5, 6, 7, 0], [8, 9, 10, 0]]
    attention_mask = [[1, 1, 1, 0], [1, 1, 1, 0]]
    model_a = _tiny_model_bundle(tmp_path, 11)
    model_b = _tiny_model_bundle(tmp_path, 22)
    payload = _tensor_request(row_ids, input_ids, attention_mask)
    result = execute_distilbert_pair_binary_scores(
        payload,
        model_a_config=model_a[0],
        model_a_weights=model_a[1],
        model_b_config=model_b[0],
        model_b_weights=model_b[1],
    )
    assert result["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert result["row_ids"] == row_ids
    expected_a = _reference_scores(model_a[0], model_a[1], input_ids, attention_mask, tmp_path)
    expected_b = _reference_scores(model_b[0], model_b[1], input_ids, attention_mask, tmp_path)
    assert result["model_a_scores"] == expected_a
    assert result["model_b_scores"] == expected_b
