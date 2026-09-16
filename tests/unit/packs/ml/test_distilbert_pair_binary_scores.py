import json
from pathlib import Path

import pytest

from portable_batch_execution.broker.planning import canonical_operation_params
from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
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
